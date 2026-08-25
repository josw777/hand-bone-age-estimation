# -*- coding: utf-8 -*-
"""
train_hand_segmentation.py

Model
-----
DeepLabV3 + MobileNetV3-Large

Original training settings preserved
------------------------------------
- Input size: 640 x 640
- Batch size: 2
- Max epochs: 200
- Early stopping patience: 15
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau(mode="max", factor=0.5, patience=5, min_lr=1e-7)
- Loss: Cross Entropy + Dice Loss + 0.5 * Auxiliary CE
- Augmentation:
    rotation ±7 degrees
    brightness 0.90~1.10
    contrast   0.90~1.10
- Model selection: validation Dice
- AMP enabled on CUDA
- Resume from last checkpoint supported

Expected dataset structure
--------------------------
<PROJECT_ROOT>/
└─ data/
   └─ hand_segmentation/
      ├─ train/
      │  ├─ sample01.png
      │  ├─ sample01_mask.png
      │  └─ ...
      └─ valid/
         ├─ sample02.png
         ├─ sample02_mask.png
         └─ ...

Outputs
-------
<PROJECT_ROOT>/outputs/hand_segmentation/
├─ best.pth
├─ last.pth
├─ training_history.csv
├─ validation_predictions.csv
├─ summary.json
└─ hand_seg_crop512_traced.pt   # optional TorchScript export

Examples
--------
python src/segmentation/train_hand_segmentation.py

Resume:
python src/segmentation/train_hand_segmentation.py --resume

Custom paths:
python src/segmentation/train_hand_segmentation.py ^
    --data-root data/hand_segmentation ^
    --output-dir outputs/hand_segmentation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


# =============================================================================
# Defaults from the original notebook
# =============================================================================

IMG_SIZE = 640
BATCH_SIZE = 2

MAX_EPOCHS = 200
PATIENCE = 15

LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 2
SEED = 42

AUG_ROTATION = 7.0
AUG_BRIGHTNESS_MIN = 0.90
AUG_BRIGHTNESS_MAX = 1.10
AUG_CONTRAST_MIN = 0.90
AUG_CONTRAST_MAX = 1.10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TORCHSCRIPT_EXPORT_SIZE = 512


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root() -> Path:
    """
    Assumes this file is stored at:
      <PROJECT_ROOT>/src/segmentation/train_hand_segmentation.py
    """
    return Path(__file__).resolve().parents[2]


def find_image_for_mask(mask_path: Path) -> Path:
    base_name = mask_path.name[:-9]  # remove "_mask.png"

    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_path.parent / f"{base_name}{ext}"

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Source image not found for mask: {mask_path}"
    )


def check_split(split_dir: Path) -> List[Tuple[Path, Path]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset split directory not found: {split_dir}"
        )

    masks = sorted(split_dir.glob("*_mask.png"))

    if not masks:
        raise RuntimeError(
            f"No '*_mask.png' files found in: {split_dir}"
        )

    samples: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    size_mismatch: List[Tuple[str, tuple, tuple]] = []

    for mask_path in masks:
        try:
            image_path = find_image_for_mask(mask_path)
        except FileNotFoundError:
            missing.append(mask_path.name)
            continue

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_size = mask.size

        if image_size != mask_size:
            size_mismatch.append(
                (
                    image_path.name,
                    image_size,
                    mask_size,
                )
            )

        samples.append(
            (
                image_path,
                mask_path,
            )
        )

    print("=" * 72)
    print(f"Split          : {split_dir.name}")
    print(f"Mask count     : {len(masks)}")
    print(f"Valid pairs    : {len(samples)}")
    print(f"Missing images : {len(missing)}")
    print(f"Size mismatch  : {len(size_mismatch)}")

    if missing:
        print("Missing examples:", missing[:5])

    if size_mismatch:
        print("Size mismatch examples:", size_mismatch[:5])

    if missing or size_mismatch:
        raise RuntimeError(
            f"Dataset validation failed for split: {split_dir}"
        )

    return samples


# =============================================================================
# Dataset
# =============================================================================

def resize_with_padding(
    image: Image.Image,
    target_size: int,
    interpolation: InterpolationMode,
    fill: int = 0,
) -> Image.Image:
    """
    Aspect-ratio preserving resize + square center padding.
    """
    width, height = image.size

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = TF.resize(
        image,
        [new_height, new_width],
        interpolation=interpolation,
    )

    pad_width = target_size - new_width
    pad_height = target_size - new_height

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    return TF.pad(
        image,
        [left, top, right, bottom],
        fill=fill,
    )


class HandSegDataset(Dataset):
    def __init__(
        self,
        folder: Path,
        img_size: int = IMG_SIZE,
        augment: bool = False,
    ) -> None:
        self.folder = Path(folder)
        self.img_size = int(img_size)
        self.augment = bool(augment)

        self.samples = check_split(
            self.folder
        )

        print(
            f"{self.folder.name}: "
            f"{len(self.samples)} pairs "
            f"(augment={self.augment})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")

        # Letterbox to 640 x 640
        image = resize_with_padding(
            image,
            self.img_size,
            InterpolationMode.BILINEAR,
            fill=0,
        )

        mask = resize_with_padding(
            mask,
            self.img_size,
            InterpolationMode.NEAREST,
            fill=0,
        )

        if self.augment:
            angle = random.uniform(
                -AUG_ROTATION,
                AUG_ROTATION,
            )

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            brightness = random.uniform(
                AUG_BRIGHTNESS_MIN,
                AUG_BRIGHTNESS_MAX,
            )

            image = TF.adjust_brightness(
                image,
                brightness,
            )

            contrast = random.uniform(
                AUG_CONTRAST_MIN,
                AUG_CONTRAST_MAX,
            )

            image = TF.adjust_contrast(
                image,
                contrast,
            )

        image_tensor = TF.to_tensor(
            image
        )

        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        mask_array = np.asarray(
            mask,
            dtype=np.uint8,
        )

        mask_array = (
            mask_array > 0
        ).astype(np.int64)

        mask_tensor = torch.from_numpy(
            mask_array
        )

        return (
            image_tensor,
            mask_tensor,
        )


# =============================================================================
# Model
# =============================================================================

def build_model(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    weights = (
        DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )

    model = deeplabv3_mobilenet_v3_large(
        weights=weights
    )

    main_in_channels = (
        model.classifier[-1].in_channels
    )

    model.classifier[-1] = nn.Conv2d(
        main_in_channels,
        num_classes,
        kernel_size=1,
    )

    if model.aux_classifier is not None:
        aux_in_channels = (
            model.aux_classifier[-1].in_channels
        )

        model.aux_classifier[-1] = nn.Conv2d(
            aux_in_channels,
            num_classes,
            kernel_size=1,
        )

    return model


# =============================================================================
# Loss / metrics
# =============================================================================

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    hand_prob = torch.softmax(
        logits,
        dim=1,
    )[:, 1]

    hand_target = (
        targets == 1
    ).float()

    intersection = (
        hand_prob * hand_target
    ).sum(
        dim=(1, 2)
    )

    denominator = (
        hand_prob.sum(dim=(1, 2))
        + hand_target.sum(dim=(1, 2))
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return (
        1.0 - dice.mean()
    )


def segmentation_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    main_logits = outputs["out"]

    ce_loss = F.cross_entropy(
        main_logits,
        targets,
    )

    d_loss = dice_loss(
        main_logits,
        targets,
    )

    main_loss = (
        ce_loss + d_loss
    )

    if "aux" in outputs:
        aux_ce = F.cross_entropy(
            outputs["aux"],
            targets,
        )

        total_loss = (
            main_loss
            + 0.5 * aux_ce
        )
    else:
        total_loss = main_loss

    return (
        total_loss,
        ce_loss,
        d_loss,
    )


def update_confusion(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[int, int, int]:
    prediction = prediction.bool()
    target = target.bool()

    tp = (
        prediction & target
    ).sum().item()

    fp = (
        prediction & ~target
    ).sum().item()

    fn = (
        ~prediction & target
    ).sum().item()

    return (
        int(tp),
        int(fp),
        int(fn),
    )


def calc_dice_iou(
    tp: int,
    fp: int,
    fn: int,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    dice = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    return (
        float(dice),
        float(iou),
    )


# =============================================================================
# Train / validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                images
            )

            (
                loss,
                ce,
                d_loss,
            ) = segmentation_loss(
                outputs,
                masks,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += float(
            loss.item()
        )
        total_ce += float(
            ce.item()
        )
        total_dice_loss += float(
            d_loss.item()
        )

    n = max(
        1,
        len(loader),
    )

    return (
        total_loss / n,
        total_ce / n,
        total_dice_loss / n,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(
                images
            )

            (
                loss,
                _,
                _,
            ) = segmentation_loss(
                outputs,
                masks,
            )

        predictions = torch.argmax(
            outputs["out"],
            dim=1,
        )

        tp, fp, fn = update_confusion(
            predictions == 1,
            masks == 1,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_loss += float(
            loss.item()
        )

    val_loss = (
        total_loss
        / max(
            1,
            len(loader),
        )
    )

    dice, iou = calc_dice_iou(
        total_tp,
        total_fp,
        total_fn,
    )

    return (
        val_loss,
        dice,
        iou,
    )


# =============================================================================
# Per-image validation diagnostics
# =============================================================================

@torch.no_grad()
def evaluate_per_image(
    model: nn.Module,
    dataset: HandSegDataset,
    device: torch.device,
    use_amp: bool,
) -> List[dict]:
    model.eval()

    results: List[dict] = []

    for index in range(
        len(dataset)
    ):
        image, gt_mask = dataset[
            index
        ]

        x = image.unsqueeze(
            0
        ).to(
            device
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                x
            )["out"]

        pred_mask = torch.argmax(
            logits,
            dim=1,
        )[0].cpu()

        pred = (
            pred_mask == 1
        )

        gt = (
            gt_mask == 1
        )

        tp = (
            pred & gt
        ).sum().item()

        fp = (
            pred & ~gt
        ).sum().item()

        fn = (
            ~pred & gt
        ).sum().item()

        dice, iou = calc_dice_iou(
            int(tp),
            int(fp),
            int(fn),
        )

        image_path, _ = (
            dataset.samples[index]
        )

        results.append({
            "index": index,
            "filename": image_path.name,
            "dice": dice,
            "iou": iou,
        })

    return sorted(
        results,
        key=lambda row: row["dice"],
    )


# =============================================================================
# TorchScript export
# =============================================================================

class SegExportWrapper(nn.Module):
    """
    Returns only the main DeepLabV3 segmentation logits.
    """
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            x
        )

        if isinstance(
            output,
            dict,
        ):
            output = output["out"]

        return output


def export_torchscript(
    model: nn.Module,
    save_path: Path,
    device: torch.device,
    export_size: int = TORCHSCRIPT_EXPORT_SIZE,
) -> None:
    model.eval()

    wrapper = (
        SegExportWrapper(
            model
        )
        .to(device)
        .eval()
    )

    example = torch.randn(
        1,
        3,
        export_size,
        export_size,
        device=device,
    )

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            example,
            strict=False,
        )

    traced.save(
        str(save_path)
    )

    print(
        "TorchScript saved:",
        save_path,
    )


# =============================================================================
# File helpers
# =============================================================================

def save_history(
    path: Path,
    rows: List[dict],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def save_rows(
    path: Path,
    rows: List[dict],
) -> None:
    save_history(
        path,
        rows,
    )


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    root = project_root()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            root
            / "data"
            / "hand_segmentation"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "outputs"
            / "hand_segmentation"
        ),
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Original notebook used 0. "
            "Increase only after verifying multiprocessing behavior."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export the final best model to TorchScript.",
    )

    parser.add_argument(
        "--export-size",
        type=int,
        default=TORCHSCRIPT_EXPORT_SIZE,
        help=(
            "TorchScript tracing input size. "
            "Training uses 640 by default; the final inference pipeline used 512."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
    )

    data_root = (
        args.data_root.resolve()
    )

    train_dir = (
        data_root / "train"
    )

    valid_dir = (
        data_root / "valid"
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir / "best.pth"
    )

    last_path = (
        output_dir / "last.pth"
    )

    history_path = (
        output_dir
        / "training_history.csv"
    )

    validation_csv = (
        output_dir
        / "validation_predictions.csv"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    torchscript_path = (
        output_dir
        / "hand_seg_crop512_traced.pt"
    )

    print("=" * 80)
    print("Hand Segmentation Training")
    print("=" * 80)
    print("Data root    :", data_root)
    print("Train dir    :", train_dir)
    print("Valid dir    :", valid_dir)
    print("Output dir   :", output_dir)
    print("Device       :", device)

    if torch.cuda.is_available():
        print(
            "GPU          :",
            torch.cuda.get_device_name(
                0
            ),
        )

    print("Train size   :", args.img_size)
    print("Batch size   :", args.batch_size)
    print("Max epochs   :", args.max_epochs)
    print("Patience     :", args.patience)
    print("LR           :", args.lr)
    print("Weight decay :", args.weight_decay)
    print("AMP          :", use_amp)
    print("=" * 80)

    train_dataset = HandSegDataset(
        train_dir,
        img_size=args.img_size,
        augment=True,
    )

    valid_dataset = HandSegDataset(
        valid_dir,
        img_size=args.img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    print()
    print(
        "Train images :",
        len(train_dataset),
    )
    print(
        "Valid images :",
        len(valid_dataset),
    )

    model = build_model().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 0
    best_dice = 0.0
    bad_epochs = 0
    history: List[dict] = []

    if (
        args.resume
        and last_path.exists()
    ):
        checkpoint = torch.load(
            last_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        scaler.load_state_dict(
            checkpoint["scaler"]
        )

        start_epoch = (
            int(
                checkpoint["epoch"]
            )
            + 1
        )

        best_dice = float(
            checkpoint["best_dice"]
        )

        bad_epochs = int(
            checkpoint["bad_epochs"]
        )

        print(
            f"Resume from epoch "
            f"{start_epoch + 1}"
        )
    else:
        print("NEW TRAINING")

    for epoch in range(
        start_epoch,
        args.max_epochs,
    ):
        start_time = time.time()

        (
            train_loss,
            train_ce,
            train_dice_loss,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
        )

        (
            val_loss,
            val_dice,
            val_iou,
        ) = validate(
            model,
            valid_loader,
            device,
            use_amp,
        )

        scheduler.step(
            val_dice
        )

        current_lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        improved = (
            val_dice > best_dice
        )

        if improved:
            best_dice = val_dice
            bad_epochs = 0
        else:
            bad_epochs += 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_dice": best_dice,
            "bad_epochs": bad_epochs,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_loss": val_loss,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
        }

        torch.save(
            checkpoint,
            last_path,
        )

        if improved:
            torch.save(
                checkpoint,
                best_path,
            )

        elapsed = (
            time.time()
            - start_time
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_dice_loss": train_dice_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "lr": current_lr,
            "elapsed_sec": elapsed,
            "is_best": int(
                improved
            ),
        }

        history.append(
            row
        )

        save_history(
            history_path,
            history,
        )

        print(
            f"[{epoch + 1:03d}/{args.max_epochs}] "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Dice {val_dice:.5f} | "
            f"IoU {val_iou:.5f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if improved:
            print(
                f"    ★ BEST Dice: "
                f"{best_dice:.5f}"
            )

        if (
            bad_epochs
            >= args.patience
        ):
            print()
            print("Early stopping")
            print(
                "Best Dice:",
                f"{best_dice:.5f}",
            )
            break

    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_path}"
        )

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model"]
    )

    model.eval()

    print()
    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(
        "Best epoch :",
        int(
            best_checkpoint["epoch"]
        ) + 1,
    )
    print(
        "Best Dice  :",
        float(
            best_checkpoint["best_dice"]
        ),
    )
    print(
        "Best IoU   :",
        float(
            best_checkpoint["val_iou"]
        ),
    )

    validation_rows = (
        evaluate_per_image(
            model,
            valid_dataset,
            device,
            use_amp,
        )
    )

    save_rows(
        validation_csv,
        validation_rows,
    )

    dice_values = np.asarray(
        [
            row["dice"]
            for row in validation_rows
        ],
        dtype=np.float64,
    )

    iou_values = np.asarray(
        [
            row["iou"]
            for row in validation_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "model": (
            "DeepLabV3-MobileNetV3-Large"
        ),
        "train_input_size": args.img_size,
        "train_samples": len(
            train_dataset
        ),
        "validation_samples": len(
            valid_dataset
        ),
        "best_epoch": (
            int(
                best_checkpoint["epoch"]
            )
            + 1
        ),
        "global_validation_dice": float(
            best_checkpoint["best_dice"]
        ),
        "global_validation_iou": float(
            best_checkpoint["val_iou"]
        ),
        "mean_per_image_dice": float(
            dice_values.mean()
        ),
        "median_per_image_dice": float(
            np.median(
                dice_values
            )
        ),
        "min_per_image_dice": float(
            dice_values.min()
        ),
        "mean_per_image_iou": float(
            iou_values.mean()
        ),
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": (
            args.patience
        ),
        "loss": (
            "CrossEntropy + DiceLoss + "
            "0.5 * AuxiliaryCrossEntropy"
        ),
        "augmentation": {
            "rotation_degrees": [
                -AUG_ROTATION,
                AUG_ROTATION,
            ],
            "brightness": [
                AUG_BRIGHTNESS_MIN,
                AUG_BRIGHTNESS_MAX,
            ],
            "contrast": [
                AUG_CONTRAST_MIN,
                AUG_CONTRAST_MAX,
            ],
        },
    }

    if not args.no_export:
        export_torchscript(
            model,
            torchscript_path,
            device,
            export_size=args.export_size,
        )

        summary[
            "torchscript_export_size"
        ] = args.export_size

        summary[
            "torchscript_path"
        ] = str(
            torchscript_path
        )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Validation CSV:",
        validation_csv,
    )
    print(
        "Summary JSON :",
        summary_path,
    )
    print(
        "Best ckpt    :",
        best_path,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()