# -*- coding: utf-8 -*-
r"""


 YOLOX-S + Segmentation 정렬/crop + masked percentile + 배경제거 데이터셋용
ConvNeXt V1-Tiny + sex-specific 240-bin Label Distribution Learning.

DATA
----
G:\Project\sinra_cho\crop_yolo_seg_maskedp_bgremove_512
  csv\train.csv
  csv\validation.csv
  train\images
  validation\images

중요
----
- test는 이 학습 코드에서 전혀 읽지 않습니다.
- 입력 전처리: native segmentation mask 내부 p1~p99 정규화 -> 512 resize/pad -> 512 mask 약 3px dilation -> background=0
- normalized grayscale -> RGB 3채널 반복
- ConvNeXt V1-Tiny: convnext_tiny.fb_in1k
- 240 bins: 1~240 months
- sigma=10
- lambda_kl=0.025
- loss = raw-month MAE + lambda * KL(G || p)
- AdamW, LR 3e-4, weight_decay 0.15
- EMA 0.999
- warmup 2 epoch + cosine 20 epoch -> min_lr 1e-6
- train weak affine: rotation 5°, translate 3%, scale 0.97~1.03
- max epochs 100
- early stopping patience 12
- best_model.pt / last_model.pt 모두 저장
- --resume 으로 중단 후 이어서 학습 가능
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config
from tqdm import tqdm


# =============================================================================
# Defaults
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_DIR = PROJECT_ROOT

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_512_input"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "bone_age_ldl"
)

IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
)


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        # cuDNN benchmark 활성화
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()

    if requested == "auto":
        return torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(requested)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA를 요청했지만 "
            "torch.cuda.is_available()이 False입니다."
        )

    return device


def amp_context(
    device: torch.device,
    enabled: bool,
):
    if enabled and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=use_amp,
        )


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def standardize_dataframe(
    csv_path: Path,
) -> pd.DataFrame:

    dataframe = pd.read_csv(
        csv_path
    ).copy()

    normalized = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    def select_column(
        candidates: Sequence[str],
    ) -> str:
        for candidate in candidates:
            key = normalize_column_name(
                candidate
            )

            if key in normalized:
                return normalized[key]

        raise KeyError(
            f"{csv_path}에서 컬럼을 찾지 못했습니다.\n"
            f"후보={list(candidates)}\n"
            f"실제={list(dataframe.columns)}"
        )

    id_column = select_column(
        [
            "id",
            "image id",
            "case id",
            "imageid",
            "caseid",
        ]
    )

    age_column = select_column(
        [
            "boneage",
            "bone age",
            "bone age months",
            "bone age (months)",
            "ground truth bone age months",
            "ground truth bone age (months)",
        ]
    )

    sex_column = select_column(
        [
            "male",
            "sex",
            "gender",
        ]
    )

    ids = pd.to_numeric(
        dataframe[id_column],
        errors="coerce",
    )

    if ids.isna().any():
        extracted = (
            dataframe[id_column]
            .astype(str)
            .str.extract(
                r"(\d+)",
                expand=False,
            )
        )

        ids = ids.fillna(
            pd.to_numeric(
                extracted,
                errors="coerce",
            )
        )

    ages = pd.to_numeric(
        dataframe[age_column],
        errors="coerce",
    )

    def parse_male(value) -> float:
        if isinstance(value, str):
            text = value.strip().lower()

            if text in {
                "m", "male", "true",
                "1", "1.0", "남", "남자",
            }:
                return 1.0

            if text in {
                "f", "female", "false",
                "0", "0.0", "여", "여자",
            }:
                return 0.0

        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"성별 값을 변환할 수 없습니다: {value}"
            ) from error

        if numeric not in {0.0, 1.0}:
            raise ValueError(
                f"성별 값은 0 또는 1이어야 합니다: {value}"
            )

        return numeric

    output = pd.DataFrame(
        {
            "id": ids,
            "boneage": ages,
            "male": dataframe[
                sex_column
            ].map(parse_male),
        }
    )

    output = (
        output
        .dropna(
            subset=[
                "id",
                "boneage",
                "male",
            ]
        )
        .reset_index(drop=True)
    )

    output["id"] = (
        output["id"]
        .astype(int)
    )

    output["boneage"] = (
        output["boneage"]
        .astype(float)
    )

    output["male"] = (
        output["male"]
        .astype(float)
    )

    if output["id"].duplicated().any():
        duplicates = (
            output.loc[
                output["id"].duplicated(
                    keep=False
                ),
                "id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{csv_path}에 중복 ID가 있습니다: "
            f"{duplicates}"
        )

    return output


def build_image_index(
    image_dir: Path,
) -> Dict[int, Path]:

    if not image_dir.exists():
        raise FileNotFoundError(
            image_dir
        )

    index: Dict[int, Path] = {}

    for path in image_dir.iterdir():
        if (
            not path.is_file()
            or path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        try:
            image_id = int(
                path.stem
            )
        except ValueError:
            continue

        if image_id in index:
            raise ValueError(
                f"중복 이미지 ID={image_id}: "
                f"{index[image_id]} / {path}"
            )

        index[
            image_id
        ] = path

    if not index:
        raise FileNotFoundError(
            f"이미지를 찾지 못했습니다: {image_dir}"
        )

    return index


def attach_image_paths(
    dataframe: pd.DataFrame,
    image_dir: Path,
    split_name: str,
) -> pd.DataFrame:

    image_index = build_image_index(
        image_dir
    )

    output = dataframe.copy()

    output[
        "image_path"
    ] = output["id"].map(
        image_index
    )

    missing = output[
        output["image_path"].isna()
    ]

    if len(missing) > 0:
        raise FileNotFoundError(
            f"[{split_name}] CSV와 매칭되지 않는 이미지가 "
            f"{len(missing)}장 있습니다.\n"
            f"예시={missing['id'].head(10).tolist()}"
        )

    extra_count = (
        len(image_index)
        - output["image_path"].notna().sum()
    )

    print(
        f"[{split_name}] "
        f"CSV={len(output)}, "
        f"folder={len(image_index)}, "
        f"matched={output['image_path'].notna().sum()}, "
        f"folder-extra={extra_count}"
    )

    return output.reset_index(
        drop=True
    )


# =============================================================================
# Dataset
# =============================================================================

class BoneAgeLDLDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform,
        *,
        target_mean: float,
        target_std: float,
    ) -> None:

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.transform = transform
        self.target_mean = float(
            target_mean
        )
        self.target_std = float(
            target_std
        )

    def __len__(self) -> int:
        return len(
            self.dataframe
        )

    @staticmethod
    def load_image(
        image_path: Path,
    ) -> Image.Image:
        with Image.open(
            image_path
        ) as image:
            # raw grayscale -> 동일값 RGB 3채널
            return (
                image
                .convert("L")
                .convert("RGB")
            )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:

        row = self.dataframe.iloc[
            index
        ]

        age_month = float(
            row["boneage"]
        )

        normalized_target = (
            age_month - self.target_mean
        ) / self.target_std

        image = self.load_image(
            Path(
                row["image_path"]
            )
        )

        return {
            "image": self.transform(
                image
            ),
            "male": torch.tensor(
                [
                    float(
                        row["male"]
                    )
                ],
                dtype=torch.float32,
            ),
            # target normalization 통계 보존용.
            # LDL loss는 raw month "age"를 직접 사용.
            "target": torch.tensor(
                normalized_target,
                dtype=torch.float32,
            ),
            "age": torch.tensor(
                age_month,
                dtype=torch.float32,
            ),
            "id": torch.tensor(
                int(
                    row["id"]
                ),
                dtype=torch.long,
            ),
        }


# =============================================================================
# Model
# =============================================================================

class ConvNeXtTinyDistributionRegression(
    nn.Module
):
    def __init__(
        self,
        *,
        model_name: str,
        image_dim: int,
        sex_dim: int,
        fusion_dim: int,
        image_dropout: float,
        fusion_dropout: float,
        pretrained: bool,
        num_bins: int,
    ) -> None:

        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        backbone_dim = int(
            self.backbone.num_features
        )

        self.image_head = nn.Sequential(
            nn.Linear(
                backbone_dim,
                image_dim,
            ),
            nn.LayerNorm(
                image_dim
            ),
            nn.GELU(),
            nn.Dropout(
                image_dropout
            ),
        )

        self.sex_embedding = nn.Sequential(
            nn.Linear(
                1,
                sex_dim,
            ),
            nn.GELU(),
        )

        self.fusion_trunk = nn.Sequential(
            nn.Linear(
                image_dim + sex_dim,
                fusion_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                fusion_dropout
            ),
        )

        self.male_output = nn.Linear(
            fusion_dim,
            num_bins,
        )

        self.female_output = nn.Linear(
            fusion_dim,
            num_bins,
        )

        self.register_buffer(
            "age_bins",
            torch.arange(
                1,
                num_bins + 1,
                dtype=torch.float32,
            ),
        )

        self.backbone_feature_dim = (
            backbone_dim
        )

    def forward(
        self,
        image: torch.Tensor,
        male: torch.Tensor,
    ):

        image_feature = (
            self.image_head(
                self.backbone(
                    image
                )
            )
        )

        sex_feature = (
            self.sex_embedding(
                male
            )
        )

        shared_feature = (
            self.fusion_trunk(
                torch.cat(
                    [
                        image_feature,
                        sex_feature,
                    ],
                    dim=1,
                )
            )
        )

        male_logits = (
            self.male_output(
                shared_feature
            )
        )

        female_logits = (
            self.female_output(
                shared_feature
            )
        )

        gate = (
            male
            .reshape(-1, 1)
            .to(
                dtype=male_logits.dtype
            )
            .clamp(
                0.0,
                1.0,
            )
        )

        logits = (
            gate * male_logits
            + (1.0 - gate)
            * female_logits
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )

        pred_age = torch.sum(
            probability
            * self.age_bins.unsqueeze(
                0
            ),
            dim=1,
        )

        return (
            logits,
            probability,
            pred_age,
        )


# =============================================================================
# LDL Loss
# =============================================================================

class DistributionAgeLoss(nn.Module):
    def __init__(
        self,
        *,
        sigma: float,
        lambda_kl: float,
    ) -> None:

        super().__init__()

        self.sigma = float(
            sigma
        )

        self.lambda_kl = float(
            lambda_kl
        )

    def make_target_distribution(
        self,
        true_age: torch.Tensor,
        age_bins: torch.Tensor,
    ) -> torch.Tensor:

        distance = (
            age_bins.unsqueeze(0)
            - true_age.reshape(
                -1,
                1,
            )
        )

        target = torch.exp(
            -0.5
            * (
                distance
                / self.sigma
            ) ** 2
        )

        target = (
            target
            / target.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(
                1e-12
            )
        )

        return target

    def forward(
        self,
        logits: torch.Tensor,
        pred_age: torch.Tensor,
        true_age: torch.Tensor,
        age_bins: torch.Tensor,
    ):

        age_loss = F.l1_loss(
            pred_age,
            true_age,
        )

        target_dist = (
            self.make_target_distribution(
                true_age,
                age_bins,
            )
        )

        log_probability = (
            F.log_softmax(
                logits,
                dim=1,
            )
        )

        kl_loss = F.kl_div(
            log_probability,
            target_dist,
            reduction="batchmean",
        )

        total_loss = (
            age_loss
            + self.lambda_kl
            * kl_loss
        )

        return (
            total_loss,
            age_loss,
            kl_loss,
        )


# =============================================================================
# EMA
# =============================================================================

class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float,
    ) -> None:

        self.decay = float(
            decay
        )

        self.module = (
            copy.deepcopy(
                model
            )
            .eval()
        )

        for parameter in (
            self.module.parameters()
        ):
            parameter.requires_grad_(
                False
            )

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
    ) -> None:

        ema_state = (
            self.module.state_dict()
        )

        model_state = (
            model.state_dict()
        )

        for name, ema_value in (
            ema_state.items()
        ):
            source_value = (
                model_state[
                    name
                ].detach()
            )

            if (
                ema_value
                .is_floating_point()
            ):
                ema_value.mul_(
                    self.decay
                ).add_(
                    source_value.to(
                        device=ema_value.device,
                        dtype=ema_value.dtype,
                    ),
                    alpha=(
                        1.0
                        - self.decay
                    ),
                )

            else:
                ema_value.copy_(
                    source_value.to(
                        device=ema_value.device,
                    )
                )

    def state_dict(self):
        return (
            self.module.state_dict()
        )

    def load_state_dict(
        self,
        state_dict,
    ) -> None:
        self.module.load_state_dict(
            state_dict,
            strict=True,
        )


# =============================================================================
# Scheduler
# =============================================================================

def build_scheduler(
    optimizer,
    *,
    warmup_epochs: int,
    cosine_schedule_epochs: int,
    min_lr: float,
    base_lr: float,
):

    min_factor = min(
        float(min_lr)
        / float(base_lr),
        1.0,
    )

    def lr_lambda(
        epoch_index: int,
    ) -> float:

        epoch_number = (
            int(epoch_index)
            + 1
        )

        if (
            warmup_epochs > 0
            and epoch_number
            <= warmup_epochs
        ):
            if warmup_epochs == 1:
                return 1.0

            progress = (
                (epoch_number - 1)
                / (warmup_epochs - 1)
            )

            return (
                0.5
                + 0.5
                * progress
            )

        cosine_epoch = (
            epoch_number
            - warmup_epochs
        )

        if (
            cosine_epoch
            <= cosine_schedule_epochs
        ):
            progress = (
                cosine_epoch
                / cosine_schedule_epochs
            )

            cosine_factor = (
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * progress
                    )
                )
            )

            return (
                min_factor
                + (
                    1.0
                    - min_factor
                )
                * cosine_factor
            )

        return min_factor

    return (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    )


# =============================================================================
# Train / Eval
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    *,
    device,
    use_amp,
    ema_model,
):

    model.train()

    total_loss = 0.0
    total_age_loss = 0.0
    total_kl = 0.0
    total_abs_error = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
        ncols=125,
    )

    for batch in progress:
        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        male = batch[
            "male"
        ].to(
            device,
            non_blocking=True,
        )

        true_age = batch[
            "age"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context(
            device,
            use_amp,
        ):
            (
                logits,
                _,
                pred_age,
            ) = model(
                images,
                male,
            )

            (
                loss,
                age_loss,
                kl_loss,
            ) = criterion(
                logits,
                pred_age,
                true_age,
                model.age_bins,
            )

        if use_amp:
            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        ema_model.update(
            model
        )

        error = (
            pred_age.detach()
            - true_age
        )

        batch_size = int(
            images.size(0)
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        total_age_loss += (
            float(
                age_loss.detach().cpu()
            )
            * batch_size
        )

        total_kl += (
            float(
                kl_loss.detach().cpu()
            )
            * batch_size
        )

        total_abs_error += float(
            torch.abs(
                error
            )
            .sum()
            .detach()
            .cpu()
        )

        total_samples += (
            batch_size
        )

        progress.set_postfix(
            mae=(
                f"{total_abs_error / total_samples:.3f}"
            ),
            kl=(
                f"{total_kl / total_samples:.3f}"
            ),
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "age_loss": (
            total_age_loss
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
        "mae": (
            total_abs_error
            / total_samples
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    *,
    device,
    use_amp,
    description,
):

    model.eval()

    total_loss = 0.0
    total_age_loss = 0.0
    total_kl = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_samples = 0

    rows = []

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        ncols=125,
    )

    for batch in progress:
        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True,
        )

        male = batch[
            "male"
        ].to(
            device,
            non_blocking=True,
        )

        true_age = batch[
            "age"
        ].to(
            device,
            non_blocking=True,
        )

        with amp_context(
            device,
            use_amp,
        ):
            (
                logits,
                _,
                pred_age,
            ) = model(
                images,
                male,
            )

            (
                loss,
                age_loss,
                kl_loss,
            ) = criterion(
                logits,
                pred_age,
                true_age,
                model.age_bins,
            )

        errors = (
            pred_age
            - true_age
        )

        batch_size = int(
            images.size(0)
        )

        total_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        total_age_loss += (
            float(
                age_loss.detach().cpu()
            )
            * batch_size
        )

        total_kl += (
            float(
                kl_loss.detach().cpu()
            )
            * batch_size
        )

        total_abs_error += float(
            torch.abs(
                errors
            )
            .sum()
            .detach()
            .cpu()
        )

        total_sq_error += float(
            torch.square(
                errors
            )
            .sum()
            .detach()
            .cpu()
        )

        total_samples += (
            batch_size
        )

        ids = (
            batch["id"]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        males = (
            male
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .tolist()
        )

        predictions = (
            pred_age
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        truths = (
            true_age
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        signed_errors = (
            errors
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        for (
            image_id,
            sex,
            prediction,
            truth,
            error,
        ) in zip(
            ids,
            males,
            predictions,
            truths,
            signed_errors,
        ):
            rows.append(
                {
                    "id": int(
                        image_id
                    ),
                    "male": float(
                        sex
                    ),
                    "true_boneage": float(
                        truth
                    ),
                    "pred_boneage": float(
                        prediction
                    ),
                    "signed_error": float(
                        error
                    ),
                    "abs_error": float(
                        abs(error)
                    ),
                }
            )

        progress.set_postfix(
            mae=(
                f"{total_abs_error / total_samples:.3f}"
            ),
            kl=(
                f"{total_kl / total_samples:.3f}"
            ),
        )

    return (
        {
            "loss": (
                total_loss
                / total_samples
            ),
            "age_loss": (
                total_age_loss
                / total_samples
            ),
            "kl": (
                total_kl
                / total_samples
            ),
            "mae": (
                total_abs_error
                / total_samples
            ),
            "rmse": math.sqrt(
                total_sq_error
                / total_samples
            ),
        },
        pd.DataFrame(
            rows
        ),
    )


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    ema_model,
    optimizer,
    scheduler,
    scaler,
    best_val_mae: float,
    best_val_rmse: float,
    no_improvement: int,
    history,
    target_mean: float,
    target_std: float,
    args,
):

    torch.save(
        {
            "epoch": int(
                epoch
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "ema_model_state_dict": (
                ema_model.state_dict()
            ),
            "ema_decay": float(
                ema_model.decay
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "scaler_state_dict": (
                scaler.state_dict()
            ),
            "best_val_mae": float(
                best_val_mae
            ),
            "best_val_rmse": float(
                best_val_rmse
            ),
            "no_improvement": int(
                no_improvement
            ),
            "history": history,
            "target_mean": float(
                target_mean
            ),
            "target_std": float(
                target_std
            ),
            "config": vars(
                args
            ),
            "architecture": {
                "backbone": (
                    "convnext_tiny.fb_in1k"
                ),
                "image_head": (
                    "GAP -> Dense512 -> LayerNorm "
                    "-> GELU -> Dropout0.20"
                ),
                "sex_head": (
                    "Dense32 -> GELU"
                ),
                "fusion_head": (
                    "Dense128 -> GELU -> Dropout0.20 "
                    "-> sex-specific Male/Female Linear240"
                ),
                "distribution": {
                    "bins": "1~240 months",
                    "sigma": float(
                        args.sigma
                    ),
                    "lambda_kl": float(
                        args.lambda_kl
                    ),
                    "loss": (
                        "raw-month MAE "
                        "+ lambda * KL(G||p)"
                    ),
                },
            },
            "input_channels": {
                "R": "raw grayscale",
                "G": "raw grayscale",
                "B": "raw grayscale",
            },
            "test_used": False,
        },
        path,
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "ConvNeXt V1-Tiny LDL "
            "for aligned 512x512 hand X-ray dataset"
        )
    )

    parser.add_argument(
        "--base_dir",
        default=str(
            DEFAULT_BASE_DIR
        ),
    )

    parser.add_argument(
        "--dataset_dir",
        default=str(
            DEFAULT_DATASET_DIR
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--model_name",
        default=(
            "convnext_tiny.fb_in1k"
        ),
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--image_dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--sex_dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--fusion_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--image_dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--fusion_dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--head_lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--num_bins",
        type=int,
        default=240,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--cosine_schedule_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--rotation",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--translate",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.97,
    )

    parser.add_argument(
        "--scale_max",
        type=float,
        default=1.03,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "중단된 동일 실험의 last_model.pt"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser


# =============================================================================
# Main
# =============================================================================

def main():

    args = build_parser().parse_args()

    base_dir = Path(
        args.base_dir
    ).resolve()

    dataset_dir = Path(
        args.dataset_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    # 프로젝트 내부 torch cache 사용
    torch_home = (
        base_dir
        / ".torch_cache"
    )

    torch_home.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ[
        "TORCH_HOME"
    ] = str(
        torch_home
    )

    torch.hub.set_dir(
        str(
            torch_home
            / "hub"
        )
    )

    train_csv = (
        dataset_dir
        / "csv"
        / "train.csv"
    )

    val_csv = (
        dataset_dir
        / "csv"
        / "validation.csv"
    )

    train_dir = (
        dataset_dir
        / "train"
        / "images"
    )

    val_dir = (
        dataset_dir
        / "validation"
        / "images"
    )

    # test 경로를 의도적으로 정의하지 않음.

    for path in [
        dataset_dir,
        train_csv,
        val_csv,
        train_dir,
        val_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "best_model.pt"
    )

    last_path = (
        output_dir
        / "last_model.pt"
    )

    history_path = (
        output_dir
        / "history.csv"
    )

    val_pred_path = (
        output_dir
        / "validation_predictions_best.csv"
    )

    config_path = (
        output_dir
        / "config.json"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    if (
        args.resume is None
        and best_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"기존 best_model.pt가 있습니다:\n{best_path}\n\n"
            "새 학습이면 output_dir을 바꾸세요.\n"
            "동일 실험 재시작이면 --overwrite,\n"
            "중단 이어학습이면 --resume last_model.pt를 사용하세요."
        )

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    use_amp = bool(
        args.amp
        and device.type == "cuda"
    )

    train_df = standardize_dataframe(
        train_csv
    )

    val_df = standardize_dataframe(
        val_csv
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit은 1 이상이어야 합니다."
            )

        train_df = (
            train_df
            .iloc[
                :args.limit
            ]
            .reset_index(
                drop=True
            )
        )

        val_df = (
            val_df
            .iloc[
                :args.limit
            ]
            .reset_index(
                drop=True
            )
        )

    train_df = attach_image_paths(
        train_df,
        train_dir,
        "training",
    )

    val_df = attach_image_paths(
        val_df,
        val_dir,
        "validation",
    )

    # train/validation ID leakage 검사
    overlap = (
        set(
            train_df["id"]
        )
        & set(
            val_df["id"]
        )
    )

    if overlap:
        raise ValueError(
            "train/validation ID가 겹칩니다. "
            f"예시={sorted(overlap)[:10]}"
        )

    target_mean = float(
        train_df[
            "boneage"
        ].mean()
    )

    target_std = float(
        train_df[
            "boneage"
        ].std(
            ddof=1
        )
    )

    if target_std <= 0:
        raise ValueError(
            "target_std가 0 이하입니다."
        )

    # normalization config
    probe = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    data_config = (
        resolve_model_data_config(
            probe
        )
    )

    del probe

    mean = data_config.get(
        "mean",
        (
            0.5,
            0.5,
            0.5,
        ),
    )

    std = data_config.get(
        "std",
        (
            0.5,
            0.5,
            0.5,
        ),
    )

    train_transform = transforms.Compose(
        [
            # 입력 크기 고정
            transforms.Resize(
                (
                    args.image_size,
                    args.image_size,
                ),
                interpolation=(
                    InterpolationMode.BICUBIC
                ),
            ),
            transforms.RandomAffine(
                degrees=args.rotation,
                translate=(
                    args.translate,
                    args.translate,
                ),
                scale=(
                    args.scale_min,
                    args.scale_max,
                ),
                interpolation=(
                    InterpolationMode.BICUBIC
                ),
                fill=0,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    args.image_size,
                    args.image_size,
                ),
                interpolation=(
                    InterpolationMode.BICUBIC
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    train_dataset = BoneAgeLDLDataset(
        train_df,
        train_transform,
        target_mean=target_mean,
        target_std=target_std,
    )

    val_dataset = BoneAgeLDLDataset(
        val_df,
        eval_transform,
        target_mean=target_mean,
        target_std=target_std,
    )

    loader_kwargs = {
        "num_workers": (
            args.workers
        ),
        "pin_memory": (
            device.type == "cuda"
        ),
        "persistent_workers": (
            args.workers > 0
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=(
            args.eval_batch_size
        ),
        shuffle=False,
        **loader_kwargs,
    )

    model = (
        ConvNeXtTinyDistributionRegression(
            model_name=args.model_name,
            image_dim=args.image_dim,
            sex_dim=args.sex_dim,
            fusion_dim=args.fusion_dim,
            image_dropout=args.image_dropout,
            fusion_dropout=args.fusion_dropout,
            pretrained=True,
            num_bins=args.num_bins,
        )
        .to(
            device
        )
    )

    ema_model = ModelEMA(
        model,
        decay=args.ema_decay,
    )

    criterion = DistributionAgeLoss(
        sigma=args.sigma,
        lambda_kl=args.lambda_kl,
    )

    # backbone / head 분리
    backbone_params = list(
        model.backbone.parameters()
    )

    head_params = (
        list(
            model.image_head.parameters()
        )
        + list(
            model.sex_embedding.parameters()
        )
        + list(
            model.fusion_trunk.parameters()
        )
        + list(
            model.male_output.parameters()
        )
        + list(
            model.female_output.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    backbone_params
                ),
                "lr": (
                    args.backbone_lr
                ),
            },
            {
                "params": (
                    head_params
                ),
                "lr": (
                    args.head_lr
                ),
            },
        ],
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=(
            args.warmup_epochs
        ),
        cosine_schedule_epochs=(
            args.cosine_schedule_epochs
        ),
        min_lr=args.min_lr,
        base_lr=(
            args.backbone_lr
        ),
    )

    scaler = make_grad_scaler(
        use_amp
    )

    best_val_mae = float(
        "inf"
    )

    best_val_rmse = float(
        "inf"
    )

    no_improvement = 0
    history: List[
        Dict[str, float]
    ] = []

    start_epoch = 1

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    if args.resume is not None:
        resume_path = Path(
            args.resume
        ).resolve()

        if not resume_path.exists():
            raise FileNotFoundError(
                resume_path
            )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        ema_state = checkpoint.get(
            "ema_model_state_dict"
        )

        if ema_state is None:
            ema_model.load_state_dict(
                model.state_dict()
            )
        else:
            ema_model.load_state_dict(
                ema_state
            )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        best_val_mae = float(
            checkpoint.get(
                "best_val_mae",
                float("inf"),
            )
        )

        best_val_rmse = float(
            checkpoint.get(
                "best_val_rmse",
                float("inf"),
            )
        )

        no_improvement = int(
            checkpoint.get(
                "no_improvement",
                0,
            )
        )

        history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        print(
            f"[RESUME] {resume_path}"
        )

        print(
            f"[RESUME] next epoch = "
            f"{start_epoch}"
        )

    config = {
        **vars(args),
        "device_resolved": str(
            device
        ),
        "dataset_dir_resolved": str(
            dataset_dir
        ),
        "train_csv": str(
            train_csv
        ),
        "validation_csv": str(
            val_csv
        ),
        "train_images": str(
            train_dir
        ),
        "validation_images": str(
            val_dir
        ),
        "output_dir_resolved": str(
            output_dir
        ),
        "train_count": len(
            train_df
        ),
        "validation_count": len(
            val_df
        ),
        "target_mean": (
            target_mean
        ),
        "target_std": (
            target_std
        ),
        "normalize_mean": list(
            mean
        ),
        "normalize_std": list(
            std
        ),
        "pretrained_source": (
            "timm convnext_tiny.fb_in1k "
            "pretrained weight"
        ),
        "evaluation_weights": (
            "EMA"
        ),
        "test_used": False,
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "Experiment       :",
        output_dir.name,
    )
    print(
        "Device           :",
        device,
    )
    print(
        "Dataset          :",
        dataset_dir,
    )
    print(
        "Train images     :",
        train_dir,
    )
    print(
        "Val images       :",
        val_dir,
    )
    print(
        "Train / Val      :",
        len(train_df),
        "/",
        len(val_df),
    )
    print(
        "Input            :",
        "512x512 raw grayscale x3",
    )
    print(
        "Backbone         :",
        args.model_name,
    )
    print(
        "LDL              :",
        f"sigma={args.sigma}, "
        f"lambda={args.lambda_kl}, "
        f"bins={args.num_bins}",
    )
    print(
        "Optimizer        :",
        f"AdamW LR={args.backbone_lr:g}/{args.head_lr:g}, "
        f"WD={args.weight_decay:g}",
    )
    print(
        "EMA              :",
        args.ema_decay,
    )
    print(
        "Scheduler        :",
        f"warmup={args.warmup_epochs}, "
        f"cosine={args.cosine_schedule_epochs}, "
        f"min_lr={args.min_lr:g}",
    )
    print(
        "Augmentation     :",
        f"rot={args.rotation}°, "
        f"translate={args.translate}, "
        f"scale={args.scale_min}~{args.scale_max}",
    )
    print(
        "Max epoch        :",
        args.epochs,
    )
    print(
        "Early stopping   :",
        args.early_stopping,
    )
    print(
        "TEST             :",
        "NOT USED",
    )
    print(
        "Best checkpoint  :",
        best_path,
    )
    print(
        "Last checkpoint  :",
        last_path,
    )
    print("=" * 100)
    print()

    if start_epoch > args.epochs:
        raise ValueError(
            "resume epoch가 최대 epoch보다 큽니다."
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device=device,
            use_amp=use_amp,
            ema_model=ema_model,
        )

        (
            val_metrics,
            val_predictions,
        ) = evaluate(
            ema_model.module,
            val_loader,
            criterion,
            device=device,
            use_amp=use_amp,
            description="Validation",
        )

        # epoch 종료 후 scheduler.step()
        scheduler.step()

        current_backbone_lr = float(
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        current_head_lr = float(
            optimizer
            .param_groups[1][
                "lr"
            ]
        )

        epoch_minutes = (
            time.time()
            - epoch_start
        ) / 60.0

        improved = (
            val_metrics[
                "mae"
            ]
            < best_val_mae
        )

        if improved:
            best_val_mae = float(
                val_metrics[
                    "mae"
                ]
            )

            best_val_rmse = float(
                val_metrics[
                    "rmse"
                ]
            )

            no_improvement = 0

        else:
            no_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "backbone_lr": (
                    current_backbone_lr
                ),
                "head_lr": (
                    current_head_lr
                ),
                "train_loss": (
                    train_metrics[
                        "loss"
                    ]
                ),
                "train_mae": (
                    train_metrics[
                        "mae"
                    ]
                ),
                "train_age_loss": (
                    train_metrics[
                        "age_loss"
                    ]
                ),
                "train_kl": (
                    train_metrics[
                        "kl"
                    ]
                ),
                "val_loss": (
                    val_metrics[
                        "loss"
                    ]
                ),
                "val_mae": (
                    val_metrics[
                        "mae"
                    ]
                ),
                "val_age_loss": (
                    val_metrics[
                        "age_loss"
                    ]
                ),
                "val_kl": (
                    val_metrics[
                        "kl"
                    ]
                ),
                "val_rmse": (
                    val_metrics[
                        "rmse"
                    ]
                ),
                "epoch_minutes": (
                    epoch_minutes
                ),
            }
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        # 매 epoch last 저장
        save_checkpoint(
            last_path,
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_mae=(
                best_val_mae
            ),
            best_val_rmse=(
                best_val_rmse
            ),
            no_improvement=(
                no_improvement
            ),
            history=history,
            target_mean=(
                target_mean
            ),
            target_std=(
                target_std
            ),
            args=args,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lrB={current_backbone_lr:.2e} | "
            f"lrH={current_head_lr:.2e} | "
            f"train MAE={train_metrics['mae']:.3f} | "
            f"val MAE={val_metrics['mae']:.3f} | "
            f"val RMSE={val_metrics['rmse']:.3f} | "
            f"KL={val_metrics['kl']:.3f} | "
            f"{epoch_minutes:.1f} min"
        )

        if improved:
            save_checkpoint(
                best_path,
                epoch=epoch,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_mae=(
                    best_val_mae
                ),
                best_val_rmse=(
                    best_val_rmse
                ),
                no_improvement=(
                    no_improvement
                ),
                history=history,
                target_mean=(
                    target_mean
                ),
                target_std=(
                    target_std
                ),
                args=args,
            )

            val_predictions.to_csv(
                val_pred_path,
                index=False,
                encoding="utf-8-sig",
            )

            print(
                "  BEST 저장: "
                f"MAE={best_val_mae:.3f}, "
                f"RMSE={best_val_rmse:.3f}"
            )

        else:
            print(
                "  개선 없음: "
                f"{no_improvement}/"
                f"{args.early_stopping}"
            )

        if (
            no_improvement
            >= args.early_stopping
        ):
            print(
                "Early stopping"
            )
            break

    # -------------------------------------------------------------------------
    # Final best validation
    # -------------------------------------------------------------------------
    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    best_eval_state = (
        best_checkpoint.get(
            "ema_model_state_dict",
            best_checkpoint[
                "model_state_dict"
            ],
        )
    )

    model.load_state_dict(
        best_eval_state,
        strict=True,
    )

    (
        final_val_metrics,
        final_val_predictions,
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device=device,
        use_amp=use_amp,
        description=(
            "Final Validation"
        ),
    )

    final_val_predictions.to_csv(
        val_pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "best_epoch": int(
            best_checkpoint[
                "epoch"
            ]
        ),
        "best_validation_mae": float(
            final_val_metrics[
                "mae"
            ]
        ),
        "best_validation_rmse": float(
            final_val_metrics[
                "rmse"
            ]
        ),
        "evaluation_weights": (
            "EMA"
        ),
        "test_used": False,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "BEST EPOCH           =",
        summary[
            "best_epoch"
        ],
    )
    print(
        "BEST VALIDATION MAE  =",
        f"{summary['best_validation_mae']:.3f}",
        "months",
    )
    print(
        "BEST VALIDATION RMSE =",
        f"{summary['best_validation_rmse']:.3f}",
        "months",
    )
    print(
        "TEST USED            =",
        False,
    )
    print("=" * 100)


if __name__ == "__main__":
    main()