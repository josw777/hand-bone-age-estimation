# -*- coding: utf-8 -*-
r"""
train_male_bias_refinement.py

저연령 남아 구간의 과대예측 편향을 완화하기 위한 targeted fine-tuning.

목표
----
기본 골연령 모델의 shared representation과 Female prediction은 고정하고,
Male LDL head만 미세 조정하여 Male <= 60 months의 positive bias를 줄인다.

학습 전략
---------
1) 기본 모델의 best EMA weight에서 시작
2) 아래 모듈은 모두 freeze
   - backbone
   - image_head
   - sex_embedding
   - fusion_trunk
   - female_output
3) male_output만 학습
4) 학습 데이터
   - Main loader: train split의 모든 Male
   - Bias loader: train split의 Male <= 60 months
5) loss
   total =
       male_main_LDL
       + lambda_bias * ReLU(mean(pred_young - GT_young))^2
       + lambda_keep * mean(|pred_older - teacher_pred_older|)

기본 설정
---------
- lambda_bias: 0.005
- lambda_keep: 0.25
- male head learning rate: 1e-5
- max epochs: 100
- early stopping patience: 6
- min_delta: 0.003
- bias_min_delta: 0.10 months
- targeted best MAE constraint: baseline MAE + 0.02 months

저장
----
best_mae_model.pt
best_targeted_model.pt
last_model.pt
history.csv
validation_predictions_best_mae.csv
validation_predictions_best_targeted.csv
subgroup_best_mae.csv
subgroup_best_targeted.csv
summary.json

학습 및 모델 선택에는 test / external evaluation 데이터를 사용하지 않는다.
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

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config
from tqdm import tqdm

try:
    import train_bone_age_ldl as base_train
except Exception as exc:
    raise ImportError(
        "같은 폴더의 train_bone_age_ldl.py가 필요합니다."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_512_input"
)

BASE_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "bone_age_ldl"
    / "best_model.pt"
)

BASE_VAL_PRED = (
    PROJECT_ROOT
    / "outputs"
    / "bone_age_ldl"
    / "validation_predictions_best.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "male_bias_refinement"
)


# =============================================================================
# Utility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(s: str):
    if s == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가")
    return d


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# =============================================================================
# Dataset
# =============================================================================

class BoneAgeDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(str(r["image_path"])) as im:
            im = im.convert("L").convert("RGB")
        return {
            "image": self.transform(im),
            "male": torch.tensor([float(r["male"])], dtype=torch.float32),
            "age": torch.tensor(float(r["boneage"]), dtype=torch.float32),
            "id": torch.tensor(int(r["id"]), dtype=torch.long),
        }


# =============================================================================
# Model helpers
# =============================================================================

def extract_shared(model, image, male):
    image_feature = model.image_head(model.backbone(image))
    sex_feature = model.sex_embedding(male)
    shared = model.fusion_trunk(
        torch.cat([image_feature, sex_feature], dim=1)
    )
    return shared


def logits_to_age(model, logits):
    prob = torch.softmax(logits, dim=1)
    return torch.sum(prob * model.age_bins.unsqueeze(0), dim=1)


def forward_full(model, image, male):
    shared = extract_shared(model, image, male)
    male_logits = model.male_output(shared)
    female_logits = model.female_output(shared)
    gate = male.reshape(-1, 1).to(male_logits.dtype).clamp(0, 1)
    logits = gate * male_logits + (1 - gate) * female_logits
    pred = logits_to_age(model, logits)
    return logits, pred


def freeze_everything_except_male_head(model):
    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.male_output.parameters():
        p.requires_grad_(True)

    # Frozen representation must stay deterministic.
    model.eval()
    model.male_output.train()


def trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Training
# =============================================================================

def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_epoch(
    *,
    model,
    teacher_male_head,
    male_loader,
    young_loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    lambda_bias,
    lambda_keep,
    young_age_max,
    grad_clip,
):
    # Keep all frozen modules in eval mode.
    model.eval()
    model.male_output.train()
    teacher_male_head.eval()

    young_iter = iter(young_loader)

    n_total = 0
    main_loss_sum = 0.0
    main_mae_sum = 0.0
    young_bias_sum = 0.0
    young_mae_sum = 0.0
    keep_sum = 0.0
    steps = 0

    progress = tqdm(
        male_loader,
        desc="Train male-head",
        leave=False,
        ncols=155,
    )

    for main_batch in progress:
        young_batch, young_iter = next_batch(
            young_iter, young_loader
        )

        x = main_batch["image"].to(device, non_blocking=True)
        male = main_batch["male"].to(device, non_blocking=True)
        age = main_batch["age"].to(device, non_blocking=True)

        yx = young_batch["image"].to(device, non_blocking=True)
        ymale = young_batch["male"].to(device, non_blocking=True)
        yage = young_batch["age"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, use_amp):
            # Frozen feature extractor; no need to keep its graph.
            with torch.no_grad():
                shared = extract_shared(model, x, male)
                yshared = extract_shared(model, yx, ymale)

            # Student male head.
            logits = model.male_output(shared)
            pred = logits_to_age(model, logits)

            main_loss, _, _ = criterion(
                logits, pred, age, model.age_bins
            )

            ylogits = model.male_output(yshared)
            ypred = logits_to_age(model, ylogits)

            young_error = ypred - yage
            young_bias = young_error.mean()
            positive_bias = F.relu(young_bias)
            bias_penalty = positive_bias.square()

            # Preserve base mapping on older males.
            older_mask = age > young_age_max
            if bool(older_mask.any()):
                with torch.no_grad():
                    teacher_logits = teacher_male_head(
                        shared[older_mask]
                    )
                    teacher_pred = logits_to_age(
                        model, teacher_logits
                    )

                keep_loss = torch.mean(
                    torch.abs(
                        pred[older_mask] - teacher_pred
                    )
                )
            else:
                keep_loss = pred.sum() * 0.0

            total_loss = (
                main_loss
                + lambda_bias * bias_penalty
                + lambda_keep * keep_loss
            )

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.male_output.parameters(),
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.male_output.parameters(),
                max_norm=grad_clip,
            )
            optimizer.step()

        n = int(age.shape[0])
        n_total += n
        main_loss_sum += float(main_loss.detach().cpu()) * n
        main_mae_sum += float(
            torch.abs(pred - age).sum().detach().cpu()
        )

        steps += 1
        young_bias_sum += float(young_bias.detach().cpu())
        young_mae_sum += float(
            torch.abs(young_error).mean().detach().cpu()
        )
        keep_sum += float(keep_loss.detach().cpu())

        progress.set_postfix(
            maleMAE=f"{main_mae_sum / n_total:.3f}",
            YMbias=f"{young_bias_sum / steps:+.2f}",
            keep=f"{keep_sum / steps:.3f}",
        )

    return {
        "male_main_loss": main_loss_sum / n_total,
        "male_train_mae": main_mae_sum / n_total,
        "young_train_bias": young_bias_sum / max(steps, 1),
        "young_train_mae": young_mae_sum / max(steps, 1),
        "older_keep_mae": keep_sum / max(steps, 1),
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, desc):
    model.eval()

    rows = []
    total_n = 0
    loss_sum = 0.0
    ae_sum = 0.0
    se_sum = 0.0

    for batch in tqdm(
        loader, desc=desc, leave=False, ncols=125
    ):
        x = batch["image"].to(device, non_blocking=True)
        male = batch["male"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)

        with amp_context(device, use_amp):
            logits, pred = forward_full(model, x, male)
            loss, _, _ = criterion(
                logits, pred, age, model.age_bins
            )

        error = pred - age
        n = int(age.shape[0])

        total_n += n
        loss_sum += float(loss.detach().cpu()) * n
        ae_sum += float(torch.abs(error).sum().detach().cpu())
        se_sum += float(torch.square(error).sum().detach().cpu())

        ids = batch["id"].cpu().numpy()
        ss = male.float().cpu().numpy().reshape(-1)
        yy = age.float().cpu().numpy()
        pp = pred.float().cpu().numpy()

        for image_id, s, y, p in zip(ids, ss, yy, pp):
            rows.append({
                "id": int(image_id),
                "male": float(s),
                "true_boneage": float(y),
                "pred_boneage": float(p),
                "signed_error": float(p - y),
                "abs_error": float(abs(p - y)),
            })

    return (
        {
            "loss": loss_sum / total_n,
            "mae": ae_sum / total_n,
            "rmse": math.sqrt(se_sum / total_n),
        },
        pd.DataFrame(rows),
    )


def subgroup_table(df):
    d = df.copy()
    d["sex_name"] = np.where(
        d["male"].to_numpy(float) >= 0.5,
        "Male",
        "Female",
    )

    age = d["true_boneage"].to_numpy(float)
    d["age_group"] = np.select(
        [
            age <= 60,
            (age > 60) & (age <= 96),
            (age > 96) & (age <= 144),
            age > 144,
        ],
        ["le60m", "61_96m", "97_144m", "gt144m"],
        default="unknown",
    )

    groups = [
        ("Overall", np.ones(len(d), dtype=bool)),
        ("Female_all", d["sex_name"].to_numpy() == "Female"),
        ("Male_all", d["sex_name"].to_numpy() == "Male"),
    ]

    for sex in ["Female", "Male"]:
        for ag in ["le60m", "61_96m", "97_144m", "gt144m"]:
            groups.append((
                f"{sex}_{ag}",
                (d["sex_name"].to_numpy() == sex)
                & (d["age_group"].to_numpy() == ag),
            ))

    rows = []

    for name, mask in groups:
        cur = d.loc[mask]
        if len(cur) == 0:
            continue

        y = cur["true_boneage"].to_numpy(float)
        p = cur["pred_boneage"].to_numpy(float)
        e = p - y
        ae = np.abs(e)

        rows.append({
            "group": name,
            "N": int(len(cur)),
            "MAE": float(ae.mean()),
            "RMSE": float(np.sqrt(np.mean(e**2))),
            "Bias": float(e.mean()),
            "MedianAE": float(np.median(ae)),
            "P90AE": float(np.percentile(ae, 90)),
            "AE15_rate_pct": float((ae >= 15).mean() * 100),
            "AE20_rate_pct": float((ae >= 20).mean() * 100),
        })

    return pd.DataFrame(rows)


def group_row(subgroups, name):
    x = subgroups[subgroups["group"] == name]
    if len(x) != 1:
        raise RuntimeError(f"missing subgroup: {name}")
    return x.iloc[0]


def verify_epoch0(current_df, saved_path, tolerance):
    if not saved_path.exists():
        print("[VERIFY] 기존 base prediction 없음 -> skip")
        return

    saved = pd.read_csv(saved_path)
    if not {"id", "pred_boneage"}.issubset(saved.columns):
        print("[VERIFY] 기존 CSV 컬럼 부족 -> skip")
        return

    m = current_df[["id", "pred_boneage"]].merge(
        saved[["id", "pred_boneage"]],
        on="id",
        suffixes=("_current", "_saved"),
    )

    delta = np.abs(
        m["pred_boneage_current"].to_numpy(float)
        - m["pred_boneage_saved"].to_numpy(float)
    )

    print(
        f"[VERIFY] epoch0 mean |Δ|={delta.mean():.6f}, "
        f"max |Δ|={delta.max():.6f}"
    )

    if delta.max() > tolerance:
        raise RuntimeError(
            f"base model reproduction mismatch: {delta.max():.6f}"
        )


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    path,
    *,
    epoch,
    model,
    optimizer,
    scaler,
    history,
    best_mae,
    best_target_bias,
    early_ref_mae,
    early_ref_bias,
    no_improvement,
    baseline_mae,
    baseline_bias,
    args,
):
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "best_mae": float(best_mae),
        "best_target_bias": float(best_target_bias),
        "early_ref_mae": float(early_ref_mae),
        "early_ref_bias": float(early_ref_bias),
        "no_improvement": int(no_improvement),
        "baseline_mae": float(baseline_mae),
        "baseline_male_le60_bias": float(baseline_bias),
        "config": vars(args),
        "trainable_scope": "male_output only",
        "test_used": False,
        "enterprise_used": False,
    }, path)


# =============================================================================
# Main
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", default=str(DATASET_ROOT))
    p.add_argument("--base_ckpt", default=str(BASE_CKPT))
    p.add_argument("--base_val_pred", default=str(BASE_VAL_PRED))
    p.add_argument("--output_dir", default=str(OUTPUT_ROOT))

    p.add_argument("--model_name", default="convnext_tiny.fb_in1k")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--image_dim", type=int, default=512)
    p.add_argument("--sex_dim", type=int, default=32)
    p.add_argument("--fusion_dim", type=int, default=128)
    p.add_argument("--image_dropout", type=float, default=0.20)
    p.add_argument("--fusion_dropout", type=float, default=0.20)
    p.add_argument("--num_bins", type=int, default=240)

    p.add_argument("--male_batch_size", type=int, default=64)
    p.add_argument("--young_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--lambda_kl", type=float, default=0.025)
    p.add_argument("--lambda_bias", type=float, default=0.005)
    p.add_argument("--lambda_keep", type=float, default=0.25)
    p.add_argument("--young_age_max", type=float, default=60.0)

    p.add_argument("--early_stopping", type=int, default=6)
    p.add_argument("--min_delta", type=float, default=0.003)
    p.add_argument("--bias_min_delta", type=float, default=0.10)
    p.add_argument("--max_mae_degradation", type=float, default=0.02)

    p.add_argument("--rotation", type=float, default=5.0)
    p.add_argument("--translate", type=float, default=0.03)
    p.add_argument("--scale_min", type=float, default=0.97)
    p.add_argument("--scale_max", type=float, default=1.03)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--verify_tolerance", type=float, default=0.02)
    p.add_argument("--resume", default=None)
    p.add_argument("--overwrite", action="store_true")

    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    dataset = Path(args.dataset_dir).resolve()
    base_ckpt = Path(args.base_ckpt).resolve()
    base_val_pred = Path(args.base_val_pred).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    train_csv = dataset / "csv" / "train.csv"
    val_csv = dataset / "csv" / "validation.csv"
    train_dir = dataset / "train" / "images"
    val_dir = dataset / "validation" / "images"

    for path in [train_csv, val_csv, train_dir, val_dir, base_ckpt]:
        if not path.exists():
            raise FileNotFoundError(path)

    best_mae_path = outdir / "best_mae_model.pt"
    best_target_path = outdir / "best_targeted_model.pt"
    last_path = outdir / "last_model.pt"
    history_path = outdir / "history.csv"

    if args.resume is None and last_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"기존 학습 결과가 있습니다: {last_path}\n"
            "--resume 또는 --overwrite를 사용하세요."
        )

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_df = base_train.attach_image_paths(
        base_train.standardize_dataframe(train_csv),
        train_dir,
        "refinement train",
    )

    val_df = base_train.attach_image_paths(
        base_train.standardize_dataframe(val_csv),
        val_dir,
        "refinement validation",
    )

    male_train_df = (
        train_df[
            train_df["male"].astype(float) >= 0.5
        ]
        .copy()
        .reset_index(drop=True)
    )

    young_train_df = (
        male_train_df[
            male_train_df["boneage"].astype(float)
            <= args.young_age_max
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(young_train_df) < args.young_batch_size:
        raise RuntimeError("young male train sample 부족")

    probe = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    data_cfg = resolve_model_data_config(probe)
    del probe

    mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
    std = data_cfg.get("std", (0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomAffine(
            degrees=args.rotation,
            translate=(args.translate, args.translate),
            scale=(args.scale_min, args.scale_max),
            interpolation=InterpolationMode.BICUBIC,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(
            (args.image_size, args.image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    male_loader = DataLoader(
        BoneAgeDataset(male_train_df, train_transform),
        batch_size=args.male_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    young_loader = DataLoader(
        BoneAgeDataset(young_train_df, train_transform),
        batch_size=args.young_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        BoneAgeDataset(val_df, eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = base_train.ConvNeXtTinyDistributionRegression(
        model_name=args.model_name,
        image_dim=args.image_dim,
        sex_dim=args.sex_dim,
        fusion_dim=args.fusion_dim,
        image_dropout=args.image_dropout,
        fusion_dropout=args.fusion_dropout,
        pretrained=False,
        num_bins=args.num_bins,
    ).to(device)

    ckpt = torch.load(
        base_ckpt,
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt.get(
        "ema_model_state_dict",
        ckpt.get("model_state_dict"),
    )
    if state is None:
        raise KeyError("base model state 없음")

    model.load_state_dict(state, strict=True)

    # Teacher = original base male head.
    teacher_male_head = copy.deepcopy(
        model.male_output
    ).to(device).eval()

    for p in teacher_male_head.parameters():
        p.requires_grad_(False)

    freeze_everything_except_male_head(model)

    criterion = base_train.DistributionAgeLoss(
        sigma=args.sigma,
        lambda_kl=args.lambda_kl,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.male_output.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    scaler = make_scaler(use_amp)

    # -------------------------------------------------------------------------
    # Epoch 0
    # -------------------------------------------------------------------------
    base_metrics, base_pred = evaluate(
        model,
        val_loader,
        criterion,
        device,
        use_amp,
        "Epoch0 Base",
    )

    verify_epoch0(
        base_pred,
        base_val_pred,
        args.verify_tolerance,
    )

    base_sub = subgroup_table(base_pred)
    base_male_young = group_row(
        base_sub, "Male_le60m"
    )
    base_female = group_row(
        base_sub, "Female_all"
    )
    base_male = group_row(
        base_sub, "Male_all"
    )

    baseline_mae = float(base_metrics["mae"])
    baseline_rmse = float(base_metrics["rmse"])
    baseline_bias = float(base_male_young["Bias"])
    baseline_ym_mae = float(base_male_young["MAE"])

    print()
    print("=" * 126)
    print("REFINEMENT | MALE-HEAD-ONLY TARGETED BIAS CORRECTION")
    print("=" * 126)
    print("Device                    :", device)
    print("Base model init               :", base_ckpt)
    print("Trainable                 : male_output ONLY")
    print("Frozen                    : backbone/image_head/sex_embedding/fusion/female_output")
    print("Male TRAIN                :", len(male_train_df))
    print("Young Male TRAIN <=60m    :", len(young_train_df))
    print("LR                        :", args.lr)
    print("lambda_bias               :", args.lambda_bias)
    print("lambda_keep older male    :", args.lambda_keep)
    print(
        "Baseline Overall          : "
        f"MAE={baseline_mae:.3f} RMSE={baseline_rmse:.3f}"
    )
    print(
        "Baseline Male<=60         : "
        f"MAE={baseline_ym_mae:.3f} Bias={baseline_bias:+.3f}"
    )
    print(
        "Baseline Female all       : "
        f"MAE={base_female['MAE']:.3f} Bias={base_female['Bias']:+.3f}"
    )
    print(
        "Baseline Male all         : "
        f"MAE={base_male['MAE']:.3f} Bias={base_male['Bias']:+.3f}"
    )
    print(
        "Targeted-best MAE ceiling : "
        f"{baseline_mae + args.max_mae_degradation:.3f}"
    )
    print("Trainable params          :", trainable_parameter_count(model))
    print("TEST                      : NOT USED")
    print("Enterprise                : NOT USED")
    print("=" * 126)

    history = [{
        "epoch": 0,
        "val_mae": baseline_mae,
        "val_rmse": baseline_rmse,
        "male_le60_mae": baseline_ym_mae,
        "male_le60_bias": baseline_bias,
        "female_all_mae": float(base_female["MAE"]),
        "male_all_mae": float(base_male["MAE"]),
    }]

    best_mae = baseline_mae
    best_target_bias = abs(baseline_bias)
    early_ref_mae = baseline_mae
    early_ref_bias = abs(baseline_bias)
    no_improvement = 0
    start_epoch = 1

    # Save epoch0 as initial bests.
    save_checkpoint(
        best_mae_path,
        epoch=0,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        history=history,
        best_mae=best_mae,
        best_target_bias=best_target_bias,
        early_ref_mae=early_ref_mae,
        early_ref_bias=early_ref_bias,
        no_improvement=0,
        baseline_mae=baseline_mae,
        baseline_bias=baseline_bias,
        args=args,
    )

    save_checkpoint(
        best_target_path,
        epoch=0,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        history=history,
        best_mae=best_mae,
        best_target_bias=best_target_bias,
        early_ref_mae=early_ref_mae,
        early_ref_bias=early_ref_bias,
        no_improvement=0,
        baseline_mae=baseline_mae,
        baseline_bias=baseline_bias,
        args=args,
    )

    base_pred.to_csv(
        outdir / "validation_predictions_best_mae.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_pred.to_csv(
        outdir / "validation_predictions_best_targeted.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_sub.to_csv(
        outdir / "subgroup_best_mae.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_sub.to_csv(
        outdir / "subgroup_best_targeted.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------
    if args.resume:
        rp = Path(args.resume).resolve()
        r = torch.load(rp, map_location=device, weights_only=False)

        model.load_state_dict(
            r["model_state_dict"], strict=True
        )
        freeze_everything_except_male_head(model)

        optimizer.load_state_dict(
            r["optimizer_state_dict"]
        )

        if "scaler_state_dict" in r:
            scaler.load_state_dict(
                r["scaler_state_dict"]
            )

        history = list(r.get("history", history))
        best_mae = float(r.get("best_mae", best_mae))
        best_target_bias = float(
            r.get("best_target_bias", best_target_bias)
        )
        early_ref_mae = float(
            r.get("early_ref_mae", early_ref_mae)
        )
        early_ref_bias = float(
            r.get("early_ref_bias", early_ref_bias)
        )
        no_improvement = int(
            r.get("no_improvement", 0)
        )
        start_epoch = int(r["epoch"]) + 1
        print("[RESUME]", rp)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model=model,
            teacher_male_head=teacher_male_head,
            male_loader=male_loader,
            young_loader=young_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            lambda_bias=args.lambda_bias,
            lambda_keep=args.lambda_keep,
            young_age_max=args.young_age_max,
            grad_clip=args.grad_clip,
        )

        val_metrics, val_pred = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            "Validation",
        )

        sub = subgroup_table(val_pred)
        ym = group_row(sub, "Male_le60m")
        female = group_row(sub, "Female_all")
        male_all = group_row(sub, "Male_all")

        val_mae = float(val_metrics["mae"])
        val_rmse = float(val_metrics["rmse"])
        ym_mae = float(ym["MAE"])
        ym_bias = float(ym["Bias"])
        abs_bias = abs(ym_bias)

        # Female predictions must remain identical aside from numerical noise.
        female_mae_delta = float(female["MAE"]) - float(base_female["MAE"])

        true_best = val_mae < best_mae
        if true_best:
            best_mae = val_mae
            save_checkpoint(
                best_mae_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                history=history,
                best_mae=best_mae,
                best_target_bias=best_target_bias,
                early_ref_mae=early_ref_mae,
                early_ref_bias=early_ref_bias,
                no_improvement=no_improvement,
                baseline_mae=baseline_mae,
                baseline_bias=baseline_bias,
                args=args,
            )
            val_pred.to_csv(
                outdir / "validation_predictions_best_mae.csv",
                index=False,
                encoding="utf-8-sig",
            )
            sub.to_csv(
                outdir / "subgroup_best_mae.csv",
                index=False,
                encoding="utf-8-sig",
            )

        eligible = (
            val_mae
            <= baseline_mae + args.max_mae_degradation
        )

        targeted_best = (
            eligible
            and abs_bias < best_target_bias
        )

        if targeted_best:
            best_target_bias = abs_bias
            save_checkpoint(
                best_target_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                history=history,
                best_mae=best_mae,
                best_target_bias=best_target_bias,
                early_ref_mae=early_ref_mae,
                early_ref_bias=early_ref_bias,
                no_improvement=no_improvement,
                baseline_mae=baseline_mae,
                baseline_bias=baseline_bias,
                args=args,
            )
            val_pred.to_csv(
                outdir / "validation_predictions_best_targeted.csv",
                index=False,
                encoding="utf-8-sig",
            )
            sub.to_csv(
                outdir / "subgroup_best_targeted.csv",
                index=False,
                encoding="utf-8-sig",
            )

        mae_meaningful = (
            val_mae < early_ref_mae - args.min_delta
        )

        bias_meaningful = (
            eligible
            and abs_bias
            < early_ref_bias - args.bias_min_delta
        )

        if mae_meaningful:
            early_ref_mae = val_mae
        if bias_meaningful:
            early_ref_bias = abs_bias

        meaningful = mae_meaningful or bias_meaningful
        if meaningful:
            no_improvement = 0
        else:
            no_improvement += 1

        elapsed = (time.time() - t0) / 60.0

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_male_mae": train_metrics["male_train_mae"],
            "train_young_mae": train_metrics["young_train_mae"],
            "train_young_bias": train_metrics["young_train_bias"],
            "older_keep_mae": train_metrics["older_keep_mae"],
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "male_all_mae": float(male_all["MAE"]),
            "male_all_bias": float(male_all["Bias"]),
            "male_le60_mae": ym_mae,
            "male_le60_bias": ym_bias,
            "female_all_mae": float(female["MAE"]),
            "female_all_bias": float(female["Bias"]),
            "female_mae_delta_vs_epoch0": female_mae_delta,
            "eligible_targeted": int(eligible),
            "true_best": int(true_best),
            "targeted_best": int(targeted_best),
            "meaningful": int(meaningful),
            "epoch_minutes": elapsed,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        save_checkpoint(
            last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            history=history,
            best_mae=best_mae,
            best_target_bias=best_target_bias,
            early_ref_mae=early_ref_mae,
            early_ref_bias=early_ref_bias,
            no_improvement=no_improvement,
            baseline_mae=baseline_mae,
            baseline_bias=baseline_bias,
            args=args,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"trainMale={train_metrics['male_train_mae']:.3f} | "
            f"trainYM bias={train_metrics['young_train_bias']:+.2f} | "
            f"keep={train_metrics['older_keep_mae']:.3f} | "
            f"val={val_mae:.3f} RMSE={val_rmse:.3f} | "
            f"MaleAll={male_all['MAE']:.3f} | "
            f"Male<=60={ym_mae:.3f} Bias={ym_bias:+.3f} | "
            f"Female={female['MAE']:.3f} Δ={female_mae_delta:+.4f} | "
            f"{elapsed:.1f} min"
        )

        if true_best:
            print("  BEST OVERALL MAE 저장")
        if targeted_best:
            print("  BEST TARGETED 저장")
        if meaningful:
            reasons = []
            if mae_meaningful:
                reasons.append("overall MAE")
            if bias_meaningful:
                reasons.append("Male<=60 bias")
            print(
                "  meaningful 개선 -> patience reset:",
                ", ".join(reasons),
            )
        else:
            print(
                f"  의미 있는 개선 없음: "
                f"{no_improvement}/{args.early_stopping}"
            )

        if no_improvement >= args.early_stopping:
            print("Early stopping")
            break

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    def eval_saved(path, label):
        c = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            c["model_state_dict"],
            strict=True,
        )
        freeze_everything_except_male_head(model)
        m, p = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            label,
        )
        s = subgroup_table(p)
        return c, m, p, s

    bmc, bmm, bmp, bms = eval_saved(
        best_mae_path, "Best MAE"
    )
    btc, btm, btp, bts = eval_saved(
        best_target_path, "Best Targeted"
    )

    bmm_ym = group_row(bms, "Male_le60m")
    btm_ym = group_row(bts, "Male_le60m")
    bmm_f = group_row(bms, "Female_all")
    btm_f = group_row(bts, "Female_all")

    summary = {
        "baseline": {
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "Male_le60_MAE": baseline_ym_mae,
            "Male_le60_Bias": baseline_bias,
            "Female_all_MAE": float(base_female["MAE"]),
        },
        "best_mae": {
            "epoch": int(bmc["epoch"]),
            "MAE": float(bmm["mae"]),
            "RMSE": float(bmm["rmse"]),
            "Male_le60_MAE": float(bmm_ym["MAE"]),
            "Male_le60_Bias": float(bmm_ym["Bias"]),
            "Female_all_MAE": float(bmm_f["MAE"]),
        },
        "best_targeted": {
            "epoch": int(btc["epoch"]),
            "MAE": float(btm["mae"]),
            "RMSE": float(btm["rmse"]),
            "Male_le60_MAE": float(btm_ym["MAE"]),
            "Male_le60_Bias": float(btm_ym["Bias"]),
            "Female_all_MAE": float(btm_f["MAE"]),
            "MAE_ceiling": float(
                baseline_mae + args.max_mae_degradation
            ),
        },
        "trainable_scope": "male_output only",
        "lambda_bias": args.lambda_bias,
        "lambda_keep": args.lambda_keep,
        "test_used": False,
        "enterprise_used": False,
    }

    with open(
        outdir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 126)
    print("REFINEMENT FINAL")
    print("=" * 126)

    for label, result in [
        ("BASELINE", summary["baseline"]),
        ("BEST MAE", summary["best_mae"]),
        ("BEST TARGETED", summary["best_targeted"]),
    ]:
        print(
            f"{label:16s} | "
            f"MAE={result['MAE']:.3f} | "
            f"RMSE={result['RMSE']:.3f} | "
            f"Male<=60 MAE={result['Male_le60_MAE']:.3f} | "
            f"Bias={result['Male_le60_Bias']:+.3f} | "
            f"Female={result['Female_all_MAE']:.3f}"
        )

    print()
    print("TEST USED       = False")
    print("ENTERPRISE USED = False")
    print("=" * 126)


if __name__ == "__main__":
    main()