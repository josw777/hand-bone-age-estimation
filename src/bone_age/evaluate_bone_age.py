# -*- coding: utf-8 -*-
r"""
evaluate_bone_age.py

최종 골연령 모델의 held-out test 평가.

원칙
----
- 모델 선택이 완료된 뒤 최종 성능 기록을 위해 실행한다.
- held-out test 결과를 다시 모델 또는 하이퍼파라미터 선택에 사용하지 않는다.
- augmentation 없이 deterministic evaluation을 수행한다.
- 기본적으로 기존 평가 결과가 있으면 재실행을 막는다.
  다시 평가해야 하는 경우에만 --overwrite를 사용한다.

출력
----
<PROJECT_ROOT>/outputs/heldout_test/
  test_predictions.csv
  test_subgroups.csv
  test_summary.json
  test_metrics.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_model_data_config

import train_bone_age_ldl as base_train
import train_male_bias_refinement as refinement


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_512_input"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "app"
    / "backend"
    / "model_package"
    / "models"
)

FINAL_CKPT = (
    MODEL_ROOT
    / "best_model.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "heldout_test"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 사용 불가")

    return device


def resolve_test_paths(dataset_root: Path):
    csv_candidates = [
        dataset_root / "csv" / "test.csv",
        dataset_root / "test.csv",
    ]

    image_candidates = [
        dataset_root / "test" / "images",
        dataset_root / "test",
        dataset_root / "Test" / "images",
    ]

    test_csv = next(
        (p for p in csv_candidates if p.is_file()),
        None,
    )

    test_dir = next(
        (p for p in image_candidates if p.is_dir()),
        None,
    )

    if test_csv is None:
        raise FileNotFoundError(
            "TEST CSV를 찾지 못했습니다.\n확인한 경로:\n"
            + "\n".join(f"- {p}" for p in csv_candidates)
        )

    if test_dir is None:
        raise FileNotFoundError(
            "TEST images 폴더를 찾지 못했습니다.\n확인한 경로:\n"
            + "\n".join(f"- {p}" for p in image_candidates)
        )

    return test_csv, test_dir


def full_metrics(df: pd.DataFrame):
    y = df["true_boneage"].to_numpy(dtype=float)
    p = df["pred_boneage"].to_numpy(dtype=float)

    e = p - y
    ae = np.abs(e)

    sse = float(np.sum((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))

    r2 = (
        float(1.0 - sse / sst)
        if sst > 0
        else float("nan")
    )

    return {
        "N": int(len(df)),
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "R2": r2,
        "Bias": float(e.mean()),
        "MedianAE": float(np.median(ae)),
        "P90AE": float(np.percentile(ae, 90)),
        "P95AE": float(np.percentile(ae, 95)),
        "MaxAE": float(ae.max()),
        "AE_ge_6_n": int(np.sum(ae >= 6.0)),
        "AE_ge_6_pct": float(np.mean(ae >= 6.0) * 100.0),
        "AE_ge_12_n": int(np.sum(ae >= 12.0)),
        "AE_ge_12_pct": float(np.mean(ae >= 12.0) * 100.0),
        "AE_ge_15_n": int(np.sum(ae >= 15.0)),
        "AE_ge_15_pct": float(np.mean(ae >= 15.0) * 100.0),
        "AE_ge_20_n": int(np.sum(ae >= 20.0)),
        "AE_ge_20_pct": float(np.mean(ae >= 20.0) * 100.0),
        "AE_ge_24_n": int(np.sum(ae >= 24.0)),
        "AE_ge_24_pct": float(np.mean(ae >= 24.0) * 100.0),
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = ckpt.get("config", {})

    model = base_train.ConvNeXtTinyDistributionRegression(
        model_name=config.get(
            "model_name",
            "convnext_tiny.fb_in1k",
        ),
        image_dim=int(config.get("image_dim", 512)),
        sex_dim=int(config.get("sex_dim", 32)),
        fusion_dim=int(config.get("fusion_dim", 128)),
        image_dropout=float(config.get("image_dropout", 0.20)),
        fusion_dropout=float(config.get("fusion_dropout", 0.20)),
        pretrained=False,
        num_bins=int(config.get("num_bins", 240)),
    ).to(device)

    state = ckpt.get("model_state_dict")

    if state is None:
        raise KeyError(
            "final checkpoint에 model_state_dict가 없습니다."
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, ckpt, config


def build_eval_transform(model_name: str, image_size: int):
    probe = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    cfg = resolve_model_data_config(probe)
    del probe

    mean = cfg.get("mean", (0.5, 0.5, 0.5))
    std = cfg.get("std", (0.5, 0.5, 0.5))

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_dir",
        default=str(DATASET_ROOT),
    )
    p.add_argument(
        "--checkpoint",
        default=str(FINAL_CKPT),
    )
    p.add_argument(
        "--output_dir",
        default=str(OUTPUT_ROOT),
    )

    p.add_argument(
        "--device",
        default="auto",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    p.add_argument(
        "--expected_test_n",
        type=int,
        default=197,
    )

    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_dir / "test_summary.json"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            "\n이미 held-out TEST 결과가 존재합니다:\n"
            f"{summary_path}\n\n"
            "이 평가는 최종 1회 평가 용도입니다.\n"
            "정말 재실행해야 하는 경우에만 --overwrite를 사용하세요."
        )

    test_csv, test_dir = resolve_test_paths(
        dataset_root
    )

    device = resolve_device(args.device)
    use_amp = bool(
        args.amp
        and device.type == "cuda"
    )

    model, checkpoint, config = load_model(
        checkpoint_path,
        device,
    )

    model_name = config.get(
        "model_name",
        "convnext_tiny.fb_in1k",
    )
    image_size = int(
        config.get("image_size", 512)
    )

    eval_transform = build_eval_transform(
        model_name,
        image_size,
    )

    test_df = base_train.attach_image_paths(
        base_train.standardize_dataframe(
            test_csv
        ),
        test_dir,
        "held-out test",
    )

    print()
    print("=" * 100)
    print("FINAL HELD-OUT TEST")
    print("=" * 100)
    print("Checkpoint :", checkpoint_path)
    print("Dataset    :", dataset_root)
    print("Test CSV   :", test_csv)
    print("Test images:", test_dir)
    print("Test N     :", len(test_df))
    print("Device     :", device)
    print("AMP        :", use_amp)
    print("TEST only  : YES")
    print("External evaluation : NOT USED")
    print("=" * 100)

    if len(test_df) != args.expected_test_n:
        print(
            f"[WARNING] 기대 TEST N={args.expected_test_n}인데 "
            f"현재 N={len(test_df)}입니다."
        )
        print(
            "데이터셋이 의도한 최종 held-out TEST인지 반드시 확인하세요."
        )

    dataset = refinement.BoneAgeDataset(
        test_df,
        eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    criterion = base_train.DistributionAgeLoss(
        sigma=float(config.get("sigma", 10.0)),
        lambda_kl=float(config.get("lambda_kl", 0.025)),
    ).to(device)

    test_metrics_basic, predictions = refinement.evaluate(
        model,
        loader,
        criterion,
        device,
        use_amp,
        "HELD-OUT TEST",
    )

    metrics = full_metrics(predictions)
    subgroups = refinement.subgroup_table(
        predictions
    )

    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subgroups.to_csv(
        output_dir / "test_subgroups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model": "male_bias_refinement",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_root),
        "test_csv": str(test_csv),
        "test_images": str(test_dir),
        "model_selected_before_test": True,
        "test_used_for_model_selection": False,
        "external_evaluation_used": False,
        "metrics": metrics,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "FINAL HELD-OUT TEST",
        "=" * 72,
        f"N        : {metrics['N']}",
        f"MAE      : {metrics['MAE']:.6f} months",
        f"RMSE     : {metrics['RMSE']:.6f} months",
        f"R²       : {metrics['R2']:.6f}",
        f"Bias     : {metrics['Bias']:+.6f} months",
        f"MedianAE : {metrics['MedianAE']:.6f}",
        f"P90AE    : {metrics['P90AE']:.6f}",
        f"P95AE    : {metrics['P95AE']:.6f}",
        f"MaxAE    : {metrics['MaxAE']:.6f}",
        "",
        f"AE >=  6 : {metrics['AE_ge_6_n']} ({metrics['AE_ge_6_pct']:.3f}%)",
        f"AE >= 12 : {metrics['AE_ge_12_n']} ({metrics['AE_ge_12_pct']:.3f}%)",
        f"AE >= 15 : {metrics['AE_ge_15_n']} ({metrics['AE_ge_15_pct']:.3f}%)",
        f"AE >= 20 : {metrics['AE_ge_20_n']} ({metrics['AE_ge_20_pct']:.3f}%)",
        f"AE >= 24 : {metrics['AE_ge_24_n']} ({metrics['AE_ge_24_pct']:.3f}%)",
        "",
        "TEST used for model selection : False",
        "External evaluation used              : False",
    ]

    metrics_txt = "\n".join(lines)

    (
        output_dir / "test_metrics.txt"
    ).write_text(
        metrics_txt,
        encoding="utf-8",
    )

    print()
    print(metrics_txt)
    print()
    print("성별/연령 subgroup:")
    print(
        subgroups.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print()
    print("저장:")
    print(" ", output_dir / "test_predictions.csv")
    print(" ", output_dir / "test_subgroups.csv")
    print(" ", summary_path)
    print(" ", output_dir / "test_metrics.txt")
    print()
    print(
        "※ 이 TEST 결과로 모델을 다시 선택하지 않습니다. "
        "최종 모델 성능 기록용입니다."
    )


if __name__ == "__main__":
    main()