# -*- coding: utf-8 -*-
"""
create_seg_aligned_dataset.py

수부 X-ray 공간 표준화 전처리.

처리 순서
---------
Raw X-ray
 -> YOLOX-S hand detection
 -> YOLO bbox + margin
 -> hand segmentation
 -> largest component + hole filling
 -> PCA coarse alignment
 -> finger-wrist direction refinement
 -> original image / mask rotation
 -> rotated mask bbox + final margin crop
 -> native-resolution aligned image / mask 저장

이 단계에서는 intensity normalization이나 512 resize를 수행하지 않는다.
최종 모델 입력 생성은 create_final_512_input.py에서 수행한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


# ============================================================
# 0. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"

RAW_TRAIN_DIR = RAW_ROOT / "train"
RAW_VAL_DIR = RAW_ROOT / "validation"
RAW_TEST_DIR = RAW_ROOT / "test"

# QC를 통과한 sample filename allowlist
QC_ROOT = DATA_ROOT / "qc"

# Native-size aligned image / mask output
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "seg_aligned_native"
)

# YOLOX source package
YOLOX_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "YOLOX"
)

# YOLOX experiment configuration
YOLOX_EXP_FILE = (
    PROJECT_ROOT
    / "src"
    / "detection"
    / "yolox_s_hand_exp.py"
)

# Trained models used by the inference pipeline
MODEL_ROOT = (
    PROJECT_ROOT
    / "app"
    / "backend"
    / "model_package"
    / "models"
)

YOLOX_CKPT = (
    MODEL_ROOT
    / "yolox_s_hand_best.pth"
)

SEG_JIT_PATH = (
    MODEL_ROOT
    / "hand_seg_crop512_traced.pt"
)


# ============================================================
# 1. 파라미터
# ============================================================

# YOLOX
YOLO_CONF = 0.20
YOLO_NMS = 0.70

# Seg 학습 때와 동일:
# YOLO bbox 기준 좌/우/상/하 각각 10%
SEG_MARGIN_X = 0.10
SEG_MARGIN_TOP = 0.10
SEG_MARGIN_BOTTOM = 0.10

# Seg inference
SEG_INPUT_SIZE = 512
SEG_THRESHOLD = 0.5

SEG_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)
SEG_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

# 최종 native crop:
# rotated mask bbox 기준
FINAL_MARGIN_X = 0.04
FINAL_MARGIN_TOP = 0.03
FINAL_MARGIN_BOTTOM = 0.02

# metadata를 몇 장마다 디스크에 다시 쓸지
META_SAVE_EVERY = 25

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ============================================================
# 2. 공통 유틸
# ============================================================

def check_paths() -> None:
    pairs = [
        ("RAW_TRAIN_DIR", RAW_TRAIN_DIR),
        ("RAW_VAL_DIR", RAW_VAL_DIR),
        ("QC_ROOT", QC_ROOT),
        ("YOLOX_ROOT", YOLOX_ROOT),
        ("YOLOX_EXP_FILE", YOLOX_EXP_FILE),
        ("YOLOX_CKPT", YOLOX_CKPT),
        ("SEG_JIT_PATH", SEG_JIT_PATH),
    ]

    print("===== PATH CHECK =====")
    bad = []

    for name, path in pairs:
        ok = path.exists()
        print(f"{name:16s}: {ok} | {path}")
        if not ok:
            bad.append((name, path))

    if bad:
        lines = "\n".join(
            f"- {name}: {path}"
            for name, path in bad
        )
        raise FileNotFoundError(
            "\n존재하지 않는 경로가 있습니다.\n"
            + lines
            + "\n\n상단 경로 설정 블록을 수정하세요."
        )


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_stem_index(root: Path) -> Dict[str, Path]:
    """
    원본 폴더를 stem -> path로 인덱싱.
    stem 중복이면 오류.
    """
    result: Dict[str, Path] = {}

    for p in collect_image_paths(root):
        key = p.stem

        if key in result:
            raise RuntimeError(
                f"동일 stem 파일이 2개 이상 있습니다: {key}\n"
                f"  1) {result[key]}\n"
                f"  2) {p}"
            )

        result[key] = p

    return result


def find_qc_split_dir(split: str) -> Path:
    """
    QC_ROOT 아래에서 training / validation 폴더를 찾는다.
    """
    candidates = {
        "training": [
            QC_ROOT / "training",
            QC_ROOT / "train",
            QC_ROOT / "training" / "images",
            QC_ROOT / "train" / "images",
        ],
        "validation": [
            QC_ROOT / "validation",
            QC_ROOT / "valid",
            QC_ROOT / "val",
            QC_ROOT / "validation" / "images",
            QC_ROOT / "valid" / "images",
            QC_ROOT / "val" / "images",
        ],
        "test": [
            QC_ROOT / "test",
            QC_ROOT / "testing",
            QC_ROOT / "test" / "images",
            QC_ROOT / "testing" / "images",
        ],
    }[split]

    for p in candidates:
        if p.exists() and len(collect_image_paths(p)) > 0:
            return p

    raise FileNotFoundError(
        f"{split} QC allowlist 폴더를 찾지 못했습니다.\n"
        f"QC_ROOT = {QC_ROOT}\n"
        f"후보 = {candidates}"
    )


def read_native_image(path: Path) -> np.ndarray:
    """
    최종 저장용.
    원본 dtype/channel을 가능한 한 그대로 읽음.
    """
    img = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if img is None:
        raise RuntimeError(
            f"원본 이미지 읽기 실패: {path}"
        )

    return img


def read_model_bgr8(path: Path) -> np.ndarray:
    """
    YOLOX/Seg 추론용.
    기존 OpenCV color input과 동일하게 uint8 BGR로 읽음.
    최종 저장 영상에는 이 배열을 사용하지 않음.
    """
    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise RuntimeError(
            f"모델용 이미지 읽기 실패: {path}"
        )

    return img


def ensure_output_dirs(split: str) -> Tuple[Path, Path]:
    output_split = (
        "train"
        if split == "training"
        else split
    )

    image_dir = OUTPUT_ROOT / output_split / "images"
    mask_dir = OUTPUT_ROOT / output_split / "masks"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return image_dir, mask_dir


# ============================================================
# 3. YOLOX 로드 / 추론
# ============================================================

def load_yolox(device: torch.device):
    if str(YOLOX_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(YOLOX_ROOT)
        )

    from yolox.exp import get_exp
    from yolox.data.data_augment import ValTransform

    exp = get_exp(
        str(YOLOX_EXP_FILE),
        None
    )

    model = exp.get_model()

    ckpt = torch.load(
        YOLOX_CKPT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True
    )

    model.to(device)
    model.eval()

    preproc = ValTransform(
        legacy=False
    )

    print("YOLOX loaded")
    print("  test_size   :", exp.test_size)
    print("  num_classes :", exp.num_classes)

    return exp, model, preproc


def yolox_detect_hand(
    image_bgr: np.ndarray,
    exp,
    model,
    preproc,
    device: torch.device,
):
    from yolox.utils import postprocess

    h, w = image_bgr.shape[:2]

    img, _ = preproc(
        image_bgr,
        None,
        exp.test_size
    )

    tensor = torch.from_numpy(
        img
    ).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        outputs = model(tensor)

        outputs = postprocess(
            outputs,
            num_classes=1,
            conf_thre=YOLO_CONF,
            nms_thre=YOLO_NMS,
            class_agnostic=True
        )

    output = outputs[0]

    if output is None or len(output) == 0:
        return None

    output = output.detach().cpu()

    scores = (
        output[:, 4]
        * output[:, 5]
    )

    best_idx = int(
        torch.argmax(scores).item()
    )

    score = float(
        scores[best_idx].item()
    )

    box = output[
        best_idx, :4
    ].numpy().astype(np.float32)

    ratio = min(
        exp.test_size[0] / h,
        exp.test_size[1] / w
    )

    box /= ratio

    x1, y1, x2, y2 = box

    x1 = int(round(np.clip(x1, 0, w - 1)))
    y1 = int(round(np.clip(y1, 0, h - 1)))
    x2 = int(round(np.clip(x2, 1, w)))
    y2 = int(round(np.clip(y2, 1, h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "box": (x1, y1, x2, y2),
        "score": score,
    }


def add_seg_margin(
    box: Tuple[int, int, int, int],
    image_shape
):
    h_img, w_img = image_shape[:2]

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    mx = bw * SEG_MARGIN_X
    mt = bh * SEG_MARGIN_TOP
    mb = bh * SEG_MARGIN_BOTTOM

    nx1 = max(
        0,
        int(round(x1 - mx))
    )
    nx2 = min(
        w_img,
        int(round(x2 + mx))
    )

    ny1 = max(
        0,
        int(round(y1 - mt))
    )
    ny2 = min(
        h_img,
        int(round(y2 + mb))
    )

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return (
        nx1, ny1,
        nx2, ny2
    )


# ============================================================
# 4. Segmentation 로드 / 추론
# ============================================================

def load_seg_jit(
    device: torch.device
):
    model = torch.jit.load(
        str(SEG_JIT_PATH),
        map_location=device
    )

    model.eval()

    print(
        "Seg TorchScript loaded:",
        SEG_JIT_PATH
    )

    return model


def predict_seg_mask(
    crop_bgr: np.ndarray,
    seg_model,
    device: torch.device
) -> np.ndarray:

    rgb = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2RGB
    )

    h, w = rgb.shape[:2]

    x = cv2.resize(
        rgb,
        (
            SEG_INPUT_SIZE,
            SEG_INPUT_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    x = (
        x.astype(np.float32)
        / 255.0
    )

    x = (
        x - SEG_MEAN
    ) / SEG_STD

    x = torch.from_numpy(
        x.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)

    with torch.inference_mode():
        logits = seg_model(x)

        # 혹시 tuple/list wrapper인 경우 방어
        if isinstance(
            logits,
            (tuple, list)
        ):
            logits = logits[0]

        if logits.shape[1] == 1:
            prob = torch.sigmoid(
                logits
            )[0, 0]
        else:
            prob = torch.softmax(
                logits,
                dim=1
            )[0, 1]

        mask = (
            prob.detach().cpu().numpy()
            >= SEG_THRESHOLD
        ).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# ============================================================
# 5. Mask 후처리
# ============================================================

def clean_hand_mask(
    mask: np.ndarray
) -> np.ndarray:

    mask = (
        mask > 0
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if n_labels <= 1:
        return mask

    largest_idx = (
        1
        + int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )
    )

    clean = (
        labels == largest_idx
    ).astype(np.uint8)

    # 내부 hole fill
    padded = np.pad(
        clean,
        1,
        mode="constant",
        constant_values=0
    )

    inv = (
        1 - padded
    ).astype(np.uint8)

    flood = inv.copy()

    ff_mask = np.zeros(
        (
            flood.shape[0] + 2,
            flood.shape[1] + 2
        ),
        dtype=np.uint8
    )

    cv2.floodFill(
        flood,
        ff_mask,
        seedPoint=(0, 0),
        newVal=2
    )

    holes = (
        flood == 1
    ).astype(np.uint8)

    filled = np.clip(
        padded + holes,
        0,
        1
    ).astype(np.uint8)

    return filled[
        1:-1,
        1:-1
    ]


# ============================================================
# 6. PCA + Finger-Wrist orientation
# ============================================================

def get_pca_axis(
    mask: np.ndarray
):
    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 100:
        return None

    points = np.column_stack(
        [xs, ys]
    ).astype(np.float32)

    mean, eigenvectors, eigenvalues = (
        cv2.PCACompute2(
            points,
            mean=None
        )
    )

    center = mean[0]
    axis = eigenvectors[0]

    theta = np.degrees(
        np.arctan2(
            axis[1],
            axis[0]
        )
    )

    theta = theta % 180.0

    # 현재 notebook에서 검증한 것과 동일
    rotation = theta - 90.0

    return {
        "center": center,
        "axis": axis,
        "theta": float(theta),
        "rotation": float(rotation),
        "eigenvalues": eigenvalues,
    }


def rotate_mask_bound(
    mask: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    orientation 계산용 mask 회전.
    """
    h, w = mask.shape[:2]

    cx = w / 2.0
    cy = h / 2.0

    M = cv2.getRotationMatrix2D(
        (cx, cy),
        angle,
        1.0
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(
        math.ceil(
            h * sin + w * cos
        )
    )

    new_h = int(
        math.ceil(
            h * cos + w * sin
        )
    )

    M[0, 2] += (
        new_w / 2.0 - cx
    )
    M[1, 2] += (
        new_h / 2.0 - cy
    )

    out = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return (
        out > 127
    ).astype(np.uint8)


def get_finger_wrist_residual(
    mask: np.ndarray
):
    """
    PCA coarse rotation 뒤의 mask에서
    가장 위쪽 fingertip component와
    아래쪽 wrist 중심을 연결한 축으로 residual 계산.

    현재 notebook에서 시각 검증한 방식과 동일.
    """

    mask = (
        mask > 0
    ).astype(np.uint8)

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 100:
        return None

    y_min = int(ys.min())
    y_max = int(ys.max())

    height = (
        y_max - y_min + 1
    )

    # ------------------------------
    # fingertip 후보
    # ------------------------------
    top_limit = int(
        y_min
        + height * 0.12
    )

    top_mask = np.zeros_like(
        mask
    )

    top_mask[
        y_min:top_limit + 1
    ] = mask[
        y_min:top_limit + 1
    ]

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            top_mask,
            connectivity=8
        )
    )

    candidates = []

    for idx in range(1, n):
        area = int(
            stats[
                idx,
                cv2.CC_STAT_AREA
            ]
        )

        if area < 10:
            continue

        comp_ys, comp_xs = np.where(
            labels == idx
        )

        candidates.append({
            "top_y": int(comp_ys.min()),
            "cx": float(comp_xs.mean()),
            "cy": float(comp_ys.mean()),
            "area": area,
        })

    if not candidates:
        return None

    finger = min(
        candidates,
        key=lambda x: x["top_y"]
    )

    finger_x = finger["cx"]
    finger_y = finger["cy"]

    # ------------------------------
    # wrist 중심
    # ------------------------------
    wrist_y1 = int(
        y_min
        + height * 0.82
    )

    wrist_y2 = int(
        y_min
        + height * 0.92
    )

    local_ys, wrist_xs = np.where(
        mask[
            wrist_y1:wrist_y2 + 1
        ] > 0
    )

    if len(wrist_xs) < 20:
        return None

    wrist_ys = (
        local_ys + wrist_y1
    )

    wrist_x = float(
        np.median(wrist_xs)
    )

    wrist_y = float(
        np.median(wrist_ys)
    )

    dx = (
        finger_x - wrist_x
    )

    dy = (
        finger_y - wrist_y
    )

    residual = float(
        np.degrees(
            np.arctan2(
                dx,
                -dy
            )
        )
    )

    # notebook에서 사용한 안전 제한
    residual = float(
        np.clip(
            residual,
            -12.0,
            12.0
        )
    )

    return {
        "residual": residual,
        "finger_x": finger_x,
        "finger_y": finger_y,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
    }


def get_total_rotation(
    mask: np.ndarray
):
    pca = get_pca_axis(
        mask
    )

    if pca is None:
        return None

    pca_angle = float(
        pca["rotation"]
    )

    # residual 계산용으로 mask만 1차 회전
    pca_mask = rotate_mask_bound(
        mask,
        pca_angle
    )

    fw = get_finger_wrist_residual(
        pca_mask
    )

    if fw is None:
        fw_angle = 0.0
        fw_status = "FW_FAIL_USE_PCA_ONLY"
    else:
        fw_angle = float(
            fw["residual"]
        )
        fw_status = "OK"

    total_angle = (
        pca_angle + fw_angle
    )

    return {
        "pca_angle": pca_angle,
        "fw_angle": fw_angle,
        "total_angle": float(total_angle),
        "fw_status": fw_status,
    }


# ============================================================
# 7. 원본 native image + mask를 단 한 번 회전
# ============================================================

def estimate_background_value(
    image: np.ndarray,
    mask: np.ndarray
):
    """
    회전으로 새로 생기는 외곽 영역을 채울 값.
    손 mask 바깥의 median 사용.
    원래 영상 내부 픽셀은 건드리지 않음.
    """
    bg_pixels = image[
        mask == 0
    ]

    if bg_pixels.size == 0:
        if image.ndim == 2:
            return 0
        return tuple(
            0 for _ in range(
                image.shape[2]
            )
        )

    if image.ndim == 2:
        return float(
            np.median(bg_pixels)
        )

    med = np.median(
        bg_pixels,
        axis=0
    )

    return tuple(
        float(v)
        for v in med
    )


def rotate_native_pair_once(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float
):
    """
    실제 저장 대상 image와 mask를 동일 affine matrix로
    딱 한 번만 회전.
    """
    h, w = image.shape[:2]

    if mask.shape[:2] != (h, w):
        raise ValueError(
            "rotate_native_pair_once: image/mask 크기가 다릅니다."
        )

    cx = w / 2.0
    cy = h / 2.0

    M = cv2.getRotationMatrix2D(
        (cx, cy),
        angle,
        1.0
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(
        math.ceil(
            h * sin + w * cos
        )
    )

    new_h = int(
        math.ceil(
            h * cos + w * sin
        )
    )

    M[0, 2] += (
        new_w / 2.0 - cx
    )

    M[1, 2] += (
        new_h / 2.0 - cy
    )

    bg = estimate_background_value(
        image,
        mask
    )

    rotated_img = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg
    )

    rotated_mask = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    rotated_mask = (
        rotated_mask > 127
    ).astype(np.uint8)

    return (
        rotated_img,
        rotated_mask,
        M
    )


# ============================================================
# 8. rotated mask bbox + final 4/3/2 margin
# ============================================================

def crop_by_rotated_mask(
    image: np.ndarray,
    mask: np.ndarray
):
    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1

    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    bw = x2 - x1
    bh = y2 - y1

    mx = int(round(
        bw * FINAL_MARGIN_X
    ))

    mt = int(round(
        bh * FINAL_MARGIN_TOP
    ))

    mb = int(round(
        bh * FINAL_MARGIN_BOTTOM
    ))

    H, W = image.shape[:2]

    fx1 = max(
        0,
        x1 - mx
    )

    fx2 = min(
        W,
        x2 + mx
    )

    fy1 = max(
        0,
        y1 - mt
    )

    fy2 = min(
        H,
        y2 + mb
    )

    if fx2 <= fx1 or fy2 <= fy1:
        return None

    crop_img = image[
        fy1:fy2,
        fx1:fx2
    ].copy()

    crop_mask = mask[
        fy1:fy2,
        fx1:fx2
    ].copy()

    return {
        "image": crop_img,
        "mask": crop_mask,
        "box": (
            fx1, fy1,
            fx2, fy2
        ),
        "raw_mask_box": (
            x1, y1,
            x2, y2
        ),
    }


# ============================================================
# 9. 한 장 전체 처리
# ============================================================

def process_one(
    raw_path: Path,
    exp,
    yolox_model,
    yolox_preproc,
    seg_model,
    device: torch.device,
):
    # --------------------------------------------------------
    # 추론용 8-bit BGR
    # --------------------------------------------------------
    model_bgr = read_model_bgr8(
        raw_path
    )

    # --------------------------------------------------------
    # 최종 저장용 native 원본
    # --------------------------------------------------------
    native = read_native_image(
        raw_path
    )

    if native.shape[:2] != model_bgr.shape[:2]:
        raise RuntimeError(
            f"native/model image 크기 불일치: {raw_path.name}"
        )

    # --------------------------------------------------------
    # YOLOX
    # --------------------------------------------------------
    det = yolox_detect_hand(
        model_bgr,
        exp,
        yolox_model,
        yolox_preproc,
        device
    )

    if det is None:
        return {
            "status": "YOLO_FAIL"
        }

    yolo_box = det["box"]
    yolo_score = det["score"]

    # --------------------------------------------------------
    # Seg input용 10% margin crop
    # --------------------------------------------------------
    seg_box = add_seg_margin(
        yolo_box,
        model_bgr.shape
    )

    if seg_box is None:
        return {
            "status": "SEG_CROP_FAIL",
            "yolo_score": yolo_score,
        }

    sx1, sy1, sx2, sy2 = seg_box

    seg_crop_bgr = model_bgr[
        sy1:sy2,
        sx1:sx2
    ].copy()

    # 최종 저장에 사용할 native crop
    native_crop = native[
        sy1:sy2,
        sx1:sx2
    ].copy()

    if native_crop.shape[:2] != seg_crop_bgr.shape[:2]:
        return {
            "status": "NATIVE_SEG_CROP_SIZE_MISMATCH",
            "yolo_score": yolo_score,
        }

    # --------------------------------------------------------
    # Seg
    # --------------------------------------------------------
    mask_raw = predict_seg_mask(
        seg_crop_bgr,
        seg_model,
        device
    )

    mask = clean_hand_mask(
        mask_raw
    )

    mask_area = int(
        mask.sum()
    )

    if mask_area < 100:
        return {
            "status": "MASK_TOO_SMALL",
            "yolo_score": yolo_score,
        }

    mask_area_ratio = float(
        mask_area
        / mask.size
    )

    # --------------------------------------------------------
    # orientation
    # --------------------------------------------------------
    orient = get_total_rotation(
        mask
    )

    if orient is None:
        return {
            "status": "PCA_FAIL",
            "yolo_score": yolo_score,
            "mask_area_ratio": mask_area_ratio,
        }

    total_angle = orient[
        "total_angle"
    ]

    # --------------------------------------------------------
    # 실제 native image + mask 회전은 여기서 딱 한 번
    # --------------------------------------------------------
    rotated_img, rotated_mask, M = (
        rotate_native_pair_once(
            native_crop,
            mask,
            total_angle
        )
    )

    # --------------------------------------------------------
    # rotated mask bbox + 4/3/2
    # --------------------------------------------------------
    final = crop_by_rotated_mask(
        rotated_img,
        rotated_mask
    )

    if final is None:
        return {
            "status": "FINAL_CROP_FAIL",
            "yolo_score": yolo_score,
            "mask_area_ratio": mask_area_ratio,
            **orient,
        }

    final_img = final["image"]
    final_mask = final["mask"]

    if final_img.shape[:2] != final_mask.shape[:2]:
        return {
            "status": "FINAL_SIZE_MISMATCH",
            "yolo_score": yolo_score,
            **orient,
        }

    # 최종 mask는 반드시 0 / 255
    final_mask_u8 = (
        final_mask * 255
    ).astype(np.uint8)

    return {
        "status": "OK",
        "image": final_img,
        "mask": final_mask_u8,

        "yolo_score": yolo_score,
        "yolo_x1": yolo_box[0],
        "yolo_y1": yolo_box[1],
        "yolo_x2": yolo_box[2],
        "yolo_y2": yolo_box[3],

        "seg_x1": seg_box[0],
        "seg_y1": seg_box[1],
        "seg_x2": seg_box[2],
        "seg_y2": seg_box[3],

        "mask_area_ratio": mask_area_ratio,

        "pca_angle": orient["pca_angle"],
        "fw_angle": orient["fw_angle"],
        "total_rotation": orient["total_angle"],
        "fw_status": orient["fw_status"],

        "final_width": int(
            final_img.shape[1]
        ),
        "final_height": int(
            final_img.shape[0]
        ),

        "final_mask_area_ratio": float(
            (final_mask > 0).mean()
        ),

        "rotation_m00": float(M[0, 0]),
        "rotation_m01": float(M[0, 1]),
        "rotation_m02": float(M[0, 2]),
        "rotation_m10": float(M[1, 0]),
        "rotation_m11": float(M[1, 1]),
        "rotation_m12": float(M[1, 2]),
    }


# ============================================================
# 10. metadata
# ============================================================

METADATA_COLUMNS = [
    "split",
    "filename",
    "source_path",
    "output_image",
    "output_mask",
    "status",

    "yolo_score",
    "yolo_x1",
    "yolo_y1",
    "yolo_x2",
    "yolo_y2",

    "seg_x1",
    "seg_y1",
    "seg_x2",
    "seg_y2",

    "mask_area_ratio",

    "pca_angle",
    "fw_angle",
    "total_rotation",
    "fw_status",

    "final_width",
    "final_height",
    "final_mask_area_ratio",

    "rotation_m00",
    "rotation_m01",
    "rotation_m02",
    "rotation_m10",
    "rotation_m11",
    "rotation_m12",

    "error",
]


def metadata_key(
    split: str,
    filename: str
) -> str:
    return f"{split}|{filename}"


def load_existing_metadata(
    csv_path: Path
) -> Dict[str, dict]:

    rows: Dict[str, dict] = {}

    if not csv_path.exists():
        return rows

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = metadata_key(
                row.get("split", ""),
                row.get("filename", "")
            )
            rows[key] = row

    return rows


def save_metadata(
    csv_path: Path,
    rows: Dict[str, dict]
) -> None:

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ordered = sorted(
        rows.values(),
        key=lambda r: (
            r.get("split", ""),
            r.get("filename", "")
        )
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=METADATA_COLUMNS,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in ordered:
            out = {
                k: row.get(k, "")
                for k in METADATA_COLUMNS
            }
            writer.writerow(out)


# ============================================================
# 11. split 처리
# ============================================================

def process_split(
    split: str,
    raw_root: Path,
    raw_index: Dict[str, Path],
    qc_dir: Path,
    exp,
    yolox_model,
    yolox_preproc,
    seg_model,
    device: torch.device,
    metadata_rows: Dict[str, dict],
    metadata_csv: Path,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> None:

    out_img_dir, out_mask_dir = (
        ensure_output_dirs(
            split
        )
    )

    qc_paths = collect_image_paths(
        qc_dir
    )

    qc_stems = []

    seen = set()

    for p in qc_paths:
        if p.stem not in seen:
            seen.add(
                p.stem
            )
            qc_stems.append(
                p.stem
            )

    qc_stems = sorted(
        qc_stems
    )

    if limit is not None:
        qc_stems = qc_stems[
            :limit
        ]

    print()
    print(
        f"===== {split.upper()} ====="
    )
    print(
        "QC allowlist:",
        len(qc_stems)
    )

    missing_raw = [
        stem for stem in qc_stems
        if stem not in raw_index
    ]

    if missing_raw:
        print(
            f"[WARNING] 원본을 못 찾은 파일: {len(missing_raw)}"
        )
        print(
            "예시:",
            missing_raw[:20]
        )

    done = 0
    skipped = 0
    failed = 0

    for i, stem in enumerate(
        tqdm(
            qc_stems,
            desc=split
        ),
        start=1
    ):
        raw_path = raw_index.get(
            stem
        )

        if raw_path is None:
            key = metadata_key(
                split,
                stem + ".png"
            )

            metadata_rows[key] = {
                "split": split,
                "filename": stem + ".png",
                "status": "RAW_NOT_FOUND",
                "error": "",
            }

            failed += 1
            continue

        out_name = (
            stem + ".png"
        )

        out_img_path = (
            out_img_dir
            / out_name
        )

        out_mask_path = (
            out_mask_dir
            / out_name
        )

        key = metadata_key(
            split,
            out_name
        )

        # ----------------------------------------------------
        # resume
        # ----------------------------------------------------
        if (
            not overwrite
            and out_img_path.exists()
            and out_mask_path.exists()
        ):
            skipped += 1

            if key not in metadata_rows:
                metadata_rows[key] = {
                    "split": split,
                    "filename": out_name,
                    "source_path": str(raw_path),
                    "output_image": str(out_img_path),
                    "output_mask": str(out_mask_path),
                    "status": "SKIP_EXISTING_NO_META",
                    "error": "",
                }

            continue

        # ----------------------------------------------------
        # process
        # ----------------------------------------------------
        row = {
            "split": split,
            "filename": out_name,
            "source_path": str(raw_path),
            "output_image": str(out_img_path),
            "output_mask": str(out_mask_path),
            "error": "",
        }

        try:
            result = process_one(
                raw_path,
                exp,
                yolox_model,
                yolox_preproc,
                seg_model,
                device
            )

            status = result.get(
                "status",
                "UNKNOWN"
            )

            row.update({
                k: v
                for k, v in result.items()
                if k not in {
                    "image",
                    "mask"
                }
            })

            if status == "OK":
                final_img = result[
                    "image"
                ]

                final_mask = result[
                    "mask"
                ]

                ok_img = cv2.imwrite(
                    str(out_img_path),
                    final_img
                )

                ok_mask = cv2.imwrite(
                    str(out_mask_path),
                    final_mask
                )

                if not ok_img:
                    raise RuntimeError(
                        f"image 저장 실패: {out_img_path}"
                    )

                if not ok_mask:
                    raise RuntimeError(
                        f"mask 저장 실패: {out_mask_path}"
                    )

                # 저장 직전 shape 보증
                if (
                    final_img.shape[:2]
                    != final_mask.shape[:2]
                ):
                    raise RuntimeError(
                        "저장 image/mask shape 불일치"
                    )

                done += 1

            else:
                failed += 1

        except Exception as e:
            row["status"] = "EXCEPTION"
            row["error"] = repr(e)
            failed += 1

        metadata_rows[key] = row

        if i % META_SAVE_EVERY == 0:
            save_metadata(
                metadata_csv,
                metadata_rows
            )

    save_metadata(
        metadata_csv,
        metadata_rows
    )

    print()
    print(
        f"[{split}] 새로 생성: {done}"
    )
    print(
        f"[{split}] 기존 skip: {skipped}"
    )
    print(
        f"[{split}] 실패: {failed}"
    )


# ============================================================
# 12. main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "training",
            "validation",
            "test",
            "all"
        ],
        default="all",
        help=(
            "처리할 split. "
            "'all'은 안전을 위해 training+validation만 처리하며 test는 포함하지 않음. "
            "held-out test는 최종 확정 후 --split test로 명시적으로 실행."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트용 앞 N장만 처리"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 생성된 image/mask도 다시 처리"
    )

    args = parser.parse_args()

    check_paths()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_csv = (
        OUTPUT_ROOT
        / "preprocessing_metadata.csv"
    )

    metadata_rows = (
        load_existing_metadata(
            metadata_csv
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)
    print("Output:", OUTPUT_ROOT)

    exp, yolox_model, yolox_preproc = (
        load_yolox(
            device
        )
    )

    seg_model = load_seg_jit(
        device
    )

    # 원본 인덱스는 split마다 한 번만 생성
    train_index = None
    val_index = None
    test_index = None

    if args.split in {
        "training",
        "all"
    }:
        qc_train = find_qc_split_dir(
            "training"
        )

        print()
        print(
            "Training QC allowlist dir:",
            qc_train
        )

        train_index = build_stem_index(
            RAW_TRAIN_DIR
        )

        process_split(
            split="training",
            raw_root=RAW_TRAIN_DIR,
            raw_index=train_index,
            qc_dir=qc_train,
            exp=exp,
            yolox_model=yolox_model,
            yolox_preproc=yolox_preproc,
            seg_model=seg_model,
            device=device,
            metadata_rows=metadata_rows,
            metadata_csv=metadata_csv,
            overwrite=args.overwrite,
            limit=args.limit,
        )

    if args.split in {
        "validation",
        "all"
    }:
        qc_val = find_qc_split_dir(
            "validation"
        )

        print()
        print(
            "Validation QC allowlist dir:",
            qc_val
        )

        val_index = build_stem_index(
            RAW_VAL_DIR
        )

        process_split(
            split="validation",
            raw_root=RAW_VAL_DIR,
            raw_index=val_index,
            qc_dir=qc_val,
            exp=exp,
            yolox_model=yolox_model,
            yolox_preproc=yolox_preproc,
            seg_model=seg_model,
            device=device,
            metadata_rows=metadata_rows,
            metadata_csv=metadata_csv,
            overwrite=args.overwrite,
            limit=args.limit,
        )

    # --------------------------------------------------------
    # Held-out test
    # --------------------------------------------------------
    # 중요:
    # --split all 에는 test가 포함되지 않는다.
    # 전처리/모델 선택이 끝난 뒤에만 아래 기능을
    # `--split test` 로 명시적으로 실행한다.
    if args.split == "test":
        if not RAW_TEST_DIR.exists():
            raise FileNotFoundError(
                f"RAW_TEST_DIR가 존재하지 않습니다: {RAW_TEST_DIR}"
            )

        qc_test = find_qc_split_dir(
            "test"
        )

        print()
        print(
            "Test QC allowlist dir:",
            qc_test
        )

        test_index = build_stem_index(
            RAW_TEST_DIR
        )

        process_split(
            split="test",
            raw_root=RAW_TEST_DIR,
            raw_index=test_index,
            qc_dir=qc_test,
            exp=exp,
            yolox_model=yolox_model,
            yolox_preproc=yolox_preproc,
            seg_model=seg_model,
            device=device,
            metadata_rows=metadata_rows,
            metadata_csv=metadata_csv,
            overwrite=args.overwrite,
            limit=args.limit,
        )

    print()
    print("===== DONE =====")
    print("Output :", OUTPUT_ROOT)
    print("Metadata:", metadata_csv)


if __name__ == "__main__":
    main()