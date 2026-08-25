# -*- coding: utf-8 -*-
r"""
수부 X-ray 뼈나이 예측
===================================================

Final pipeline:
    raw hand X-ray
      -> YOLOX-S hand detection
      -> YOLO bbox + 10% margin (segmentation input)
      -> hand segmentation
      -> largest component + hole filling
      -> PCA coarse alignment + finger/wrist residual alignment
      -> native image/mask 1회 회전
      -> rotated hand mask bbox + final margin (X 4%, top 3%, bottom 2%)
      -> native foreground p1/p99 percentile normalization
      -> aspect-ratio-preserving resize + centered black padding 512x512
      -> mask 약 3px dilation
      -> background = 0
      -> grayscale repeated to RGB
      -> Run138 ConvNeXt V1-Tiny + sex-specific 240-bin LDL
      -> expected bone age in months

보존한 기존 기업 코드 동작:
- test.csv column alias 인식:
  id/image_id/imageid/patient_id/patientid/case_id/caseid
  sex/gender/male
  filename/file/image/imagefile/imagepath/path
- sex 값 M/F, male/female, 남/여, 1/0 등 인식
- predictions.csv 형식 유지:
  id, filename, sex, predicted_age
- 진행률/소요시간 출력 유지
- YOLO/전처리 실패 시 fallback하여 prediction은 계속 생성

GT bone age는 추론에 사용하지 않습니다.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

import timm
from timm.data import resolve_model_data_config


# =============================================================================
# 기업 테스트 설정
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

IMAGES_DIR = PROJECT_DIR / "Images"
METADATA_CSV = PROJECT_DIR / "test.csv"

YOLOX_DIR = PROJECT_DIR / "YOLOX"
YOLOX_EXP = PROJECT_DIR / "yolox_s_hand.py"
YOLOX_MODEL = PROJECT_DIR / "models" / "yolox_s_hand_best.pth"
SEG_MODEL = PROJECT_DIR / "models" / "hand_seg_crop512_traced.pt"

BONEAGE_MODEL = PROJECT_DIR / "models" / "best_model.pt"

OUTPUT_CSV = PROJECT_DIR / "predictions.csv"
DEVICE = "auto"

# True면 최종 ConvNeXt 입력 512x512 이미지를 crops_512에 저장
SAVE_CROPS = False

# "raw"             : 원본 X-ray -> YOLOX-S crop -> bone age
# "already_cropped" : 이미 손 crop된 영상 -> YOLOX-S 생략
INPUT_MODE = "raw"

BATCH_SIZE = 32


# =============================================================================
# Fixed final settings
# =============================================================================

MODEL_NAME = "convnext_tiny.fb_in1k"
IMAGE_SIZE = 512

IMAGE_DIM = 512
SEX_DIM = 32
FUSION_DIM = 128
IMAGE_DROPOUT = 0.20
FUSION_DROPOUT = 0.20
NUM_BINS = 240

YOLOX_CONF = 0.20
YOLOX_NMS = 0.70

# YOLO bbox -> segmentation input margin
SEG_MARGIN_X = 0.10
SEG_MARGIN_TOP = 0.10
SEG_MARGIN_BOTTOM = 0.10

# Segmentation
SEG_INPUT_SIZE = 512
SEG_THRESHOLD = 0.5
SEG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SEG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# final rotated-mask crop margin
MARGIN_LEFT = 0.04
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.03
MARGIN_BOTTOM = 0.02

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
DILATE_KERNEL_SIZE = 7
DILATE_ITERATIONS = 1
BACKGROUND_VALUE = 0

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# =============================================================================
# Bone-age model
# =============================================================================

class ConvNeXtTinyDistributionRegression(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        image_dim: int = IMAGE_DIM,
        sex_dim: int = SEX_DIM,
        fusion_dim: int = FUSION_DIM,
        image_dropout: float = IMAGE_DROPOUT,
        fusion_dropout: float = FUSION_DROPOUT,
        num_bins: int = NUM_BINS,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

        backbone_dim = int(self.backbone.num_features)

        self.image_head = nn.Sequential(
            nn.Linear(backbone_dim, image_dim),
            nn.LayerNorm(image_dim),
            nn.GELU(),
            nn.Dropout(image_dropout),
        )

        self.sex_embedding = nn.Sequential(
            nn.Linear(1, sex_dim),
            nn.GELU(),
        )

        self.fusion_trunk = nn.Sequential(
            nn.Linear(image_dim + sex_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        self.male_output = nn.Linear(fusion_dim, num_bins)
        self.female_output = nn.Linear(fusion_dim, num_bins)

        self.register_buffer(
            "age_bins",
            torch.arange(1, num_bins + 1, dtype=torch.float32),
        )

    def forward(self, image: torch.Tensor, male: torch.Tensor):
        image_feature = self.image_head(self.backbone(image))
        sex_feature = self.sex_embedding(male)

        shared_feature = self.fusion_trunk(
            torch.cat([image_feature, sex_feature], dim=1)
        )

        male_logits = self.male_output(shared_feature)
        female_logits = self.female_output(shared_feature)

        gate = (
            male.reshape(-1, 1)
            .to(dtype=male_logits.dtype)
            .clamp(0.0, 1.0)
        )

        logits = (
            gate * male_logits
            + (1.0 - gate) * female_logits
        )

        probability = torch.softmax(logits, dim=1)

        pred_age = torch.sum(
            probability * self.age_bins.unsqueeze(0),
            dim=1,
        )

        return logits, probability, pred_age


# =============================================================================
# Environment / metadata
# =============================================================================

def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(spec)


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(
    dataframe: pd.DataFrame,
    aliases: Iterable[str],
) -> Optional[str]:
    normalized = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def parse_male(value) -> float:
    if pd.isna(value):
        raise ValueError("sex 값이 비어 있습니다.")

    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return float(int(value))

    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return float(value)

    text = str(value).strip().lower()

    male_values = {
        "m", "male", "man", "boy", "남", "남자", "1", "true"
    }
    female_values = {
        "f", "female", "woman", "girl", "여", "여자", "0", "false"
    }

    if text in male_values:
        return 1.0

    if text in female_values:
        return 0.0

    raise ValueError(f"인식할 수 없는 sex 값: {value!r}")


def build_image_index(images_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = path.stem

            if stem in index:
                raise RuntimeError(
                    "동일한 파일 stem이 여러 개 있습니다: "
                    f"{stem}\n{index[stem]}\n{path}"
                )

            index[stem] = path

    if not index:
        raise FileNotFoundError(f"지원되는 이미지가 없습니다: {images_dir}")

    return index


def prepare_metadata(
    metadata_csv: Path,
    images_dir: Path,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv, dtype=str)

    if len(df) == 0:
        raise ValueError("metadata CSV가 비어 있습니다.")

    id_col = find_column(
        df,
        ["id", "image_id", "imageid", "patient_id", "patientid", "case_id", "caseid"],
    )
    sex_col = find_column(df, ["sex", "gender", "male"])
    filename_col = find_column(
        df,
        ["filename", "file", "image", "imagefile", "imagepath", "path"],
    )

    if id_col is None:
        raise ValueError("metadata CSV에 id 열이 필요합니다.")

    if sex_col is None:
        raise ValueError(
            "metadata CSV에 sex/gender/male 열이 필요합니다. "
            "최종 모델은 성별 입력을 사용합니다."
        )

    image_index = build_image_index(images_dir)
    rows = []

    for _, row in df.iterrows():
        image_id = str(row[id_col]).strip()

        if not image_id:
            raise ValueError("빈 id가 있습니다.")

        male = parse_male(row[sex_col])

        image_path = None

        if filename_col is not None and not pd.isna(row[filename_col]):
            raw_name = str(row[filename_col]).strip()
            candidate = Path(raw_name)

            if not candidate.is_absolute():
                candidate = images_dir / candidate

            if candidate.is_file():
                image_path = candidate
            else:
                image_path = image_index.get(Path(raw_name).stem)

        if image_path is None:
            image_path = image_index.get(Path(image_id).stem)

        if image_path is None:
            raise FileNotFoundError(
                f"id={image_id}에 해당하는 이미지를 찾지 못했습니다."
            )

        rows.append(
            {
                "id": image_id,
                "filename": image_path.name,
                "image_path": str(image_path.resolve()),
                "male": male,
                "sex": "M" if male >= 0.5 else "F",
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# YOLOX-S detector
# =============================================================================

def load_yolox_detector(
    *,
    yolox_dir: Path,
    exp_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    if not yolox_dir.is_dir():
        raise FileNotFoundError(
            f"YOLOX source 폴더가 없습니다: {yolox_dir}\n"
            "패키지의 YOLOX/yolox 폴더를 확인하세요."
        )

    if str(yolox_dir) not in sys.path:
        sys.path.insert(0, str(yolox_dir))

    try:
        from yolox.exp import get_exp
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess
    except ImportError as exc:
        raise ImportError(
            "YOLOX import에 실패했습니다. "
            "패키지의 YOLOX 폴더와 requirements 설치 상태를 확인하세요."
        ) from exc

    exp = get_exp(str(exp_path), None)
    model = exp.get_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    preproc = ValTransform(legacy=False)

    return {
        "exp": exp,
        "model": model,
        "preproc": preproc,
        "postprocess": postprocess,
    }


def to_detector_bgr8(image: np.ndarray) -> np.ndarray:
    """
    학습용 tight YOLOX crop 생성 코드와 동일하게 detector 입력만
    임시 8-bit BGR로 변환합니다. 실제 crop은 이 배열에서 만들지 않습니다.
    """
    if image is None:
        raise ValueError("image is None")

    x = image

    if x.ndim == 2:
        gray = x
    elif x.ndim == 3:
        if x.shape[2] == 1:
            gray = x[..., 0]
        elif x.shape[2] == 3:
            if x.dtype == np.uint8:
                return x
            gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        elif x.shape[2] == 4:
            if x.dtype == np.uint8:
                return cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(x, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"지원하지 않는 channel shape: {x.shape}")
    else:
        raise ValueError(f"지원하지 않는 image shape: {x.shape}")

    if gray.dtype == np.uint8:
        gray8 = gray
    else:
        arr = gray.astype(np.float32, copy=False)
        finite = np.isfinite(arr)

        if not finite.any():
            raise ValueError("finite pixel이 없습니다.")

        vals = arr[finite]
        lo = float(vals.min())
        hi = float(vals.max())

        if hi <= lo:
            gray8 = np.zeros(gray.shape, dtype=np.uint8)
        else:
            scaled = (arr - lo) * (255.0 / (hi - lo))
            scaled[~finite] = 0.0
            gray8 = np.clip(scaled, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)


def expand_box(
    box_xyxy: Tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box_xyxy

    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)

    x0 = x0 - width * MARGIN_LEFT
    x1 = x1 + width * MARGIN_RIGHT
    y0 = y0 - height * MARGIN_TOP
    y1 = y1 + height * MARGIN_BOTTOM

    x0 = int(max(0, math.floor(x0)))
    y0 = int(max(0, math.floor(y0)))
    x1 = int(min(image_width, math.ceil(x1)))
    y1 = int(min(image_height, math.ceil(y1)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("유효하지 않은 crop box입니다.")

    return x0, y0, x1, y1


def detect_and_crop(
    *,
    detector,
    image_path: Path,
    device: torch.device,
) -> Tuple[Image.Image, bool]:
    """
    기존 기업 전달 버전과 동일한 동작.

    YOLOX-S 검출 성공:
        margin을 적용한 hand crop 반환, used_fallback=False

    YOLOX-S 검출 실패:
        원본 전체 grayscale 이미지 반환, used_fallback=True
    """
    # 학습용 tight YOLOX crop 생성 코드와 동일하게 원본을 unchanged로 읽고,
    # detector 입력만 임시 8-bit BGR로 변환합니다.
    original = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if original is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {image_path}")

    h, w = original.shape[:2]
    detector_input = to_detector_bgr8(original)

    exp = detector["exp"]
    model = detector["model"]
    preproc = detector["preproc"]
    postprocess = detector["postprocess"]

    ratio = min(
        exp.test_size[0] / h,
        exp.test_size[1] / w,
    )

    tensor, _ = preproc(
        detector_input,
        None,
        exp.test_size,
    )

    tensor = (
        torch.from_numpy(tensor)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.inference_mode():
        outputs = model(tensor)

        output = postprocess(
            outputs,
            num_classes=1,
            conf_thre=YOLOX_CONF,
            nms_thre=YOLOX_NMS,
            class_agnostic=True,
        )[0]

    with Image.open(image_path) as image:
        gray = image.convert("L").copy()

    if output is None or len(output) == 0:
        return gray, True

    output = output.detach().cpu()

    boxes = output[:, :4] / ratio
    scores = output[:, 4] * output[:, 5]

    best_index = int(torch.argmax(scores).item())
    xyxy = boxes[best_index].numpy().tolist()

    image_width, image_height = gray.size

    crop_box = expand_box(
        tuple(float(v) for v in xyxy),
        image_width=image_width,
        image_height=image_height,
    )

    return gray.crop(crop_box).copy(), False



# =============================================================================
# Hand segmentation + final aligned preprocessing
# =============================================================================

def load_segmentation_model(
    checkpoint_path: Path,
    device: torch.device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"segmentation model 없음: {checkpoint_path}"
        )

    model = torch.jit.load(
        str(checkpoint_path),
        map_location=device,
    )
    model.eval()
    return model


def predict_seg_mask(
    crop_bgr: np.ndarray,
    seg_model,
    device: torch.device,
) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    x = cv2.resize(
        rgb,
        (SEG_INPUT_SIZE, SEG_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    x = x.astype(np.float32) / 255.0
    x = (x - SEG_MEAN) / SEG_STD

    x = (
        torch.from_numpy(x.transpose(2, 0, 1))
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.inference_mode():
        logits = seg_model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.ndim != 4:
            raise RuntimeError(
                f"segmentation output shape 오류: {tuple(logits.shape)}"
            )

        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)[0, 0]
        else:
            prob = torch.softmax(logits, dim=1)[0, 1]

        mask = (
            prob.detach().cpu().numpy() >= SEG_THRESHOLD
        ).astype(np.uint8)

    return cv2.resize(
        mask,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def clean_hand_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if n_labels <= 1:
        return mask

    largest_idx = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    clean = (labels == largest_idx).astype(np.uint8)

    padded = np.pad(
        clean,
        1,
        mode="constant",
        constant_values=0,
    )

    inv = (1 - padded).astype(np.uint8)
    flood = inv.copy()
    ff_mask = np.zeros(
        (flood.shape[0] + 2, flood.shape[1] + 2),
        dtype=np.uint8,
    )

    cv2.floodFill(
        flood,
        ff_mask,
        seedPoint=(0, 0),
        newVal=2,
    )

    holes = (flood == 1).astype(np.uint8)
    filled = np.clip(
        padded + holes,
        0,
        1,
    ).astype(np.uint8)

    return filled[1:-1, 1:-1]


def get_pca_axis(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) < 100:
        return None

    points = np.column_stack(
        [xs, ys]
    ).astype(np.float32)

    _, eigenvectors, _ = cv2.PCACompute2(
        points,
        mean=None,
    )

    axis = eigenvectors[0]

    theta = np.degrees(
        np.arctan2(axis[1], axis[0])
    ) % 180.0

    return float(theta - 90.0)


def rotate_mask_bound(
    mask: np.ndarray,
    angle: float,
) -> np.ndarray:
    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D(
        (cx, cy),
        angle,
        1.0,
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    out = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (out > 127).astype(np.uint8)


def get_finger_wrist_residual(mask: np.ndarray):
    mask = (mask > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return None

    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1

    top_limit = int(
        y_min + height * 0.12
    )

    top_mask = np.zeros_like(mask)
    top_mask[y_min:top_limit + 1] = mask[y_min:top_limit + 1]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        top_mask,
        connectivity=8,
    )

    candidates = []

    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < 10:
            continue

        comp_ys, comp_xs = np.where(
            labels == idx
        )

        candidates.append(
            {
                "top_y": int(comp_ys.min()),
                "cx": float(comp_xs.mean()),
                "cy": float(comp_ys.mean()),
            }
        )

    if not candidates:
        return None

    finger = min(
        candidates,
        key=lambda item: item["top_y"],
    )

    wrist_y1 = int(
        y_min + height * 0.82
    )
    wrist_y2 = int(
        y_min + height * 0.92
    )

    local_ys, wrist_xs = np.where(
        mask[wrist_y1:wrist_y2 + 1] > 0
    )

    if len(wrist_xs) < 20:
        return None

    wrist_ys = local_ys + wrist_y1

    wrist_x = float(np.median(wrist_xs))
    wrist_y = float(np.median(wrist_ys))

    dx = finger["cx"] - wrist_x
    dy = finger["cy"] - wrist_y

    residual = float(
        np.degrees(
            np.arctan2(dx, -dy)
        )
    )

    return float(
        np.clip(residual, -12.0, 12.0)
    )


def get_total_rotation(mask: np.ndarray):
    pca_angle = get_pca_axis(mask)

    if pca_angle is None:
        return None

    pca_mask = rotate_mask_bound(
        mask,
        pca_angle,
    )

    residual = get_finger_wrist_residual(
        pca_mask
    )

    if residual is None:
        residual = 0.0

    return float(pca_angle + residual)


def estimate_background_value(
    image: np.ndarray,
    mask: np.ndarray,
):
    bg_pixels = image[mask == 0]

    if bg_pixels.size == 0:
        if image.ndim == 2:
            return 0
        return tuple(
            0 for _ in range(image.shape[2])
        )

    if image.ndim == 2:
        return float(
            np.median(bg_pixels)
        )

    med = np.median(
        bg_pixels,
        axis=0,
    )

    return tuple(
        float(v) for v in med
    )


def rotate_native_pair_once(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float,
):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D(
        (cx, cy),
        angle,
        1.0,
    )

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(
        math.ceil(h * sin + w * cos)
    )
    new_h = int(
        math.ceil(h * cos + w * sin)
    )

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    bg = estimate_background_value(
        image,
        mask,
    )

    rotated_img = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )

    rotated_mask = cv2.warpAffine(
        (mask * 255).astype(np.uint8),
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        rotated_img,
        (rotated_mask > 127).astype(np.uint8),
    )


def crop_by_rotated_mask(
    image: np.ndarray,
    mask: np.ndarray,
):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    bw = x2 - x1
    bh = y2 - y1

    ml = int(round(bw * MARGIN_LEFT))
    mr = int(round(bw * MARGIN_RIGHT))
    mt = int(round(bh * MARGIN_TOP))
    mb = int(round(bh * MARGIN_BOTTOM))

    H, W = image.shape[:2]

    fx1 = max(0, x1 - ml)
    fx2 = min(W, x2 + mr)
    fy1 = max(0, y1 - mt)
    fy2 = min(H, y2 + mb)

    if fx2 <= fx1 or fy2 <= fy1:
        return None

    return (
        image[fy1:fy2, fx1:fx2].copy(),
        mask[fy1:fy2, fx1:fx2].copy(),
    )


def to_gray_native(image: np.ndarray):
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    raise ValueError(
        f"지원하지 않는 image shape: {image.shape}"
    )


def masked_percentile_normalize_native(
    image_gray: np.ndarray,
    mask: np.ndarray,
):
    values = (
        image_gray.astype(np.float32)[mask > 0]
    )

    if values.size == 0:
        raise ValueError(
            "mask 내부 픽셀이 없습니다."
        )

    p1 = float(
        np.percentile(
            values,
            LOW_PERCENTILE,
        )
    )
    p99 = float(
        np.percentile(
            values,
            HIGH_PERCENTILE,
        )
    )

    if (
        not np.isfinite(p1)
        or not np.isfinite(p99)
        or p99 <= p1 + 1e-6
    ):
        raise ValueError(
            f"invalid percentile range: {p1} ~ {p99}"
        )

    x = (
        image_gray.astype(np.float32) - p1
    ) / (
        p99 - p1
    )

    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return np.rint(
        x * 255.0
    ).astype(np.uint8)


def resize_pad_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
):
    h, w = image.shape[:2]

    scale = min(
        IMAGE_SIZE / w,
        IMAGE_SIZE / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )
    new_h = max(
        1,
        int(round(h * scale)),
    )

    image_interp = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=image_interp,
    )

    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas_image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )

    canvas_mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )

    x0 = (IMAGE_SIZE - new_w) // 2
    y0 = (IMAGE_SIZE - new_h) // 2

    canvas_image[
        y0:y0+new_h,
        x0:x0+new_w,
    ] = resized_image

    canvas_mask[
        y0:y0+new_h,
        x0:x0+new_w,
    ] = (
        resized_mask > 0
    ).astype(np.uint8)

    return canvas_image, canvas_mask


def final_masked_512(
    image_gray: np.ndarray,
    mask: np.ndarray,
):
    normalized = masked_percentile_normalize_native(
        image_gray,
        mask,
    )

    image_512, mask_512 = resize_pad_image_and_mask(
        normalized,
        mask,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            DILATE_KERNEL_SIZE,
            DILATE_KERNEL_SIZE,
        ),
    )

    keep = cv2.dilate(
        mask_512,
        kernel,
        iterations=DILATE_ITERATIONS,
    ) > 0

    output = image_512.copy()
    output[~keep] = BACKGROUND_VALUE

    return Image.fromarray(
        output,
        mode="L",
    )


def detect_segment_align_prepare(
    *,
    detector,
    seg_model,
    image_path: Path,
    device: torch.device,
) -> Tuple[Image.Image, bool]:
    """
    정상:
        YOLOX -> seg -> align -> masked p1/p99 -> 512 bgremove

    fallback:
        기존 기업 코드처럼 prediction 자체는 계속 생성합니다.
        YOLO 미검출이면 원본 전체, 이후 단계 오류면 기존 YOLO crop을 사용합니다.
    """
    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED,
    )

    if original is None:
        raise RuntimeError(
            f"이미지를 읽지 못했습니다: {image_path}"
        )

    h, w = original.shape[:2]
    detector_input = to_detector_bgr8(
        original
    )

    exp = detector["exp"]
    model = detector["model"]
    preproc = detector["preproc"]
    postprocess = detector["postprocess"]

    ratio = min(
        exp.test_size[0] / h,
        exp.test_size[1] / w,
    )

    tensor, _ = preproc(
        detector_input,
        None,
        exp.test_size,
    )

    tensor = (
        torch.from_numpy(tensor)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.inference_mode():
        outputs = model(tensor)

        output = postprocess(
            outputs,
            num_classes=1,
            conf_thre=YOLOX_CONF,
            nms_thre=YOLOX_NMS,
            class_agnostic=True,
        )[0]

    # Preserve previous enterprise fallback behavior.
    if output is None or len(output) == 0:
        with Image.open(image_path) as image:
            return (
                resize_pad_512(
                    image.convert("L")
                ),
                True,
            )

    output = output.detach().cpu()

    boxes = output[:, :4] / ratio
    scores = output[:, 4] * output[:, 5]

    best_index = int(
        torch.argmax(scores).item()
    )

    x0, y0, x1, y1 = (
        float(v)
        for v in boxes[best_index].numpy().tolist()
    )

    # Keep old crop ready as fallback.
    fallback_box = expand_box(
        (x0, y0, x1, y1),
        image_width=w,
        image_height=h,
    )

    fallback_gray = to_gray_native(
        original
    )
    fx0, fy0, fx1, fy1 = fallback_box
    fallback_crop = Image.fromarray(
        fallback_gray[
            fy0:fy1,
            fx0:fx1,
        ].astype(np.uint8),
        mode="L",
    )

    try:
        # Wider crop only for segmentation input.
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)

        sx0 = int(
            max(
                0,
                math.floor(
                    x0 - bw * SEG_MARGIN_X
                ),
            )
        )
        sx1 = int(
            min(
                w,
                math.ceil(
                    x1 + bw * SEG_MARGIN_X
                ),
            )
        )
        sy0 = int(
            max(
                0,
                math.floor(
                    y0 - bh * SEG_MARGIN_TOP
                ),
            )
        )
        sy1 = int(
            min(
                h,
                math.ceil(
                    y1 + bh * SEG_MARGIN_BOTTOM
                ),
            )
        )

        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError(
                "segmentation crop box 오류"
            )

        seg_crop_bgr = detector_input[
            sy0:sy1,
            sx0:sx1,
        ].copy()

        native_crop = original[
            sy0:sy1,
            sx0:sx1,
        ].copy()

        mask = clean_hand_mask(
            predict_seg_mask(
                seg_crop_bgr,
                seg_model,
                device,
            )
        )

        if int(mask.sum()) < 100:
            raise ValueError(
                "segmentation mask too small"
            )

        total_angle = get_total_rotation(
            mask
        )

        if total_angle is None:
            raise ValueError(
                "orientation estimation failed"
            )

        rotated_image, rotated_mask = (
            rotate_native_pair_once(
                native_crop,
                mask,
                total_angle,
            )
        )

        cropped = crop_by_rotated_mask(
            rotated_image,
            rotated_mask,
        )

        if cropped is None:
            raise ValueError(
                "rotated-mask crop failed"
            )

        final_image, final_mask = cropped

        gray = to_gray_native(
            final_image
        )

        prepared = final_masked_512(
            gray,
            final_mask,
        )

        return prepared, False

    except Exception:
        # Do not change the old enterprise output contract.
        # If segmentation-specific preprocessing fails, use the previous YOLO crop.
        return (
            resize_pad_512(
                fallback_crop
            ),
            True,
        )


# =============================================================================
# Final preprocessing
# =============================================================================

def resize_pad_512(gray_image: Image.Image) -> Image.Image:
    """
    run71 학습 데이터와 동일한 최종 전처리:
      PIL convert("L")
      -> aspect ratio 유지
      -> OpenCV INTER_AREA resize
      -> 512x512 centered black padding
    """
    gray = gray_image.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    height, width = arr.shape

    if width <= 0 or height <= 0:
        raise ValueError("빈 이미지입니다.")

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height,
    )

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        arr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )

    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width,
    ] = resized

    return Image.fromarray(canvas, mode="L")


def build_model_transform(model_name: str):
    probe = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    data_cfg = resolve_model_data_config(probe)
    del probe

    mean = data_cfg.get("mean", (0.5, 0.5, 0.5))
    std = data_cfg.get("std", (0.5, 0.5, 0.5))

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# =============================================================================
# Bone-age model load / inference
# =============================================================================

def load_boneage_model(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    cfg = dict(checkpoint.get("config", {}))

    model = ConvNeXtTinyDistributionRegression(
        model_name=cfg.get("model_name", MODEL_NAME),
        image_dim=int(cfg.get("image_dim", IMAGE_DIM)),
        sex_dim=int(cfg.get("sex_dim", SEX_DIM)),
        fusion_dim=int(cfg.get("fusion_dim", FUSION_DIM)),
        image_dropout=float(cfg.get("image_dropout", IMAGE_DROPOUT)),
        fusion_dropout=float(cfg.get("fusion_dropout", FUSION_DROPOUT)),
        num_bins=int(cfg.get("num_bins", NUM_BINS)),
    ).to(device)

    # Run138 final checkpoint uses model_state_dict.
    # Keep EMA as fallback only for older checkpoints.
    state = checkpoint.get(
        "model_state_dict",
        checkpoint.get("ema_model_state_dict"),
    )

    if state is None:
        raise KeyError(
            "bone-age checkpoint에 model state가 없습니다."
        )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model, checkpoint, cfg


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remain = divmod(seconds, 3600)
    minutes, seconds = divmod(remain, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@torch.no_grad()
def predict_batch(
    *,
    model,
    tensors: List[torch.Tensor],
    males: List[float],
    device: torch.device,
) -> np.ndarray:
    images = torch.stack(tensors, dim=0).to(
        device,
        non_blocking=True,
    )

    male_tensor = torch.tensor(
        males,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    _, _, pred_age = model(images, male_tensor)

    return (
        pred_age
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.perf_counter()

    images_dir = IMAGES_DIR.expanduser().resolve()
    metadata_csv = METADATA_CSV.expanduser().resolve()
    yolox_dir = YOLOX_DIR.expanduser().resolve()
    yolox_exp = YOLOX_EXP.expanduser().resolve()
    yolox_model_path = YOLOX_MODEL.expanduser().resolve()
    seg_model_path = SEG_MODEL.expanduser().resolve()
    boneage_model_path = BONEAGE_MODEL.expanduser().resolve()
    output_csv = OUTPUT_CSV.expanduser().resolve()

    input_mode = str(INPUT_MODE).strip().lower()
    save_crops = bool(SAVE_CROPS)
    batch_size = int(BATCH_SIZE)

    if input_mode not in {"raw", "already_cropped"}:
        raise ValueError(
            'INPUT_MODE는 "raw" 또는 "already_cropped"여야 합니다.'
        )

    for path, label in [
        (images_dir, "Images 폴더"),
        (metadata_csv, "test.csv"),
        (boneage_model_path, "models/best_model.pt"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 없음: {path}")

    if input_mode == "raw":
        for path, label in [
            (yolox_dir, "YOLOX source 폴더"),
            (yolox_exp, "yolox_s_hand.py"),
            (yolox_model_path, "models/yolox_s_hand_best.pth"),
            (seg_model_path, "models/hand_seg_crop512_traced.pt"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} 없음: {path}")

    if batch_size <= 0:
        raise ValueError("BATCH_SIZE는 1 이상이어야 합니다.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    crop_dir = output_csv.parent / "crops_512"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(DEVICE)

    metadata = prepare_metadata(
        metadata_csv,
        images_dir,
    )

    total_cases = len(metadata)

    print(
        f"\r추론 준비 중... | 0/{total_cases} (0.0%) "
        f"| 경과 {format_elapsed(time.perf_counter() - total_start)}",
        end="",
        flush=True,
    )

    model, _, cfg = load_boneage_model(
        boneage_model_path,
        device,
    )

    model_transform = build_model_transform(
        cfg.get("model_name", MODEL_NAME)
    )

    detector = None
    seg_model = None

    if input_mode == "raw":
        detector = load_yolox_detector(
            yolox_dir=yolox_dir,
            exp_path=yolox_exp,
            checkpoint_path=yolox_model_path,
            device=device,
        )

        seg_model = load_segmentation_model(
            seg_model_path,
            device,
        )

    results = []

    pending_tensors: List[torch.Tensor] = []
    pending_males: List[float] = []
    pending_result_indices: List[int] = []

    fallback_count = 0
    error_count = 0

    def flush_pending():
        if not pending_tensors:
            return

        predictions = predict_batch(
            model=model,
            tensors=pending_tensors,
            males=pending_males,
            device=device,
        )

        for result_index, prediction in zip(
            pending_result_indices,
            predictions,
        ):
            results[result_index]["predicted_age"] = float(prediction)

        pending_tensors.clear()
        pending_males.clear()
        pending_result_indices.clear()

    for row_index, row in metadata.iterrows():
        image_path = Path(row["image_path"])

        record = {
            "id": row["id"],
            "filename": row["filename"],
            "sex": row["sex"],
            "predicted_age": np.nan,
        }

        try:
            if input_mode == "raw":
                prepared, used_fallback = detect_segment_align_prepare(
                    detector=detector,
                    seg_model=seg_model,
                    image_path=image_path,
                    device=device,
                )

                if used_fallback:
                    fallback_count += 1
            else:
                with Image.open(image_path) as image:
                    prepared = resize_pad_512(
                        image.convert("L")
                    )

            if save_crops:
                prepared.save(
                    crop_dir / f"{row['id']}.png",
                    format="PNG",
                )

            tensor = model_transform(
                prepared.convert("RGB")
            )

            results.append(record)

            pending_tensors.append(tensor)
            pending_males.append(float(row["male"]))
            pending_result_indices.append(len(results) - 1)

            if len(pending_tensors) >= batch_size:
                flush_pending()

        except Exception:
            error_count += 1
            results.append(record)

        done = row_index + 1
        elapsed = time.perf_counter() - total_start

        print(
            f"\r추론 진행: {done}/{total_cases} "
            f"({done / total_cases * 100:.1f}%) "
            f"| 경과 {format_elapsed(elapsed)}",
            end="",
            flush=True,
        )

    flush_pending()

    result_df = pd.DataFrame(
        results,
        columns=[
            "id",
            "filename",
            "sex",
            "predicted_age",
        ],
    )

    result_df["predicted_age"] = (
        result_df["predicted_age"].round(6)
    )

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    total_elapsed = time.perf_counter() - total_start
    predicted_count = int(
        result_df["predicted_age"].notna().sum()
    )

    print(
        f"\r추론 완료: {predicted_count}/{total_cases} "
        f"| 총 소요시간 {format_elapsed(total_elapsed)}"
        + " " * 20
    )

    print(f"결과 저장: {output_csv.name}")

    if fallback_count > 0:
        print(
            f"전처리 fallback 사용: {fallback_count}건"
        )

    if error_count > 0:
        print(
            f"처리 오류: {error_count}건"
        )


if __name__ == "__main__":
    main()