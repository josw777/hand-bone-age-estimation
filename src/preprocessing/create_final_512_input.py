# -*- coding: utf-8 -*-
r"""
create_final_512_input.py

수부 X-ray 최종 512x512 모델 입력 생성.

처리 순서
---------
1) Seg-aligned native image + segmentation mask 로드
2) native mask foreground에서 p1 / p99 계산
3) p1~p99 기준 intensity를 0~255로 선형 정규화
4) aspect ratio를 유지한 채 512x512 resize + center padding
5) 최종 mask를 약 3px dilation
6) dilated mask 밖 background를 0으로 제거
7) 최종 image / mask 저장

percentile 통계는 resize/interpolation 이전의 native foreground에서 계산한다.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# 기본 설정
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "seg_aligned_native"
)

DST_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_512_input"
)

SPLITS = ("train", "validation", "test")

IMAGE_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

TARGET_SIZE = 512

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

# 512x512 좌표계에서 약 3px margin
DILATE_KERNEL_SIZE = 7
DILATE_ITERATIONS = 1

BACKGROUND_VALUE = 0


# =============================================================================
# 유틸
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=Path,
        default=SRC_ROOT,
    )

    parser.add_argument(
        "--dst",
        type=Path,
        default=DST_ROOT,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 출력 폴더가 있으면 삭제 후 다시 생성",
    )

    return parser.parse_args()


def collect_images(folder: Path):
    files = []

    for path in folder.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTS
        ):
            files.append(path)

    return sorted(files)


def find_same_stem(folder: Path, stem: str):
    for ext in IMAGE_EXTS:
        path = folder / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def read_gray(path: Path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"읽기 실패: {path}"
        )

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR,
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:
        raise RuntimeError(
            f"지원하지 않는 shape: {image.shape}"
        )

    # 다양한 bit-depth 입력을 처리하되,
    # 실제 percentile 계산은 float32에서 수행
    return gray


def make_binary_mask(mask_gray: np.ndarray):
    # mask가 0/1 또는 0/255 어느 쪽이어도 대응
    if mask_gray.max() <= 1:
        binary = (
            mask_gray > 0
        ).astype(np.uint8)
    else:
        binary = (
            mask_gray > 127
        ).astype(np.uint8)

    if binary.sum() == 0:
        raise RuntimeError(
            "empty segmentation mask"
        )

    return binary


# =============================================================================
# 원해상도 masked percentile
# =============================================================================

def masked_percentile_normalize_native(
    image_gray: np.ndarray,
    binary_mask: np.ndarray,
):
    """
    percentile 통계는 반드시 원해상도 segmentation foreground에서 계산.

    반환:
      normalized uint8 [0,255]
      p_low
      p_high
    """

    image_float = image_gray.astype(
        np.float32
    )

    values = image_float[
        binary_mask > 0
    ]

    if values.size == 0:
        raise RuntimeError(
            "mask 내부 픽셀이 없습니다."
        )

    p_low = float(
        np.percentile(
            values,
            LOW_PERCENTILE,
        )
    )

    p_high = float(
        np.percentile(
            values,
            HIGH_PERCENTILE,
        )
    )

    if (
        not np.isfinite(p_low)
        or not np.isfinite(p_high)
    ):
        raise RuntimeError(
            f"비정상 percentile: {p_low}, {p_high}"
        )

    if p_high <= p_low + 1e-6:
        raise RuntimeError(
            f"percentile 범위가 너무 좁음: "
            f"{p_low} ~ {p_high}"
        )

    normalized = (
        image_float - p_low
    ) / (
        p_high - p_low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized = np.rint(
        normalized * 255.0
    ).astype(np.uint8)

    return normalized, p_low, p_high


# =============================================================================
# 512 resize + padding
# =============================================================================

def resize_pad_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = TARGET_SIZE,
):
    """
    aspect ratio 유지 + center padding.

    image와 mask에 동일한 scale/padding 적용.
    """

    h, w = image.shape[:2]

    if h <= 0 or w <= 0:
        raise RuntimeError(
            f"잘못된 image shape: {image.shape}"
        )

    scale = min(
        target_size / w,
        target_size / h,
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    if scale < 1.0:
        image_interp = cv2.INTER_AREA
    else:
        image_interp = cv2.INTER_CUBIC

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
        (target_size, target_size),
        dtype=np.uint8,
    )

    canvas_mask = np.zeros(
        (target_size, target_size),
        dtype=np.uint8,
    )

    x0 = (
        target_size - new_w
    ) // 2

    y0 = (
        target_size - new_h
    ) // 2

    x1 = x0 + new_w
    y1 = y0 + new_h

    canvas_image[
        y0:y1,
        x0:x1,
    ] = resized_image

    canvas_mask[
        y0:y1,
        x0:x1,
    ] = (
        resized_mask > 0
    ).astype(np.uint8)

    return (
        canvas_image,
        canvas_mask,
        scale,
        x0,
        y0,
        new_w,
        new_h,
    )


# =============================================================================
# 최종 3px margin + background removal
# =============================================================================

def make_keep_mask_512(
    mask_512: np.ndarray,
):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            DILATE_KERNEL_SIZE,
            DILATE_KERNEL_SIZE,
        ),
    )

    keep_mask = cv2.dilate(
        mask_512.astype(np.uint8),
        kernel,
        iterations=DILATE_ITERATIONS,
    )

    return (
        keep_mask > 0
    )


# =============================================================================
# Split 처리
# =============================================================================

def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
):
    src_img_dir = (
        src_root
        / split
        / "images"
    )

    src_mask_dir = (
        src_root
        / split
        / "masks"
    )

    dst_img_dir = (
        dst_root
        / split
        / "images"
    )

    dst_mask_dir = (
        dst_root
        / split
        / "masks"
    )

    if not src_img_dir.exists():
        raise FileNotFoundError(
            f"image 폴더 없음: {src_img_dir}"
        )

    if not src_mask_dir.exists():
        raise FileNotFoundError(
            f"mask 폴더 없음: {src_mask_dir}"
        )

    dst_img_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = collect_images(
        src_img_dir
    )

    print()
    print("=" * 100)
    print(split)
    print("=" * 100)
    print(
        f"원본 image 수: {len(image_paths)}"
    )

    rows = []
    success = 0
    failed = 0

    for image_path in tqdm(
        image_paths,
        desc=split,
        ncols=110,
    ):
        try:
            mask_path = find_same_stem(
                src_mask_dir,
                image_path.stem,
            )

            if mask_path is None:
                raise FileNotFoundError(
                    f"mask 없음: {image_path.stem}"
                )

            image_native = read_gray(
                image_path
            )

            mask_native_gray = read_gray(
                mask_path
            )

            if (
                image_native.shape[:2]
                != mask_native_gray.shape[:2]
            ):
                raise RuntimeError(
                    "image/mask shape mismatch: "
                    f"{image_native.shape[:2]} vs "
                    f"{mask_native_gray.shape[:2]}"
                )

            mask_native = make_binary_mask(
                mask_native_gray
            )

            # -----------------------------------------------------------------
            # 1) 원해상도 mask 내부에서 p1 / p99
            # 2) 원해상도 intensity normalization
            # -----------------------------------------------------------------
            (
                normalized_native,
                p_low,
                p_high,
            ) = masked_percentile_normalize_native(
                image_native,
                mask_native,
            )

            # -----------------------------------------------------------------
            # 3) image / mask 동일 geometry로 512 resize + padding
            # -----------------------------------------------------------------
            (
                image_512,
                mask_512,
                scale,
                pad_x,
                pad_y,
                resized_w,
                resized_h,
            ) = resize_pad_image_and_mask(
                normalized_native,
                mask_native,
                TARGET_SIZE,
            )

            # -----------------------------------------------------------------
            # 4) 최종 512 mask에서 약 3px margin
            # -----------------------------------------------------------------
            keep_mask_512 = make_keep_mask_512(
                mask_512
            )

            # -----------------------------------------------------------------
            # 5) background=0
            # -----------------------------------------------------------------
            output_512 = image_512.copy()

            output_512[
                ~keep_mask_512
            ] = BACKGROUND_VALUE

            # 저장 mask는 정확한 segmentation mask 자체
            mask_save = (
                mask_512 * 255
            ).astype(np.uint8)

            out_img = (
                dst_img_dir
                / f"{image_path.stem}.png"
            )

            out_mask = (
                dst_mask_dir
                / f"{image_path.stem}.png"
            )

            if not cv2.imwrite(
                str(out_img),
                output_512,
            ):
                raise RuntimeError(
                    f"image 저장 실패: {out_img}"
                )

            if not cv2.imwrite(
                str(out_mask),
                mask_save,
            ):
                raise RuntimeError(
                    f"mask 저장 실패: {out_mask}"
                )

            native_values = (
                image_native.astype(np.float32)
                [mask_native > 0]
            )

            out_values = (
                output_512.astype(np.float32)
                [mask_512 > 0]
            )

            rows.append(
                {
                    "id": image_path.stem,
                    "split": split,

                    "native_height": int(
                        image_native.shape[0]
                    ),
                    "native_width": int(
                        image_native.shape[1]
                    ),

                    "resize_scale": float(
                        scale
                    ),
                    "resized_width": int(
                        resized_w
                    ),
                    "resized_height": int(
                        resized_h
                    ),
                    "pad_x": int(
                        pad_x
                    ),
                    "pad_y": int(
                        pad_y
                    ),

                    "percentile_low": float(
                        p_low
                    ),
                    "percentile_high": float(
                        p_high
                    ),
                    "percentile_range": float(
                        p_high - p_low
                    ),

                    "native_mask_ratio": float(
                        mask_native.mean()
                    ),
                    "final_mask_ratio": float(
                        mask_512.mean()
                    ),
                    "final_keep_ratio": float(
                        keep_mask_512.mean()
                    ),

                    "hand_mean_before": float(
                        native_values.mean()
                    ),
                    "hand_std_before": float(
                        native_values.std()
                    ),

                    "hand_mean_after": float(
                        out_values.mean()
                    ),
                    "hand_std_after": float(
                        out_values.std()
                    ),

                    "source_image": str(
                        image_path
                    ),
                    "source_mask": str(
                        mask_path
                    ),
                    "output_image": str(
                        out_img
                    ),
                    "output_mask": str(
                        out_mask
                    ),
                }
            )

            success += 1

        except Exception as exc:
            failed += 1

            print()
            print(
                f"[FAIL] {image_path.name}"
            )
            print(
                repr(exc)
            )

    metadata = pd.DataFrame(
        rows
    )

    metadata_path = (
        dst_root
        / f"{split}_preprocessing_metadata.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"[{split}] 성공={success}, 실패={failed}"
    )

    if len(metadata) > 0:
        print(
            "native p1 median =",
            round(
                float(
                    metadata[
                        "percentile_low"
                    ].median()
                ),
                3,
            ),
        )

        print(
            "native p99 median =",
            round(
                float(
                    metadata[
                        "percentile_high"
                    ].median()
                ),
                3,
            ),
        )

        print(
            "hand mean before median =",
            round(
                float(
                    metadata[
                        "hand_mean_before"
                    ].median()
                ),
                3,
            ),
        )

        print(
            "hand mean after median =",
            round(
                float(
                    metadata[
                        "hand_mean_after"
                    ].median()
                ),
                3,
            ),
        )

    return success, failed


# =============================================================================
# CSV 복사
# =============================================================================

def copy_label_csv_folder(
    dst_root: Path,
):
    """
    data/labels/ 폴더가 존재하면 최종 입력 dataset의 csv/로 복사한다.
    전처리 자체에는 필수 조건이 아니다.
    """
    src_csv_dir = (
        PROJECT_ROOT
        / "data"
        / "labels"
    )

    dst_csv_dir = (
        dst_root
        / "csv"
    )

    if not src_csv_dir.exists():
        print()
        print(
            "[INFO] data/labels 폴더가 없어 CSV 복사를 건너뜁니다."
        )
        return

    if dst_csv_dir.exists():
        shutil.rmtree(
            dst_csv_dir
        )

    shutil.copytree(
        src_csv_dir,
        dst_csv_dir,
    )

    print()
    print(
        "label csv 폴더 복사 완료:"
    )
    print(
        dst_csv_dir
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    src_root = args.src
    dst_root = args.dst

    if not src_root.exists():
        raise FileNotFoundError(
            f"source dataset 없음:\n{src_root}"
        )

    if dst_root.exists():
        if args.overwrite:
            print(
                "기존 출력 폴더 삭제:"
            )
            print(
                dst_root
            )

            shutil.rmtree(
                dst_root
            )

        elif any(
            dst_root.iterdir()
        ):
            raise FileExistsError(
                "\n출력 폴더가 이미 존재하고 비어있지 않습니다.\n"
                f"{dst_root}\n\n"
                "다시 만들려면 --overwrite 를 붙이세요."
            )

    dst_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_success = 0
    total_failed = 0

    for split in SPLITS:
        success, failed = process_split(
            src_root,
            dst_root,
            split,
        )

        total_success += success
        total_failed += failed

    copy_label_csv_folder(
        dst_root,
    )

    config = {
        "source_dataset": str(
            src_root
        ),
        "output_dataset": str(
            dst_root
        ),
        "target_size": TARGET_SIZE,

        "processing_order": [
            "native segmentation mask",
            "native masked percentile p1-p99",
            "native intensity normalization",
            "image/mask resize+center padding to 512",
            "512 mask dilation approx 3px",
            "outside keep-mask set to 0",
        ],

        "percentile_low": LOW_PERCENTILE,
        "percentile_high": HIGH_PERCENTILE,
        "percentile_statistics_region": (
            "native original segmentation mask foreground"
        ),

        "image_downscale_interpolation": (
            "cv2.INTER_AREA"
        ),
        "image_upscale_interpolation": (
            "cv2.INTER_CUBIC"
        ),
        "mask_interpolation": (
            "cv2.INTER_NEAREST"
        ),

        "dilation_kernel": (
            f"{DILATE_KERNEL_SIZE}x"
            f"{DILATE_KERNEL_SIZE} ellipse"
        ),
        "dilation_iterations": (
            DILATE_ITERATIONS
        ),
        "approx_margin_px_at_512": (
            DILATE_KERNEL_SIZE // 2
        ),

        "background_value": (
            BACKGROUND_VALUE
        ),

        "clahe": False,
        "histogram_equalization": False,

        "total_success": (
            total_success
        ),
        "total_failed": (
            total_failed
        ),
    }

    with (
        dst_root
        / "preprocessing_config.json"
    ).open(
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
    print("완료")
    print("=" * 100)
    print(
        f"총 성공: {total_success}"
    )
    print(
        f"총 실패: {total_failed}"
    )
    print()
    print(
        "출력:"
    )
    print(
        dst_root
    )
    print()
    print(
        "최종 처리:"
    )
    print(
        "Native masked p1~p99"
        " -> 512 resize/pad"
        " -> 512 mask 3px dilation"
        " -> background=0"
    )


if __name__ == "__main__":
    main()