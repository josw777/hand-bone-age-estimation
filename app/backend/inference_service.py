from __future__ import annotations

import base64
import importlib.util
import io
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


HERE = Path(__file__).resolve().parent
MODEL_PACKAGE = HERE / "model_package"
MODEL_SCRIPT = MODEL_PACKAGE / "convnext-v1_model.py"


def _load_pipeline_module():
    if not MODEL_SCRIPT.is_file():
        raise FileNotFoundError(f"모델 추론 코드가 없습니다: {MODEL_SCRIPT}")

    spec = importlib.util.spec_from_file_location("boneage_enterprise_pipeline", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"모델 추론 코드를 불러올 수 없습니다: {MODEL_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pil_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class BoneAgeService:
    """기업 전달용 추론 코드를 재사용하는 1장 추론 서비스.

    모델은 최초 요청 시 한 번만 로드되고 이후 요청에서는 재사용합니다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._module = None
        self._device = None
        self._boneage_model = None
        self._model_transform = None
        self._detector = None
        self._seg_model = None
        self._cfg: Dict[str, Any] = {}
        self._load_seconds: Optional[float] = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device_name(self) -> Optional[str]:
        return None if self._device is None else str(self._device)

    @property
    def load_seconds(self) -> Optional[float]:
        return self._load_seconds

    def load(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            started = time.perf_counter()
            m = _load_pipeline_module()
            device = m.resolve_device("auto")

            boneage_model, _, cfg = m.load_boneage_model(
                m.BONEAGE_MODEL.expanduser().resolve(),
                device,
            )
            model_transform = m.build_model_transform(
                cfg.get("model_name", m.MODEL_NAME)
            )

            detector = m.load_yolox_detector(
                yolox_dir=m.YOLOX_DIR.expanduser().resolve(),
                exp_path=m.YOLOX_EXP.expanduser().resolve(),
                checkpoint_path=m.YOLOX_MODEL.expanduser().resolve(),
                device=device,
            )
            seg_model = m.load_segmentation_model(
                m.SEG_MODEL.expanduser().resolve(),
                device,
            )

            self._module = m
            self._device = device
            self._boneage_model = boneage_model
            self._model_transform = model_transform
            self._detector = detector
            self._seg_model = seg_model
            self._cfg = cfg
            self._load_seconds = time.perf_counter() - started
            self._loaded = True

    def predict(
        self,
        image_path: Path,
        sex: str,
        chronological_age_months: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.load()

        assert self._module is not None
        assert self._device is not None
        assert self._boneage_model is not None
        assert self._model_transform is not None
        assert self._detector is not None
        assert self._seg_model is not None

        m = self._module
        male = float(m.parse_male(sex))

        started = time.perf_counter()
        with self._lock:
            prepared, used_fallback = m.detect_segment_align_prepare(
                detector=self._detector,
                seg_model=self._seg_model,
                image_path=image_path,
                device=self._device,
            )

            tensor = self._model_transform(prepared.convert("RGB"))
            pred = m.predict_batch(
                model=self._boneage_model,
                tensors=[tensor],
                males=[male],
                device=self._device,
            )

        predicted_months = float(pred[0])
        predicted_years = predicted_months / 12.0
        whole_years = int(predicted_months // 12)
        remaining_months = int(round(predicted_months - whole_years * 12))
        if remaining_months == 12:
            whole_years += 1
            remaining_months = 0

        difference_months = None
        if chronological_age_months is not None:
            difference_months = predicted_months - float(chronological_age_months)

        return {
            "predicted_age_months": round(predicted_months, 3),
            "predicted_age_years": round(predicted_years, 3),
            "predicted_age_display": f"{whole_years}년 {remaining_months}개월",
            "sex": "M" if male >= 0.5 else "F",
            "difference_months": None if difference_months is None else round(difference_months, 3),
            "preprocessed_image": _pil_png_data_url(prepared),
            "used_fallback": bool(used_fallback),
            "processing_time_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "device": str(self._device),
            "input_size": "512x512",
            "model_name": "ConvNeXt V1-Tiny + sex-specific 240-bin LDL",
            "pipeline": [
                "YOLOX-S hand detection",
                "Hand segmentation",
                "PCA + finger/wrist alignment",
                "Masked percentile p1-p99",
                "512x512 resize/padding + background removal",
                "ConvNeXt V1-Tiny + sex-specific LDL",
            ],
        }
