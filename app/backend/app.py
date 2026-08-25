from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference_service import BoneAgeService


app = FastAPI(
    title="Shilla Bone Age AI API",
    version="1.0.0",
    description="YOLOX-S + Hand Segmentation + ConvNeXt V1-Tiny bone-age inference API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = BoneAgeService()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_loaded": service.loaded,
        "device": service.device_name,
        "model_load_seconds": service.load_seconds,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    sex: str = Form(...),
    chronological_age_months: Optional[float] = Form(None),
):
    filename = image.filename or "upload.png"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 이미지 형식입니다. jpg/jpeg/png/bmp/tif/tiff를 사용하세요.",
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 이미지 파일입니다.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지 파일이 너무 큽니다. 최대 50MB입니다.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        result = service.predict(
            image_path=tmp_path,
            sex=sex,
            chronological_age_months=chronological_age_months,
        )
        return result

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"추론 중 오류가 발생했습니다: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
