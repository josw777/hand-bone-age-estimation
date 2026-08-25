# Shilla BoneAge AI Dashboard - 통합 버전

이 폴더는 두 개의 기존 결과물을 하나로 합친 실행형 데모입니다.

- 기존 React 대시보드 UI
- 기업 전달용 최종 뼈나이 추론 코드 및 모델

실제 추론 흐름은 기업 전달 코드와 동일한 코드를 재사용합니다.

원본 수부 X-ray
→ YOLOX-S 손 검출
→ Hand Segmentation
→ PCA + Finger/Wrist 방향 정렬
→ 손 영역 재크롭
→ Masked Percentile p1~p99
→ 512×512 Resize + Padding
→ 손 바깥 배경 제거
→ ConvNeXt V1-Tiny + 성별 정보 + sex-specific 240-bin LDL
→ 뼈나이 예측(개월)

## 1. 폴더 구조

```text
boneage_dashboard_integrated/
├─ backend/
│  ├─ app.py
│  ├─ inference_service.py
│  ├─ requirements.txt
│  └─ model_package/
│     ├─ convnext-v1_model.py
│     ├─ yolox_s_hand.py
│     ├─ models/
│     │  ├─ best_model.pt
│     │  ├─ hand_seg_crop512_traced.pt
│     │  └─ yolox_s_hand_best.pth
│     └─ YOLOX/
├─ frontend/
│  ├─ src/
│  ├─ public/
│  └─ package.json
├─ setup.bat
├─ run_backend.bat
├─ run_frontend.bat
└─ run_all.bat
```

## 2. 처음 한 번 설치

Windows에서 프로젝트 루트의 `setup.bat`를 실행합니다.

수동 설치가 필요한 경우:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\backend\requirements.txt

cd .\frontend
npm install
```

기업 전달용 모델을 이미 실행하던 Python 환경이 있다면 그 환경에 다음 세 패키지만 추가해도 됩니다.

```powershell
python -m pip install fastapi "uvicorn[standard]" python-multipart
```

그 경우 `run_backend.bat`의 Python 경로를 기존 환경에 맞게 수정하면 됩니다.

## 3. 실행

가장 간단한 방법:

```text
run_all.bat
```

백엔드와 프론트엔드 창이 각각 열립니다.

- Dashboard: http://127.0.0.1:3000
- API: http://127.0.0.1:8000
- API 문서: http://127.0.0.1:8000/docs

첫 추론 시 YOLOX-S, Segmentation, ConvNeXt 모델을 메모리에 올리기 때문에 시간이 더 걸릴 수 있습니다. 이후 요청에서는 모델을 재사용합니다.

## 4. 사용 방법

1. Dashboard의 `Upload X-ray`를 누릅니다.
2. 수부 X-ray 이미지 파일을 선택합니다.
3. 성별 Female/Male을 선택합니다.
4. 실제 나이는 선택 입력입니다. 입력하면 예측 뼈나이와의 단순 차이를 함께 표시합니다.
5. `Run AI Prediction`을 누릅니다.
6. 추론 후 화면에 다음이 표시됩니다.
   - 예측 뼈나이(년/개월 및 개월)
   - 실제 최종 512×512 모델 입력 이미지
   - 추론 시간
   - 사용 장치(cpu/cuda)
   - 전처리 fallback 사용 여부

## 5. 지원 이미지

기업 전달 코드와 동일하게 다음 확장자를 지원합니다.

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

현재 버전은 DICOM `.dcm`을 직접 읽지 않습니다.

## 6. 중요한 구현 사항

- React에서 AI 모델을 직접 실행하지 않습니다.
- React가 이미지를 FastAPI의 `/api/predict`로 전송합니다.
- FastAPI는 서버 시작 후 최초 요청에서 모델을 한 번 로드합니다.
- 이후에는 동일한 모델 객체를 재사용하여 매 요청마다 가중치를 다시 로드하지 않습니다.
- 화면의 `전처리` 탭은 CSS 효과가 아니라 실제 기업 전달 파이프라인이 생성한 최종 512×512 입력 이미지입니다.
- 기존 대시보드에 있던 성인 키 예측, 임의 앙상블 결과 등 현재 모델이 실제로 출력하지 않는 값은 제거했습니다.

## 7. GPU 설치 관련

`backend/requirements.txt`는 기업 전달 패키지의 PyTorch CUDA 설정을 유지했습니다. PC의 CUDA/PyTorch 환경이 다르면 기존에 모델이 정상 실행되던 torch/torchvision 설치를 유지하고 나머지 패키지만 설치하는 것이 가장 안전합니다.

## 8. 용도

본 통합 버전은 프로젝트 시연 및 연구용 UI입니다. 의료진의 임상 판독이나 진단을 대체하는 용도로 사용하지 않습니다.
