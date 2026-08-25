수부 X-ray 뼈나이 예측 AI

본 패키지는 수부 X-ray 이미지와 성별 정보를 입력받아 뼈나이(Bone Age)를 개월(month) 단위로 예측합니다.

최종 추론 과정은 다음과 같습니다.

원본 수부 X-ray
    ↓
YOLOX-S 손 영역 검출
    ↓
Hand Segmentation
    ↓
손 방향 정렬
(PCA + Finger/Wrist 보정)
    ↓
손 영역 재크롭
    ↓
Masked Percentile 정규화 (p1~p99)
    ↓
512×512 Resize + Padding
    ↓
손 바깥 배경 제거
    ↓
ConvNeXt V1-Tiny
+ 성별 정보
+ 240-bin Label Distribution Learning
    ↓
뼈나이 예측 (개월)

1. 폴더 구성

최종 전달 폴더는 아래와 같이 구성합니다.

boneage_enterprise_final_package/
│
├─ Images/
│   └─ 추론할 수부 X-ray 이미지
│
├─ models/
│   ├─ best_model.pt
│   ├─ hand_seg_crop512_traced.pt
│   └─ yolox_s_hand_best.pth
│
├─ YOLOX/
│   └─ yolox/
│
├─ convnext-v1_model.py
├─ yolox_s_hand.py
├─ test.csv
├─ requirements.txt
└─ README_KR.md

모델 파일

models/best_model.pt

최종 뼈나이 예측 모델

ConvNeXt V1-Tiny 기반

성별 정보를 함께 사용

출력 단위: 개월(month)

models/hand_seg_crop512_traced.pt

손 segmentation 및 정렬 전처리에 사용하는 TorchScript 모델

models/yolox_s_hand_best.pth

원본 X-ray에서 손 영역을 검출하는 YOLOX-S 모델

2. 실행 환경

권장 Python 버전:

Python 3.11

가상환경 생성:

py -3.11 -m venv .venv

가상환경 활성화:

.\.venv\Scripts\Activate.ps1

필요 라이브러리 설치:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

3. 입력 데이터 준비

3-1. 이미지

추론할 X-ray 이미지를 Images 폴더에 넣습니다.

예:

Images/
├─ 1001.png
├─ 1002.png
├─ 1003.jpg
└─ ...

지원 이미지 확장자:

.jpg
.jpeg
.png
.bmp
.tif
.tiff

3-2. test.csv

test.csv에는 최소한 이미지 식별 정보와 성별 정보가 필요합니다.

기존 기업 전달 코드의 CSV 읽기 방식을 유지하고 있어 다음과 같은 열 이름을 인식할 수 있습니다.

ID 계열

id
image_id
imageid
patient_id
patientid
case_id
caseid

성별 계열

sex
gender
male

파일명 계열

파일명을 CSV에 직접 넣는 경우 다음과 같은 열도 인식합니다.

filename
file
image
imagefile
imagepath
path

파일명 열이 없으면 ID와 Images 폴더의 파일명을 기준으로 자동 매칭합니다.

3-3. 성별 값

다음과 같은 형식을 인식합니다.

남성 예:

M
Male
male
남
남성
1

여성 예:

F
Female
female
여
여성
0

test.csv 예시 1

id,sex
1001,M
1002,F
1003,M

test.csv 예시 2

image_id,gender
1001,male
1002,female
1003,male

test.csv 예시 3

patient_id,filename,sex
1001,1001.png,M
1002,1002.png,F
1003,1003.jpg,M

실제 뼈나이 정답(GT Bone Age)은 추론 과정에 필요하지 않습니다.

4. 실행 방법

프로젝트 폴더에서 다음 명령을 실행합니다.

python .\convnext-v1_model.py

가상환경의 Python을 직접 지정하는 경우:

.\.venv\Scripts\python.exe .\convnext-v1_model.py

GPU가 사용 가능한 환경에서는 CUDA를 자동으로 사용합니다.

5. 실제 추론 과정

5-1. 손 영역 검출

원본 수부 X-ray에서 YOLOX-S를 이용해 손 영역을 검출합니다.

검출된 bounding box에 여유 영역을 포함시켜 segmentation 입력 영역을 생성합니다.

5-2. Hand Segmentation

검출된 손 영역에 segmentation 모델을 적용하여 손 foreground mask를 생성합니다.

생성된 mask에서:

가장 큰 connected component 유지

내부 hole 보정

을 수행합니다.

5-3. 손 방향 정렬

손 mask의 주축을 이용해 PCA 기반 1차 방향 정렬을 수행합니다.

이후 손가락 방향과 손목 위치를 이용해 잔여 회전각을 보정합니다.

PCA coarse alignment
        +
Finger/Wrist residual correction

최종 회전은 원본 영상과 mask에 동일하게 적용됩니다.

5-4. 손 영역 재크롭

회전된 손 mask의 실제 영역을 기준으로 다시 crop합니다.

최종 crop에는 손 끝부분이 잘리지 않도록 소량의 margin을 포함합니다.

5-5. Masked Percentile 정규화

손 foreground 내부 픽셀만 사용하여:

p1  = 1 percentile
p99 = 99 percentile

을 계산합니다.

이 범위를 기준으로 intensity를 0~255로 정규화합니다.

배경 픽셀은 percentile 계산에 사용하지 않습니다.

5-6. 512×512 입력 생성

정규화된 손 이미지는 종횡비를 유지하면서 512×512 크기로 resize 및 center padding됩니다.

손 mask는 약 3px 정도 확장한 뒤, mask 바깥 영역을 0으로 설정합니다.

최종 모델 입력:

512 × 512
grayscale
→ RGB 3채널 반복

5-7. 뼈나이 예측

최종 512×512 손 영상과 성별 정보를 ConvNeXt V1-Tiny 기반 모델에 입력합니다.

모델은 240개 age bin에 대한 분포를 예측하고, 해당 분포의 기대값을 최종 뼈나이로 사용합니다.

출력 단위는 개월(month) 입니다.

6. 출력 결과

추론이 완료되면 프로젝트 폴더에:

predictions.csv

가 생성됩니다.

기본 출력 형식:

id,filename,sex,predicted_age
1001,1001.png,M,132.41
1002,1002.png,F,108.73
1003,1003.jpg,M,145.82

각 열 설명

id

입력 데이터 식별자

filename

실제 추론에 사용한 이미지 파일명

sex

성별

predicted_age

모델이 예측한 뼈나이

단위: 개월(month)

7. 참고 사항

이미지 파일명 매칭

가능하면 test.csv의 ID와 이미지 파일명을 동일하게 구성하는 것을 권장합니다.

예:

test.csv id = 1001
Images/1001.png

CSV에 파일명 열이 있는 경우 해당 파일명을 우선 사용할 수 있습니다.

손 검출 또는 전처리 실패

기존 기업 전달 코드의 동작 방식을 유지하여, 일부 전처리 단계에서 문제가 발생하더라도 가능한 경우 기존 방식의 fallback 입력을 사용해 추론을 계속합니다.

따라서 실행 후에는 출력 건수가 입력 데이터 건수와 일치하는지 확인하는 것을 권장합니다.

추론 시 필요한 정보

추론에 필요한 정보:

1. 수부 X-ray 이미지
2. 성별

추론에 필요하지 않은 정보:

실제 Bone Age 정답
Chronological Age

8. 빠른 실행 요약

# 1. 가상환경 생성
py -3.11 -m venv .venv

# 2. 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 3. 라이브러리 설치
python -m pip install -r requirements.txt

# 4. Images 폴더와 test.csv 준비

# 5. 추론 실행
python .\convnext-v1_model.py

추론 완료 후:

predictions.csv

파일을 확인합니다.