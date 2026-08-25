# Model Experiment Log

이 폴더는 최종 모델 개발 과정에서 수행한 주요 구조 실험과 의사결정을 요약합니다.  
전체 실험 스크립트를 나열하기보다, **무엇을 바꾸었고 / 어떤 결과가 나왔고 / 왜 채택 또는 기각했는지**를 중심으로 정리했습니다.

> 내부 실험 번호는 공개 문서에서 제외하고 구조와 목적 중심의 이름으로 표기했습니다.

## 1. 기준 모델

기준 구조는 다음과 같습니다.

- **Backbone:** ConvNeXt V1-Tiny
- **Input:** 512×512 grayscale repeated to 3 channels
- **Sex information:** sex embedding
- **Prediction head:** separate Male / Female 240-bin heads
- **Objective:** Label Distribution Learning (LDL)
- **Distribution:** 1–240 months, Gaussian target distribution
- **Core validation:** MAE **5.842 months**, RMSE **8.118 months**

이 구조를 기준으로 이후 실험에서는 한 번에 하나의 목적을 중심으로 구조를 추가하거나 변경했습니다.

## 2. 주요 실험 방향

### Sex conditioning

Backbone의 여러 stage에 성별 조건을 직접 주입하는 **Stage-wise Sex-FiLM**을 실험했습니다.  
FiLM parameter가 실제로 학습되었지만 전체 validation 성능은 악화되어, backbone 내부까지 성별 정보를 강하게 주입하는 방식은 채택하지 않았습니다.

### Ordinal regularization

골연령이 순서가 있는 연속 값이라는 점을 반영하기 위해 CDF/Wasserstein-1 loss를 추가했습니다.

- 강한 regularization (`lambda=0.02`)은 RMSE와 일부 구간 성능을 악화시켰습니다.
- 약한 regularization (`lambda=0.005`)은 MAE **5.807**, RMSE **8.090**으로 소폭 개선되었습니다.

이를 통해 ordinal signal 자체는 유효할 가능성이 있지만, LDL의 주 objective를 방해하지 않을 정도로 약하게 적용해야 함을 확인했습니다.

### Spatial attention

손의 성장 단계에 따라 중요한 위치가 달라질 수 있다는 가설로, Stage-3 attention map의 중심 위치에 연령 순서를 부여하는 continuous spatial prior를 실험했습니다.

Attention ordering 자체는 매우 높은 정확도로 학습되었지만, bone-age MAE 개선은 제한적이었습니다.  
즉 모델이 이미 유사한 위치 정보를 representation 내부에서 활용하고 있을 가능성이 있다고 판단했습니다.

### Local ROI experts

CAM/localization 분석을 이용해 wrist/middle/upper local ROI를 자동 생성하고 별도 local expert를 학습했습니다.

Local-only 모델은 whole-hand 모델보다 성능이 낮았지만, 일부 hard case에서는 whole-hand 모델과 다른 오류 패턴을 보였습니다.  
특히 Upper ROI는 보완 신호가 가장 뚜렷했습니다.

### Residual feature fusion

Whole-hand expert와 Upper ROI expert의 feature를 결합하고, **zero-initialized residual LDL head**만 추가 학습했습니다.

- Validation MAE: **5.788 months**
- Validation RMSE: **8.066 months**

구조 실험 중 매우 좋은 validation 성능을 기록했지만, 별도의 ROI 생성과 두 expert inference가 필요했습니다.  
약 **0.05개월 수준의 개선**에 비해 배포 파이프라인 복잡도가 크게 증가하여 최종 배포 모델로 선택하지 않았습니다.

### Relation-gated fusion

여러 local ROI feature를 sample-wise gate로 선택적으로 사용할 수 있도록 relation-gated fusion도 실험했습니다.  
그러나 gate가 거의 상수로 붕괴했고 기준 모델을 개선하지 못해 추가 복잡화는 중단했습니다.

## 3. Subgroup error analysis와 targeted refinement

전체 평균 성능만 비교하지 않고 **성별 × 연령 구간**으로 validation error를 분석했습니다.

특히 저연령 남아에서 systematic positive bias가 관찰되어, 전체 모델을 다시 학습하는 대신 다음과 같이 제한적인 fine-tuning을 수행했습니다.

- backbone: frozen
- image head: frozen
- sex embedding / fusion: frozen
- Female LDL head: frozen
- **Male LDL head only trainable**
- 모든 Male sample로 기존 LDL objective 유지
- Male ≤60 months에 positive bias penalty 적용
- Male >60 months에는 base prediction과의 teacher-consistency 적용

결과적으로 Male ≤60 months에서:

- MAE: **5.880 months**
- Bias: **+3.528 months**

로 개선되었고, Female branch는 그대로 보존되었습니다.

이 모델을 최종 모델로 선택했습니다.

## 4. 왜 validation MAE가 가장 낮은 모델을 최종 선택하지 않았는가?

실험 중 Whole-hand + Upper ROI residual fusion은 MAE **5.788**로 더 낮은 validation MAE를 기록했습니다.

그러나 최종 모델 선택에서는 단순히 가장 낮은 단일 validation MAE만 보지 않았습니다.

- ROI expert 추가에 따른 inference pipeline 복잡도
- 추가 weight 및 연산 비용
- 개선 폭의 크기
- subgroup bias 개선 여부
- 실제 dashboard/deployment 구성의 단순성

을 함께 고려했습니다.

따라서 **단일 whole-hand inference 구조를 유지하면서 실제 오류 분석에서 발견된 저연령 남아 편향을 직접 개선한 targeted refinement**를 최종 배포 모델로 선택했습니다.

## 5. Held-out test

Held-out test는 **모델 선택이 완료된 이후 한 번만 평가**했으며, 모델 또는 하이퍼파라미터 선택에 사용하지 않았습니다.

최종 held-out test 결과는 `../results/heldout_test/`에 저장되어 있습니다.

- N = **197**
- MAE = **4.245 months**
- RMSE = **5.412 months**
- R² = **0.9837**
- Bias = **+0.794 months**

## 6. Files

- `experiment_summary.csv`  
  주요 구조 실험의 변경점, validation 성능, 관찰 결과, 채택 여부를 표 형태로 정리한 파일입니다.

실험 간 비교는 동일 validation 구성에서 수행된 주요 구조 실험을 중심으로 정리했으며, 데이터 구성이 다른 초기 실험은 직접적인 수치 비교에서 제외했습니다.
