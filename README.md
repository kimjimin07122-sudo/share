# dronev2 — 드론 텔레메트리 이상탐지 파이프라인 (구조 요약)

실제로 돌아가는 데 필요한 최소 구성(19개 파일)만 모았습니다. import 의존성까지 확인해서, 이 파일들만 있으면 각 모델을 처음부터 학습·평가할 수 있습니다. 데이터·학습된 가중치·실험 결과/그림은 용량 문제로 뺐습니다.

## 전체 흐름

```
원시 CSV (16채널 센서)
   → preprocessing.py (공통 리샘플·인과적 per-flight 정규화·실제초 윈도우)
   → feature_engineering.py (raw/1차미분/2차미분/에너지)
   → 모델 (networks/) — 재구성 or 예측
   → Isolation Forest로 2차 판단
   → 융합 점수 → 이상 탐지 / 채널별 진단
```

## 모델 4가지 + 각각의 진입점

| 모델 | 구조 (`networks/`) | 학습/평가 진입점 |
|---|---|---|
| VAE (원 논문 재현) | `vae.py` | `train.py` |
| GRU 예측기 | `gru_predictor.py` | `train_gru.py` |
| **LSTM-AE + Isolation Forest** (메인, 재구성 기반) | `lstm_ae.py` | `train_eval_lstm_ae_heldout.py` ⭐ |
| LSTM 예측기 (다음 시점 1개만 예측) | `lstm_predictor.py` | `train_eval_lstm_predictor.py` |
| LSTM 15초 앞 예측기 (교수님 요구: 최소 15초 lead time) | `lstm_predictor.py` (동일 구조 재사용) | `train_eval_lstm_predictor_15s.py`(v1) → `train_eval_lstm_predictor_15s_multisession.py`(v2, v1을 import해서 씀) |

## 공통 지원 모듈 (여러 모델이 같이 씀)

- `config.py` — 채널명, 하이퍼파라미터 등 전역 설정
- `data_loader.py` — StandardScaler
- `gru_data_loader.py` — CSV 로딩 + 실제 타임스탬프 파싱(`read_raw_csv_with_timestamp`, 이름과 달리 GRU 전용 아니고 거의 모든 모델이 씀)
- `feature_engineering.py` — feature 계산
- `preprocessing.py` — 파이프라인 v4(가장 최신 버전, 세션드리프트 근본원인 3가지를 고친 버전)
- `eval.py` — 공통 평가 지표(`compute_binary_metrics` 등)
- `eval_lstm_ae.py` — LSTM-AE 재구성 오차 계산 함수(`train_eval_lstm_ae_heldout.py`가 여기서 import함 — 이름은 "eval_"지만 핵심 의존성)

## 여기 없는 것 (필요하면 요청)

- 연구 과정에서 나온 검증/실험 스크립트 30여 개 (세션드리프트 원인 규명, negative result들, 예측기 비교 등) — 구조 파악보다 "왜 이렇게 설계했는지"에 필요
- 온보드 배포 스크립트(ONNX 변환, 스트리밍 검출기, Jetson 벤치마크)
- 데이터셋, 학습된 모델 가중치, 결과 그림
