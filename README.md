# dronev2 — 드론 텔레메트리 이상탐지 파이프라인 (구조 요약)

코드 구조만 보기 쉽게 정리한 저장소입니다. 데이터·학습된 모델 가중치·실험 결과 그림은 용량 문제로 뺐습니다(전체 히스토리·수치·그림은 팀 내부 `together` 저장소 참고).

## 전체 흐름

```
원시 CSV (16채널 센서)
   → 전처리/feature engineering
   → 모델 (재구성 or 예측)
   → Isolation Forest로 2차 판단
   → 융합 점수 → 이상 탐지 / 채널별 진단
```

## 시도한 모델 4가지 (시간순)

| # | 모델 | 핵심 파일 |
|---|---|---|
| 1 | VAE (원 논문 재현) | `networks/vae.py`, `train.py`, `eval.py` |
| 2 | GRU 예측기 | `networks/gru_predictor.py`, `gru_data_loader.py`, `train_gru.py`, `eval_gru.py` |
| 3 | **LSTM-AE + Isolation Forest** (메인, 재구성 기반) | `networks/lstm_ae.py`, `lstm_ae_data_loader.py`, `train_eval_lstm_ae_heldout.py` ⭐ |
| 4 | LSTM 예측기 (다음 시점 예측 기반) | `networks/lstm_predictor.py`, `lstm_predictor_data_loader.py`, `train_eval_lstm_predictor.py` |
| 4.5 | LSTM 15초 앞 예측기 (교수님 요구: 최소 15초 lead time) | `train_eval_lstm_predictor_15s*.py`, `eval_lstm_predictor_15s*.py` |

## 공통 파이프라인 구성요소

- `config.py` — 전역 설정(채널명, 하이퍼파라미터)
- `data_loader.py` — StandardScaler
- `feature_engineering.py` — raw/1차미분/2차미분/에너지 feature 계산
- `preprocessing.py` — 파이프라인 v4: 공통 리샘플·인과적 per-flight 정규화·실제 초 단위 윈도우(가장 최근 버전, 세션드리프트 근본원인 3가지를 고친 버전)

## 검증·실험 스크립트 (연구 과정에서 나온 것들)

| 스크립트 | 뭘 확인했는지 |
|---|---|
| `eval_pipeline_fixes_ablation.py` | 세션드리프트 근본원인 규명(입력 파이프라인 결함 3건) |
| `eval_robust_baseline.py` | file-wide baseline을 raw 탐지점수에 적용 시도 (negative result) |
| `verify_sample_size_bias.py` | AUROC 수치의 표본편향 검증 |
| `experiment_leadtime_and_fault_types.py` | lead-time·fault 유형별 사전경보 능력 스윕 |
| `experiment_early_warning.py` | forecast-then-detect 사전경보 시도 (negative result) |
| `experiment_forecaster_comparison.py`, `experiment_ensemble_uncertainty.py`, `experiment_health_indicator_rul.py` | 사전경보용 예측기 7종 비교 (건강지표+RUL 외삽이 최선) |
| `experiment_swarm_pair_diff.py` | 편대 짝 비교로 세션드리프트 우회 시도 |

## 온보드 배포

- `onboard_streaming_detector.py`, `onboard_streaming_predictor.py` — 스트리밍(실시간) 추론
- `export_predictor_deploy.py`, `export_benchmark_forests.py` — ONNX/경량화 변환
- `soak_test_onboard.py`, `benchmark_iforest_onboard.py`, `figure_jetson_benchmark.py` — 지연시간·안정성 실측
- `make_onboard_bundle.sh` — 보드에 올릴 배포 패키지 생성

## 참고

전체 실험 히스토리, 수치, 결론, 향후 계획은 팀 내부 `together` 저장소의 `PROJECT_SUMMARY.md`에 있습니다. 이 저장소는 코드 구조 공유용입니다.
