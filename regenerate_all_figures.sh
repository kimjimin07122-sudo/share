#!/usr/bin/env bash
# Regenerate every figure under the CURRENT pipeline (v4.1, causal).
#
# The figures directory had accumulated output from three different pipeline
# versions across three days, so the numbers on different slides would not
# have agreed with each other. Order matters: the trainers write the .pth /
# .joblib artifacts that the evaluators then load.
set -uo pipefail
cd "$(dirname "$0")"
LOG=figures/_regen.log
: > "$LOG"

run () {
  printf '%-52s' "  $1"
  if python "$1" >> "$LOG" 2>&1; then echo "ok"; else echo "FAILED (see $LOG)"; fi
}

echo "== 1. train the models the evaluators depend on =="
run train_eval_lstm_ae_heldout.py
run train_eval_lstm_predictor.py

echo "== 2. pipeline + detection analyses =="
run eval_pipeline_fixes_ablation.py
run eval_robust_baseline.py
run verify_sample_size_bias.py
run experiment_swarm_pair_diff.py

echo "== 3. the 15s predictor line =="
run train_eval_lstm_predictor_15s.py
run train_eval_lstm_predictor_15s_multisession.py
run eval_lstm_predictor_15s_magx_specific.py
run eval_lstm_predictor_15s_magx_synthetic_faults.py

echo "== 4. forecasting / early-warning experiments =="
run experiment_leadtime_and_fault_types.py
run experiment_early_warning.py
run experiment_forecaster_comparison.py
run experiment_ensemble_uncertainty.py
run experiment_health_indicator_rul.py
run summarize_forecaster_methods.py

echo
echo "figures: $(ls figures/*.png | wc -l)"
