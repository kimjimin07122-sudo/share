"""
Build the deployable artifact set for the 15s-ahead PREDICTOR (2026-08-20).

The detector package in deploy/ ships the LSTM-AE, which flags a fault as it
happens. This is the other line of work: the predictor, which forecasts the
state 15s ahead and scores that forecast, so an alarm can in principle fire
before the fault.

Two scorers are exported, because they behave very differently and the
comparison is the point (PROJECT_SUMMARY sec 3.16):

  lstm_forecast   the neural forecaster. Trained on normal flight, so it
                  regresses to the mean and scored 0/4 on a gradual injected
                  ramp -- kept because it is the model the requirement asked
                  for, and because having it onboard is what lets the board
                  reproduce that comparison rather than take it on trust.

  health_rul      trend of a scalar health indicator, extrapolated to the
                  failure level, reporting seconds-to-failure. No network at
                  all. Best of the seven forecasters tried: 4/4 on the ramp,
                  +5.0s median lead, 4.6s absolute error on the time estimate.

Everything is fit under the CAUSAL configuration used in the validated
experiment: each flight normalized on its first WARMUP_SEC seconds only, since
whole-flight statistics cannot exist onboard (sec 3.13).
"""
import glob
import os

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from config import Config
from data_loader import StandardScaler
from gru_data_loader import read_raw_csv_with_timestamp
from networks.lstm_predictor import LSTMPredictor
from train_eval_lstm_predictor_15s import CONTEXT_BINS, INPUT_DIM, build_pairs, flight_to_binned
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY
from experiment_early_warning import HORIZON, WARMUP_SEC
from experiment_leadtime_and_fault_types import train_for_horizon, MAG_X_COLS

DEPLOY = os.path.join(Config.BASE_DIR, "deploy")
ONNX_PATH = os.path.join(DEPLOY, "lstm_predictor_15s.onnx")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(DEPLOY, exist_ok=True)
    print(f"device: {device}   context {CONTEXT_BINS}s -> +{HORIZON}s   warm-up {WARMUP_SEC:.0f}s")

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]

    raws = [read_raw_csv_with_timestamp(f) for f in [train_csv] + augment]
    binned_raw = [flight_to_binned(r, t, warmup_sec=WARMUP_SEC) for r, t in raws]
    fscaler = StandardScaler()
    fscaler.fit(np.vstack(binned_raw))
    binned = [fscaler.transform(b) for b in binned_raw]
    print(f"fit feature scaler on {len(binned)} flights")

    model, n_pairs = train_for_horizon(HORIZON, binned, device)
    print(f"trained predictor on {n_pairs} pairs")

    # normality model over STATES -- the forecast is scored against this
    normal_states = np.concatenate(binned, axis=0)
    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS,
                              random_state=Config.RANDOM_SEED, n_jobs=-1).fit(normal_states)
    magx_mu = normal_states[:, MAG_X_COLS].mean(axis=0)
    magx_sd = np.where(normal_states[:, MAG_X_COLS].std(axis=0) == 0, 1e-6,
                       normal_states[:, MAG_X_COLS].std(axis=0))

    # thresholds from clean held-out flights, matching the experiment
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    s_if, s_mz = [], []
    for f in eval_files:
        raw, t = read_raw_csv_with_timestamp(f)
        b = flight_to_binned(raw, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC)
        pasts, _, _ = build_pairs(b, CONTEXT_BINS, HORIZON, stride=1)
        if len(pasts) == 0:
            continue
        with torch.no_grad():
            pred = model(torch.tensor(pasts).to(device)).cpu().numpy()
        s_if.append(-iforest.score_samples(pred))
        s_mz.append(np.abs((pred[:, MAG_X_COLS] - magx_mu) / magx_sd).mean(axis=1))
    thr_if = float(np.percentile(np.concatenate(s_if), 99))
    thr_mz = float(np.percentile(np.concatenate(s_mz), 99))
    print(f"alarm thresholds (99th pct on clean flights): IF={thr_if:.4f}  mag_x={thr_mz:.4f}")

    model.eval().to("cpu")
    dummy = torch.randn(1, CONTEXT_BINS, INPUT_DIM)
    torch.onnx.export(model, dummy, ONNX_PATH,
                      input_names=["context"], output_names=["forecast"],
                      dynamic_axes={"context": {0: "batch"}, "forecast": {0: "batch"}},
                      opset_version=17)
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"context": dummy.numpy()})[0]
    print(f"exported {os.path.basename(ONNX_PATH)} "
          f"({os.path.getsize(ONNX_PATH)/1024:.0f} KB)  max diff {np.abs(ref-got).max():.2e}")

    joblib.dump(fscaler, os.path.join(DEPLOY, "predictor_feature_scaler.joblib"))
    joblib.dump(iforest, os.path.join(DEPLOY, "predictor_normality_iforest.joblib"))
    np.savez(os.path.join(DEPLOY, "predictor_stats.npz"),
             magx_mu=magx_mu, magx_sd=magx_sd,
             thr_if=thr_if, thr_magx=thr_mz,
             context_bins=CONTEXT_BINS, horizon=HORIZON, warmup_sec=WARMUP_SEC)
    print(f"saved predictor artifacts to {DEPLOY}/")


if __name__ == "__main__":
    main()
