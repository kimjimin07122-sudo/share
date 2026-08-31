"""
Onboard streaming 15s-ahead PREDICTOR (pipeline v4.1, 2026-08-20).

Companion to onboard_streaming_detector.py. The detector flags a fault as it
happens; this forecasts the state 15 seconds ahead and scores that forecast,
so an alarm can fire before the fault rather than with it.

Emits three signals per second, all computed from data that has already
arrived:

  lstm_if     how anomalous the neural forecast of T+15 looks, judged by an
              IsolationForest over normal flight states
  lstm_magx   the same forecast, but only its mag_x features, against the
              normal mag_x distribution
  ttf         seconds until a scalar health indicator, extended along its
              recent trend, reaches the failure level -- "about N seconds
              until this crosses"

Read the third one first. Across seven forecasters the neural point forecast
placed LAST on a gradual injected ramp (0/4) while this training-free
extrapolation placed first (4/4, +5.0s median lead, 4.6s error on the estimate)
-- a normal-trained network learns to emit normal-looking futures, so asking
whether its forecast looks anomalous rarely fires. Both are shipped so the
board can reproduce that comparison rather than take it on trust.

WHAT THIS DOES AND DOES NOT DELIVER
  * best measured lead time is ~5s, against a 15s requirement
  * that ceiling is a property of the DATA, not the code: the one verified
    fault in this dataset is abrupt, sitting 0.81 sigma from baseline 15s
    beforehand, so there is no precursor for any method to find
  * lead time above is measured on synthetic gradual ramps
  See PROJECT_SUMMARY secs 3.12, 3.13, 3.16.

No torch: forecasting runs through ONNX Runtime.
"""
import os
import time
from collections import deque

import joblib
import numpy as np
import onnxruntime as ort

from config import Config
from data_loader import build_scaler
from feature_engineering import compute_sequence_features
from gru_data_loader import read_raw_csv_with_timestamp

DEPLOY = os.path.join(Config.BASE_DIR, "deploy")
ONNX_PATH = os.path.join(DEPLOY, "lstm_predictor_15s.onnx")
BIN_HZ = 1.0                      # the predictor works on 1-second bins
MAG_X = Config.CHANNEL_NAMES.index("mag_x")
MAG_X_COLS = slice(MAG_X * 4, MAG_X * 4 + 4)
GRADIENT_MARGIN = 2               # extra bins so the context is interior
TREND_SEC = 10                    # health-indicator slope window (best of 10/15/25/40)
FAILURE_SIGMA = 3.0               # health indicator level counted as failure
MIN_SLOPE = 1e-3


class StreamingPredictor:
    """Push raw samples in arrival order; get a forecast verdict each second.

    Memory is bounded: one context window plus a small margin, never the
    whole flight."""

    def __init__(self, sess, feature_scaler, iforest, stats):
        self.sess = sess
        self.fscaler = feature_scaler
        self.iforest = iforest
        self.magx_mu = stats["magx_mu"]
        self.magx_sd = stats["magx_sd"]
        self.thr_if = float(stats["thr_if"])
        self.thr_magx = float(stats["thr_magx"])
        self.context = int(stats["context_bins"])
        self.horizon = int(stats["horizon"])
        self.warmup_sec = float(stats["warmup_sec"])

        self.bin_idx = None
        self._acc, self._n = None, 0
        self.warm_bins = []
        self.scaler = None
        self.hi_mu = self.hi_sd = None          # health-indicator reference
        self.buf = deque(maxlen=self.context + GRADIENT_MARGIN)
        self.buf_t = deque(maxlen=self.context + GRADIENT_MARGIN)
        self.hi_hist = deque(maxlen=TREND_SEC)  # health indicator history
        self.latencies = {"feature": [], "onnx": [], "score": [], "total": []}

    def push(self, sample, t):
        idx = int(t * BIN_HZ)
        if self.bin_idx is None:
            self.bin_idx, self._acc, self._n = idx, np.zeros_like(sample, dtype=np.float64), 0
        out = None
        if idx != self.bin_idx:
            out = self._close_bin(self._acc / max(self._n, 1), self.bin_idx / BIN_HZ)
            self.bin_idx, self._acc, self._n = idx, np.zeros_like(sample, dtype=np.float64), 0
        self._acc += sample
        self._n += 1
        return out

    def _close_bin(self, binned, t):
        if self.scaler is None:                             # calibration phase
            self.warm_bins.append(binned)
            if t >= self.warmup_sec and len(self.warm_bins) >= 10:
                warm = np.asarray(self.warm_bins, dtype=np.float32)
                self.scaler = build_scaler()
                self.scaler.fit(warm)
                self.hi_mu = float(warm[:, MAG_X].mean())
                self.hi_sd = float(warm[:, MAG_X].std()) or 1e-6
                self.warm_bins = []
            return None

        self.hi_hist.append(abs((binned[MAG_X] - self.hi_mu) / self.hi_sd))
        self.buf.append(binned.astype(np.float32))
        self.buf_t.append(t)
        if len(self.buf) < self.context + GRADIENT_MARGIN:
            return None
        return self._verdict(t)

    def _time_to_failure(self):
        """Least-squares slope of the health indicator, extended to the failure
        level. inf when it is flat or falling -- nothing to predict."""
        if len(self.hi_hist) < TREND_SEC:
            return float("inf"), float(self.hi_hist[-1])
        y = np.asarray(self.hi_hist, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        xc = x - x.mean()
        slope = float((xc * y).sum() / (xc ** 2).sum())
        if slope <= MIN_SLOPE:
            return float("inf"), float(y[-1])
        remaining = (FAILURE_SIGMA - y[-1]) / slope / BIN_HZ
        return max(remaining, 0.0), float(y[-1])

    def _verdict(self, t):
        t0 = time.perf_counter()
        raw = np.asarray(self.buf, dtype=np.float32)
        taxis = np.asarray(self.buf_t, dtype=np.float64)
        feats = compute_sequence_features(self.scaler.transform(raw), taxis)
        ctx = self.fscaler.transform(feats)[-self.context:][None, :, :].astype(np.float32)
        t1 = time.perf_counter()

        forecast = self.sess.run(None, {"context": ctx})[0]
        t2 = time.perf_counter()

        lstm_if = float(-self.iforest.score_samples(forecast)[0])
        lstm_magx = float(np.abs((forecast[0, MAG_X_COLS] - self.magx_mu) / self.magx_sd).mean())
        ttf, hi = self._time_to_failure()
        t3 = time.perf_counter()

        self.latencies["feature"].append((t1 - t0) * 1000)
        self.latencies["onnx"].append((t2 - t1) * 1000)
        self.latencies["score"].append((t3 - t2) * 1000)
        self.latencies["total"].append((t3 - t0) * 1000)

        return {
            "time_sec": t,
            "target_sec": t + self.horizon,
            "lstm_if": lstm_if,
            "lstm_magx": lstm_magx,
            "health_indicator": hi,
            "seconds_to_failure": ttf,
            "alarm_lstm_if": lstm_if > self.thr_if,
            "alarm_lstm_magx": lstm_magx > self.thr_magx,
            "alarm_ttf": ttf <= self.horizon,
        }


def load(deploy_dir=DEPLOY):
    sess = ort.InferenceSession(os.path.join(deploy_dir, "lstm_predictor_15s.onnx"),
                                providers=["CPUExecutionProvider"])
    fscaler = joblib.load(os.path.join(deploy_dir, "predictor_feature_scaler.joblib"))
    iforest = joblib.load(os.path.join(deploy_dir, "predictor_normality_iforest.joblib"))
    stats = np.load(os.path.join(deploy_dir, "predictor_stats.npz"))
    return sess, fscaler, iforest, stats


def run_flight(path, sess, fscaler, iforest, stats):
    raw, t = read_raw_csv_with_timestamp(path)
    pred = StreamingPredictor(sess, fscaler, iforest, stats)
    out = []
    for i in range(len(raw)):
        v = pred.push(raw[i], float(t[i]))
        if v is not None:
            out.append(v)
    return out, pred


def main():
    sess, fscaler, iforest, stats = load()
    print(f"15s predictor loaded: context {int(stats['context_bins'])}s -> +{int(stats['horizon'])}s, "
          f"warm-up {float(stats['warmup_sec']):.0f}s")
    print(f"alarm thresholds: lstm_if {float(stats['thr_if']):.3f}   "
          f"lstm_magx {float(stats['thr_magx']):.3f}   ttf <= {int(stats['horizon'])}s")

    # The predictor needs far more runway than the detector: warm-up
    # (30s) + context (15 bins) + margin must all fit before the first
    # forecast, i.e. ~47s minimum. Config.BURST_FILE is only 45.7s, so the
    # default demo uses a longer held-out sortie; pass a path to override.
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = os.path.join(DEPLOY, "sample", "x70_20180601_x76_083916.csv")
        if not os.path.exists(target):
            target = os.path.join(Config.TEST_DATA_PATH, "x70_20180601_x76_083916.csv")
    print(f"\n=== streaming {os.path.basename(target)} ===")

    wall0 = time.perf_counter()
    verdicts, pred = run_flight(target, sess, fscaler, iforest, stats)
    wall = time.perf_counter() - wall0
    raw, t = read_raw_csv_with_timestamp(target)
    if not verdicts:
        need = float(stats["warmup_sec"]) + int(stats["context_bins"]) + GRADIENT_MARGIN
        print(f"  no forecasts: this flight is {t[-1]:.1f}s but the predictor needs "
              f"~{need:.0f}s (warm-up {float(stats['warmup_sec']):.0f}s + context "
              f"{int(stats['context_bins'])}s + margin) before its first output.")
        return
    print(f"  {len(raw)} raw samples over {t[-1]:.1f}s -> {len(verdicts)} forecasts, "
          f"first at t={verdicts[0]['time_sec']:.0f}s")
    print(f"  wall clock {wall:.2f}s -> {t[-1]/wall:.1f}x real time")

    print(f"\n{'now':>5s} {'about':>7s} {'lstm_if':>8s} {'lstm_magx':>10s} {'HI':>6s} {'sec→fail':>9s}  alarms")
    show = verdicts if len(verdicts) <= 40 else verdicts[::max(1, len(verdicts)//30)]
    for v in show:
        flags = "".join([
            "I" if v["alarm_lstm_if"] else ".",
            "M" if v["alarm_lstm_magx"] else ".",
            "T" if v["alarm_ttf"] else ".",
        ])
        ttf = "  inf" if not np.isfinite(v["seconds_to_failure"]) else f"{v['seconds_to_failure']:7.1f}"
        print(f"{v['time_sec']:5.0f} {v['target_sec']:7.0f} {v['lstm_if']:8.3f} "
              f"{v['lstm_magx']:10.3f} {v['health_indicator']:6.2f} {ttf:>9s}  {flags}")
    if len(show) < len(verdicts):
        print(f"  ({len(show)} of {len(verdicts)} forecasts shown)")
    print("  alarms: I=lstm_if  M=lstm_magx  T=time-to-failure within horizon")

    print("\n=== latency per forecast (CPU) ===")
    for stage in ("feature", "onnx", "score", "total"):
        a = np.array(pred.latencies[stage])
        print(f"  {stage:8s} mean {a.mean():6.2f} ms   p95 {np.percentile(a,95):6.2f} ms")
    tot = np.array(pred.latencies["total"])
    print(f"  a forecast is due every 1000 ms; p95 is {np.percentile(tot,95):.2f} ms "
          f"-> {1000/np.percentile(tot,95):.0f}x headroom")

    n_alarm = sum(v["alarm_lstm_if"] or v["alarm_lstm_magx"] or v["alarm_ttf"] for v in verdicts)
    print(f"\n{n_alarm}/{len(verdicts)} forecasts raised at least one alarm on this flight.")
    print("Note this sortie's real fault is ABRUPT (0.81 sigma precursor at -15s), so no")
    print("forecaster can anticipate it -- expect alarms at fault time, not before.")


if __name__ == "__main__":
    main()
