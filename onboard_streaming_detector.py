"""
Onboard streaming detector for pipeline v4.1 (2026-08-20).

Everything in this repo so far has been an OFFLINE evaluator: read a whole
CSV, compute features over the full sequence, then window it. That cannot run
on the aircraft -- at inference time the rest of the flight has not happened
yet. This module is the deployable form: samples arrive one at a time, and a
fused anomaly score is emitted as soon as each window closes, using only data
that has already arrived.

What "causal" costs, concretely:
  * The scaler is fit during a WARMUP_SEC calibration phase at the start of
    the flight and then frozen -- the offline code fit it on the whole flight,
    which measurably let a fault at t=100s change the values at t=10s
    (PROJECT_SUMMARY sec 3.13).
  * Derivatives come from np.gradient over a bounded rolling buffer instead of
    the full sequence, so the buffer's leading edge uses one-sided differences.
    The buffer carries GRADIENT_MARGIN extra bins beyond the window so the
    scored region always sits in the interior; check_agreement() measures what
    residual difference remains against the offline path.

Inference runs through ONNX Runtime on CPU, which is the realistic Jetson
path, and per-stage latency is reported so the budget can be compared with
Reis & Reis's Jetson Nano figures (42ms end-to-end per window, ~1.2MB LSTM,
127MB peak RAM). No board yet, so this is a CPU proxy, not a Jetson
measurement -- see PROJECT_SUMMARY sec 4 on why the two are not comparable.
"""
import os
import time
from collections import deque

import joblib
import numpy as np
import onnxruntime as ort

from config import Config
from data_loader import build_scaler
from feature_engineering import compute_sequence_features, compute_window_summary
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import window_length, stride_length

# torch is deliberately NOT imported at module level. Scoring runs entirely
# through ONNX Runtime, so a flight computer only needs onnxruntime, numpy,
# scikit-learn, joblib and pandas. The two functions below that do need torch
# -- exporting the ONNX file and cross-checking against the offline pipeline
# -- import it locally, since both are development-time steps that never run
# on the aircraft.

ONNX_PATH = os.path.join(Config.BASE_DIR, "lstm_ae_v4.onnx")
INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL
GRADIENT_MARGIN = 4       # extra bins kept so the scored window is interior


def load_session(path=ONNX_PATH):
    """Load the prebuilt ONNX graph. This is the onboard path -- no torch."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It ships in deploy/; run export_onnx() on a "
            f"machine with torch to rebuild it.")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def export_onnx():
    """Development-time only: rebuild the ONNX file from the .pth checkpoint.
    Needs torch, which a flight computer does not have -- main() only calls
    this when the prebuilt graph is missing AND torch happens to be available."""
    import torch
    from networks.lstm_ae import LSTMAutoencoder
    model = LSTMAutoencoder(INPUT_DIM, Config.LSTM_AE_HIDDEN_SIZE,
                            Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM)
    model.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                     map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, window_length(), INPUT_DIM)
    torch.onnx.export(model, dummy, ONNX_PATH,
                      input_names=["window"], output_names=["reconstruction"],
                      dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
                      opset_version=17)
    with torch.no_grad():
        ref = model(dummy).numpy()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"window": dummy.numpy()})[0]
    return sess, float(np.abs(ref - got).max()), os.path.getsize(ONNX_PATH) / 1024


class StreamingDetector:
    """Feed it raw samples in arrival order; it yields (time, score) as each
    window closes. Holds a bounded buffer -- memory does not grow with flight
    length, which is the whole point of a streaming form."""

    def __init__(self, sess, feature_scaler, iforest, stats, hz=None,
                 warmup_sec=None, window=None, stride=None):
        self.sess = sess
        self.fscaler = feature_scaler
        self.iforest = iforest
        self.channel_mean, self.channel_std = stats["channel_mean"], stats["channel_std"]
        self.if_mean, self.if_std = float(stats["if_mean"]), float(stats["if_std"])
        self.hz = hz or Config.RESAMPLE_HZ
        self.warmup_sec = Config.WARMUP_SEC if warmup_sec is None else warmup_sec
        self.window = window or window_length()
        self.stride = stride or stride_length(stride_sec=Config.EVAL_STRIDE_SEC)

        self.bin_idx = None
        self._acc, self._n = None, 0
        self.warm_bins = []
        self.scaler = None
        self.buf = deque(maxlen=self.window + GRADIENT_MARGIN)
        self.buf_t = deque(maxlen=self.window + GRADIENT_MARGIN)
        self.since_last = 0
        self.latencies = {"feature": [], "onnx": [], "iforest": [], "total": []}

    def push(self, sample, t):
        """One raw sample at real time t (seconds since takeoff)."""
        idx = int(t * self.hz)
        if self.bin_idx is None:
            self.bin_idx = idx
            self._acc = np.zeros_like(sample, dtype=np.float64)
            self._n = 0
        out = None
        if idx != self.bin_idx:                       # current bin just closed
            out = self._close_bin(self._acc / max(self._n, 1), self.bin_idx / self.hz)
            self.bin_idx = idx
            self._acc = np.zeros_like(sample, dtype=np.float64)
            self._n = 0
        self._acc += sample
        self._n += 1
        return out

    def _close_bin(self, binned, t):
        if self.scaler is None:                       # still calibrating
            self.warm_bins.append(binned)
            if t >= self.warmup_sec and len(self.warm_bins) >= 10:
                self.scaler = build_scaler()
                self.scaler.fit(np.asarray(self.warm_bins, dtype=np.float32))
                self.warm_bins = []
            return None

        self.buf.append(binned.astype(np.float32))
        self.buf_t.append(t)
        self.since_last += 1
        if len(self.buf) < self.window + GRADIENT_MARGIN or self.since_last < self.stride:
            return None
        self.since_last = 0
        return self._score(t)

    def _score(self, t):
        t0 = time.perf_counter()
        raw = np.asarray(self.buf, dtype=np.float32)
        taxis = np.asarray(self.buf_t, dtype=np.float64)
        feats = compute_sequence_features(self.scaler.transform(raw), taxis)
        win = self.fscaler.transform(feats)[-self.window:][None, :, :].astype(np.float32)
        t1 = time.perf_counter()

        recon = self.sess.run(None, {"window": win})[0]
        t2 = time.perf_counter()

        residual = np.abs(recon - win)
        lstm_norm = (residual.mean() - self.channel_mean.mean()) / self.channel_std.mean()
        if_feat = np.concatenate([compute_window_summary(win[0]),
                                  compute_window_summary(residual[0])])[None, :]
        if_norm = (-self.iforest.score_samples(if_feat)[0] - self.if_mean) / self.if_std
        t3 = time.perf_counter()

        self.latencies["feature"].append((t1 - t0) * 1000)
        self.latencies["onnx"].append((t2 - t1) * 1000)
        self.latencies["iforest"].append((t3 - t2) * 1000)
        self.latencies["total"].append((t3 - t0) * 1000)
        fused = Config.LSTM_AE_FUSION_ALPHA * lstm_norm + (1 - Config.LSTM_AE_FUSION_ALPHA) * if_norm
        return t, float(fused)


def run_flight(path, sess, fscaler, iforest, stats):
    raw, t = read_raw_csv_with_timestamp(path)
    det = StreamingDetector(sess, fscaler, iforest, stats)
    out = []
    for i in range(len(raw)):
        r = det.push(raw[i], float(t[i]))
        if r is not None:
            out.append(r)
    return out, det


def check_agreement(path, scored, fscaler, iforest, stats):
    """The streaming path differs from the offline one only in how derivatives
    see the sequence edge, so scores should track closely but not identically."""
    import torch
    from networks.lstm_ae import LSTMAutoencoder
    from eval_lstm_ae import compute_window_residuals, fused_score_batch
    from preprocessing import prepare_flight_file
    model = LSTMAutoencoder(INPUT_DIM, Config.LSTM_AE_HIDDEN_SIZE,
                            Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM)
    model.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                     map_location="cpu"))
    model.eval()
    feats_raw, time_axis, _ = prepare_flight_file(path)
    feats = fscaler.transform(feats_raw)
    W, S = window_length(), stride_length(stride_sec=Config.EVAL_STRIDE_SEC)
    res, win = compute_window_residuals(model, feats, torch.device("cpu"), window=W, stride=S)
    offline = fused_score_batch(res, win, iforest, stats["channel_mean"], stats["channel_std"],
                                float(stats["if_mean"]), float(stats["if_std"]))
    off_t = time_axis[np.minimum(np.arange(len(offline)) * S + W - 1, len(time_axis) - 1)]

    st_t = np.array([s[0] for s in scored])
    st_v = np.array([s[1] for s in scored])
    b = np.array([offline[int(np.argmin(np.abs(off_t - tt)))] for tt in st_t])
    print(f"  correlation streaming vs offline : {np.corrcoef(st_v, b)[0,1]:.4f}")
    print(f"  mean |difference|                : {np.abs(st_v-b).mean():.4f}  (offline score sd {b.std():.4f})")
    print(f"  peak location  streaming t={st_t[st_v.argmax()]:.2f}s   offline t={off_t[offline.argmax()]:.2f}s")


DEPLOY = os.path.join(Config.BASE_DIR, "deploy")


def load_artifacts():
    """Prefer deploy/, which is what a cloned board actually has; fall back to
    the repo root where training writes them."""
    for d in (DEPLOY, Config.BASE_DIR):
        fs = os.path.join(d, "lstm_ae_feature_scaler_heldout.joblib")
        if os.path.exists(fs):
            return (joblib.load(fs),
                    joblib.load(os.path.join(d, "iforest_model_heldout.joblib")),
                    np.load(os.path.join(d, "lstm_ae_stats_heldout.npz")), d)
    raise FileNotFoundError("detector artifacts not found in deploy/ or the repo root")


def main():
    # Use the shipped ONNX graph. Only rebuild it when it is missing, and only
    # if torch is around -- re-exporting on every run made this script require
    # torch on the flight computer, which defeats the point of the ONNX path.
    onnx_path = ONNX_PATH if os.path.exists(ONNX_PATH) else os.path.join(DEPLOY, "lstm_ae_v4.onnx")
    if os.path.exists(onnx_path):
        sess = load_session(onnx_path)
        print(f"=== loaded {os.path.relpath(onnx_path, Config.BASE_DIR)} "
              f"({os.path.getsize(onnx_path)/1024:.0f} KB) ===")
    else:
        print("=== ONNX graph missing, rebuilding from checkpoint (needs torch) ===")
        sess, max_err, size_kb = export_onnx()
        print(f"  exported {ONNX_PATH} ({size_kb:.0f} KB)   torch-vs-onnx max diff = {max_err:.2e}")

    fscaler, iforest, stats, src = load_artifacts()
    print(f"    artifacts from {os.path.relpath(src, Config.BASE_DIR)}/")

    target = os.path.join(DEPLOY, "sample", Config.BURST_FILE)
    if not os.path.exists(target):
        target = os.path.join(Config.TEST_DATA_PATH, Config.BURST_FILE)
    print(f"\n=== streaming {Config.BURST_FILE} sample-by-sample ===")
    wall0 = time.perf_counter()
    scored, det = run_flight(target, sess, fscaler, iforest, stats)
    wall = time.perf_counter() - wall0
    raw, t = read_raw_csv_with_timestamp(target)
    print(f"  fed {len(raw)} raw samples covering {t[-1]:.1f}s of flight")
    print(f"  emitted {len(scored)} window scores "
          f"(first at t={scored[0][0]:.2f}s -- after the {Config.WARMUP_SEC:.0f}s calibration phase)")
    print(f"  wall clock {wall:.2f}s for {t[-1]:.1f}s of flight -> {t[-1]/wall:.1f}x real time")

    print("\n=== per-window latency on CPU (Jetson proxy) ===")
    for stage in ("feature", "onnx", "iforest", "total"):
        v = np.array(det.latencies[stage])
        print(f"  {stage:8s} mean {v.mean():6.2f} ms   p95 {np.percentile(v, 95):6.2f} ms   max {v.max():6.2f} ms")
    budget = np.array(det.latencies["total"])
    fits = np.percentile(budget, 95) < Config.EVAL_STRIDE_SEC * 1000
    print(f"  window closes every {Config.EVAL_STRIDE_SEC*1000:.0f} ms; "
          f"p95 compute is {np.percentile(budget,95):.2f} ms "
          f"-> {'FITS' if fits else 'DOES NOT FIT'} the real-time budget")
    print("  (Reis & Reis report 42 ms/window end-to-end on a Jetson Nano at 5s windows)")

    # The cross-check reconstructs the offline path in torch, so it only runs
    # where torch exists. Its absence is normal on a flight computer.
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n(skipping the offline cross-check: it needs torch, which this "
              "machine does not have -- expected onboard)")
        return
    print("\n=== does streaming agree with the offline pipeline? ===")
    check_agreement(target, scored, fscaler, iforest, stats)


if __name__ == "__main__":
    main()
