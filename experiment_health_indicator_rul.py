"""
Health-indicator extrapolation with an explicit time-to-failure estimate
(2026-08-20). The prognostics/RUL formulation, and the sixth forecaster
tried for this problem.

Everything before this predicted a 64-dimensional STATE and then asked how
anomalous that state looked. This inverts the framing to the one the
prognostics literature actually uses for lead time:

  1. collapse the flight to a scalar HEALTH INDICATOR -- here how far mag_x
     sits from the level it held during the causal warm-up window, which is
     the quantity a magnetometer fault actually moves;
  2. fit its recent trend;
  3. extrapolate to the level at which the fault becomes visible, and report
     the TIME UNTIL THAT CROSSING.

Two things make this different in kind from the earlier attempts rather than
just another variant. It is trained on nothing, so it cannot regress to a
normal-data mean the way the LSTM did (0/4 on a gradual ramp, losing to a
training-free extrapolator at 4/4). And its output is an interpretable
"about N seconds until this crosses the failure level" instead of an opaque
anomaly score -- which is the form an operator, or a fail-safe controller,
can actually act on.

Also sweeps the trend window, since how much history the slope is fitted over
directly controls how early a slow ramp becomes visible in the trend, and 15s
was never justified as a choice.

Same harness as the rest: causal normalization, attributable alarms (fires on
the injected run but not on the clean run of the same flight), slow_ramp as
the forecastable case, step as the control that must stay silent.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from data_loader import build_scaler
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import resample_to_rate
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY
from experiment_early_warning import (
    inject, HORIZON, FAULT_AMPLITUDE_STD, RAMP_SEC, SUSTAIN_SEC,
    VISIBLE_SIGMA, ONSET_STARTS, WARMUP_SEC,
)

MAG_X = Config.CHANNEL_NAMES.index("mag_x")
HZ = 1.0                       # health indicator sampled at 1Hz
TREND_WINDOWS = [10, 15, 25, 40]   # seconds of history the slope is fitted over
MIN_SLOPE = 1e-3               # below this the trend is flat; no crossing predicted

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_health_indicator")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def health_indicator(raw, real_time, warmup_sec=WARMUP_SEC):
    """|mag_x - warm-up mean| / warm-up sd, at 1Hz. Causal: the reference is
    fixed during the warm-up window and never revisited."""
    binned, taxis = resample_to_rate(raw, real_time, HZ)
    if len(binned) == 0:
        return None, None
    warm = binned[taxis <= warmup_sec, MAG_X]
    if len(warm) < 5:
        return None, None
    mu, sd = float(warm.mean()), float(warm.std()) or 1e-6
    return np.abs((binned[:, MAG_X] - mu) / sd), taxis


def time_to_crossing(hi, taxis, trend_sec, level):
    """At each time step, fit a slope over the last `trend_sec` of the health
    indicator and report the predicted seconds until it reaches `level`.
    inf where the trend is flat, falling, or the crossing is already past."""
    n = int(trend_sec * HZ)
    out = np.full(len(hi), np.inf)
    for i in range(n, len(hi)):
        y = hi[i - n:i]
        x = np.arange(n, dtype=np.float64)
        xc = x - x.mean()
        slope = float((xc * y).sum() / (xc ** 2).sum())
        if slope <= MIN_SLOPE:
            continue
        remaining = (level - y[-1]) / slope / HZ
        out[i] = remaining if remaining > 0 else 0.0
    return out


def main():
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]
    eval_raw = [(os.path.basename(f), *read_raw_csv_with_timestamp(f)) for f in eval_files]
    print(f"eval flights: {[n for n, _, _ in eval_raw]}")
    print(f"health indicator: |mag_x - warmup mean| / warmup sd, warm-up {WARMUP_SEC:.0f}s, "
          f"failure level {VISIBLE_SIGMA} sigma\n")

    rows = []
    for trend_sec in TREND_WINDOWS:
        for kind in ["slow_ramp", "step"]:
            leads, ttf_errors = [], []
            for fn, raw, t in eval_raw:
                hi_c, taxis = health_indicator(raw, t)
                if hi_c is None:
                    continue
                ttf_c = time_to_crossing(hi_c, taxis, trend_sec, VISIBLE_SIGMA)
                for onset in ONSET_STARTS:
                    if onset + RAMP_SEC + SUSTAIN_SEC > t[-1]:
                        continue
                    inj, dev = inject(raw, t, kind, onset, FAULT_AMPLITUDE_STD)
                    vis = t[dev >= VISIBLE_SIGMA]
                    if len(vis) == 0:
                        continue
                    manifest = float(vis[0])
                    hi_i, _ = health_indicator(inj, t)
                    ttf_i = time_to_crossing(hi_i, taxis, trend_sec, VISIBLE_SIGMA)

                    # alarm: predicted crossing within HORIZON seconds, on the
                    # injected run but not the clean one at the same instant
                    fires = (ttf_i <= HORIZON) & ~(ttf_c <= HORIZON) & (taxis <= manifest)
                    if fires.any():
                        first = float(taxis[fires][0])
                        leads.append(manifest - first)
                        ttf_errors.append(abs(ttf_i[fires][0] - (manifest - first)))
                    else:
                        leads.append(np.nan)

            a = np.array(leads, dtype=float)
            det = ~np.isnan(a)
            med = float(np.median(a[det])) if det.any() else float("nan")
            mae = float(np.mean(ttf_errors)) if ttf_errors else float("nan")
            rows.append({"trend_sec": trend_sec, "fault_type": kind, "n": len(a),
                         "detected": int(det.sum()), "median_lead_sec": med,
                         "ttf_abs_error_sec": mae})
            msg = (f"median lead {med:+.1f}s   time-to-failure |error| {mae:.1f}s"
                   if det.any() else "never fired early")
            print(f"  trend {trend_sec:2d}s  {kind:10s} detected {int(det.sum())}/{len(a)}   {msg}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "health_indicator_rul.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(rows, os.path.join(FIGURES_DIR, "health_indicator_rul.png"))
    print(f"\nsaved results to {RESULTS_DIR}/ and figure to {FIGURES_DIR}/")


def plot(rows, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TREND_WINDOWS)); w = 0.36
    for i, kind in enumerate(["slow_ramp", "step"]):
        vals, dets = [], []
        for ts in TREND_WINDOWS:
            r = next((r for r in rows if r["trend_sec"] == ts and r["fault_type"] == kind), None)
            v = r["median_lead_sec"] if r else float("nan")
            vals.append(0.0 if (v is None or np.isnan(v)) else v)
            dets.append(f"{r['detected']}/{r['n']}" if r else "-")
        xs = x + (i - 0.5) * w
        ax.bar(xs, vals, w * 0.9, color=PALETTE[i], label=kind)
        for xi, v, d in zip(xs, vals, dets):
            ax.text(xi, v + 0.4, (f"{v:.0f}s\n({d})" if v > 0 else f"none\n({d})"),
                    ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.axhline(HORIZON, color=INK_PRIMARY, linestyle="--", linewidth=1.0)
    ax.text(len(TREND_WINDOWS) - 0.55, HORIZON + 0.5, f"{HORIZON}s required",
            fontsize=9, color=INK_SECONDARY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}s" for t in TREND_WINDOWS])
    ax.set_xlabel("trend window the slope is fitted over")
    ax.set_ylabel("median lead time before the fault is visible (s)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Health-indicator extrapolation: predicted time until the fault level is crossed",
                 color=INK_PRIMARY)
    fig.tight_layout(); fig.savefig(save_path, dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
