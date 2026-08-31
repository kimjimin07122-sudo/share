"""
Forecast-then-detect: turning the 15s predictor into an actual early-warning
system, and measuring which fault types can be warned about at all
(2026-08-20).

WHY THIS EXISTS
---------------
experiment_leadtime_and_fault_types.py showed the previous design gives ZERO
advance warning (PROJECT_SUMMARY sec 3.12). The anomaly score was
|prediction - actual| at the target, and "actual" only exists once the fault
has already happened -- so the alarm always fires at fault time, whatever the
horizon.

THE FIX: score the FORECAST, not the error
------------------------------------------
At time T the model forecasts the state at T+H. Instead of waiting for T+H to
compare against, we ask *how anomalous the forecast itself looks* against a
model of normal built from the training flights. That verdict is available at
T, so an alarm at T is a genuine H-second warning about T+H.

  error-based  (old):  score(T+H) = |pred - actual(T+H)|   -> alarm at T+H
  forecast-based (new): score(T)   = anomaly(pred(T+H))     -> alarm at T

Two normality models are scored, since they answer slightly different
questions: an Isolation Forest over the whole predicted feature vector, and a
mag_x-specific z-score against the training distribution of mag_x features.

BUT A FORECAST CAN ONLY WARN ABOUT WHAT IS FORESEEABLE
------------------------------------------------------
Changing the scoring rule is necessary but not sufficient: if a fault has no
precursor in the preceding context, no forecast can anticipate it. The
literature draws exactly this line between *incipient* faults (gradual
degradation -- forecastable) and *abrupt* faults (sudden -- not). Our own
data looks abrupt: 15s before the verified mag_x burst, mag_x sits only 0.81
baseline-sigma from its earlier-flight level, i.e. essentially no precursor.

So this script measures the CEILING rather than assuming one. Four injected
fault types span the spectrum -- from `slow_ramp` (a 25s gradual onset, the
kind the dataset paper's Fig. 8 describes, where a yaw error builds from the
start of the flight until control can no longer compensate) to `step` and
`noise_burst` (instantaneous, no precursor by construction). Lead time is
measured as: how long before the fault becomes *visible* (first crosses
VISIBLE_SIGMA) does each method first raise an alarm.

Expected and worth stating up front: slow_ramp should earn real lead time and
step should earn none. If step also shows lead time, something is leaking.

AND IT DID LEAK, THE FIRST TIME
-------------------------------
The first run gave `step` and `noise_burst` +32.5s of "lead time" -- for
faults that are instantaneous by construction. The cause was per-flight
normalization being NON-CAUSAL: fitting each flight's mean/std over the whole
flight means a fault injected at t=100s changes the normalized values at
t=10s (measured mean |diff| 0.358), so the pre-fault windows were no longer
the clean data they were supposed to be. Fixed by fitting each flight's
scaler on a WARMUP_SEC prefix instead, which drops the pre-fault difference
to exactly 0.000000.

This matters well beyond this script: whole-flight normalization cannot run
onboard either, since future samples do not exist yet at inference time. The
main pipeline (preprocessing.py) still uses the whole-flight form and should
move to the causal one -- see PROJECT_SUMMARY sec 3.13.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from config import Config
from data_loader import StandardScaler
from gru_data_loader import read_raw_csv_with_timestamp
from train_eval_lstm_predictor_15s import (
    CONTEXT_BINS, RANDOM_SEED, INK_PRIMARY, INK_SECONDARY, GRIDLINE, SURFACE,
    build_pairs, flight_to_binned,
)
from train_eval_lstm_predictor_15s_multisession import DUPLICATES, EVAL_ONLY
from experiment_leadtime_and_fault_types import train_for_horizon, MAG_X, MAG_X_COLS

HORIZON = 15                 # seconds of lead time we are trying to earn
FAULT_AMPLITUDE_STD = 6.0
RAMP_SEC = 25.0              # onset length for the gradual fault
SUSTAIN_SEC = 10.0
VISIBLE_SIGMA = 3.0          # fault counts as "manifest" once it crosses this
ONSET_STARTS = [50.0, 75.0, 100.0]
WARMUP_SEC = 30.0            # causal normalization: fit each flight's scaler on
                             # its first 30s only. Whole-flight statistics let a
                             # fault at t=100s alter the data at t=10s, which
                             # fabricated lead time for precursor-free faults
                             # (see the note below and PROJECT_SUMMARY sec 3.13).

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_early_warning")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def inject(raw, real_time, kind, onset, amplitude_std):
    """Returns (raw, deviation_in_sigma_per_row). All four types reach the
    same peak amplitude; they differ only in how fast they get there, which
    is exactly the axis that decides forecastability."""
    raw = raw.copy()
    col = raw[:, MAG_X].astype(np.float64)
    sd = col.std()
    dev = np.zeros_like(real_time, dtype=np.float64)
    t = real_time - onset

    if kind == "slow_ramp":                       # incipient: 25s gradual onset
        ramp = np.clip(t / RAMP_SEC, 0, 1)
        dev = np.where(t >= 0, ramp, 0.0)
        dev = np.where(t > RAMP_SEC + SUSTAIN_SEC, 0.0, dev)
    elif kind == "bias_drift":                    # 3s triangular bump
        d = 3.0
        dev = np.where((t >= 0) & (t < d), 1 - np.abs(2 * t / d - 1), 0.0)
    elif kind == "step":                          # abrupt, no precursor
        dev = ((t >= 0) & (t < SUSTAIN_SEC)).astype(float)
    elif kind == "noise_burst":                   # abrupt, no precursor
        rng = np.random.default_rng(RANDOM_SEED + int(onset))
        m = (t >= 0) & (t < SUSTAIN_SEC)
        dev = np.where(m, np.abs(rng.normal(0, 1, size=len(t))), 0.0)
    else:
        raise ValueError(kind)

    dev_sigma = dev * amplitude_std
    col += dev_sigma * sd
    raw[:, MAG_X] = col.astype(np.float32)
    return raw, dev_sigma


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    augment = [f for f in test_files if os.path.basename(f) not in EVAL_ONLY | DUPLICATES]
    eval_files = [f for f in test_files if os.path.basename(f) in EVAL_ONLY]

    train_raw = [read_raw_csv_with_timestamp(f) for f in [train_csv] + augment]
    train_binned_raw = [flight_to_binned(r, t, warmup_sec=WARMUP_SEC) for r, t in train_raw]
    fscaler = StandardScaler()
    fscaler.fit(np.vstack(train_binned_raw))
    train_binned = [fscaler.transform(b) for b in train_binned_raw]

    model, n_pairs = train_for_horizon(HORIZON, train_binned, device)
    print(f"predictor: context {CONTEXT_BINS}s -> +{HORIZON}s, {n_pairs} training pairs")

    # --- normality model over STATES (not errors): what does a normal
    # feature vector look like? The forecast is scored against this. ---
    normal_states = np.concatenate([b for b in train_binned], axis=0)
    iforest = IsolationForest(n_estimators=Config.IFOREST_N_ESTIMATORS,
                              random_state=RANDOM_SEED, n_jobs=-1).fit(normal_states)
    magx_mu = normal_states[:, MAG_X_COLS].mean(axis=0)
    magx_sd = np.where(normal_states[:, MAG_X_COLS].std(axis=0) == 0, 1e-6,
                       normal_states[:, MAG_X_COLS].std(axis=0))

    def forecast_scores(binned):
        """-> (context_end_time, IF score of forecast, mag_x z of forecast,
        error score at target). The first three are available at
        context_end_time; the last only at context_end_time + HORIZON."""
        pasts, nexts, target_idx = build_pairs(binned, CONTEXT_BINS, HORIZON, stride=1)
        if len(pasts) == 0:
            return None
        with torch.no_grad():
            pred = model(torch.tensor(pasts).to(device)).cpu().numpy()
        ctx_end = np.asarray(target_idx) - HORIZON
        if_score = -iforest.score_samples(pred)
        magx_z = np.abs((pred[:, MAG_X_COLS] - magx_mu) / magx_sd).mean(axis=1)
        err = np.abs(pred[:, MAG_X_COLS] - nexts[:, MAG_X_COLS]).mean(axis=1)
        return ctx_end, if_score, magx_z, err, np.asarray(target_idx)

    # --- thresholds from the CLEAN eval files (no injection) ---
    eval_raw = [(os.path.basename(f), *read_raw_csv_with_timestamp(f)) for f in eval_files]
    clean = {"if": [], "magx": [], "err": []}
    for _, raw, t in eval_raw:
        out = forecast_scores(flight_to_binned(raw, t, feature_scaler=fscaler, warmup_sec=WARMUP_SEC))
        if out is None:
            continue
        _, s_if, s_mz, s_err, _ = out
        clean["if"].append(s_if); clean["magx"].append(s_mz); clean["err"].append(s_err)
    thr = {k: float(np.percentile(np.concatenate(v), 99)) for k, v in clean.items()}
    print(f"thresholds (99th pct on clean eval flights): "
          + "  ".join(f"{k}={v:.3f}" for k, v in thr.items()))

    kinds = ["slow_ramp", "bias_drift", "step", "noise_burst"]
    rows = []
    for kind in kinds:
        per_method = {"forecast_IF": [], "forecast_magx": [], "error_based": []}
        for name, raw, t in eval_raw:
            clean_out = forecast_scores(flight_to_binned(raw, t, feature_scaler=fscaler,
                                                          warmup_sec=WARMUP_SEC))
            if clean_out is None:
                continue
            for onset in ONSET_STARTS:
                if onset + RAMP_SEC + SUSTAIN_SEC > t[-1]:
                    continue
                inj, dev_sigma = inject(raw, t, kind, onset, FAULT_AMPLITUDE_STD)
                vis = t[dev_sigma >= VISIBLE_SIGMA]
                if len(vis) == 0:
                    continue
                manifest = float(vis[0])          # when the fault becomes visible
                out = forecast_scores(flight_to_binned(inj, t, feature_scaler=fscaler,
                                                        warmup_sec=WARMUP_SEC))
                if out is None:
                    continue
                ctx_end, s_if, s_mz, s_err, tgt = out
                _, c_if, c_mz, c_err, _ = clean_out

                def first_attributable(times, injected, clean, threshold, cutoff):
                    """Earliest alarm that the FAULT caused: it must fire in the
                    injected run and NOT in the clean run of the same flight at
                    the same time. Without this, `min()` just picks up the
                    earliest background false positive -- at a 99th-percentile
                    threshold ~1% of windows alarm anyway, which is what gave
                    precursor-free `step` a bogus 48.5s of "lead time"."""
                    n = min(len(times), len(injected), len(clean))
                    m = (injected[:n] > threshold) & ~(clean[:n] > threshold) & (times[:n] <= cutoff)
                    return float(times[:n][m].min()) if m.any() else None

                for key, s_inj, s_cln, tname in (
                        ("forecast_IF", s_if, c_if, "if"),
                        ("forecast_magx", s_mz, c_mz, "magx")):
                    a = first_attributable(ctx_end, s_inj, s_cln, thr[tname], manifest)
                    per_method[key].append(manifest - a if a is not None else np.nan)
                a = first_attributable(tgt, s_err, c_err, thr["err"], manifest + SUSTAIN_SEC)
                per_method["error_based"].append(manifest - a if a is not None else np.nan)

        print(f"\n=== {kind} ===")
        for meth, leads in per_method.items():
            leads = np.array(leads, dtype=float)
            n = len(leads)
            detected = ~np.isnan(leads)
            med = float(np.median(leads[detected])) if detected.any() else float("nan")
            rows.append({"fault_type": kind, "method": meth, "n_events": n,
                         "detected": int(detected.sum()),
                         "median_lead_sec": med})
            print(f"  {meth:14s} detected {int(detected.sum())}/{n}   "
                  f"median lead time {med:+.1f}s" if detected.any() else
                  f"  {meth:14s} detected 0/{n}   (never fired before the fault was visible)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "early_warning.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved results to {RESULTS_DIR}/")
    plot(rows, kinds, os.path.join(FIGURES_DIR, "early_warning_lead_time.png"))
    print(f"saved figure to {FIGURES_DIR}/")


def plot(rows, kinds, save_path):
    methods = ["forecast_IF", "forecast_magx", "error_based"]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(kinds))
    w = 0.26
    for i, meth in enumerate(methods):
        vals = []
        for k in kinds:
            r = next((r for r in rows if r["fault_type"] == k and r["method"] == meth), None)
            v = r["median_lead_sec"] if r else float("nan")
            vals.append(0.0 if (v is None or np.isnan(v)) else v)
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w * 0.9, color=PALETTE[i], label=meth)
        for xi, v, k in zip(xs, vals, kinds):
            label = f"{v:.0f}s" if v > 0 else "none"
            ax.text(xi, v + 0.3, label, ha="center", fontsize=8.5, color=INK_SECONDARY)

    ax.axhline(HORIZON, color=INK_PRIMARY, linestyle="--", linewidth=1.0)
    ax.text(len(kinds) - 0.4, HORIZON + 0.4, f"{HORIZON}s required", fontsize=9, color=INK_SECONDARY, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["slow_ramp\n(gradual, 25s onset)", "bias_drift\n(3s bump)",
                        "step\n(abrupt)", "noise_burst\n(abrupt)"], fontsize=9)
    ax.set_ylabel("median lead time before the fault becomes visible (s)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Early warning is only possible for faults that have a precursor", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
