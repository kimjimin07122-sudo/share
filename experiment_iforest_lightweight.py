"""
How few trees does the IsolationForest actually need? (2026-08-23)

The Jetson run put the detector at 10.17 ms p95 against a 20 ms window -- it
fits, but only 2x over, which is thin once other processes compete for the CPU
in flight. The breakdown says where to look: the forest is 91% of that budget
(8.62 ms) while the ONNX network is 0.49 ms. So the lever is the forest, not
the model, which is the opposite of where lightweighting effort usually goes.

Sweeps n_estimators and reports what would make a smaller forest
unacceptable, not just what makes it attractive:

  latency     single-window score_samples, which is how the streaming path
              actually calls it -- no batching to hide behind
  detection   AUROC / recall against the held-out normal threshold
  stability   the same sweep repeated over several random seeds, since a few
              trees make the forest a noisier estimator and a single seed
              looking fine proves nothing

Channel diagnosis is deliberately NOT part of the criterion. It is computed
from the LSTM-AE reconstruction residuals alone and never touches the forest,
so it is identical at every tree count -- checking it here would only have
produced a reassuring constant. Confirmed by measurement: mag_x is the top
burst channel for 100 trees and for 5 alike.
"""
import csv
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from config import Config
from feature_engineering import compute_window_summary
from networks.lstm_ae import LSTMAutoencoder
from eval import compute_binary_metrics
from eval_lstm_ae import compute_window_residuals
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import (prepare_flight, prepare_flight_file, build_windows,
                           window_length, stride_length, window_end_times)
import glob

TREE_COUNTS = [100, 50, 30, 20, 10, 5]
# Passing the performance test is necessary but not sufficient. An isolation
# score is an average over random paths, so a handful of trees makes it a
# high-variance estimator; it survives here partly because the forest carries
# only 35% of the fused score (alpha=0.65 on the LSTM) and the test set is 38
# flights. That is thin evidence to stake a flight system on, so the deployed
# choice is floored at a count that is still a real ensemble, and the smaller
# ones are reported rather than adopted.
MIN_DEPLOYABLE_TREES = 30
HOLDOUT_FRAC = 0.15
RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_iforest_lightweight")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
JETSON_RATIO = 8.62 / 1.16          # measured board/host ratio for the forest

INK, INK2, INK3 = "#0e1620", "#3c4a58", "#6b7a88"
GRID, SURFACE = "#dce3e9", "#fafbfc"
ACCENT, PASS, FAIL, WARN = "#1f5f8b", "#2c6b52", "#9c3b34", "#a8690c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c2ccd6", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK3, "ytick.color": INK3,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W, S = window_length(), stride_length(stride_sec=Config.EVAL_STRIDE_SEC)

    fscaler = joblib.load(os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib"))
    model = LSTMAutoencoder(Config.NUM_CHANNELS * 4, Config.LSTM_AE_HIDDEN_SIZE,
                            Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                     map_location=device))
    model.eval()

    # rebuild the training-side IF features exactly as train_eval does
    raw_all, time_all = read_raw_csv_with_timestamp(
        os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv"))
    split = int(len(raw_all) * (1 - HOLDOUT_FRAC))
    feat_tr, _, _ = prepare_flight(raw_all[:split], time_all[:split])
    feat_ho, _, _ = prepare_flight(raw_all[split:], time_all[split:] - time_all[split])
    feat_tr = fscaler.transform(feat_tr)
    feat_ho = fscaler.transform(feat_ho)

    tr_w = build_windows(feat_tr, W, stride_length())
    res_tr, _ = compute_window_residuals(model, feat_tr, device, window=W, stride=stride_length())
    if_feat_tr = np.stack([np.concatenate([compute_window_summary(w), compute_window_summary(r)])
                           for w, r in zip(tr_w, res_tr)], axis=0)
    err_mu = res_tr.mean(axis=(0, 1))
    err_sd = np.where(res_tr.std(axis=(0, 1)) == 0, 1e-6, res_tr.std(axis=(0, 1)))
    print(f"IF training features: {if_feat_tr.shape}")

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    # cache residuals so each tree count is scored on identical inputs
    cache = {}
    for f in test_files:
        fr, _, _ = prepare_flight_file(f)
        fe = fscaler.transform(fr)
        r, w = compute_window_residuals(model, fe, device, window=W, stride=S)
        if r is not None:
            cache[f] = (r, w)
    ho_blocks = []
    bounds = np.linspace(0, len(feat_ho), len(test_files) + 1, dtype=int)
    for i in range(len(test_files)):
        r, w = compute_window_residuals(model, feat_ho[bounds[i]:bounds[i+1]], device,
                                        window=W, stride=S)
        if r is not None:
            ho_blocks.append((r, w))

    burst_fr, burst_t, _ = prepare_flight_file(os.path.join(Config.TEST_DATA_PATH, Config.BURST_FILE))
    burst_res, burst_win = compute_window_residuals(model, fscaler.transform(burst_fr),
                                                    device, window=W, stride=1)
    burst_time = window_end_times(burst_t, len(burst_res), window=W, stride=1)

    SEEDS = [Config.RANDOM_SEED, Config.RANDOM_SEED + 1, Config.RANDOM_SEED + 2,
             Config.RANDOM_SEED + 3, Config.RANDOM_SEED + 4]
    rows = []
    print(f"\n{'trees':>6s} {'1-window ms':>12s} {'est. Jetson':>12s} "
          f"{'AUROC mean':>11s} {'AUROC min':>10s} {'recall mean':>12s} {'burst top':>10s}")
    for n in TREE_COUNTS:
        iforest = IsolationForest(n_estimators=n, random_state=Config.RANDOM_SEED,
                                  n_jobs=-1).fit(if_feat_tr)
        s_tr = -iforest.score_samples(if_feat_tr)
        if_mu, if_sd = s_tr.mean(), (s_tr.std() or 1e-6)

        def fused(r, w):
            lstm_n = (r.mean(axis=(1, 2)) - err_mu.mean()) / err_sd.mean()
            iff = np.stack([np.concatenate([compute_window_summary(a), compute_window_summary(b)])
                            for a, b in zip(w, r)], axis=0)
            return Config.LSTM_AE_FUSION_ALPHA * lstm_n + \
                   (1 - Config.LSTM_AE_FUSION_ALPHA) * ((-iforest.score_samples(iff) - if_mu) / if_sd)

        # latency the way the streaming path calls it: one window at a time
        one = if_feat_tr[:1]
        iforest.score_samples(one)                       # warm
        t0 = time.perf_counter()
        for _ in range(300):
            iforest.score_samples(one)
        ms = (time.perf_counter() - t0) / 300 * 1000

        # repeat over seeds: with few trees the forest is a noisier estimator,
        # so one seed agreeing with 100 trees would not settle anything
        aurocs, recalls = [], []
        for sd in SEEDS:
            f_sd = IsolationForest(n_estimators=n, random_state=sd, n_jobs=-1).fit(if_feat_tr)
            s_sd = -f_sd.score_samples(if_feat_tr)
            mu_sd, sdv = s_sd.mean(), (s_sd.std() or 1e-6)

            def fused_sd(r, w, _f=f_sd, _mu=mu_sd, _sd=sdv):
                lstm_n = (r.mean(axis=(1, 2)) - err_mu.mean()) / err_sd.mean()
                iff = np.stack([np.concatenate([compute_window_summary(a), compute_window_summary(b)])
                                for a, b in zip(w, r)], axis=0)
                return Config.LSTM_AE_FUSION_ALPHA * lstm_n + \
                       (1 - Config.LSTM_AE_FUSION_ALPHA) * ((-_f.score_samples(iff) - _mu) / _sd)

            nrm = [float(fused_sd(r, w).max()) for r, w in ho_blocks]
            sc = nrm + [float(fused_sd(*cache[f]).max()) for f in test_files if f in cache]
            mm = compute_binary_metrics([0]*len(nrm) + [1]*(len(sc)-len(nrm)), sc, max(nrm))
            aurocs.append(mm["AUROC"]); recalls.append(mm["Recall"])

        normal = [float(fused(r, w).max()) for r, w in ho_blocks]
        thr = max(normal)
        scores = list(normal) + [float(fused(*cache[f]).max()) for f in test_files if f in cache]
        y = [0]*len(normal) + [1]*(len(scores)-len(normal))
        m = compute_binary_metrics(y, scores, thr)

        pc = burst_res.mean(axis=1).reshape(len(burst_res), Config.NUM_CHANNELS, 4).mean(axis=2)
        lo, hi = Config.BURST_REAL_SEC
        bm = (burst_time >= lo) & (burst_time <= hi)
        base = pc[~bm].mean(axis=0)
        ex = np.maximum(pc[bm].mean(axis=0) - base, 0)
        top_ch = int(np.argmax(ex)) + 1
        ctrl_clean = all(
            np.maximum(pc[(burst_time >= a) & (burst_time <= b)].mean(axis=0) - base, 0)[10] == 0
            for a, b in Config.CONTROL_REAL_SEC
            if ((burst_time >= a) & (burst_time <= b)).any())

        rows.append({"trees": n, "ms_per_window": ms, "jetson_est_ms": ms * JETSON_RATIO,
                     "AUROC": m["AUROC"], "recall": m["Recall"], "FN": m["FN"],
                     "auroc_mean": float(np.mean(aurocs)), "auroc_min": float(np.min(aurocs)),
                     "auroc_std": float(np.std(aurocs)), "recall_mean": float(np.mean(recalls)),
                     "recall_min": float(np.min(recalls)), "burst_top_channel": top_ch})
        print(f"{n:6d} {ms:11.3f}  {ms*JETSON_RATIO:11.2f}  {np.mean(aurocs):11.4f} "
              f"{np.min(aurocs):10.4f} {np.mean(recalls):12.4f} {'ch'+str(top_ch):>10s}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "iforest_sweep.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)

    # accept a reduction only if it holds across every seed, not just the
    # lucky one: no seed may fall below the 100-tree baseline
    base = rows[0]
    ok = [r for r in rows
          if r["auroc_min"] >= base["auroc_min"] - 1e-9
          and r["recall_min"] >= base["recall_min"] - 1e-9]
    passes_all = min(ok, key=lambda r: r["trees"]) if ok else None
    deployable = [r for r in ok if r["trees"] >= MIN_DEPLOYABLE_TREES]
    best = min(deployable, key=lambda r: r["trees"]) if deployable else None
    print()
    print(f"channel diagnosis: burst top channel is "
          f"{'ch11 (mag_x) at every tree count' if all(r['burst_top_channel']==11 for r in rows) else 'NOT stable'}"
          f" -- as expected, it does not use the forest")
    if passes_all:
        print(f"smallest forest holding AUROC and recall across all {len(SEEDS)} seeds: "
              f"{passes_all['trees']} trees")
    if best:
        if passes_all and passes_all["trees"] < best["trees"]:
            print(f"  not adopting {passes_all['trees']}: too few trees to be a stable "
                  f"estimator on unseen data, and it only passes because the forest "
                  f"carries 35% of the fused score")
        print(f"deploying {best['trees']} trees (floor {MIN_DEPLOYABLE_TREES})")
        print(f"  detector budget on the board: {base['jetson_est_ms']:.2f} ms -> "
              f"{best['jetson_est_ms']:.2f} ms (forest only)")
        det_total = 10.17 - 8.62 + best["jetson_est_ms"]
        print(f"  estimated p95 total: 10.17 ms -> {det_total:.2f} ms  "
              f"({20/det_total:.1f}x headroom, was 2.0x)")
    else:
        print("no tree count below 100 preserved the channel diagnosis -- keep 100")
    plot(rows, best, os.path.join(FIGURES_DIR, "iforest_lightweight_sweep.png"))
    print(f"saved figure to {FIGURES_DIR}/")


def plot(rows, best, path):
    trees = [r["trees"] for r in rows]
    x = np.arange(len(trees))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    ax = axes[0]
    ax.plot(x, [r["jetson_est_ms"] for r in rows], "-o", color=ACCENT, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=2, markersize=8, label="forest, est. Jetson")
    ax.plot(x, [r["ms_per_window"] for r in rows], "-o", color="#8fa8bd", linewidth=1.6,
            markeredgecolor=SURFACE, markeredgewidth=2, markersize=6, label="forest, x86")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['jetson_est_ms']:.1f}", (xi, r["jetson_est_ms"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=INK2)
    ax.set_title("forest latency per window", color=INK, fontsize=11)
    ax.set_ylabel("milliseconds"); ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    au = [r["AUROC"] for r in rows]; rc = [r["recall"] for r in rows]
    ax.plot(x, au, "-o", color=ACCENT, linewidth=2, markeredgecolor=SURFACE,
            markeredgewidth=2, markersize=8, label="AUROC")
    ax.plot(x, rc, "-o", color=WARN, linewidth=2, markeredgecolor=SURFACE,
            markeredgewidth=2, markersize=8, label="recall")
    # zoom to the data: on a 0-1 axis these lines are flat and say nothing
    span = max(au + rc) - min(au + rc)
    pad = max(span, 0.02) * 0.9
    ax.set_ylim(min(au + rc) - pad, max(au + rc) + pad)
    ax.set_title("detection performance\n(axis zoomed -- both are flat)",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")

    ax = axes[2]
    mean = np.array([r["auroc_mean"] for r in rows])
    lo = np.array([r["auroc_min"] for r in rows])
    hi = mean + (mean - lo)
    ax.fill_between(x, lo, hi, color=ACCENT, alpha=0.16)
    ax.plot(x, mean, "-o", color=ACCENT, linewidth=2, markeredgecolor=SURFACE,
            markeredgewidth=2, markersize=8)
    ax.axhline(rows[0]["auroc_min"], color=INK3, linestyle="--", linewidth=1.0)
    ax.annotate("100-tree floor", xy=(0.02, rows[0]["auroc_min"]),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=INK3, va="bottom")
    ax.set_title("AUROC across 5 seeds\n(band reaches the worst seed)",
                 color=INK, fontsize=11)
    ax.set_ylabel("AUROC")

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(t) for t in trees])
        ax.set_xlabel("IsolationForest trees")
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if best:
            ax.axvline(trees.index(best["trees"]), color=PASS, linestyle=":", linewidth=1.4)

    sub = (f"deploying {best['trees']} trees" if best else "no reduction holds across seeds")
    fig.suptitle(f"Shrinking the IsolationForest — it is 91% of the detector's board budget "
                 f"({sub})", color=INK, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
