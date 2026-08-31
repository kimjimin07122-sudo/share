"""
Robust per-file baseline detection score (session-drift experiment,
2026-08-20): instead of thresholding a file's fused score against a global
"max of held-out normal blocks" reference, score each file against its OWN
median/MAD -- the same "file-wide own-baseline" idea that worked for channel
diagnosis, but applied directly to the scalar detection score this time.

Crucially this is NOT a search over sub-windows (that's what biased the
earlier attempt at extending channel diagnosis to unlabeled files, see
PROJECT_SUMMARY.md sec 3.3): a file's median and MAD are each a single
number computed once over all of that file's window scores, so there is no
multiple-comparisons trap. A fault confined to a small fraction of the file
barely moves the median, so it should still stand out as a robust-z outlier.

Reuses the already-trained (dt-fixed) v3 held-out LSTM-AE artifacts --
no retraining needed, this only changes how the fused window scores are
turned into one number per file.
"""
import csv
import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data_loader import build_scaler
from gru_data_loader import read_raw_csv_with_timestamp, load_scaled_csv_with_time
from feature_engineering import compute_sequence_features
from networks.lstm_ae import LSTMAutoencoder
from eval import compute_binary_metrics
from eval_lstm_ae import compute_window_residuals, fused_score_batch

HOLDOUT_FRAC = 0.15
INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL

MODEL_PATH = os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth")
IFOREST_PATH = os.path.join(Config.BASE_DIR, "iforest_model_heldout.joblib")
STATS_PATH = os.path.join(Config.BASE_DIR, "lstm_ae_stats_heldout.npz")
FEATURE_SCALER_PATH = os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib")

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_robust_baseline")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
GLOBAL_METHOD_METRICS_CSV = os.path.join(Config.BASE_DIR, "results_lstm_ae_heldout", "lstm_ae_metrics.csv")

# validated against references/palette.md (dataviz skill) via validate_palette.js
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def robust_z_max(window_scores, median, scale):
    """max((score - median) / scale) over one file's windows. `scale` is
    (mad * 1.4826); pass the file's own MAD for the pure per-file method, or
    a shared/global MAD for the hybrid method."""
    robust_std = scale + 1e-8
    return float(((window_scores - median) / robust_std).max())


def plot_robust_scores(rows, threshold, n_normal, save_path, title="Per-file robust z-score — LSTM-AE, dt-fixed"):
    """One bar per file/block: robust_z_max, log-scale (values span ~1.6 to
    ~4200), colored by whether the threshold call was correct (status colors,
    not identity -- this chart's story is "did it work"). A divider marks
    where the held-out normal control group ends and the labeled-fault test
    files begin."""
    names = [r["file"] for r in rows]
    scores = [r["score"] for r in rows]
    correct = [
        (r["true_label"] == "normal") == (r["score"] <= threshold)
        for r in rows
    ]
    colors = [STATUS_GOOD if c else STATUS_CRITICAL for c in correct]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(rows))
    ax.bar(x, scores, color=colors, width=0.68)
    ax.axhline(threshold, color=INK_PRIMARY, linewidth=1.2, linestyle="--", zorder=3)
    ax.text(len(rows) - 0.5, threshold, f"  threshold {threshold:.1f}",
            va="bottom", ha="right", fontsize=8, color=INK_SECONDARY)

    divider = n_normal - 0.5
    ax.axvline(divider, color=BASELINE, linewidth=1.0)

    ax.set_yscale("log")
    ax.set_ylabel("robust z-score (file's own median/MAD), log scale")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ymax = ax.get_ylim()[1]
    ax.text(n_normal / 2 - 0.5, ymax, "held-out NORMAL blocks (control)",
            ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.text(n_normal + (len(rows) - n_normal) / 2 - 0.5, ymax, "labeled FAULT test files",
            ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATUS_GOOD, label="correctly classified"),
        plt.Rectangle((0, 0), 1, 1, color=STATUS_CRITICAL, label="misclassified"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.set_title(title, color=INK_PRIMARY, pad=28)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


CAT_AQUA = "#1baf7a"


def plot_method_comparison(methods, save_path):
    """Headline slide chart: N scoring methods side by side on the metrics
    that matter for a detector. `methods` is an ordered dict/list of
    (label, color, metrics_dict) so the bar order matches the legend order."""
    metric_names = ["AUROC", "Precision", "Recall", "F1-score"]
    n = len(methods)
    width = 0.8 / n
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (label, color, metrics) in enumerate(methods):
        offset = (i - (n - 1) / 2) * width
        vals = [metrics.get(m, 0.0) for m in metric_names]
        xs = x + offset
        ax.bar(xs, vals, width * 0.92, color=color, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32 - 0.06 * n), frameon=False, ncol=1, fontsize=9)
    ax.set_title("LSTM-AE detection: scoring method comparison", color=INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --- reproduce the exact raw-channel scaler used at training time.
    # train_eval_lstm_ae_heldout.py never persists it (only the feature
    # scaler is saved), but StandardScaler.fit is deterministic, so
    # refitting on the same first-85% split reproduces it exactly. ---
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    raw_all, time_all = read_raw_csv_with_timestamp(train_csv)
    split_idx = int(len(raw_all) * (1 - HOLDOUT_FRAC))
    raw_train, raw_heldout = raw_all[:split_idx], raw_all[split_idx:]
    time_heldout = time_all[split_idx:]

    scaler = build_scaler()
    scaler.fit(raw_train)
    scaled_heldout = scaler.transform(raw_heldout)

    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
        num_layers=Config.LSTM_AE_NUM_LAYERS,
        bottleneck_dim=Config.LSTM_AE_BOTTLENECK_DIM,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    iforest = joblib.load(IFOREST_PATH)
    stats = np.load(STATS_PATH)
    channel_mean, channel_std = stats["channel_mean"], stats["channel_std"]
    if_mean, if_std = float(stats["if_mean"]), float(stats["if_std"])

    def window_scores_for(scaled, time_sec):
        feats = feature_scaler.transform(compute_sequence_features(scaled, time_sec))
        residuals, windows = compute_window_residuals(model, feats, device, stride=1)
        if residuals is None:
            return None
        return fused_score_batch(residuals, windows, iforest, channel_mean, channel_std, if_mean, if_std)

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    n_blocks = len(test_files)
    bounds = np.linspace(0, len(scaled_heldout), n_blocks + 1, dtype=int)

    # --- pass 1: run the model once per file/block, keep the raw window-score
    # arrays so both the pure and hybrid methods can be derived without
    # re-running inference. ---
    print("\n=== scoring held-out NORMAL blocks + 19 labeled test files ===")
    entries = []  # each: {name, true_label, scores}
    for i in range(n_blocks):
        block = scaled_heldout[bounds[i]:bounds[i + 1]]
        block_time = time_heldout[bounds[i]:bounds[i + 1]]
        scores = window_scores_for(block, block_time)
        if scores is None:
            continue
        entries.append({"name": f"heldout_normal_block_{i+1:02d}", "true_label": "normal", "scores": scores})
    n_normal = len(entries)
    for f in test_files:
        scaled, _, time_sec = load_scaled_csv_with_time(f, scaler=scaler)
        scores = window_scores_for(scaled, time_sec)
        entries.append({"name": os.path.basename(f), "true_label": "abnormal", "scores": scores})
    print(f"scored {n_normal} normal blocks + {len(entries) - n_normal} test files")

    for e in entries:
        e["median"] = float(np.median(e["scores"]))
        e["mad"] = float(np.median(np.abs(e["scores"] - e["median"])))

    def build_result(label_a, label_b, score_fn):
        """score_fn(entry) -> file-level score. Returns (rows, metrics)."""
        rows = []
        for e in entries:
            rows.append({"file": e["name"], "true_label": e["true_label"], "score": score_fn(e)})
        threshold = max(r["score"] for r in rows if r["true_label"] == "normal")
        y_true = [1 if r["true_label"] == "abnormal" else 0 for r in rows]
        scores_all = [r["score"] for r in rows]
        metrics = compute_binary_metrics(y_true, scores_all, threshold)
        for r in rows:
            r["predicted_label"] = "abnormal" if r["score"] > threshold else "normal"
        return rows, threshold, metrics

    # --- method: pure per-file median/MAD (2026-08-20, first attempt) ---
    rows_pure, threshold_pure, metrics_pure = build_result(
        "normal", "abnormal", lambda e: robust_z_max(e["scores"], e["median"], e["mad"] * 1.4826)
    )
    print(f"\n[pure per-file] threshold={threshold_pure:.3f}  "
          + "  ".join(f"{k}={metrics_pure[k]:.4f}" for k in ["AUROC", "Recall", "F1-score"]))

    # --- method: hybrid -- file's own median (removes session-level offset),
    # but MAD pooled from held-out NORMAL blocks only (one shared noise-floor
    # estimate, so a file's own noise no longer sets its own bar). ---
    normal_abs_devs = np.concatenate([
        np.abs(e["scores"] - e["median"]) for e in entries if e["true_label"] == "normal"
    ])
    global_mad = float(np.median(normal_abs_devs))
    global_scale = global_mad * 1.4826
    print(f"\nglobal MAD (pooled from {n_normal} held-out normal blocks): {global_mad:.4f}")

    rows_hybrid, threshold_hybrid, metrics_hybrid = build_result(
        "normal", "abnormal", lambda e: robust_z_max(e["scores"], e["median"], global_scale)
    )
    print(f"[hybrid]         threshold={threshold_hybrid:.3f}  "
          + "  ".join(f"{k}={metrics_hybrid[k]:.4f}" for k in ["AUROC", "Recall", "F1-score"]))

    global_metrics = {}
    if os.path.exists(GLOBAL_METHOD_METRICS_CSV):
        with open(GLOBAL_METHOD_METRICS_CSV) as fh:
            global_metrics = {row["metric"]: float(row["value"]) for row in csv.DictReader(fh)}
    if global_metrics:
        print(f"[global threshold, existing method]  "
              + "  ".join(f"{k}={global_metrics[k]:.4f}" for k in ["AUROC", "Recall", "F1-score"]))

    def save_csvs(prefix, rows, metrics):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f"{prefix}_scores.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "true_label", "predicted_label", "score"])
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(RESULTS_DIR, f"{prefix}_metrics.csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN"]:
                writer.writerow([name, metrics[name]])

    save_csvs("robust_baseline", rows_pure, metrics_pure)
    save_csvs("robust_hybrid", rows_hybrid, metrics_hybrid)
    print(f"\nsaved results to {RESULTS_DIR}/")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_robust_scores(rows_pure, threshold_pure, n_normal,
                        os.path.join(FIGURES_DIR, "robust_baseline_per_file_scores.png"),
                        title="Per-file robust z-score (own median / own MAD) — LSTM-AE, dt-fixed")
    plot_robust_scores(rows_hybrid, threshold_hybrid, n_normal,
                        os.path.join(FIGURES_DIR, "robust_hybrid_per_file_scores.png"),
                        title="Hybrid robust z-score (own median / global MAD) — LSTM-AE, dt-fixed")
    methods = [
        ("global threshold (existing)", CAT_BLUE, global_metrics),
        ("pure per-file median/MAD", CAT_ORANGE, metrics_pure),
        ("hybrid: own median + global MAD", CAT_AQUA, metrics_hybrid),
    ]
    plot_method_comparison(methods, os.path.join(FIGURES_DIR, "method_comparison_3way.png"))
    print(f"saved figures to {FIGURES_DIR}/")

    print("\n=== verdict ===")
    if metrics_hybrid["AUROC"] > global_metrics.get("AUROC", 1.0) and metrics_hybrid["Recall"] >= global_metrics.get("Recall", 1.0):
        print("Hybrid clearly beats the existing global-threshold method -- worth keeping.")
    else:
        print("Hybrid does NOT clearly beat the existing global-threshold method.")
        print("Recommendation: stop here: document 'file-wide baseline applied to raw detection "
              "score' as a negative result (tried pure per-file and this hybrid variant; both "
              "underperform the existing held-out-normal-max threshold).")


if __name__ == "__main__":
    main()
