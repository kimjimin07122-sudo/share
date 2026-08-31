"""
Evaluation for the GRU forecasting baseline (TiTAD adoption plan, step 2).
Implements TiTAD's scoring recipe (Eqs. 15-17): per-channel MAE, median/IQR
normalization using training-set residual stats, then per-timestep max as
the detection score.

Two parts:
  1) Channel-ranking check on the known mag_x fault burst
     (x70_20180611_x76_070620.csv, t~1053s) vs. control windows -- does this
     prediction-based approach do any better than the VAE at surfacing Ch11
     over baro_altitude/baro_temperature?
  2) Full detection evaluation across all 19 test sorties (AUROC/Accuracy/
     Precision/Recall/F1), using training-data blocks as the normal-proxy
     baseline -- same convention as eval.py's VAE evaluation, for a fair
     side-by-side comparison.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from gru_data_loader import GRUPredictionDataset, load_scaled_csv
from networks.gru_predictor import GRUPredictor
from eval import compute_binary_metrics

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_gru")


def load_model_and_stats(device):
    model = GRUPredictor(
        num_channels=Config.NUM_CHANNELS,
        hidden_size=Config.GRU_HIDDEN_SIZE,
        num_layers=Config.GRU_NUM_LAYERS,
    ).to(device)
    model.load_state_dict(torch.load(Config.GRU_MODEL_SAVE_PATH, map_location=device))
    model.eval()

    stats = np.load(Config.GRU_RESIDUAL_STATS_PATH)
    return model, stats["median"], stats["iqr"]


def predict_residuals(model, scaled, device, window=None, stride=1):
    window = window or Config.GRU_WINDOW
    xs, ys = [], []
    for i in range(0, len(scaled) - window, stride):
        xs.append(scaled[i:i + window])
        ys.append(scaled[i + window])
    if not xs:
        return None, None

    X = torch.tensor(np.asarray(xs, dtype=np.float32))
    y = np.asarray(ys, dtype=np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            preds.append(model(X[i:i + 1024].to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    residuals = np.abs(preds - y)  # (N, 16), raw MAE per channel per timestep
    time_sec = (np.arange(len(residuals)) * stride + window) / Config.DATA_HZ
    return residuals, time_sec


def channel_rank(mean_channel_score):
    ranking = np.argsort(-mean_channel_score) + 1
    ch11_rank = int(np.where(ranking == 11)[0][0]) + 1
    return ranking, ch11_rank


def report_window(residuals, normalized, time_sec, lo, hi, label):
    idx = np.where((time_sec >= lo) & (time_sec <= hi))[0]
    raw_mean = residuals[idx].mean(axis=0)
    norm_mean = normalized[idx].mean(axis=0)

    raw_rank, raw_ch11 = channel_rank(raw_mean)
    norm_rank, norm_ch11 = channel_rank(norm_mean)

    print(f"[{label}] {lo}-{hi}s (n={len(idx)})")
    print(f"  raw MAE:              top5={raw_rank[:5].tolist()} ch11_rank={raw_ch11}")
    print(f"  median/IQR normalized: top5={norm_rank[:5].tolist()} ch11_rank={norm_ch11}")
    return norm_mean


def plot_burst_vs_control(burst_scores, control_scores_list, save_path):
    channels = np.arange(1, Config.NUM_CHANNELS + 1)
    control_mean = np.mean(control_scores_list, axis=0)

    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars1 = ax.bar(channels - width / 2, burst_scores, width, label="burst (mag_x fault, 1043-1063s)", color="#c43b3b")
    bars2 = ax.bar(channels + width / 2, control_mean, width, label="control windows (avg of 3, fault-free)", color="#4f81bd")
    ax.set_xticks(channels)
    ax.set_xlabel("channel number")
    ax.set_ylabel("median/IQR-normalized MAE")
    ax.set_title("GRU baseline: fault burst vs. control windows, per channel")
    ax.axvline(11, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"saved {save_path}")


def plot_sortie_scores(rows, threshold, save_path):
    names = [r["file"] for r in rows]
    scores = [r["score"] for r in rows]
    colors = ["#c43b3b" if r["predicted_label"] == "abnormal" else "#4f81bd" for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rows)), scores, color=colors, width=0.7)
    plt.axhline(threshold, color="black", linewidth=1.0, linestyle="--")
    plt.xticks(np.arange(len(rows)), names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("sortie score: max normalized prediction error")
    plt.title("GRU baseline: per-sortie detection scores")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"saved {save_path}")


def sortie_score(model, scaler, median, iqr, filepath, device):
    scaled, _ = load_scaled_csv(filepath, scaler=scaler)
    residuals, _ = predict_residuals(model, scaled, device, stride=Config.GRU_STRIDE)
    if residuals is None:
        return None
    normalized = (residuals - median) / iqr
    per_t_score = normalized.max(axis=1)
    return float(per_t_score.max())


def evaluate_channel_ranking(model, scaler, median, iqr, device):
    target_file = os.path.join(Config.TEST_DATA_PATH, "x70_20180611_x76_070620.csv")
    scaled, _ = load_scaled_csv(target_file, scaler=scaler)
    residuals, time_sec = predict_residuals(model, scaled, device, stride=1)
    normalized = (residuals - median) / iqr

    print(f"\nfile: {os.path.basename(target_file)}, length: {time_sec[-1]:.0f}s, {len(residuals)} prediction points\n")

    burst_norm = report_window(residuals, normalized, time_sec, 1043, 1063, "burst (mag_x fault)")
    print()
    control_norms = []
    for lo, hi in [(200, 230), (500, 530), (900, 930)]:
        norm_mean = report_window(residuals, normalized, time_sec, lo, hi, f"control {lo}-{hi}s")
        control_norms.append(norm_mean)
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_burst_vs_control(burst_norm, control_norms, os.path.join(RESULTS_DIR, "gru_burst_vs_control.png"))


def evaluate_detection(model, scaler, median, iqr, device):
    print("\n=== Full-sortie detection evaluation (19 test files) ===")

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))

    # Normal proxy: split scaled training data into len(test_files) contiguous blocks,
    # same convention as eval.py's VAE evaluation (no separate normal test sortie exists).
    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    scaled_train, _ = load_scaled_csv(train_csv, scaler=scaler)
    n_blocks = len(test_files)
    block_bounds = np.linspace(0, len(scaled_train), n_blocks + 1, dtype=int)

    normal_scores = []
    rows = []
    for i in range(n_blocks):
        block = scaled_train[block_bounds[i]:block_bounds[i + 1]]
        residuals, _ = predict_residuals(model, block, device, stride=Config.GRU_STRIDE)
        if residuals is None:
            continue
        normalized = (residuals - median) / iqr
        score = float(normalized.max(axis=1).max())
        normal_scores.append(score)
        rows.append({"file": f"normal_block_{i+1:02d}", "true_label": "normal", "score": score})

    threshold = max(normal_scores) if normal_scores else 0.0

    y_true = [0] * len(normal_scores)
    scores = list(normal_scores)

    for f in test_files:
        score = sortie_score(model, scaler, median, iqr, f, device)
        predicted = "abnormal" if score > threshold else "normal"
        rows.append({"file": os.path.basename(f), "true_label": "abnormal", "predicted_label": predicted, "score": score})
        y_true.append(1)
        scores.append(score)
        print(f"[{os.path.basename(f)}] score={score:.4f} pred={predicted}")

    for r in rows:
        r.setdefault("predicted_label", "normal")

    metrics = compute_binary_metrics(y_true, scores, threshold)
    print(f"\nThreshold: {threshold:.4f}")
    for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score"]:
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Confusion matrix: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sortie_rows = [r for r in rows if r["true_label"] == "abnormal"]
    plot_sortie_scores(sortie_rows, threshold, os.path.join(RESULTS_DIR, "gru_sortie_scores.png"))

    import csv
    with open(os.path.join(RESULTS_DIR, "gru_sortie_scores.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "true_label", "predicted_label", "score"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "gru_metrics.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score", "TP", "TN", "FP", "FN"]:
            writer.writerow([name, metrics[name]])
    print(f"saved results to {RESULTS_DIR}/")


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_csv = os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv")
    train_dataset = GRUPredictionDataset(train_csv)
    scaler = train_dataset.scaler

    model, median, iqr = load_model_and_stats(device)

    evaluate_channel_ranking(model, scaler, median, iqr, device)
    evaluate_detection(model, scaler, median, iqr, device)


if __name__ == "__main__":
    evaluate_all()
