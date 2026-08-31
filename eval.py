import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from data_loader import DroneSensorDataset
from networks.vae import Conv1DVAE


PAPER_VAE_BO_METRICS = {
    "AUROC": 0.9132,
    "Accuracy": 0.9231,
    "Precision": 0.8696,
    "Recall": 1.0000,
    "F1-score": 0.9302,
}


def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle, drop_last=False)


def collect_errors(model, loader, device):
    all_abs_diff = []
    with torch.no_grad():
        for data in loader:
            if isinstance(data, (tuple, list)):
                data = data[0]
            data = data.to(device)
            recon, _, _, _ = model(data)
            all_abs_diff.append(torch.abs(data - recon).cpu().numpy())

    abs_diff = np.concatenate(all_abs_diff, axis=0)
    clipped = np.clip(abs_diff, 0.0, 1.0)
    sample_errors = clipped.mean(axis=(1, 2))
    channel_errors = clipped.mean(axis=(0, 2))
    channel_probs = channel_errors / channel_errors.sum() if channel_errors.sum() > 0 else channel_errors
    return abs_diff, sample_errors, channel_errors, channel_probs


def split_dataset_blocks(dataset, n_blocks):
    indices = np.array_split(np.arange(len(dataset)), max(1, n_blocks))
    blocks = []
    for idx in indices:
        if len(idx) == 0:
            continue
        samples = torch.stack([dataset[int(i)] for i in idx])
        blocks.append(TensorDataset(samples))
    return blocks


def compute_auroc(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        wins += np.sum(pos > negatives)
        wins += 0.5 * np.sum(pos == negatives)
    return wins / total


def compute_binary_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores > threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "AUROC": compute_auroc(y_true, scores),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }



def select_fig7_window(abs_diff, sample_errors, threshold, filename, baseline_channel_errors=None):
    mode = Config.FIG7_MODE.lower()
    total = len(sample_errors)

    if mode == "full":
        indices = np.arange(total)
        label = "full sortie"
    elif mode == "threshold":
        indices = np.where(sample_errors > threshold)[0]
        if len(indices) == 0:
            peak = int(np.argmax(sample_errors))
            indices = np.array([peak])
            label = "threshold fallback: peak sample"
        else:
            label = "threshold-exceeding windows"
    elif mode == "peak":
        peak = int(np.argmax(sample_errors))
        window = int(Config.FIG7_PEAK_WINDOW)
        start = max(0, peak - window)
        end = min(total, peak + window + 1)
        indices = np.arange(start, end)
        label = f"peak +/- {window} windows"
    elif mode == "manual":
        if Config.FIG7_MANUAL_FILE and Config.FIG7_MANUAL_FILE != filename:
            indices = np.arange(total)
            label = "manual inactive for this file; full sortie"
        else:
            if Config.FIG7_MANUAL_START_SEC is None or Config.FIG7_MANUAL_END_SEC is None:
                raise ValueError("Manual Fig. 7 mode requires FIG7_MANUAL_START_SEC and FIG7_MANUAL_END_SEC.")
            step_sec = Config.STRIDE / Config.DATA_HZ
            start = max(0, int(Config.FIG7_MANUAL_START_SEC / step_sec))
            end = min(total, int(Config.FIG7_MANUAL_END_SEC / step_sec) + 1)
            indices = np.arange(start, end)
            label = f"manual {Config.FIG7_MANUAL_START_SEC}-{Config.FIG7_MANUAL_END_SEC} sec"
    else:
        raise ValueError(f"Unsupported FIG7_MODE: {Config.FIG7_MODE}")

    if len(indices) == 0:
        raise ValueError(f"Fig. 7 window for {filename} is empty.")

    window_abs_diff = abs_diff[indices]
    channel_errors = window_abs_diff.mean(axis=(0, 2))
    if baseline_channel_errors is not None:
        method = Config.CHANNEL_ATTRIBUTION_METHOD.lower()
        if method == "excess":
            channel_errors = np.maximum(channel_errors - baseline_channel_errors, 0.0)
        elif method == "ratio":
            channel_errors = channel_errors / (baseline_channel_errors + 1e-8)
        elif method != "raw":
            raise ValueError(f"Unsupported CHANNEL_ATTRIBUTION_METHOD: {Config.CHANNEL_ATTRIBUTION_METHOD}")
    return channel_errors, indices, label

def plot_reconstruction(time_sec, sample_errors, threshold, save_path, title):
    plt.figure(figsize=(10, 4))
    plt.plot(time_sec, sample_errors, color="black", linewidth=0.7)
    plt.axhline(threshold, color="#c43b3b", linewidth=1.0, linestyle="--")
    plt.xlabel("time (sec)")
    plt.ylabel("reconstruction error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_fault_identification(channel_scores, save_path, title):
    channels = np.arange(1, Config.NUM_CHANNELS + 1)
    colors = ["#4f81bd"] * Config.NUM_CHANNELS
    colors[int(np.argmax(channel_scores))] = "#c43b3b"

    plt.figure(figsize=(10, 4.5))
    plt.bar(channels, channel_scores, width=0.65, color=colors)
    plt.xticks(channels)
    plt.xlabel("channel number")
    plt.ylabel("average reconstruction error")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_sortie_scores(rows, threshold, save_path):
    names = [row["file"] for row in rows]
    scores = [float(row["score"]) for row in rows]
    colors = ["#c43b3b" if row["predicted_label"] == "abnormal" else "#4f81bd" for row in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rows)), scores, color=colors, width=0.7)
    plt.axhline(threshold, color="black", linewidth=1.0, linestyle="--")
    plt.xticks(np.arange(len(rows)), names, rotation=75, ha="right", fontsize=7)
    plt.ylabel("sortie score: max mean reconstruction error")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_metric_table(metric_rows, save_path):
    names = [row["metric"] for row in metric_rows if row["metric"] in PAPER_VAE_BO_METRICS]
    current = [float(row["current_vae"]) for row in metric_rows if row["metric"] in PAPER_VAE_BO_METRICS]
    paper = [float(row["paper_vae_bo"]) for row in metric_rows if row["metric"] in PAPER_VAE_BO_METRICS]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    table_data = [[name, f"{cur:.4f}", f"{ref:.4f}"] for name, cur, ref in zip(names, current, paper)]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Current VAE", "Paper VAE B.O."],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("VAE evaluation metrics", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_representative_reconstruction_grid(case_items, threshold, save_path):
    if not case_items:
        return

    fig, axes = plt.subplots(len(case_items), 1, figsize=(10, 2.8 * len(case_items)), sharex=False)
    if len(case_items) == 1:
        axes = [axes]

    for ax, item in zip(axes, case_items):
        time_sec = np.arange(len(item["sample_errors"])) * (Config.STRIDE / Config.DATA_HZ)
        ax.plot(time_sec, item["sample_errors"], color="black", linewidth=0.7)
        ax.axhline(threshold, color="#c43b3b", linewidth=1.0, linestyle="--")
        ax.set_ylabel("reconstruction error")
        ax.set_title(item["title"], fontsize=10)

    axes[-1].set_xlabel("time (sec)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_paper_fig7(channel_errors, filename, save_path):
    channels = np.arange(1, Config.NUM_CHANNELS + 1)
    colors = ["#4f81bd"] * Config.NUM_CHANNELS
    colors[10] = "#c43b3b"

    plt.figure(figsize=(8.5, 4.2))
    plt.bar(channels, channel_errors, width=0.65, color=colors)
    plt.xticks(channels)
    plt.xlabel("Channel number")
    plt.ylabel("Average reconstruction error")
    plt.title(f"Results of fault identification (VAE): {filename}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def select_representative_cases(sortie_rows, test_artifacts):
    normal_like = [row for row in sortie_rows if row["predicted_label"] == "normal"]
    abnormal = [row for row in sortie_rows if row["predicted_label"] == "abnormal"]

    cases = []
    if normal_like:
        row = min(normal_like, key=lambda r: float(r["score"]))
        artifact = test_artifacts.get(row["file"])
        if artifact:
            cases.append({
                "title": f"Anomaly Case I / minor-noise-like: {row['file']}",
                "sample_errors": artifact["sample_errors"],
            })

    ch11_rows = [row for row in abnormal if int(row["channel_11_rank"]) == 1]
    if ch11_rows:
        row = max(ch11_rows, key=lambda r: float(r["score"]))
    elif abnormal:
        row = min(abnormal, key=lambda r: int(r["channel_11_rank"]))
    else:
        row = None
    if row is not None:
        artifact = test_artifacts.get(row["file"])
        if artifact:
            cases.append({
                "title": f"Anomaly Case II / channel-11-like: {row['file']}",
                "sample_errors": artifact["sample_errors"],
            })

    if abnormal:
        row = max(abnormal, key=lambda r: float(r["score"]))
        artifact = test_artifacts.get(row["file"])
        if artifact and all(item["title"].split(": ")[-1] != row["file"] for item in cases):
            cases.append({
                "title": f"Anomaly Case III / high-error-like: {row['file']}",
                "sample_errors": artifact["sample_errors"],
            })

    return cases[:3]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all():
    os.makedirs(Config.EVAL_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DroneSensorDataset(Config.TRAIN_DATA_PATH, is_train=True)

    model = Conv1DVAE().to(device)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    if not test_files:
        raise ValueError(f"No test CSV files found in {Config.TEST_DATA_PATH}")

    if Config.NORMAL_TEST_DATA_PATH:
        normal_dataset = DroneSensorDataset(Config.NORMAL_TEST_DATA_PATH, scaler=train_dataset.scaler, is_train=False)
        normal_blocks = split_dataset_blocks(normal_dataset, len(test_files))
        normal_source_note = Config.NORMAL_TEST_DATA_PATH
    else:
        normal_blocks = split_dataset_blocks(train_dataset, len(test_files))
        normal_source_note = "training data blocks used as normal proxy; set Config.NORMAL_TEST_DATA_PATH for paper-exact metrics"

    train_abs_diff, _, train_channel_errors, _ = collect_errors(model, make_loader(train_dataset), device)

    normal_score_rows = []
    normal_scores = []
    for i, block in enumerate(normal_blocks):
        _, sample_errors, _, _ = collect_errors(model, make_loader(block), device)
        score = float(np.max(sample_errors))
        normal_scores.append(score)
        normal_score_rows.append({"file": f"normal_block_{i + 1:02d}", "true_label": "normal", "score": score})

    threshold = max(normal_scores) if normal_scores else 0.0

    sortie_rows = []
    channel_rows = []
    test_artifacts = {}
    y_true = [0] * len(normal_scores)
    scores = list(normal_scores)

    for test_file in test_files:
        filename = os.path.basename(test_file)
        prefix = filename.replace(".csv", "")
        dataset = DroneSensorDataset(test_file, scaler=train_dataset.scaler, is_train=False)
        abs_diff, sample_errors, full_channel_errors, _ = collect_errors(model, make_loader(dataset), device)
        sortie_score = float(np.max(sample_errors))
        predicted_label = "abnormal" if sortie_score > threshold else "normal"
        channel_errors, fig7_indices, fig7_window_label = select_fig7_window(
            abs_diff, sample_errors, threshold, filename, train_channel_errors
        )
        channel_probs = channel_errors / channel_errors.sum() if channel_errors.sum() > 0 else channel_errors
        test_artifacts[filename] = {
            "sample_errors": sample_errors,
            "channel_errors": channel_errors,
            "full_channel_errors": full_channel_errors,
            "fig7_indices": fig7_indices,
            "fig7_window_label": fig7_window_label,
        }

        top_channel = int(np.argmax(channel_errors) + 1)
        ch11_rank = int(np.where(np.argsort(-channel_errors) == 10)[0][0] + 1)

        time_sec = np.arange(len(sample_errors)) * (Config.STRIDE / Config.DATA_HZ)
        plot_reconstruction(
            time_sec,
            sample_errors,
            threshold,
            os.path.join(Config.EVAL_OUTPUT_DIR, f"paper_vae_reconstruction_{prefix}.png"),
            f"VAE reconstruction error: {prefix}",
        )
        plot_fault_identification(
            channel_errors,
            os.path.join(Config.EVAL_OUTPUT_DIR, f"paper_vae_fault_identification_{prefix}.png"),
            f"VAE fault identification: {prefix}",
        )

        sortie_rows.append({
            "file": filename,
            "true_label": "abnormal",
            "predicted_label": predicted_label,
            "score": sortie_score,
            "threshold": threshold,
            "top_channel": top_channel,
            "top_channel_name": Config.CHANNEL_NAMES[top_channel - 1],
            "channel_11_rank": ch11_rank,
            "channel_11_score": float(channel_errors[10]),
            "fig7_mode": Config.FIG7_MODE,
            "fig7_window": fig7_window_label,
            "fig7_window_count": int(len(fig7_indices)),
        })

        for idx, (error, prob) in enumerate(zip(channel_errors, channel_probs), start=1):
            channel_rows.append({
                "file": filename,
                "channel": idx,
                "channel_name": Config.CHANNEL_NAMES[idx - 1],
                "avg_reconstruction_error": float(error),
                "normalized_score": float(prob),
                "rank": int(np.where(np.argsort(-channel_errors) == (idx - 1))[0][0] + 1),
                "fig7_mode": Config.FIG7_MODE,
                "fig7_window": fig7_window_label,
                "fig7_window_count": int(len(fig7_indices)),
            })

        y_true.append(1)
        scores.append(sortie_score)
        print(
            f"[{filename}] score={sortie_score:.6f}, pred={predicted_label}, "
            f"top_channel={top_channel} ({Config.CHANNEL_NAMES[top_channel - 1]}), ch11_rank={ch11_rank}"
        )

    metrics = compute_binary_metrics(y_true, scores, threshold)
    score_rows = [
        {**row, "predicted_label": "normal", "threshold": threshold, "top_channel": "", "top_channel_name": "", "channel_11_rank": "", "channel_11_score": "", "fig7_mode": "", "fig7_window": "", "fig7_window_count": ""}
        for row in normal_score_rows
    ] + sortie_rows

    write_csv(
        os.path.join(Config.EVAL_OUTPUT_DIR, "paper_vae_sortie_scores.csv"),
        score_rows,
        ["file", "true_label", "predicted_label", "score", "threshold", "top_channel", "top_channel_name", "channel_11_rank", "channel_11_score", "fig7_mode", "fig7_window", "fig7_window_count"],
    )
    write_csv(
        os.path.join(Config.EVAL_OUTPUT_DIR, "paper_vae_fault_identification.csv"),
        channel_rows,
        ["file", "channel", "channel_name", "avg_reconstruction_error", "normalized_score", "rank", "fig7_mode", "fig7_window", "fig7_window_count"],
    )

    metric_rows = []
    for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score"]:
        metric_rows.append({"metric": name, "current_vae": metrics[name], "paper_vae_bo": PAPER_VAE_BO_METRICS[name]})
    for name in ["TP", "TN", "FP", "FN"]:
        metric_rows.append({"metric": name, "current_vae": metrics[name], "paper_vae_bo": ""})

    write_csv(os.path.join(Config.EVAL_OUTPUT_DIR, "paper_vae_metrics.csv"), metric_rows, ["metric", "current_vae", "paper_vae_bo"])
    plot_sortie_scores(sortie_rows, threshold, os.path.join(Config.EVAL_OUTPUT_DIR, "paper_vae_sortie_scores.png"))
    plot_metric_table(metric_rows, os.path.join(Config.EVAL_OUTPUT_DIR, "paper_like_table8_vae_metrics.png"))

    representative_cases = select_representative_cases(sortie_rows, test_artifacts)
    plot_representative_reconstruction_grid(
        representative_cases,
        threshold,
        os.path.join(Config.EVAL_OUTPUT_DIR, "paper_like_fig4_6_vae_reconstruction_cases.png"),
    )

    ch11_candidates = [row for row in sortie_rows if int(row["channel_11_rank"]) == 1]
    if not ch11_candidates:
        ch11_candidates = sorted(sortie_rows, key=lambda row: int(row["channel_11_rank"]))[:1]
    if ch11_candidates:
        fig7_row = max(ch11_candidates, key=lambda row: float(row["score"]))
        fig7_artifact = test_artifacts[fig7_row["file"]]
        plot_paper_fig7(
            fig7_artifact["channel_errors"],
            fig7_row["file"],
            os.path.join(Config.EVAL_OUTPUT_DIR, "paper_like_fig7_fault_identification.png"),
        )

    print("\n--- Paper-style VAE evaluation ---")
    print(f"Scaling method: {Config.SCALING_METHOD}")
    print(f"Normal reference: {normal_source_note}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Fig. 7 mode: {Config.FIG7_MODE}")
    print(f"Channel attribution: {Config.CHANNEL_ATTRIBUTION_METHOD}")
    for name in ["AUROC", "Accuracy", "Precision", "Recall", "F1-score"]:
        print(f"{name}: {metrics[name]:.4f} (paper VAE B.O.: {PAPER_VAE_BO_METRICS[name]:.4f})")
    print(f"Confusion matrix: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"Results saved to: {Config.EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate_all()
