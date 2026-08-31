"""
Is our detection performance real, or an extreme-value sampling artifact?
(2026-08-20, verification of the v4 pipeline's AUROC 0.99-1.00.)

Both the paper's protocol and ours score a flight by the MAX over its
windows. But the two sides of the comparison get wildly different numbers of
draws: the held-out normal stretch is only ~19.6s, split into 19 blocks, so
each normal score is a max over ~2.6 windows -- while each test sortie is
4-131s, so its score is a max over ~4,126 windows on average. That is
~1,568x more draws. The max of many samples exceeds the max of few even when
both are drawn from the SAME distribution, so some of the separation between
"normal" and "abnormal" is guaranteed regardless of any real anomaly.

(The v4 pipeline made this worse, not better: a 1.0s window on ~1.03s blocks
leaves 2-3 windows per block, where v3's 32-sample window left ~251. The
underlying cause is the same data limitation as everywhere else -- we only
have 19.6s of held-out normal, see PROJECT_SUMMARY sec 6.)

Two tests here, and the SECOND is the one that actually answers the question:

  1. Random matched-size subsampling. Score each test sortie using only k
     windows (k = what a normal block gets) instead of all of them. This
     confirms the max-over-many advantage is large, BUT it is not a clean
     measure of it: a random k-window snippet of an abnormal flight is
     almost always drawn from normal flight, since the fault occupies well
     under 1s of a 45-131s sortie. So a low matched recall is partly the
     sampling bias and partly just "the fault is rare inside the file" --
     the two are confounded and this test cannot separate them. Reported
     for completeness, not as the bias estimate.

  2. Within-file fault localization (the real test). On the one sortie whose
     fault time we have verified (Config.BURST_FILE, 42.19-42.99s), slide a
     burst-length window across the whole flight and rank every candidate
     position by its score. Every position gets the same number of draws
     from the same flight, so sample size and session identity are both
     controlled out. If the true fault window ranks at or near the top of
     ~50 candidates, the model is genuinely locating the fault; if it ranks
     mid-pack, the file-level score was riding on something else.
"""
import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from networks.lstm_ae import LSTMAutoencoder
from networks.lstm_predictor import LSTMPredictor
from eval import compute_binary_metrics
from eval_lstm_ae import compute_window_residuals, fused_score_batch
from train_eval_lstm_predictor import predict_full_sequence, fused_score_batch_pred
from gru_data_loader import read_raw_csv_with_timestamp
from preprocessing import prepare_flight, prepare_flight_file, window_length, stride_length

HOLDOUT_FRAC = 0.15
N_DRAWS = 2000
RNG = np.random.default_rng(Config.RANDOM_SEED)
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"
CAT_BLUE, CAT_ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load_ae(device):
    m = LSTMAutoencoder(Config.NUM_CHANNELS * 4, Config.LSTM_AE_HIDDEN_SIZE,
                        Config.LSTM_AE_NUM_LAYERS, Config.LSTM_AE_BOTTLENECK_DIM).to(device)
    m.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                 map_location=device))
    m.eval()
    st = np.load(os.path.join(Config.BASE_DIR, "lstm_ae_stats_heldout.npz"))
    return (m, joblib.load(os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib")),
            joblib.load(os.path.join(Config.BASE_DIR, "iforest_model_heldout.joblib")),
            st["channel_mean"], st["channel_std"], float(st["if_mean"]), float(st["if_std"]))


def load_pred(device):
    m = LSTMPredictor(input_dim=Config.NUM_CHANNELS * 4, hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
                      num_layers=Config.LSTM_AE_NUM_LAYERS).to(device)
    m.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_predictor_model.pth"),
                                 map_location=device))
    m.eval()
    st = np.load(os.path.join(Config.BASE_DIR, "lstm_predictor_stats.npz"))
    return (m, joblib.load(os.path.join(Config.BASE_DIR, "lstm_predictor_feature_scaler.joblib")),
            joblib.load(os.path.join(Config.BASE_DIR, "iforest_model_predictor.joblib")),
            st["channel_mean"], st["channel_std"], float(st["if_mean"]), float(st["if_std"]))


def collect_scores(kind, device):
    """-> (list of per-normal-block window-score arrays, list of per-test-file arrays)"""
    W, S = window_length(), stride_length(stride_sec=Config.EVAL_STRIDE_SEC)
    if kind == "lstm_ae":
        model, fscaler, iforest, cm, cs, im, isd = load_ae(device)

        def score(feats):
            r, w = compute_window_residuals(model, feats, device, window=W, stride=S)
            if r is None:
                return np.array([])
            return fused_score_batch(r, w, iforest, cm, cs, im, isd)
    else:
        model, fscaler, iforest, cm, cs, im, isd = load_pred(device)

        def score(feats):
            e, c = predict_full_sequence(model, feats, device, W, stride=S)
            if e is None or len(e) == 0:
                return np.array([])
            return fused_score_batch_pred(e, c, iforest, cm, cs, im, isd)

    raw, t = read_raw_csv_with_timestamp(os.path.join(Config.TRAIN_DATA_PATH, "x70_all.csv"))
    sp = int(len(raw) * (1 - HOLDOUT_FRAC))
    fh_raw, _, _ = prepare_flight(raw[sp:], t[sp:] - t[sp])
    fh = fscaler.transform(fh_raw)

    test_files = sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv")))
    bounds = np.linspace(0, len(fh), len(test_files) + 1, dtype=int)
    normal = [score(fh[bounds[i]:bounds[i + 1]]) for i in range(len(test_files))]
    normal = [s for s in normal if len(s)]

    abnormal, names = [], []
    for f in test_files:
        fr, _, _ = prepare_flight_file(f)
        s = score(fscaler.transform(fr))
        if len(s):
            abnormal.append(s)
            names.append(os.path.basename(f))
    return normal, abnormal, names


def analyse(kind, device):
    normal, abnormal, names = collect_scores(kind, device)
    k = int(round(np.mean([len(s) for s in normal])))
    n_ab = np.mean([len(s) for s in abnormal])
    print(f"\n=== {kind} ===")
    print(f"windows per normal block : mean {k}   per test file: mean {n_ab:.0f}  ({n_ab/max(k,1):.0f}x more draws)")

    normal_max = np.array([s.max() for s in normal])
    threshold = normal_max.max()

    # --- protocol as reported: max over ALL windows of a file ---
    full_max = np.array([s.max() for s in abnormal])
    y = [0] * len(normal_max) + [1] * len(full_max)
    m_full = compute_binary_metrics(y, list(normal_max) + list(full_max), threshold)

    # --- same, but each test file limited to k windows (matched to normal) ---
    matched_recall, matched_medians = [], []
    for s in abnormal:
        draws = np.array([RNG.choice(s, size=min(k, len(s)), replace=False).max() for _ in range(N_DRAWS)])
        matched_recall.append(float((draws > threshold).mean()))
        matched_medians.append(float(np.median(draws)))
    matched_recall_mean = float(np.mean(matched_recall))
    y_m = [0] * len(normal_max) + [1] * len(matched_medians)
    m_matched = compute_binary_metrics(y_m, list(normal_max) + matched_medians, threshold)

    print(f"  [test 1] as-reported (max over whole file) : AUROC {m_full['AUROC']:.4f}  Recall {m_full['Recall']:.4f}")
    print(f"  [test 1] random {k}-window subsample        : Recall {matched_recall_mean:.4f}  "
          f"(confounded: a random snippet usually contains no fault at all)")

    loc = localize_fault(kind, device)
    print(f"  [test 2] within-file fault localization on {Config.BURST_FILE}:")
    print(f"           true fault window ranks {loc['rank']} of {loc['n_positions']} "
          f"burst-length positions  (percentile {loc['percentile']:.1f})")
    print(f"           fault score {loc['fault_score']:.4f} vs file median {loc['median_score']:.4f} "
          f"and best non-fault {loc['best_other']:.4f}")
    return {"kind": kind, "k": k, "full": m_full, "matched_recall": matched_recall_mean, "loc": loc}


def localize_fault(kind, device):
    """Slide a burst-length window over the one sortie with verified fault
    timing and rank every position. Same flight, same window count for every
    candidate -- so neither sample size nor session offset can explain a win."""
    W, S = window_length(), stride_length(stride_sec=Config.EVAL_STRIDE_SEC)
    if kind == "lstm_ae":
        model, fscaler, iforest, cm, cs, im, isd = load_ae(device)

        def score(feats):
            r, w = compute_window_residuals(model, feats, device, window=W, stride=S)
            return np.array([]) if r is None else fused_score_batch(r, w, iforest, cm, cs, im, isd)
    else:
        model, fscaler, iforest, cm, cs, im, isd = load_pred(device)

        def score(feats):
            e, c = predict_full_sequence(model, feats, device, W, stride=S)
            return np.array([]) if e is None or len(e) == 0 else fused_score_batch_pred(e, c, iforest, cm, cs, im, isd)

    fr, time_axis, _ = prepare_flight_file(os.path.join(Config.TEST_DATA_PATH, Config.BURST_FILE))
    s = score(fscaler.transform(fr))
    end_t = time_axis[np.minimum(np.arange(len(s)) * S + W - 1, len(time_axis) - 1)]

    lo, hi = Config.BURST_REAL_SEC
    span = hi - lo
    # tile the flight into non-overlapping burst-length positions
    starts = np.arange(end_t.min(), end_t.max() - span, span)
    pos_scores, pos_is_fault = [], []
    for st in starts:
        m = (end_t >= st) & (end_t < st + span)
        if not m.any():
            continue
        pos_scores.append(float(s[m].max()))
        pos_is_fault.append(not (st + span < lo or st > hi))
    pos_scores = np.array(pos_scores)
    pos_is_fault = np.array(pos_is_fault)

    fault_score = float(pos_scores[pos_is_fault].max())
    order = np.argsort(-pos_scores)
    rank = int(np.where(pos_scores[order] == fault_score)[0][0]) + 1
    return {
        "rank": rank, "n_positions": len(pos_scores),
        "percentile": 100.0 * (1 - (rank - 1) / max(len(pos_scores) - 1, 1)),
        "fault_score": fault_score,
        "median_score": float(np.median(pos_scores)),
        "best_other": float(pos_scores[~pos_is_fault].max()),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    results = [analyse(k, device) for k in ("lstm_ae", "lstm_predictor")]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    w = 0.36
    full = [r["full"]["Recall"] for r in results]
    matched = [r["loc"]["percentile"] / 100.0 for r in results]
    ax.bar(x - w / 2, full, w, color=CAT_BLUE, label="file-level recall (max over whole flight)")
    ax.bar(x + w / 2, matched, w, color=CAT_ORANGE,
           label="within-file: percentile of the TRUE fault window")
    for xi, v in zip(x - w / 2, full):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
    for xi, v in zip(x + w / 2, matched):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([r["kind"] for r in results])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), frameon=False, fontsize=9)
    ax.set_title("File-level detection vs. within-file fault localization", color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sample_size_bias_check.png"), dpi=180)
    plt.close(fig)
    print(f"\nsaved figure to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
