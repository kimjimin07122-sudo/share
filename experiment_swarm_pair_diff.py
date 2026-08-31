"""
Swarm-pair differential diagnosis (2026-08-20): use the fact that this is a
FORMATION-FLIGHT dataset, which every model in this project has so far
ignored.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE HERE
-----------------------------------------------
Every previous approach compared a flight against a model of "normal" built
from a DIFFERENT session, which is precisely what produces session drift --
the score ends up measuring how far apart two sessions are rather than how
faulty a drone is (PROJECT_SUMMARY secs 3.1, 3.9).

The filenames reveal simultaneous pairs: same date, same takeoff time,
different airframe.

    20180601 08:39:16   x71 + x76
    20180611 07:20:30   x66 + x76
    20180611 07:50:47   x66 + x76
    20180611 08:22:53   x66 + x76

These are two aircraft flying the same mission at the same moment in the same
airspace, which their barometric altitude confirms (correlation 0.855 between
partners). Comparing partners to each other cancels session identity by
construction: same weather, same location, same magnetic environment, same
clock. No stored normal model and no held-out normal file is needed.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Raw channel values do NOT track between partners -- only altitude does
(gyro/accel 0.06-0.46, mag 0.11-0.35), because the two aircraft hold
different formation slots and headings, so their instantaneous manoeuvres and
magnetic projections legitimately differ. Subtracting raw channels would be
meaningless.

What IS comparable is each drone's own error PROFILE. For drone D and channel
c we take the model's mean per-channel error and divide by that drone's mean
error across all channels. That within-drone normalization removes overall
scale differences (one drone simply being noisier, or differently
calibrated), leaving the SHAPE of the profile: which channels are unusually
bad *for this drone*. Then we difference the two profiles.

THE TESTABLE PREDICTION
-----------------------
The dataset paper (Ahn & Chung, ESWA 2024) singles out drone X76 -- its Fig. 8
is titled "Time history of the locational variables of drone X76" -- and
identifies channel 11 (mag_x) as the fault channel. x76 appears in all four
pairs. So if this works, mag_x should be the channel where x76 most exceeds
its partner, in pair after pair.

Honest limits, stated up front:
  * All 19 test sorties are labelled abnormal (paper Table 3), so the partner
    is not a clean control -- x66/x71 may carry their own faults. This blunts
    the contrast rather than invalidating it.
  * The one fault whose timing we verified (x76_070620) has no partner, so it
    cannot be used here.
  * Four pairs is a small sample: even a perfect 4/4 sign test is only
    p = 1/16 ~ 0.06. Treat a positive result as corroboration, not proof.
"""
import csv
import glob
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from networks.lstm_ae import LSTMAutoencoder
from eval_lstm_ae import compute_window_residuals, fused_score_batch
from preprocessing import prepare_flight_file, window_length, stride_length

FAULTY_AIRFRAME = "x76"          # per Ahn & Chung Fig. 8
EXPECTED_CHANNEL = 11            # mag_x, 1-indexed as the paper numbers them

RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_swarm_pair")
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"
STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"
CAT_BLUE = "#2a78d6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def find_pairs():
    pat = re.compile(r"x70_(\d{8})_(x\d+)_(\d{6})\.csv")
    groups = {}
    for f in sorted(glob.glob(os.path.join(Config.TEST_DATA_PATH, "*.csv"))):
        m = pat.match(os.path.basename(f))
        if m:
            groups.setdefault((m.group(1), m.group(3)), []).append((m.group(2), f))
    return {k: v for k, v in sorted(groups.items()) if len(v) == 2}


def load_model(device):
    model = LSTMAutoencoder(Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL,
                            Config.LSTM_AE_HIDDEN_SIZE, Config.LSTM_AE_NUM_LAYERS,
                            Config.LSTM_AE_BOTTLENECK_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.BASE_DIR, "best_lstm_ae_model_heldout.pth"),
                                     map_location=device))
    model.eval()
    stats = np.load(os.path.join(Config.BASE_DIR, "lstm_ae_stats_heldout.npz"))
    return (model,
            joblib.load(os.path.join(Config.BASE_DIR, "lstm_ae_feature_scaler_heldout.joblib")),
            joblib.load(os.path.join(Config.BASE_DIR, "iforest_model_heldout.joblib")),
            stats["channel_mean"], stats["channel_std"],
            float(stats["if_mean"]), float(stats["if_std"]))


def profile(path, bundle, device):
    """-> (file-level fused score, within-drone-normalized per-channel error profile)"""
    model, fscaler, iforest, cm, cs, im, isd = bundle
    W, S = window_length(), stride_length(stride_sec=Config.EVAL_STRIDE_SEC)
    feats_raw, _, _ = prepare_flight_file(path)
    feats = fscaler.transform(feats_raw)
    residuals, windows = compute_window_residuals(model, feats, device, window=W, stride=S)
    if residuals is None:
        return None, None
    fused = fused_score_batch(residuals, windows, iforest, cm, cs, im, isd)
    per_channel = residuals.mean(axis=(0, 1)).reshape(
        Config.NUM_CHANNELS, Config.LSTM_AE_FEATURES_PER_CHANNEL).mean(axis=1)
    # divide by this drone's own mean error, so only the SHAPE of the profile
    # survives -- a globally noisier drone does not win on every channel
    return float(fused.max()), per_channel / per_channel.mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    bundle = load_model(device)

    pairs = find_pairs()
    print(f"simultaneous pairs found: {len(pairs)}\n")

    rows, diff_profiles = [], []
    faulty_higher = 0
    for (date, tm), members in pairs.items():
        members = sorted(members, key=lambda x: x[0] != FAULTY_AIRFRAME)  # faulty airframe first
        (fa, fpath), (pa, ppath) = members
        f_score, f_prof = profile(fpath, bundle, device)
        p_score, p_prof = profile(ppath, bundle, device)
        if f_prof is None or p_prof is None:
            continue

        diff = f_prof - p_prof            # positive = worse on the faulty airframe
        diff_profiles.append(diff)
        top = int(np.argmax(diff)) + 1
        rank_expected = int(np.where(np.argsort(-diff) + 1 == EXPECTED_CHANNEL)[0][0]) + 1
        higher = f_score > p_score
        faulty_higher += higher

        label = f"{date} {tm[:2]}:{tm[2:4]}  {fa} vs {pa}"
        print(f"{label}")
        print(f"   file-level fused score : {fa} {f_score:8.3f}   {pa} {p_score:8.3f}   "
              f"-> {'FAULTY airframe higher' if higher else 'partner higher'}")
        print(f"   channel profile diff   : top channel = ch{top}   "
              f"ch{EXPECTED_CHANNEL}(mag_x) rank = {rank_expected}/16   "
              f"ch{EXPECTED_CHANNEL} diff = {diff[EXPECTED_CHANNEL-1]:+.4f}")
        rows.append({"date": date, "time": tm, "faulty": fa, "partner": pa,
                     "faulty_score": f_score, "partner_score": p_score,
                     "faulty_higher": int(higher), "top_channel": top,
                     "magx_rank": rank_expected, "magx_diff": diff[EXPECTED_CHANNEL-1]})

    n = len(rows)
    mean_diff = np.mean(diff_profiles, axis=0)
    magx_rank_mean = int(np.where(np.argsort(-mean_diff) + 1 == EXPECTED_CHANNEL)[0][0]) + 1
    print(f"\n=== summary over {n} pairs ===")
    print(f"faulty airframe ({FAULTY_AIRFRAME}) scored higher in {faulty_higher}/{n} pairs "
          f"(sign test p = {0.5**n if faulty_higher == n else float('nan'):.3f} if unanimous)")
    print(f"averaged channel-profile difference: top channel = ch{int(np.argmax(mean_diff))+1}, "
          f"ch{EXPECTED_CHANNEL}(mag_x) rank = {magx_rank_mean}/16")
    print(f"per-pair mag_x rank: {[r['magx_rank'] for r in rows]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "swarm_pair_diff.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(mean_diff, rows, os.path.join(FIGURES_DIR, "swarm_pair_channel_diff.png"))
    print(f"\nsaved results to {RESULTS_DIR}/ and figure to {FIGURES_DIR}/")


def plot(mean_diff, rows, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ch = np.arange(1, Config.NUM_CHANNELS + 1)
    colors = [STATUS_CRITICAL if c == EXPECTED_CHANNEL else CAT_BLUE for c in ch]
    ax.bar(ch, mean_diff, color=colors, width=0.65)
    ax.axhline(0, color="#c3c2b7", linewidth=1.0)
    ax.set_xticks(ch)
    ax.set_xticklabels([f"{c}\n{Config.CHANNEL_NAMES[c-1][:9]}" for c in ch], fontsize=7)
    ax.set_ylabel(f"error profile: {FAULTY_AIRFRAME} minus partner")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Same-moment formation partners compared directly "
                 f"({len(rows)} pairs, session effects cancel)", color=INK_PRIMARY)
    ax.text(0.5, -0.26, f"red = ch{EXPECTED_CHANNEL} (mag_x), the fault channel the dataset paper "
                        f"identifies for drone {FAULTY_AIRFRAME}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
