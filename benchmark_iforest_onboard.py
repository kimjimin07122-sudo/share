"""
Measure the tree-count trade-off ON THE BOARD (2026-08-23).

experiment_iforest_lightweight.py swept tree counts on the x86 host and scaled
to the Jetson with a measured ratio. The estimate held up -- 4.4 ms predicted
against 4.39 ms measured -- but for a presentation the board should speak for
itself, and a measured curve needs no scaling assumption to defend.

Runs the real streaming detector end to end against each forest in
deploy/benchmark_forests/ and reports what actually happened: latency per
window, the share the forest takes, headroom against the 20 ms window, and
whether the detection output is unchanged.

Measuring and plotting are separate on purpose. The flight computer only has
the inference dependencies -- matplotlib is not among them, and installing it
there just to draw a chart would be backwards. So the board writes a CSV, and
the plotting happens wherever matplotlib already lives:

    # on the Jetson
    python benchmark_iforest_onboard.py

    # copy the result back
    scp <user>@<jetson>:~/together/results_iforest_onboard/*.csv \
        results_iforest_onboard/

    # on the host, with both CSVs present
    python benchmark_iforest_onboard.py --plot

Each row is tagged with the machine it came from, and --plot draws every CSV
in the directory as its own series -- so the board and the host end up on one
chart, which is the comparison worth showing.
"""
import argparse
import csv
import json
import os
import platform
import time

import joblib
import numpy as np
import onnxruntime as ort

from config import Config
import onboard_streaming_detector as osd

DEPLOY = os.path.join(Config.BASE_DIR, "deploy")
FORESTS = os.path.join(DEPLOY, "benchmark_forests")
RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_iforest_onboard")
WINDOW_BUDGET_MS = 20.0


def machine_label():
    """A short name for whatever this is running on, so host and board rows
    stay distinguishable in the CSV."""
    m = platform.machine()
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path) as fh:
                return fh.read().strip("\x00 \n")
        except OSError:
            pass
    return f"{platform.system()}-{m}"


def run_one(n_trees, meta, sample, sess, fscaler, shared):
    forest = joblib.load(os.path.join(FORESTS, meta["file"]))
    stats = {"channel_mean": shared["channel_mean"], "channel_std": shared["channel_std"],
             "if_mean": np.array(meta["if_mean"]), "if_std": np.array(meta["if_std"])}
    t0 = time.perf_counter()
    scored, det = osd.run_flight(sample, sess, fscaler, forest, stats)
    wall = time.perf_counter() - t0
    lat = {k: np.array(v) for k, v in det.latencies.items()}
    peak = max(scored, key=lambda s: s[1])
    return {
        "trees": n_trees,
        "model_kb": meta["size_kb"],
        "windows": len(scored),
        "p95_total_ms": float(np.percentile(lat["total"], 95)),
        "mean_total_ms": float(lat["total"].mean()),
        "mean_forest_ms": float(lat["iforest"].mean()),
        "mean_onnx_ms": float(lat["onnx"].mean()),
        "mean_feature_ms": float(lat["feature"].mean()),
        "forest_share_pct": float(100 * lat["iforest"].sum() / lat["total"].sum()),
        "headroom_x": float(WINDOW_BUDGET_MS / np.percentile(lat["total"], 95)),
        "peak_score": float(peak[1]),
        "peak_time_sec": float(peak[0]),
        "wall_sec": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="draw the comparison figure")
    ap.add_argument("--plot-only", action="store_true",
                    help="draw from the CSVs already collected, without measuring")
    args = ap.parse_args()

    if args.plot_only:
        plot_all()
        return

    machine = machine_label()
    print(f"machine: {machine}\n")
    with open(os.path.join(FORESTS, "manifest.json")) as fh:
        manifest = json.load(fh)
    shared = np.load(os.path.join(FORESTS, "shared_error_stats.npz"))
    fscaler = joblib.load(os.path.join(DEPLOY, "lstm_ae_feature_scaler_heldout.joblib"))
    sess = ort.InferenceSession(os.path.join(DEPLOY, "lstm_ae_v4.onnx"),
                                providers=["CPUExecutionProvider"])
    sample = os.path.join(DEPLOY, "sample", Config.BURST_FILE)

    print(f"{'trees':>6s} {'size':>9s} {'p95 ms':>8s} {'forest ms':>10s} "
          f"{'forest %':>9s} {'headroom':>9s} {'peak score':>11s}")
    rows = []
    for n in sorted((int(k) for k in manifest), reverse=True):
        r = run_one(n, manifest[str(n)], sample, sess, fscaler, shared)
        r["machine"] = machine
        rows.append(r)
        print(f"{n:6d} {r['model_kb']:8.0f}K {r['p95_total_ms']:8.2f} {r['mean_forest_ms']:10.2f} "
              f"{r['forest_share_pct']:8.1f}% {r['headroom_x']:8.1f}x {r['peak_score']:11.2f}")

    base = rows[0]
    print(f"\nrelative to {base['trees']} trees:")
    for r in rows[1:]:
        print(f"  {r['trees']:3d} trees: {base['p95_total_ms']/r['p95_total_ms']:.2f}x faster, "
              f"headroom {base['headroom_x']:.1f}x -> {r['headroom_x']:.1f}x, "
              f"peak score {r['peak_score']:.2f} vs {base['peak_score']:.2f}")

    spread = max(r["peak_score"] for r in rows) - min(r["peak_score"] for r in rows)
    print(f"\npeak detection score varies by {spread:.3f} across every tree count "
          f"({100*spread/base['peak_score']:.2f}% of its value) -- the forest is "
          f"{100-base['forest_share_pct']:.0f}% of the score, not of the answer")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"iforest_onboard_{machine.replace(' ', '_')}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {out}")
    if args.plot:
        try:
            plot_all()
        except ImportError:
            print("\nmatplotlib is not installed here -- that is expected on the flight\n"
                  "computer. Copy this CSV to a machine that has it and run --plot there:\n"
                  f"  scp {out} <host>:<repo>/results_iforest_onboard/")


def plot_all():
    """Plot every machine's CSV together, so host and board sit side by side."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, INK3 = "#0e1620", "#3c4a58", "#6b7a88"
    GRID, SURFACE = "#dce3e9", "#fafbfc"
    COLORS = ["#1f5f8b", "#a8690c", "#2c6b52"]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "axes.edgecolor": "#c2ccd6", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    })

    def short(name):
        """Board model strings run to 60+ characters and blow out the legend."""
        n = name.replace("NVIDIA ", "").replace(" Engineering Reference Developer Kit", "")
        n = n.replace(" Super", "").replace("Linux-x86_64", "x86 host")
        return n.strip()

    runs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))):
        with open(f) as fh:
            rs = sorted(csv.DictReader(fh), key=lambda r: -int(r["trees"]))
        if rs:
            runs[short(rs[0]["machine"])] = rs
    if not runs:
        print("no result CSVs found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5))
    trees = [int(r["trees"]) for r in next(iter(runs.values()))]
    x = np.arange(len(trees))

    ax = axes[0]
    for i, (m, rs) in enumerate(runs.items()):
        y = [float(r["p95_total_ms"]) for r in rs]
        ax.plot(x, y, "-o", color=COLORS[i % 3], linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=m)
        for xi, v in zip(x, y):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    ax.axhline(WINDOW_BUDGET_MS, color="#9c3b34", linestyle="--", linewidth=1.2)
    ax.annotate(f"{WINDOW_BUDGET_MS:.0f} ms window deadline",
                xy=(0.98, WINDOW_BUDGET_MS), xycoords=("axes fraction", "data"),
                xytext=(0, -13), textcoords="offset points",
                fontsize=8.5, color="#9c3b34", ha="right")
    ax.set_ylim(0, WINDOW_BUDGET_MS * 1.12)
    ax.set_ylabel("p95 latency per window (ms)")
    ax.set_title("measured latency", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    ax = axes[1]
    for i, (m, rs) in enumerate(runs.items()):
        y = [float(r["forest_share_pct"]) for r in rs]
        ax.plot(x, y, "-o", color=COLORS[i % 3], linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=m)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of the per-window budget")
    ax.set_title("how much is the forest?", color=INK, fontsize=11)

    ax = axes[2]
    for i, (m, rs) in enumerate(runs.items()):
        y = [float(r["peak_score"]) for r in rs]
        # widen the marker on the lower layer so identical series stay visible
        ax.plot(x, y, "-o", color=COLORS[i % 3], linewidth=2,
                markersize=13 - 4 * i, markeredgecolor=SURFACE,
                markeredgewidth=2, alpha=1.0 if i else 0.85, label=m, zorder=2 + i)
    all_scores = [float(r["peak_score"]) for rs in runs.values() for r in rs]
    lo, hi = min(all_scores), max(all_scores)
    pad = max(hi - lo, 1e-6) * 1.6
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("peak detection score")
    spread = hi - lo
    ax.set_title(f"does the answer change?\nno -- total spread {spread:.3f} "
                 f"({100*spread/hi:.2f}%)", color=INK, fontsize=11)
    if len(runs) > 1:
        ax.annotate("both machines score identically,\nso the series coincide",
                    xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                    fontsize=8.5, color=INK3)

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([str(t) for t in trees])
        ax.set_xlabel("IsolationForest trees")
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("IsolationForest tree count, measured end to end on the target hardware",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(Config.BASE_DIR, "figures", "iforest_onboard_measured.png")
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
