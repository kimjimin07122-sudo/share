"""
Jetson Orin Nano vs x86 latency, measured 2026-08-23.

The first real-hardware numbers for this project. Both onboard scripts were
run on the board through the deployment package, and the outputs matched the
x86 reference exactly -- same window count, same first-score time, same peak
score and location -- so the only thing that differs is speed.

Two things worth a slide. The board meets its real-time budget on both
scripts, but with very different margins. And in both cases nearly all of the
time goes to the IsolationForest, not the neural network, which is the
opposite of where lightweighting effort would normally be aimed.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config

FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

# measured on the board, 2026-08-23 (JetPack, python 3.12.3, torch-free venv)
STAGES = ["feature", "ONNX network", "IsolationForest"]
DETECTOR = {"x86": [0.07, 0.08, 1.16], "jetson": [0.39, 0.49, 8.62],
            "budget_ms": 20.0, "p95": {"x86": 1.34, "jetson": 10.17}}
PREDICTOR = {"x86": [0.04, 0.03, 1.11], "jetson": [0.26, 0.19, 8.28],
             "budget_ms": 1000.0, "p95": {"x86": 1.22, "jetson": 9.04}}

INK, INK2, INK3 = "#0e1620", "#3c4a58", "#6b7a88"
GRID, SURFACE = "#dce3e9", "#fafbfc"
X86, JETSON = "#8fa8bd", "#1f5f8b"
PASS, WARN = "#2c6b52", "#a8690c"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c2ccd6", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK3, "ytick.color": INK3,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.95]})

    # --- panels 1 & 2: per-stage breakdown, log scale (0.03 ms to 8.6 ms) ---
    for ax, (name, d) in zip(axes[:2], [("detector", DETECTOR), ("15s predictor", PREDICTOR)]):
        y = np.arange(len(STAGES)); h = 0.36
        ax.barh(y + h/2, d["x86"], h, color=X86, label="x86 (RTX 5070 host)")
        ax.barh(y - h/2, d["jetson"], h, color=JETSON, label="Jetson Orin Nano")
        for yi, v in zip(y + h/2, d["x86"]):
            ax.text(v * 1.12, yi, f"{v:.2f}", va="center", fontsize=8, color=INK3)
        for yi, v in zip(y - h/2, d["jetson"]):
            ax.text(v * 1.12, yi, f"{v:.2f}", va="center", fontsize=8.5,
                    color=INK2, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(0.02, 60)
        ax.set_ylim(len(STAGES) - 0.45, -1.05)     # headroom above the top bar
        ax.set_yticks(y); ax.set_yticklabels(STAGES, fontsize=10)
        ax.set_xlabel("milliseconds per window (log)", labelpad=6)
        # Share of MEASURED per-window time, not of the deadline: the forest is
        # 8.28 of the predictor's 8.73 ms, but only 0.8% of its 1000 ms budget.
        # The old label said "of the board's budget", which stated the wrong
        # quantity on a slide.
        share = 100 * d["jetson"][2] / sum(d["jetson"])
        ax.set_title(f"{name}\np95 total: x86 {d['p95']['x86']:.2f} ms  ·  "
                     f"Jetson {d['p95']['jetson']:.2f} ms",
                     color=INK, fontsize=11, pad=26)
        # sits in the reserved band above the bars, clear of axis and legend
        ax.text(0.024, -0.72,
                f"IsolationForest: {share:.0f}% of measured per-window time   ·   "
                f"network: {d['jetson'][1]:.2f} ms",
                fontsize=8.5, color=WARN, va="center")
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    # top-right of the first panel: the short `feature` bars leave that corner
    # empty, so the legend sits clear of every bar and value label
    axes[0].legend(frameon=False, fontsize=9, loc="upper right",
                   bbox_to_anchor=(1.0, 0.90))

    # --- panel 3: headroom against each script's own deadline ---
    ax = axes[2]
    names = ["detector\n(20 ms window)", "predictor\n(1000 ms cadence)"]
    head = [DETECTOR["budget_ms"] / DETECTOR["p95"]["jetson"],
            PREDICTOR["budget_ms"] / PREDICTOR["p95"]["jetson"]]
    colors = [WARN if h < 5 else PASS for h in head]
    bars = ax.bar([0, 1], head, color=colors, width=0.55)
    for b, h in zip(bars, head):
        ax.text(b.get_x() + b.get_width()/2, h * 1.08, f"{h:.0f}x",
                ha="center", fontsize=13, fontweight="bold", color=INK)
    ax.axhline(1.0, color="#9c3b34", linestyle="--", linewidth=1.2)
    ax.text(1.46, 1.06, "deadline", fontsize=8.5, color="#9c3b34", ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(0.5, 400)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("headroom on the Jetson (x real-time budget)")
    ax.set_title("both fit, with very different margins", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

    fig.suptitle("On-device latency: Jetson Orin Nano vs the x86 host  "
                 "— outputs identical, only speed differs", color=INK, fontsize=13)
    fig.text(0.5, 0.005,
             "Board reproduced the host exactly: 731 detector windows, first score at 31.08 s, "
             "peak 186.27 at t=31.72 s, 84 forecasts, 0 false alarms on a clean flight.",
             ha="center", fontsize=9, color=INK3)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIGURES_DIR, "jetson_vs_x86_latency.png")
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved {out}")

    print(f"\n{'':16s} {'x86 p95':>9s} {'Jetson p95':>11s} {'ratio':>7s} {'headroom':>9s}")
    for name, d in (("detector", DETECTOR), ("predictor", PREDICTOR)):
        r = d["p95"]["jetson"] / d["p95"]["x86"]
        print(f"{name:16s} {d['p95']['x86']:8.2f}ms {d['p95']['jetson']:10.2f}ms "
              f"{r:6.1f}x {d['budget_ms']/d['p95']['jetson']:8.0f}x")


if __name__ == "__main__":
    main()
