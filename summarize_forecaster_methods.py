"""
Consolidated scoreboard for every forecasting approach tried (2026-08-20).

Seven methods were built for the "warn 15s before a fault" requirement. They
are scattered across several experiment scripts with their own CSVs; this
pulls them into one table and one figure, on the one comparison that matters:
lead time on a forecastable fault (slow_ramp), with the unforecastable
control (step) alongside as the sanity check that a method is measuring
anticipation rather than leaking.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config

FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")
INK_PRIMARY, INK_SECONDARY = "#0b0b0b", "#52514e"
GRIDLINE, SURFACE = "#e1e0d9", "#fcfcfb"
STATUS_GOOD, STATUS_CRITICAL, MUTED = "#0ca30c", "#d03b3b", "#898781"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# (label, detected, n, median lead on slow_ramp, note)
METHODS = [
    ("LSTM point forecast",      0, 4, 0.0,  "trained on normal only -> regresses to the mean"),
    ("LSTM residual",            1, 4, 2.5,  "persistence + learned correction"),
    ("Ensemble spread (all)",    1, 4, 1.5,  "5 members disagreeing about T+15"),
    ("Ensemble spread (mag_x)",  1, 4, 6.5,  "longest lead, but fires rarely"),
    ("Persistence",              3, 4, 5.5,  "CANNOT anticipate -- the floor"),
    ("Linear extrapolation",     4, 4, 3.0,  "training-free slope of the feature vector"),
    ("Health indicator + RUL",   4, 4, 5.0,  "best: 4/4, and outputs seconds-to-failure"),
]
REQUIRED_LEAD = 15.0


def main():
    print(f"{'method':28s} {'slow_ramp detect':>17s} {'median lead':>12s}   note")
    for label, det, n, lead, note in METHODS:
        lead_s = f"{lead:+.1f}s" if det else "  --"
        print(f"{label:28s} {f'{det}/{n}':>17s} {lead_s:>12s}   {note}")
    print(f"\nevery method scored 0/4 on the `step` control, as it must -- "
          f"an abrupt fault has no precursor to find")
    print(f"best lead achieved: {max(m[3] for m in METHODS):.1f}s against a {REQUIRED_LEAD:.0f}s requirement")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    labels = [m[0] for m in METHODS]
    leads = [m[3] for m in METHODS]
    rates = [m[1] / m[2] for m in METHODS]
    y = np.arange(len(labels))
    colors = [STATUS_GOOD if r == 1.0 else (MUTED if r == 0 else "#2a78d6") for r in rates]
    ax.barh(y, leads, color=colors, height=0.6)
    for yi, (lead, m) in enumerate(zip(leads, METHODS)):
        txt = f"{lead:.1f}s   ({m[1]}/{m[2]} detected)" if m[1] else f"never fired  ({m[1]}/{m[2]})"
        ax.text(lead + 0.25, yi, txt, va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.axvline(REQUIRED_LEAD, color=STATUS_CRITICAL, linestyle="--", linewidth=1.2)
    ax.text(REQUIRED_LEAD - 0.3, len(labels) - 0.4, f"{REQUIRED_LEAD:.0f}s required",
            fontsize=9, color=STATUS_CRITICAL, ha="right")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, REQUIRED_LEAD + 4)
    ax.set_xlabel("median lead time on a gradual (forecastable) fault, seconds")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Seven forecasters, one requirement -- none reaches 15s", color=INK_PRIMARY)
    ax.text(0.0, len(labels) - 0.15,
            "green = detected every injected fault; grey = never fired.  All seven methods correctly stayed "
            "silent on the abrupt control fault.",
            fontsize=8, color=INK_SECONDARY, transform=ax.get_yaxis_transform(which="grid"))
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "forecaster_scoreboard.png")
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"\nsaved figure to {out}")


if __name__ == "__main__":
    main()
