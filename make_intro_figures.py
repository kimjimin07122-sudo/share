"""
Two explanatory diagrams for the introduction of the lab-meeting deck
(2026-08-23). Unlike every other script in figures/, these plot no measured
data -- they draw the two architectures and the scoring rule that separates
them, because the deck's whole argument depends on the audience seeing that
difference before any result is shown.

Figure 1  architecture_detector_vs_predictor.png
    The detector and the predictor share an encoder and differ by exactly one
    structural change: the decoder is replaced by a linear head. Drawing them
    side by side makes "we changed one thing" visible, which is what licenses
    comparing their numbers at all.

Figure 2  scoring_trap.png
    Why the inherited scoring rule gave ZERO lead time, and what replaced it.
    Numbers are the real measurement from experiment_leadtime_and_fault_types
    (bias_drift at 80-83s, horizon 15s, threshold 1.020) -- see PROJECT_SUMMARY
    sec 3.12.
"""
import os

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from config import Config

INK_PRIMARY, INK_SECONDARY, GRIDLINE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, GREEN, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8d8b85"
FIGURES_DIR = os.path.join(Config.BASE_DIR, "figures")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK_PRIMARY, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def box(ax, x, y, w, h, label, sub=None, face=SURFACE, edge=INK_SECONDARY, lw=1.2, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center", va="center",
            fontsize=fs, color=INK_PRIMARY, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha="center", va="center",
                fontsize=7.8, color=MUTED, zorder=4)


def arrow(ax, x, y0, y1, color=INK_SECONDARY):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, linewidth=1.1, zorder=2))


# ------------------------------------------------------- figure 1: architecture
def architecture():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.0, 0.975, "One structural change separates the two models",
            fontsize=14, color=INK_PRIMARY, va="top")
    ax.text(0.0, 0.925, "Shared: engineered 64-d feature space, per-flight causal normalization, "
                        "IsolationForest fusion, channel diagnosis",
            fontsize=9, color=MUTED, va="top")

    cols = [
        dict(x=0.045, w=0.40, accent=BLUE, title="DETECTOR   LSTM Autoencoder",
             q='"Is it anomalous right now?"',
             blocks=[("Input window   1.0 s", "50 samples x 64 features", SURFACE),
                     ("Encoder LSTM   2 x 32", "shared design", "#e8f1fc"),
                     ("FC bottleneck   64 - 16 - 64", "Reis & Reis sec 6.5", "#e8f1fc"),
                     ("Decoder LSTM   2 x 32", "reconstructs the input", "#dceafb"),
                     ("Reconstruction   1.0 s", "same shape as input", SURFACE)],
             score="score  =  | reconstruction  -  CURRENT |",
             note="'current' is already in hand, so the score is available immediately",
             params="46,224 parameters"),
        dict(x=0.555, w=0.40, accent=ORANGE, title="PREDICTOR   LSTM Forecaster",
             q='"Will it become anomalous?"',
             blocks=[("Input context   15 s", "15 bins x 64 features", SURFACE),
                     ("Encoder LSTM   2 x 32", "identical to the left", "#e8f1fc"),
                     ("final hidden state", "32-d", "#e8f1fc"),
                     ("Linear head   32 - 64", "replaces the decoder", "#fdeae0"),
                     ("Forecast of  t + 15 s", "one timestep", SURFACE)],
             score="score  =  | forecast  -  ACTUAL |",
             note="'actual' is 15 s away -- the trap on the next slide",
             params="23,104 parameters"),
    ]

    for c in cols:
        x, w = c["x"], c["w"]
        ax.add_patch(Rectangle((x - 0.022, 0.055), w + 0.044, 0.80, facecolor="none",
                               edgecolor=GRIDLINE, linewidth=1.0, zorder=0))
        ax.add_patch(Rectangle((x - 0.022, 0.828), w + 0.044, 0.027, facecolor=c["accent"],
                               edgecolor="none", zorder=1))
        ax.text(x + w / 2, 0.795, c["title"], ha="center", fontsize=10.5, color=c["accent"], weight="bold")
        ax.text(x + w / 2, 0.762, c["q"], ha="center", fontsize=9, color=INK_SECONDARY, style="italic")

        top, bh, gap = 0.715, 0.072, 0.038
        for i, (label, sub, face) in enumerate(c["blocks"]):
            y = top - i * (bh + gap) - bh
            box(ax, x, y, w, bh, label, sub, face=face,
                edge=c["accent"] if i in (3,) else INK_SECONDARY,
                lw=1.6 if i == 3 else 1.1)
            if i:
                arrow(ax, x + w / 2, y + bh + gap - 0.004, y + bh + 0.004)

        ybot = top - 4 * (bh + gap) - bh
        ax.text(x + w / 2, ybot - 0.048, c["score"], ha="center", fontsize=9.6,
                color=c["accent"], family="monospace")
        ax.text(x + w / 2, ybot - 0.082, c["note"], ha="center", fontsize=8.2, color=MUTED)
        ax.text(x + w / 2, 0.072, c["params"], ha="center", fontsize=8.6, color=INK_SECONDARY)

    # Point at the row that actually differs (index 3: decoder vs linear head),
    # not at the shared rows above it.
    y_swap = 0.715 - 3 * (0.072 + 0.038) - 0.072 / 2
    ax.annotate("", xy=(0.553, y_swap), xytext=(0.447, y_swap),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.8))
    ax.text(0.5, y_swap + 0.030, "remove\ndecoder", ha="center", fontsize=8.4,
            color=ORANGE, weight="bold")

    out = os.path.join(FIGURES_DIR, "architecture_detector_vs_predictor.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------- figure 2: the trap
def scoring_trap():
    """Timeline convention: T is 'now'. The model reads context [T-15, T] and
    forecasts T+15. The fault becomes observable at 81 s. The two panels differ
    only in WHEN a verdict about that forecast can be issued."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.9), gridspec_kw=dict(hspace=0.16))
    fig.text(0.008, 0.985, "Same forecast, two scoring rules -- only one earns lead time",
             fontsize=14.5, color=INK_PRIMARY, va="top")
    fig.text(0.008, 0.940,
             "Measured: bias_drift injected at 80-83 s, horizon 15 s, alarm threshold 1.020",
             fontsize=9, color=MUTED, va="top")

    NOW, FAULT = 66.0, 81.0          # 'now', and when the fault becomes observable
    CTX = 15.0                        # context length == horizon == 15 s
    XL, XR = 46.0, 132.0              # timeline lives left of 108; text column right of it
    TEXT_X = 109.0

    panels = [
        dict(ax=axes[0], accent=ORANGE, tag="INHERITED FROM THE DETECTOR",
             formula="score(T+15)  =  | forecast  -  actual(T+15) |",
             alarm=FAULT, lead=0.0,
             verdict="alarm fires AT fault time -- zero advance warning",
             detail=["target 79 s   fault in neither       0.413   --",
                     "target 81 s   fault in TARGET        6.403   ALARM",
                     "target 96 s   fault in context only  0.219   --",
                     "",
                     "with the fault in context but not the target,",
                     "the score is BELOW the no-fault baseline"]),
        dict(ax=axes[1], accent=GREEN, tag="REPLACEMENT",
             formula="score(T)  =  anomaly( forecast of T+15 )",
             alarm=NOW, lead=CTX,
             verdict="verdict available at T -- a genuine 15 s warning",
             detail=["the forecast is scored against a model of NORMAL",
                     "(IsolationForest over the predicted 64-d vector,",
                     "plus a mag_x-specific z-score),",
                     "",
                     "not against an observation that cannot exist",
                     "until the fault has already happened"]),
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_xlim(XL, XR); ax.set_ylim(0, 1); ax.axis("off")

        ax.text(XL, 0.95, p["tag"], fontsize=8.2, color=p["accent"], weight="bold", va="top")
        ax.text(XL, 0.82, p["formula"], fontsize=11, color=INK_PRIMARY,
                family="monospace", va="top")

        # timeline axis
        ax.plot([XL + 2, 106], [0.30, 0.30], color=GRIDLINE, linewidth=1.4, zorder=0)
        for t in range(50, 106, 5):
            ax.plot([t, t], [0.272, 0.30], color=GRIDLINE, linewidth=1.0, zorder=0)
            ax.text(t, 0.205, f"{t}", ha="center", fontsize=7.6, color=MUTED)
        ax.text(106.5, 0.30, "s", fontsize=7.6, color=MUTED, va="center")

        # context window [T-15, T]
        ax.add_patch(Rectangle((NOW - CTX, 0.44), CTX, 0.125, facecolor="#e8f1fc",
                               edgecolor=BLUE, linewidth=1.1, zorder=2))
        ax.text(NOW - CTX / 2, 0.502, "context  15 s", ha="center", fontsize=8.4,
                color=BLUE, zorder=3)
        ax.plot([NOW, NOW], [0.30, 0.565], color=BLUE, linewidth=1.0,
                linestyle=":", zorder=1)
        ax.text(NOW, 0.615, "T = now", ha="center", fontsize=8, color=BLUE)

        # forecast reach: now -> now + 15
        ax.annotate("", xy=(NOW + CTX, 0.385), xytext=(NOW, 0.385),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.1,
                                    linestyle=":"))
        ax.text(NOW + CTX / 2, 0.335, "forecast  +15 s", ha="center", fontsize=7.9, color=MUTED)

        # fault
        ax.plot([FAULT, FAULT], [0.30, 0.72], color=INK_PRIMARY, linewidth=1.3,
                linestyle="--", zorder=2)
        ax.text(FAULT + 0.8, 0.70, "fault observable", fontsize=8.4, color=INK_PRIMARY)

        # alarm marker
        ax.plot([p["alarm"]], [0.30], marker="v", markersize=12,
                color=p["accent"], zorder=5)
        ax.text(p["alarm"], 0.115, "ALARM", ha="center", fontsize=8.8,
                color=p["accent"], weight="bold")

        if p["lead"] > 0:
            ax.annotate("", xy=(FAULT, 0.045), xytext=(p["alarm"], 0.045),
                        arrowprops=dict(arrowstyle="<|-|>", color=p["accent"], linewidth=1.6))
            ax.text((p["alarm"] + FAULT) / 2, -0.045, f"lead time  +{p['lead']:.0f} s",
                    ha="center", fontsize=9.4, color=p["accent"], weight="bold")
        else:
            ax.text(FAULT + 1.2, 0.03, "lead time  0 s", fontsize=9.4,
                    color=p["accent"], weight="bold")

        # right-hand text column, clear of every timeline element
        ax.text(TEXT_X, 0.92, p["verdict"], fontsize=9, color=p["accent"],
                va="top", weight="bold", wrap=True)
        for i, line in enumerate(p["detail"]):
            ax.text(TEXT_X, 0.74 - i * 0.105, line, fontsize=7.5,
                    color=MUTED, family="monospace", va="top")

    out = os.path.join(FIGURES_DIR, "scoring_trap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ------------------------------------------- figure 3: health indicator + RUL
def health_indicator_mechanism():
    """How the adopted forecaster actually works, in three labelled steps.

    A schematic, not measured data: the point is the MECHANISM (build a health
    indicator, fit its recent slope, extrapolate to the failure level), which
    no measured plot shows directly -- health_indicator_rul.png is the
    trend-window sweep, i.e. the justification for step 2's window length, and
    assumes the reader already knows what is being swept.

    Korean labels here (the other figures are English) because this one is
    read as an explanation rather than as a result.
    """
    import matplotlib.font_manager as fm
    ko = next((f.name for f in fm.fontManager.ttflist
               if f.name == "Noto Sans CJK KR"), "DejaVu Sans")

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_facecolor(SURFACE)

    THRESH, NOW, TREND, ONSET, SLOPE = 3.0, 75.0, 10.0, 55.0, 0.08
    XMAX = 95.0

    t = np.linspace(0, XMAX, 800)
    rng = np.random.default_rng(7)
    # flat-ish baseline, then a steady rise -- an incipient fault's signature.
    # SLOPE is chosen so the extrapolation lands ~12s out: far enough to be
    # worth acting on, close enough that a straight-line fit is credible.
    hi = 0.45 + 0.13 * rng.standard_normal(t.size).cumsum() / np.sqrt(t.size)
    hi += np.where(t > ONSET, SLOPE * (t - ONSET), 0.0)
    hi = np.clip(hi, 0.02, None)

    past = t <= NOW
    ax.plot(t[past], hi[past], color=BLUE, linewidth=1.9, zorder=4)
    ax.plot(t[~past], hi[~past], color=MUTED, linewidth=1.4, alpha=.55,
            linestyle=(0, (4, 3)), zorder=3)

    win = past & (t >= NOW - TREND)
    ax.axvspan(NOW - TREND, NOW, color=BLUE, alpha=.10, zorder=1)
    m, b = np.polyfit(t[win], hi[win], 1)
    hit = (THRESH - b) / m
    xf = np.linspace(NOW - TREND, min(hit + 3, XMAX), 60)
    ax.plot(xf, m * xf + b, color=ORANGE, linewidth=2.1,
            linestyle=(0, (6, 3)), zorder=5)

    ax.axhline(THRESH, color=INK_PRIMARY, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(XMAX - 1, THRESH + .14, "고장 수준  3σ", ha="right", fontsize=10.5,
            color=INK_PRIMARY, fontname=ko)
    ax.axvline(NOW, color=BLUE, linewidth=1.1, linestyle=":", zorder=2)
    ax.plot([hit], [THRESH], marker="o", markersize=9, color=ORANGE,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=6)

    ax.annotate("", xy=(hit, .42), xytext=(NOW, .42),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, linewidth=1.8))
    ax.text((NOW + hit) / 2, .60, f"고장까지  {hit - NOW:.0f}초",
            ha="center", fontsize=12.5, color=ORANGE, fontname=ko, weight="bold")

    # All three step labels ride in a band above the data so nothing collides
    # with the threshold line or the curve.
    steps = [
        (2.0, "left", "① 건강지표", "정상에서 얼마나 벗어났나\n|mag_x − 평균| / 표준편차", BLUE),
        (NOW - TREND / 2, "center", "② 최근 추세", f"최근 {TREND:.0f}초 기울기를\n직선으로 맞춤", BLUE),
        (XMAX - 1, "right", "③ 외삽", "그 직선이 고장 수준에\n닿는 시각 = RUL", ORANGE),
    ]
    for x, ha, head, sub, c in steps:
        ax.text(x, 5.02, head, ha=ha, fontsize=11.5, color=c, fontname=ko, weight="bold")
        ax.text(x, 4.70, sub, ha=ha, fontsize=9.2, color=MUTED,
                fontname=ko, linespacing=1.5, va="top")

    ax.set_xlim(0, XMAX); ax.set_ylim(0, 5.35)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("비행 경과 시간 (초)", fontsize=10.5, color=INK_SECONDARY, fontname=ko)
    ax.set_ylabel("건강지표  (기준선 대비 σ)", fontsize=10.5, color=INK_SECONDARY, fontname=ko)
    ax.set_title("건강지표 + RUL — 이상 점수 대신 '고장까지 남은 시간'을 내놓는다",
                 fontsize=14, color=INK_PRIMARY, loc="left", pad=14, fontname=ko)
    ax.grid(axis="y", color=GRIDLINE, linewidth=.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(NOW - 1.2, .08, "현재", ha="right", fontsize=9.5, color=BLUE, fontname=ko)

    out = os.path.join(FIGURES_DIR, "health_indicator_mechanism.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ------------------------------------------ figure 4: predictor-only pipeline
def predictor_architecture():
    """The predictor end to end, for a deck that drops the detector entirely.

    architecture_detector_vs_predictor.png exists to CONTRAST the two models,
    so it shows only the layers that differ and stops at the forecast. Once the
    detector is gone there is room -- and need -- for the parts that actually
    make the forecast usable: how a window becomes 64 features, and the three
    ways the forecast gets turned into an alarm. The IsolationForest lives in
    that last group: it never touches the network, it scores the network's
    OUTPUT against a model of normal flight states.
    """
    import matplotlib.font_manager as fm
    ko = next((f.name for f in fm.fontManager.ttflist
               if f.name == "Noto Sans CJK KR"), "DejaVu Sans")

    fig, ax = plt.subplots(figsize=(14, 6.4))
    ax.set_xlim(0, 1); ax.set_ylim(0.20, 1); ax.axis("off")

    ax.text(0.0, 0.985, "15초 예측기 — 입력에서 경보까지",
            fontsize=16, color=INK_PRIMARY, va="top", fontname=ko)
    ax.text(0.0, 0.938,
            "IsolationForest는 신경망 안이 아니라 뒤에 있다. 예측된 미래 상태를 정상 분포와 비교하는 채점기다.",
            fontsize=10, color=MUTED, va="top", fontname=ko)

    BW, BH, GAP, TOP = 0.245, 0.086, 0.038, 0.760
    PAD = 0.030
    rows = [TOP - i * (BH + GAP) - BH for i in range(4)]   # block bottoms
    FRAME_BOT, FRAME_TOP = rows[3] - 0.032, 0.848

    bands = [
        dict(x=0.030, accent=BLUE, tag="① 입력 준비", flow="serial",
             blocks=[("원시 텔레메트리  16채널", "50Hz로 리샘플링"),
                     ("인과 정규화", "비행 첫 30초 통계만 사용"),
                     ("특징 공학  16 × 4 = 64차원", "원값 · 1차미분 · 2차미분 · 에너지"),
                     ("1초 bin 집계", "최근 15개 bin = 정확히 15초")]),
        dict(x=0.378, accent=BLUE, tag="② 예측", flow="serial",
             blocks=[("LSTM 인코더", "2층 × 32유닛"),
                     ("최종 hidden state", "32차원"),
                     ("Linear Head", "32 → 64  ·  디코더를 대체"),
                     ("15초 뒤 예측", "64차원 특징 벡터")]),
        dict(x=0.726, accent=ORANGE, tag="③ 예측을 채점", flow="parallel",
             blocks=[("IsolationForest", "정상 비행 상태와 비교  →  이상도"),
                     ("mag_x z-score", "고장 채널만 따로  →  이상도"),
                     ("건강지표 → 추세 → RUL", "채택안  →  \"고장까지 N초\""),
                     ("경보", "세 신호 중 임계 초과 시 발령")]),
    ]

    for band in bands:
        x, acc, par = band["x"], band["accent"], band["flow"] == "parallel"
        ax.add_patch(Rectangle((x - PAD, FRAME_BOT), BW + 2 * PAD, FRAME_TOP - FRAME_BOT,
                               facecolor="none", edgecolor=GRIDLINE, linewidth=1.0, zorder=0))
        ax.text(x + BW / 2, FRAME_TOP - 0.030, band["tag"], ha="center", fontsize=11.5,
                color=acc, fontname=ko, weight="bold")

        for i, (label, sub) in enumerate(band["blocks"]):
            y, last = rows[i], i == 3
            adopted = par and i == 2
            face = "#fdeae0" if adopted else ("#e8f1fc" if (not par and last) else SURFACE)
            edge = ORANGE if (adopted or (par and last)) else (acc if last else INK_SECONDARY)
            ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                        boxstyle="round,pad=0.009,rounding_size=0.017",
                                        facecolor=face, edgecolor=edge,
                                        linewidth=1.9 if (last or adopted) else 1.1, zorder=3))
            ax.text(x + BW / 2, y + BH / 2 + 0.015, label, ha="center", va="center",
                    fontsize=10.2, color=INK_PRIMARY, fontname=ko, zorder=4)
            ax.text(x + BW / 2, y + BH / 2 - 0.020, sub, ha="center", va="center",
                    fontsize=8.2, color=MUTED, fontname=ko, zorder=4)
            if i and not par:
                arrow(ax, x + BW / 2, y + BH + GAP - 0.004, y + BH + 0.004)

        if par:
            # three scorers run in PARALLEL off the forecast, then merge into
            # the alarm -- rails on either side, never a top-to-bottom chain.
            lx, rx = x - 0.016, x + BW + 0.016
            cy = [rows[i] + BH / 2 for i in range(3)]
            ax.plot([lx, lx], [cy[0], cy[2]], color=MUTED, linewidth=1.0, zorder=2)
            ax.plot([rx, rx], [cy[0], rows[3] + BH / 2], color=MUTED, linewidth=1.0, zorder=2)
            for yc in cy:
                ax.annotate("", xy=(x - 0.001, yc), xytext=(lx, yc),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.0))
                ax.plot([x + BW, rx], [yc, yc], color=MUTED, linewidth=1.0, zorder=2)
            ax.annotate("", xy=(x + BW + 0.001, rows[3] + BH / 2),
                        xytext=(rx, rows[3] + BH / 2),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.4))

    mid = (rows[0] + BH + rows[3]) / 2
    for x0, x1 in [(0.030 + BW + PAD, 0.378 - PAD), (0.378 + BW + PAD, 0.726 - PAD - 0.016)]:
        ax.annotate("", xy=(x1, mid), xytext=(x0, mid),
                    arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, linewidth=1.6))

    ax.text(0.5, FRAME_BOT - 0.055,
            "학습은 ②에서만 일어난다.  ③의 채택안(건강지표 + RUL)은 학습이 필요 없다.",
            ha="center", fontsize=9.8, color=MUTED, fontname=ko)

    out = os.path.join(FIGURES_DIR, "predictor_architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for f in (architecture(), scoring_trap(), health_indicator_mechanism(),
              predictor_architecture()):
        print("wrote", f, os.path.getsize(f), "bytes")
