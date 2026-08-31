"""
Sustained-load soak test for the flight computer (2026-08-23).

The board runs so far have each been a single 45.7s flight. A real sortie is
minutes long, and the two questions that answer cannot touch are the ones that
decide whether this survives an actual flight:

  does it slow down?   Sustained load heats the SoC and the governor drops
                       clocks. A detector that meets its 20 ms window for the
                       first minute and misses it at minute ten has not met it.
  does it leak?        The streaming detector holds a bounded ring buffer by
                       design, so resident memory should be flat. If it climbs
                       over hundreds of windows, the design claim is wrong.

And one the paper we reproduce never addresses at all: a drone flies on a
battery, so "how much flight time does monitoring cost" is a question that
will be asked. Where tegrastats exists, this samples power, temperature and
CPU clock alongside the latency, so the answer is measured rather than
estimated.

    python soak_test_onboard.py                # ~20 passes, a few minutes
    python soak_test_onboard.py --passes 100   # longer, for a real soak
    python soak_test_onboard.py --plot         # draw from collected CSVs

Runs anywhere. Without tegrastats it still records latency and memory, and
says which columns are missing rather than inventing them.
"""
import argparse
import csv
import os
import platform
import re
import shutil
import subprocess
import time

import joblib
import numpy as np
import onnxruntime as ort

from config import Config
import onboard_streaming_detector as osd

DEPLOY = os.path.join(Config.BASE_DIR, "deploy")
RESULTS_DIR = os.path.join(Config.BASE_DIR, "results_soak")
WINDOW_BUDGET_MS = 20.0


def machine_label():
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path) as fh:
                return fh.read().strip("\x00 \n")
        except OSError:
            pass
    return f"{platform.system()}-{platform.machine()}"


def rss_mb():
    """Resident memory without pulling in psutil -- the board only carries the
    inference dependencies and this is not worth adding one for."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


class TegraStats:
    """One tegrastats reading. Absent off-Jetson, in which case every field
    comes back NaN and the summary says so."""

    PATTERNS = {
        "power_mw": re.compile(r"VDD_IN\s+(\d+)mW"),
        "soc_temp_c": re.compile(r"(?:soc0|SOC0|tj)@([\d.]+)C"),
        "cpu_temp_c": re.compile(r"(?:cpu|CPU)@([\d.]+)C"),
        "ram_used_mb": re.compile(r"RAM\s+(\d+)/\d+MB"),
    }

    def __init__(self):
        self.available = shutil.which("tegrastats") is not None
        self.proc = None
        self.last = ""
        if self.available:
            try:
                self.proc = subprocess.Popen(
                    ["tegrastats", "--interval", "500"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except OSError:
                self.available = False

    def sample(self):
        out = {k: float("nan") for k in self.PATTERNS}
        out["cpu_mhz"] = float("nan")
        if not (self.available and self.proc and self.proc.stdout):
            return out
        line = self.proc.stdout.readline()
        if not line:
            return out
        self.last = line
        for key, pat in self.PATTERNS.items():
            m = pat.search(line)
            if m:
                out[key] = float(m.group(1))
        clocks = re.findall(r"\d+%@(\d+)", line)
        if clocks:
            out["cpu_mhz"] = float(np.mean([float(c) for c in clocks]))
        return out

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=20, help="streaming passes over the sample flight")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot_all()
        return

    machine = machine_label()
    tegra = TegraStats()
    print(f"machine: {machine}")
    tegra_note = ("available" if tegra.available else
                  "not available -- power/temperature/clock will be blank")
    print(f"tegrastats: {tegra_note}")

    fscaler = joblib.load(os.path.join(DEPLOY, "lstm_ae_feature_scaler_heldout.joblib"))
    iforest = joblib.load(os.path.join(DEPLOY, "iforest_model_heldout.joblib"))
    stats = np.load(os.path.join(DEPLOY, "lstm_ae_stats_heldout.npz"))
    sess = ort.InferenceSession(os.path.join(DEPLOY, "lstm_ae_v4.onnx"),
                                providers=["CPUExecutionProvider"])
    sample = os.path.join(DEPLOY, "sample", Config.BURST_FILE)

    rows = []
    t_start = time.perf_counter()
    print(f"\n{'pass':>5s} {'elapsed':>8s} {'p95 ms':>8s} {'RSS MB':>8s} "
          f"{'power W':>8s} {'SoC C':>7s} {'MHz':>6s} {'peak':>9s}")
    for i in range(1, args.passes + 1):
        scored, det = osd.run_flight(sample, sess, fscaler, iforest, stats)
        lat = np.array(det.latencies["total"])
        t = tegra.sample()
        peak = max(scored, key=lambda s: s[1])
        row = {
            "machine": machine, "pass": i,
            "elapsed_sec": time.perf_counter() - t_start,
            "p95_ms": float(np.percentile(lat, 95)),
            "mean_ms": float(lat.mean()), "max_ms": float(lat.max()),
            "rss_mb": rss_mb(), "windows": len(scored),
            "peak_score": float(peak[1]), "peak_time_sec": float(peak[0]),
            "power_w": t["power_mw"] / 1000 if np.isfinite(t["power_mw"]) else float("nan"),
            "soc_temp_c": t["soc_temp_c"], "cpu_temp_c": t["cpu_temp_c"],
            "cpu_mhz": t["cpu_mhz"], "ram_used_mb": t["ram_used_mb"],
        }
        rows.append(row)
        def f(v, w, p=1):
            return f"{v:{w}.{p}f}" if np.isfinite(v) else " " * (w - 1) + "-"
        print(f"{i:5d} {row['elapsed_sec']:7.1f}s {row['p95_ms']:8.2f} {f(row['rss_mb'],8)} "
              f"{f(row['power_w'],8,2)} {f(row['soc_temp_c'],7)} {f(row['cpu_mhz'],6,0)} "
              f"{row['peak_score']:9.2f}")
    tegra.stop()

    p95 = np.array([r["p95_ms"] for r in rows])
    rss = np.array([r["rss_mb"] for r in rows])
    # The first pass or two run cold -- caches empty, ONNX Runtime still
    # settling -- so including them reads as improvement-then-drift and can
    # flip the verdict on a board that is behaving perfectly. Drift is
    # measured over the steady-state passes only.
    WARMUP_PASSES = 2
    steady = p95[WARMUP_PASSES:] if len(p95) > WARMUP_PASSES * 2 else p95
    n = max(len(steady) // 4, 1)
    first, last = steady[:n].mean(), steady[-n:].mean()
    drift = 100 * (last - first) / first

    print(f"\n=== over {len(rows)} passes ({rows[-1]['elapsed_sec']:.0f}s) ===")
    warm_note = (f" (first {WARMUP_PASSES} passes excluded as warm-up)"
                 if len(steady) < len(p95) else "")
    print(f"  latency p95     first quarter {first:.2f} ms -> last quarter {last:.2f} ms "
          f"({drift:+.1f}%){warm_note}")
    print(f"  worst pass      {p95.max():.2f} ms  "
          f"({'still within' if p95.max() < WINDOW_BUDGET_MS else 'EXCEEDS'} "
          f"the {WINDOW_BUDGET_MS:.0f} ms window, {WINDOW_BUDGET_MS/p95.max():.1f}x headroom)")
    if np.isfinite(rss).all():
        print(f"  resident memory {rss[0]:.1f} MB -> {rss[-1]:.1f} MB "
              f"({rss[-1]-rss[0]:+.1f} MB over the run)")
    scores = np.array([r["peak_score"] for r in rows])
    print(f"  peak score      {scores.min():.2f} to {scores.max():.2f} "
          f"(spread {scores.max()-scores.min():.3f})")

    pw = np.array([r["power_w"] for r in rows])
    if np.isfinite(pw).any():
        tp = np.array([r["soc_temp_c"] for r in rows])
        mhz = np.array([r["cpu_mhz"] for r in rows])
        print(f"  power draw      {np.nanmean(pw):.2f} W mean, {np.nanmax(pw):.2f} W peak")
        if np.isfinite(tp).any():
            print(f"  SoC temperature {np.nanmin(tp):.1f} -> {np.nanmax(tp):.1f} C")
        if np.isfinite(mhz).any():
            # Throttling is a sustained DOWNWARD trend under heat, not the
            # spread between extremes. tegrastats samples asynchronously at
            # 500 ms while a pass takes ~3.4 s, so each row is a point sample:
            # isolated dips are the governor between bursts, and the run
            # typically *starts* low because the CPU was idle. Comparing
            # min to max called a healthy run "thermal throttling" -- it is
            # the steady-state trend, cross-checked against temperature,
            # that means anything.
            steady_mhz = mhz[WARMUP_PASSES:] if len(mhz) > WARMUP_PASSES * 2 else mhz
            k = max(len(steady_mhz) // 4, 1)
            early = np.nanmedian(steady_mhz[:k])
            late = np.nanmedian(steady_mhz[-k:])
            trend = 100 * (1 - late / early) if np.isfinite(early) and early else 0.0
            hot = np.isfinite(tp).any() and np.nanmax(tp) > 75
            note = ""
            if trend > 5 and hot:
                note = "  -- sustained drop under heat: thermal throttling"
            elif trend > 5:
                note = "  -- clock fell but the SoC stayed cool; check other load"
            print(f"  CPU clock       {early:.0f} -> {late:.0f} MHz steady-state median "
                  f"({trend:+.1f}%){note}")
            print(f"                  range {np.nanmin(steady_mhz):.0f}-{np.nanmax(steady_mhz):.0f} MHz "
                  f"(point samples; isolated dips are normal governor behaviour)")
    else:
        print("  power/temperature/clock: not recorded (tegrastats unavailable here)")

    verdict = "PASS" if steady.max() < WINDOW_BUDGET_MS and abs(drift) < 15 else "REVIEW"
    # latency is the thing that matters; clock wobble that does not move
    # latency is not a finding
    print(f"\n  verdict: {verdict}"
          + ("" if verdict == "PASS" else
             "  -- latency drifted or exceeded the window; check thermals and load"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"soak_{machine.replace(' ', '_')}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {out}")
    if args.plot:
        try:
            plot_all()
        except ImportError:
            print("matplotlib is not installed here (expected onboard) -- copy the CSV "
                  "to a machine that has it and run --plot-only there")


def plot_all():
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

    runs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))):
        with open(f) as fh:
            rs = list(csv.DictReader(fh))
        if rs:
            name = rs[0]["machine"].replace("NVIDIA ", "")
            name = name.replace(" Engineering Reference Developer Kit", "").replace(" Super", "")
            runs[name.strip()] = rs
    if not runs:
        print("no soak CSVs found")
        return

    has_power = any(np.isfinite(float(r["power_w"] or "nan")) for rs in runs.values() for r in rs)
    ncols = 3 if has_power else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.8))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for i, (m, rs) in enumerate(runs.items()):
        # x is the pass index, not wall clock: both machines run the same
        # number of passes, but the board takes ~6x longer to do so, which
        # crushed the host into the left margin on a time axis
        t = [int(r["pass"]) for r in rs]
        y = [float(r["p95_ms"]) for r in rs]
        mins = float(rs[-1]["elapsed_sec"]) / 60
        ax.plot(t, y, "-o", color=COLORS[i % 3], linewidth=1.8, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.4,
                label=f"{m}  ({mins:.1f} min total)")
    ax.axhline(WINDOW_BUDGET_MS, color="#9c3b34", linestyle="--", linewidth=1.2)
    ax.annotate(f"{WINDOW_BUDGET_MS:.0f} ms window deadline", xy=(0.98, WINDOW_BUDGET_MS),
                xycoords=("axes fraction", "data"), xytext=(0, -13),
                textcoords="offset points", fontsize=8.5, color="#9c3b34", ha="right")
    ax.set_ylim(0, WINDOW_BUDGET_MS * 1.12)
    ax.set_ylabel("p95 latency per window (ms)")
    ax.set_title("does it slow down under load?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")

    ax = axes[1]
    # Growth, not absolute footprint. The two machines start ~600 MB apart --
    # a conda interpreter with the full training stack against a lean
    # inference venv -- and plotting absolutes lets that offset own the axis
    # and flatten both curves. "Does it leak" is a question about the slope.
    for i, (m, rs) in enumerate(runs.items()):
        t = [int(r["pass"]) for r in rs]
        base = float(rs[0]["rss_mb"])
        y = [float(r["rss_mb"]) - base for r in rs]
        ax.plot(t, y, "-o", color=COLORS[i % 3], linewidth=1.8, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.4,
                label=f"{m}  (from {base:.0f} MB)")
    ax.axhline(0, color="#c2ccd6", linewidth=1.0)
    ax.set_ylabel("resident memory growth (MB)")
    ax.set_title("does it leak?\n(growth from the first pass)", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    if has_power:
        ax = axes[2]
        for i, (m, rs) in enumerate(runs.items()):
            t = [int(r["pass"]) for r in rs]
            y = [float(r["power_w"] or "nan") for r in rs]
            if np.isfinite(y).any():
                ax.plot(t, y, "-o", color=COLORS[i % 3], linewidth=1.8, markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.4, label=m)
        ax.set_ylabel("board power draw (W)")
        mean_w = np.nanmean([float(r["power_w"] or "nan")
                             for rs in runs.values() for r in rs])
        ax.set_title(f"what does monitoring cost the battery?\n{mean_w:.1f} W mean on the board",
                     color=INK, fontsize=11)

    for ax in axes:
        ax.set_xlabel("streaming pass over the sample flight")
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Sustained-load soak: latency, memory and power over a full run",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(Config.BASE_DIR, "figures", "soak_test.png")
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved figure to {out}")


if __name__ == "__main__":
    main()
