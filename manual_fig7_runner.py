import sys
import os
from pathlib import Path

if len(sys.argv) != 4:
    print("Usage: python manual_fig7_runner.py <filename.csv> <start_sec> <end_sec>")
    sys.exit(1)

fname = sys.argv[1]
start_sec = float(sys.argv[2])
end_sec = float(sys.argv[3])

# Ensure working dir is dronev2
base = Path(__file__).resolve().parent
os.chdir(base)

import config as cfg
# set manual fig7 options
cfg.Config.FIG7_MODE = 'manual'
cfg.Config.FIG7_MANUAL_FILE = fname
cfg.Config.FIG7_MANUAL_START_SEC = start_sec
cfg.Config.FIG7_MANUAL_END_SEC = end_sec

# write to a unique output folder for this run
outdir = os.path.join(cfg.Config.EVAL_OUTPUT_DIR, f"manual_{Path(fname).stem}_{int(start_sec)}_{int(end_sec)}")
cfg.Config.EVAL_OUTPUT_DIR = outdir
os.makedirs(outdir, exist_ok=True)

# run evaluation
import eval as ev
print(f"Running manual Fig.7 for {fname} {start_sec}-{end_sec} -> {outdir}")
ev.evaluate_all()
print("Done")
