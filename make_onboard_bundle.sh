#!/usr/bin/env bash
# Build a self-contained tarball for the flight computer.
#
# The repo is private, so a Jetson without a registered SSH key cannot clone
# it. This packages only what inference needs -- no git, no credentials, no
# training data -- so it can travel by scp or USB stick.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-/tmp/dronev2_onboard.tar.gz}"

tar czf "$OUT" \
  deploy/ \
  onboard_streaming_detector.py \
  onboard_streaming_predictor.py \
  preprocessing.py \
  config.py \
  data_loader.py \
  feature_engineering.py \
  gru_data_loader.py \
  networks/

echo "built $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "copy it over:"
echo "  scp $OUT <user>@<jetson-ip>:~/"
echo
echo "then on the board:"
echo "  tar xzf $(basename "$OUT") && cd \$(tar tzf $(basename "$OUT") | head -1 | cut -d/ -f1) 2>/dev/null || true"
echo "  pip install onnxruntime numpy pandas scikit-learn joblib"
echo "  python deploy/preflight.py"
