#!/usr/bin/env bash
# Run one track and plot it.
# usage:  conda activate surge && bash run.sh [config.toml] [output dir]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
CFG=${1:-$HERE/example.toml}
OUT=${2:-$HERE/runs/$(basename "$CFG" .toml)}
python "$HERE/run_track.py" "$CFG" "$OUT"
python "$HERE/plot_track.py" "$OUT/track.npz" "$OUT/track.png"
