#!/usr/bin/env bash
set -euo pipefail
# Orchestrate STS fitting for Model A (month) and Model B (season), then plots
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$ROOT_DIR/outputs"
mkdir -p "$OUT_DIR"

Rscript "$ROOT_DIR/scripts/sts_fit_ab.R"
# Existing plotting script uses generic coeffs_* files; run it after A model saved as default
Rscript "$ROOT_DIR/scripts/sts_plot.R"

echo "All tasks finished. Outputs are under $OUT_DIR"