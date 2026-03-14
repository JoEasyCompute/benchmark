#!/usr/bin/env bash
# Not used directly (run_all embeds it), but kept for reference or custom runs.
set -euo pipefail
tag="${1:-untitled}"
mkdir -p results/power
nvidia-smi dmon -s pucvmt -o TD -f "results/power/${tag}.csv"
