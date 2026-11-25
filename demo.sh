#!/usr/bin/env bash
# Demo launcher.
#   ./demo.sh live     run the ladder for N = 2^14 .. 2^20 on this GPU, then print the summary
#   ./demo.sh replay   print the last saved good run (made by ./preflight.sh), clearly labeled as a replay
#   ./demo.sh summary  re-print the summary of the last live run
set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-live}"
BIN=./fft_bench
MINLOG="${MINLOG:-14}"
MAXLOG="${MAXLOG:-20}"

banner() { printf '\n===== %s =====\n' "$1"; }

case "$MODE" in
  live)
    if [ ! -x "$BIN" ]; then
      echo "fft_bench not built; building (set ARCH=sm_XX if the default fails)..."
      make ARCH="${ARCH:-native}" || { echo "BUILD FAILED -> use: ./demo.sh replay"; exit 1; }
    fi
    banner "LIVE RUN"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    "$BIN" --min-log "$MINLOG" --max-log "$MAXLOG" --csv demo_last.csv | tee demo_last.log
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then echo "fft_bench exited with status $status -> use: ./demo.sh replay"; exit "$status"; fi
    banner "SUMMARY"
    python3 demo_summary.py demo_last.csv
    ;;
  replay)
    if [ ! -f demo_saved.log ] || [ ! -f demo_saved.csv ]; then
      echo "No saved run. Run ./preflight.sh on a GPU node first."; exit 1
    fi
    banner "REPLAY of a saved run (not live)"
    cat demo_saved.meta 2>/dev/null
    cat demo_saved.log
    banner "SUMMARY (from the saved run)"
    python3 demo_summary.py demo_saved.csv
    ;;
  summary)
    python3 demo_summary.py demo_last.csv
    ;;
  *)
    echo "usage: $0 [live|replay|summary]"; exit 2 ;;
esac
