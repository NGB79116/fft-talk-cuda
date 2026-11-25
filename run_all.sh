#!/usr/bin/env bash
# Runs every experiment for the talk on whatever GPU node you're on.
# Submit/launch it the same way you ran Project 2 on GAIVI (interactive GPU shell or a batch wrapper);
# this script itself is scheduler-agnostic.
set -euo pipefail
cd "$(dirname "$0")"

nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
nvcc --version | tail -n 2

make clean
make ARCH="${ARCH:-native}"
make nopad ARCH="${ARCH:-native}"

./fft_bench --min-log 10 --max-log 24 --csv results_single.csv  | tee log_single.txt
./fft_bench --batched --csv results_batched.csv                 | tee log_batched.txt
# bank-conflict experiment: same ladder, padding off, mid-size only
./fft_bench_nopad --min-log 14 --max-log 22 --csv results_nopad.csv | tee log_nopad.txt

echo
echo "Done. Next:  python3 plot_results.py"
echo "Optional profiling (if Nsight Compute is available):"
echo "  ncu --set full -k regex:fft_tile_kernel -c 1 ./fft_bench --min-log 20 --max-log 20"
