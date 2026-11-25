# Speeding Up the Fast Fourier Transform on a GPU

CUDA FFT kernels, benchmarked against cuFFT. Built for COP 6527/4520 (Massively Parallel Computing) at USF.

Four kernels, each one step up the optimization ladder:

| Kernel | Idea |
|---|---|
| `v0_dft_naive` | One thread per output bin, O(N²) work |
| `v1_radix2_global` | Radix-2 FFT, one kernel launch per stage |
| `v2_radix2_tiled` | Bit reversal plus ten stages fused in shared memory, with bank-conflict padding |
| `v3` Stockham (batched) | Ping-pong shared buffers, one block per FFT, for batches of small transforms (N up to 2048) |

cuFFT is the performance control, and a device-to-device copy is the memory-bandwidth ceiling.

## Build and run

Tested on a GeForce GTX 1080 Ti (sm_61) with CUDA 11.4 on the GAIVI cluster.

```
make ARCH=sm_61            # use your GPU's architecture, e.g. sm_86
./fft_bench --min-log 10 --max-log 24 --csv results_single.csv
./fft_bench --batched --csv results_batched.csv
```

Other scripts:

- `./preflight.sh` checks the environment and a short demo run.
- `./demo.sh live` runs the demo; `./demo.sh replay` prints the last saved run.
- `./run_all.sh` runs every benchmark and writes the logs and CSVs.
- `python3 check_index_math.py` checks the kernels' index math on the CPU (no GPU needed).
- `python3 plot_results.py` makes the figures from the CSVs.

Every kernel is checked against a double-precision CPU FFT before it is timed.

## Results (1D, complex float, power-of-two sizes)

| N | v1 global | v2 tiled | cuFFT |
|---|---|---|---|
| 2²² | 6.24 ms | 4.33 ms | 0.607 ms |
| 2²⁴ | 26.58 ms | 19.00 ms | 2.46 ms |

- Fusing stages in shared memory (v2) is 1.4 to 2.7× faster than v1; cuFFT is still about 7.7× faster than v2 at N = 2²⁴.
- Maximum error against the CPU reference: below 1e-5 in all runs.
- Shared-memory padding showed no measurable effect on this GPU.
- Batched Stockham reaches 52 to 75% of cuFFT's speed for N = 128 to 2048.

## Limits

- One GPU on a shared cluster; timings vary a few percent between runs.
- 1D, single precision, power-of-two sizes only.
- Nsight Compute does not support Pascal, so explanations about caches and occupancy are hypotheses, not measurements.
- The main sweep was run once per size.

## Files

- `fft_bench.cu`: all kernels and the benchmark harness
- `results_*.csv`, `log_*.txt`: raw results
- `fig_*.png`: figures made by `plot_results.py`
- `DEMO.md`: demo run-of-show and failure playbook
- `OUTLINE.md`: talk plan
