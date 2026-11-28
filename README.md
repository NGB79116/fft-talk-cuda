# Speeding Up the Fast Fourier Transform on a GPU

CUDA FFT kernels, benchmarked against cuFFT. Built for COP 6527/4520 (Massively Parallel Computing) at USF.

Five kernels, each one step up the optimization ladder:

| Kernel | Idea |
|---|---|
| `v0_dft_naive` | One thread per output bin, O(N²) work |
| `v1_radix2_global` | Radix-2 FFT, one kernel launch per stage |
| `v2_radix2_tiled` | Bit reversal plus ten stages fused in shared memory, with bank-conflict padding |
| `v4_fourstep` | Two-pass four-step FFT: N = N1 x N2, column FFTs, twiddle multiply, row FFTs with a transposed store; two passes over memory (N from 2^12 to 2^24) |
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

| N | v1 global | v2 tiled | v4 four-step | cuFFT |
|---|---|---|---|---|
| 2²⁰ | 1.35 ms | 0.859 ms | 0.272 ms | 0.132 ms |
| 2²² | 6.24 ms | 4.33 ms | 1.12 ms | 0.604 ms |
| 2²⁴ | 26.57 ms | 19.02 ms | 8.89 ms | 2.45 ms |

- Fusing stages in shared memory (v2) is 1.4 to 2.7× faster than v1; cuFFT is still about 7.8× faster than v2 at N = 2²⁴.
- Two passes (v4) are 2.1 to 3.9× faster than v2 for N from 2¹⁹ to 2²⁴ and reach 47 to 57% of cuFFT's speed from 2¹⁹ to 2²². v4 is slower than v2 below 2¹⁹ and falls to 38% and 28% of cuFFT at 2²³ and 2²⁴. The likely causes (too few blocks at small N, uncoalesced access at the largest N) are not profiled.
- Four-step is a known technique; the work here is a from-scratch implementation and measurement against cuFFT.
- Maximum error against the CPU reference: about 6.5e-7 or less for every kernel in the single-FFT sweep.
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
