# FFT on the GPU: From O(N²) to cuFFT — 30 min + 5 min Q&A

Solo talk. Platform: GAIVI. Jetson comparison is out of scope (add later if permitted).
Grading is by classmates and Dr. Tu on: prepared documents, teaching delivery, Q&A.

## Teaching goal
A classmate who has seen tiled matmul, coalescing, reduction, and scan should leave able to:
1. read a butterfly diagram and say why FFT is O(N log N);
2. explain why a naive port is memory-bound and what **fusing stages into shared memory** buys;
3. explain when a different algorithm layout (Stockham, batched) beats the textbook one.

## Timeline (30 min)

| Min | Slide | Content | Course tie-in |
|---|---|---|---|
| 0-2 | 1. Hook | "Your phone does ~1000 FFTs a second." One-line applications: audio, radar, convolution (link to 10/21 lecture: FFT turns convolution into pointwise multiply), signals. State the roadmap ladder v0→v3→cuFFT. | Convolution |
| 2-5 | 2. DFT → FFT | DFT = matrix-vector, O(N²). Split into even/odd, recurse. Butterfly diagram for N=8 (draw it; no derivation). Twiddle factor = rotation. | — |
| 5-7 | 3. Mapping to GPU | N/2 independent butterflies per stage, log₂N stages, barrier between stages. Arithmetic intensity: ~5N log N flops over 16N bytes minimum, so memory is the story. | Roofline / memory |
| 7-10 | 4. v0 Naive DFT | One thread per output; every thread reads the same x[n] (broadcast); sincos-bound. Show time exploding at N=2^15. Code: ~10 lines. | Warps, broadcast |
| 10-14 | 5. v1 Radix-2 global | Bit-reversal pass + one kernel per stage. Index formula `i = ((t>>s)<<(s+1)) + j`. Why it is correct, why it launches log N kernels and re-reads all of memory each time. | Thread organization |
| 14-20 | 6. v2 Tiled + fused | Key insight: after bit reversal, the first K stages stay inside aligned 2^K chunks → load once, K stages in shared memory, write once. Stage-count figure: global passes drop from log N to 1 + (log N − K). Bank-conflict padding (`i + i/32`) and the `-DNO_PAD` experiment. | Shared memory, tiling, bank conflicts, sync |
| 20-24 | 7. v3 Stockham batched | Ping-pong buffers, output in natural order, no bit reversal. Batched small FFTs, one block per FFT. Compare with cuFFT batched. | Shared memory, __syncthreads |
| 24-27 | 8. Results | Charts: time vs N; fraction of cuFFT; bandwidth vs D2D copy ceiling. Honest takeaway: hand-written gets close to the memory floor only for the sizes it fits; cuFFT wins big sizes via multi-pass radix-4/8 and fused kernels. | Profiling |
| 27-30 | 9. FPGA bridge + takeaways | GPU = batched/throughput, FPGA = streaming/deterministic latency. One diagram: pipelined SDF FFT (butterfly + delay line per stage) vs GPU stage-per-kernel. Three takeaways. | — |
| 30-35 | Q&A | See below | |

Live demo (2 min, inside slide 8): run `./fft_bench --min-log 20 --max-log 20` on GAIVI; keep a recorded fallback and a screenshot.

## Slide design notes
- Butterfly diagram and the "global passes" bar figure are the two visuals worth the time.
- One code listing per slide, ≤ 15 lines, highlight the index line.
- Every results chart: title states the takeaway, direct labels, same color per kernel throughout (v0 gray, v1 orange, v2 blue, v3 green, cuFFT black).
- Put the correctness line (`err=…e-07`) on the results slide; classmates trust numbers that were checked.

## Q&A prep
1. **Why not just use cuFFT?** You always should in production; the point is to understand why it is fast and what it fuses.
2. **Why is the naive DFT even there?** It is the correctness oracle's cousin and the baseline that makes the O(N log N) win concrete.
3. **Where does the precision loss come from?** float accumulation across log N stages plus twiddle error; error grows roughly with log N; the benchmark reports max error relative to a double-precision CPU FFT.
4. **Why bit reversal at all? Why Stockham?** DIT needs the permuted input; Stockham trades the permutation for ping-pong buffers (2× shared memory) and a stride pattern that stays regular.
5. **Why does padding by i>>5 help (or not)?** Butterfly partners are 2^s apart; for power-of-two strides, many lanes hit the same bank. Padding shifts addresses. Report what you measured with `make nopad`; do not claim a speedup you did not see.
6. **Why not fuse more than TILE_LOG2=10 stages?** Shared memory per block and occupancy; blocks of 512 threads; beyond a tile, partners live in other blocks (needs global memory or cooperative launch).
7. **What would cuFFT do better?** Radix-4/8 (fewer passes, more work per thread = thread coarsening), twiddle tables, large sizes done as Cooley-Tukey 2D decomposition ("four-step") so each pass is coalesced.
8. **How would you do non-power-of-two?** Mixed radix, or Bluestein (convolution trick).
9. **Is it compute- or memory-bound?** Mostly memory-bound at large N; show eff GB/s vs the copy ceiling.
10. **FPGA vs GPU?** FPGA: fixed latency, streaming one sample/cycle, tiny batch OK; GPU: needs big batches to fill, higher peak throughput.

## Schedule to the 11/13 deadline (adjust when you know your slot)
- **Fri 10/30-Sun 11/1:** claim topic; email Dr. Tu (solo, Jetson maybe later); log into GAIVI; `make`; fix compile errors. Run `check_index_math.py` anywhere.
- **Mon 11/2-Wed 11/4:** `./run_all.sh`; confirm all `err` < 1e-3 (expect ~1e-6); collect CSVs.
- **Thu 11/5-Sat 11/7:** plots; Nsight Compute screenshot of v2 (if available); build slides.
- **11/8-11/10:** rehearse twice with a timer; trim to 30 min; record demo backup.
- **Escape hatch:** if v2 is not correct and profiled by 11/6, drop Stockham and fall back to stencil with the same talk structure.
- **After the talk:** submit slides and code to the course link.

## FPGA portfolio extension (after the talk)
Pipelined radix-2 SDF FFT in SystemVerilog (butterfly, twiddle ROM, delay-line memory), N=1024, fixed-point; UVM-style testbench against a NumPy golden model; report resource use and latency. Reuse `check_index_math.py` as the golden model generator.
