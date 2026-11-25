# Demo kit: run-of-show and failure playbook

The demo is 2 minutes inside slide 8 (results). Its job: let the class watch the ladder run, see that every
kernel is checked for correctness, and see the gap between hand-written kernels and cuFFT with their own eyes.
It is not a benchmark session; the slide charts come from the full `./run_all.sh` run.

## Files
| File | Purpose |
|---|---|
| `demo.sh live` | Runs N = 2^14 to 2^20 (naive DFT only up to 2^15), prints the ladder, then the summary table |
| `demo.sh replay` | Prints the last saved good run, labeled "REPLAY (not live)" |
| `demo_summary.py` | Turns the CSV into one compact table: v2 vs v1, v2 as % of cuFFT, GB/s, error, PASS/FAIL |
| `preflight.sh` | Checks GPU, binary, smoke run; times the live demo; saves the replay fallback |

## Day-before routine (about 10 minutes)
1. Open a GPU session the same way you did for Project 2, in `tmux` or `screen` so it survives a dropped connection.
2. `make ARCH=sm_XX` if `native` fails (compute capability from `nvidia-smi --query-gpu=compute_cap --format=csv`).
3. `./preflight.sh`. All lines `[ ok ]`, and note the printed seconds: that is your real demo length.
4. If it takes over 60 s, use `MAXLOG=19 ./demo.sh live` on the day.
5. Screenshot the summary table and the raw ladder output: these go on a backup slide.
6. Record the terminal: `script -q demo_recording.txt -c "./demo.sh live"` for a text record, or use the laptop's screen recorder for a video of the whole run.

## Ten minutes before your slot
- Re-open the GPU session if it timed out; `./preflight.sh` again (also refreshes the saved replay).
- Terminal font 20 pt or larger, window maximized, notifications off, projector tested.
- Leave the terminal ready at the prompt in the project directory, with the command typed but not run.

## Run-of-show (2 minutes)
| Time | You do | You say |
|---|---|---|
| 0:00 | Press Enter on `./demo.sh live` | "This is the whole ladder on one GAIVI GPU. Every line is one kernel at one size. Watch the error column first: each kernel is checked against a double-precision CPU FFT before it is timed." |
| 0:10 | While the first rows print | "At 2^14 the naive DFT is already orders of magnitude slower. That gap is O(N squared) against O(N log N)." (Only if the v0 numbers actually show this.) |
| 0:40 | When the summary table appears, point at the `v2 vs v1` column | "Same butterflies, same math. The only change: ten stages run inside shared memory in one pass instead of ten separate trips to global memory." |
| 1:10 | Point at `v2 % cuFFT` | "This is what the library still does better: fewer passes, higher radix, tuned memory access. Closing this gap is the rest of the optimization story." |
| 1:40 | Point at the last line | "ALL PASS: errors are around single-precision rounding. Back to the slides." |

Say what the numbers say, not what you hoped. If v2 is only slightly faster than v1, say "fusing helps most
when the transform is large enough that the extra passes are expensive" only if the data shows that trend;
otherwise just state the measured ratio.

## What the audience should take away (3 points)
1. Correctness is checked, not assumed.
2. Fusing stages in shared memory cuts global-memory passes, and that is where the speedup comes from.
3. A tuned library is still ahead, and you know why.

## If something goes wrong
| Symptom | Do this |
|---|---|
| `fft_bench: not found` or build error | `./demo.sh replay` and say "this is a saved run from earlier today on the same cluster" |
| No GPU or the job will not start | `./demo.sh replay`; do not wait in a queue in front of the class |
| `FAIL` shows in the error column | Do not hide it. Say "this size did not pass the check", switch to replay, and investigate after the talk |
| Live numbers differ from the slides | "The slides are from a full run; the cluster GPU is shared, so timings vary run to run." Keep the ratios, not the exact milliseconds, as the message |
| Session timed out or terminal froze | Replay mode, or the backup screenshot slide |
| It is taking too long | `Ctrl-C`, then `./demo.sh replay` |
| Time is short | Skip the demo and show the screenshot slide; the talk does not depend on it |

## Questions the demo invites
- "Why is the naive DFT missing for large N?" It is O(N squared); it is capped at 2^15 on purpose.
- "Why do the numbers change between runs?" Shared GPU, clock boost, cache state; the ratios are stable, milliseconds are not.
- "Is the error acceptable?" Max error relative to the largest output value, against a double-precision CPU FFT; the threshold is 1e-3 and single precision should land far below it.
