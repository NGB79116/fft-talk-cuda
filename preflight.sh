#!/usr/bin/env bash
# Run the night before and again ~10 minutes before the talk, on the GPU node you will present from.
# Checks the environment, times the live demo, and saves that run as the replay fallback.
cd "$(dirname "$0")"
fail=0
ok()   { printf '  [ ok ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }
warn() { printf '  [warn] %s\n' "$1"; }

echo "FFT demo preflight  $(date '+%Y-%m-%d %H:%M:%S')  host=$(hostname)"

if command -v nvidia-smi >/dev/null && gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1) && [ -n "$gpu" ]; then
  ok "GPU visible: $gpu"
else
  bad "no GPU visible here (are you in a GPU session?)"
fi

[ -x ./fft_bench ] && ok "fft_bench binary present" || bad "fft_bench missing: run 'make' (ARCH=sm_XX if native fails)"
command -v python3 >/dev/null && ok "python3 available" || bad "python3 missing (needed for the summary table)"
[ -f results_single.csv ] && ok "results_single.csv present (slide numbers)" || warn "results_single.csv missing: run ./run_all.sh before building slides"

if [ -x ./fft_bench ]; then
  if ./fft_bench --min-log 10 --max-log 10 >/dev/null 2>&1; then ok "tiny smoke run passes"; else bad "smoke run failed: run ./fft_bench --min-log 10 --max-log 10 and read the error"; fi
fi

echo
echo "Timing the full live demo (this is the real thing; it also saves the replay fallback)..."
start=$SECONDS
if ./demo.sh live > demo_preflight_run.txt 2>&1; then
  secs=$((SECONDS - start))
  ok "live demo completed in ${secs}s"
  [ "$secs" -le 60 ] || warn "over a minute: consider MAXLOG=19 ./demo.sh live to fit the slot"
  grep -q "ALL PASS" demo_preflight_run.txt && ok "all errors under threshold" || bad "summary does not say ALL PASS: open demo_preflight_run.txt"
  cp demo_last.log demo_saved.log
  cp demo_last.csv demo_saved.csv
  printf 'saved %s on %s (%s)\n' "$(date '+%Y-%m-%d %H:%M')" "$(hostname)" "${gpu:-unknown GPU}" > demo_saved.meta
  ok "saved run for ./demo.sh replay"
else
  bad "live demo failed: read demo_preflight_run.txt"
fi

echo
echo "Manual checklist:"
echo "  [ ] GPU session opened and kept alive (tmux/screen, no idle timeout) before class starts"
echo "  [ ] Terminal font 20pt+, dark text on light or high contrast, window maximized"
echo "  [ ] Slide with screenshot of the saved run is in the deck as the last-resort backup"
echo "  [ ] Laptop on power, notifications off, projector tested"
echo
[ $fail -eq 0 ] && echo "PREFLIGHT PASSED" || echo "PREFLIGHT HAS FAILURES"
exit $fail
