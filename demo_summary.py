#!/usr/bin/env python3
"""Compact, projector-friendly summary of a fft_bench CSV (stdlib only, no pandas needed).

Usage: python3 demo_summary.py results.csv
Exit status 1 if any kernel's error is above the threshold or a needed row is missing.
"""
import csv
import sys

THRESH = 1e-3


def main(path):
    rows = list(csv.DictReader(open(path)))
    by = {}
    ceiling = None
    for r in rows:
        if r["kernel"] == "d2d_copy_ceiling":
            ceiling = float(r["eff_gbps"])
            continue
        by.setdefault(int(r["N"]), {})[r["kernel"]] = r
    sizes = sorted(by)
    if not sizes:
        print("no data rows in", path)
        return 1

    def lg(n):
        return n.bit_length() - 1

    hdr = f"{'N':>7} {'v0 DFT':>10} {'v1 global':>10} {'v2 tiled':>10} {'cuFFT':>9} {'v2 vs v1':>9} {'v2 % cuFFT':>11} {'v2 GB/s':>8} {'max err':>9}"
    print()
    print("times in ms per transform (complex float, one GPU)")
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for n in sizes:
        k = by[n]
        need = ("v1_radix2_global", "v2_radix2_tiled", "cufft")
        if not all(x in k for x in need):
            print(f"2^{lg(n):<5d} missing rows: {[x for x in need if x not in k]}")
            ok = False
            continue
        v1, v2, cu = (float(k[x]["ms"]) for x in need)
        v0 = f"{float(k['v0_dft_naive']['ms']):10.3f}" if "v0_dft_naive" in k else f"{'-':>10}"
        err = max(float(x["rel_err"]) for x in k.values())
        bad = err >= THRESH
        ok &= not bad
        print(f"2^{lg(n):<5d} {v0} {v1:10.3f} {v2:10.3f} {cu:9.3f} {v1 / v2:8.1f}x {100 * cu / v2:10.0f}% "
              f"{float(k['v2_radix2_tiled']['eff_gbps']):8.0f} {err:9.1e}{'  <-- FAIL' if bad else ''}")
    print()
    if ceiling:
        print(f"device-to-device copy ceiling: {ceiling:.0f} GB/s  (v2 GB/s column = 16*N bytes / time)")
    print("v2 vs v1: how much fusing 10 stages into one shared-memory pass buys.")
    print("v2 % cuFFT: 100% would match the library; lower means the library is faster.")
    print(f"correctness (max error vs double-precision CPU FFT, threshold {THRESH:g}): {'ALL PASS' if ok else 'PROBLEM'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
