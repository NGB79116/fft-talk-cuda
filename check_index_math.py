#!/usr/bin/env python3
"""CPU emulation of the CUDA kernels' index math, checked against numpy.fft.

Each emulated kernel runs all "threads" at once (vectorised over t) using exactly the same index
formulas as fft_bench.cu. This validates the algorithms and indexing -- it does NOT validate CUDA
syntax, launch configuration, shared-memory sizes or timing; those need nvcc on a GPU node.
"""
import numpy as np

rng = np.random.default_rng(0)


def tw(num, den):  # exp(-i*pi*num/den)
    return np.exp(-1j * np.pi * num / den)


def bitrev(i, bits):
    r = np.zeros_like(i)
    v = i.copy()
    for _ in range(bits):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r


def dft_naive(x):
    N = len(x)
    k = np.arange(N)[:, None]
    n = np.arange(N)[None, :]
    r = (k * n) & (N - 1)
    return (x[None, :] * tw(2 * r, N)).sum(axis=1)


def radix2_stage(x, s):
    N = len(x)
    t = np.arange(N // 2)
    h = 1 << s
    j = t & (h - 1)
    i = ((t >> s) << (s + 1)) + j
    a = x[i].copy()
    b = x[i + h] * tw(j, h)
    x[i] = a + b
    x[i + h] = a - b


def radix2_global(x):
    N = len(x)
    logN = N.bit_length() - 1
    out = x[bitrev(np.arange(N), logN)].copy()
    for s in range(logN):
        radix2_stage(out, s)
    return out


def radix2_tiled(x, K):
    N = len(x)
    logN = N.bit_length() - 1
    K = min(K, logN)
    B = 1 << K
    out = np.zeros_like(x)
    for blk in range(N // B):                       # one CUDA block per tile
        base = blk * B
        sm = x[bitrev(base + np.arange(B), logN)].copy()
        for s in range(K):
            radix2_stage(sm, s)                     # N is B here: same formulas inside the tile
        out[base:base + B] = sm
    for s in range(K, logN):
        radix2_stage(out, s)
    return out


def stockham(x):
    N = len(x)
    logN = N.bit_length() - 1
    t = np.arange(N // 2)
    src, dst = x.copy(), np.zeros_like(x)
    n, s = N, 1
    for stage in range(logN):
        m = n >> 1
        q = t & (s - 1)
        p = t >> stage
        a = src[q + s * p]
        b = src[q + s * (p + m)]
        dst[q + s * (2 * p)] = a + b
        dst[q + s * (2 * p + 1)] = (a - b) * tw(p, m)
        src, dst = dst, src
        n, s = m, s << 1
    return src


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def main():
    ok = True
    for logN in range(4, 13):
        N = 1 << logN
        x = rng.uniform(-1, 1, N) + 1j * rng.uniform(-1, 1, N)
        ref = np.fft.fft(x)
        tests = {"naive": dft_naive(x), "radix2_global": radix2_global(x), "stockham": stockham(x)}
        for K in (1, 3, 5, 10):
            tests[f"tiled_K{K}"] = radix2_tiled(x, K)
        for name, y in tests.items():
            e = rel(y, ref)
            bad = e > 1e-10
            ok &= not bad
            if bad or name in ("stockham", "tiled_K3") and logN in (4, 12):
                print(f"N=2^{logN:<2d} {name:14s} rel_err={e:.2e} {'FAIL' if bad else 'ok'}")
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
