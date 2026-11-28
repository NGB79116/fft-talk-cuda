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


def fourstep(x, tile_log2=12):
    """Two-pass four-step FFT, N = N1*N2, mirroring fourstep_cols_kernel / fourstep_rows_kernel.

    pass 1: G1 adjacent columns per block (stride-N2 loads), N1-point FFTs in "shared memory",
            twiddle W_N^(c*k1), store Y[k1*N2 + c]
    pass 2: G2 adjacent rows per block, N2-point FFTs, store transposed X[k2*N1 + k1].
    G = 2^(tile_log2 - log2(L)) FFTs of length L share one block (G*L = 2^tile_log2 points).
    """
    N = len(x)
    logN = N.bit_length() - 1
    logN1, logN2 = (logN + 1) // 2, logN // 2
    N1, N2 = 1 << logN1, 1 << logN2
    logG1, logG2 = tile_log2 - logN1, tile_log2 - logN2
    assert logG1 >= 0 and logG2 >= 0, "N too large for this tile size"
    G1, G2 = 1 << logG1, 1 << logG2
    assert N2 % G1 == 0 and N1 % G2 == 0
    Y = np.zeros(N, dtype=complex)
    for blk in range(N2 // G1):
        c0 = blk * G1
        sm = np.zeros((G1, N1), dtype=complex)
        f = np.arange(G1 * N1)
        g, m = f & (G1 - 1), f >> logG1
        sm[g, bitrev(m, logN1)] = x[m * N2 + c0 + g]
        for s in range(logN1):
            for gg in range(G1):
                radix2_stage(sm[gg], s)
        k1, c = f >> logG1, c0 + g
        Y[k1 * N2 + c] = sm[g, k1] * tw(c * k1, N // 2)
    out = np.zeros(N, dtype=complex)
    for blk in range(N1 // G2):
        r0 = blk * G2
        sm = np.zeros((G2, N2), dtype=complex)
        f = np.arange(G2 * N2)
        g, n2 = f >> logN2, f & (N2 - 1)
        sm[g, bitrev(n2, logN2)] = Y[(r0 + g) * N2 + n2]
        for s in range(logN2):
            for gg in range(G2):
                radix2_stage(sm[gg], s)
        g, k2 = f & (G2 - 1), f >> logG2
        out[k2 * N1 + r0 + g] = sm[g, k2]
    return out


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
    for logN in range(12, 21):
        N = 1 << logN
        x = rng.uniform(-1, 1, N) + 1j * rng.uniform(-1, 1, N)
        ref = np.fft.fft(x)
        for tl in (12, 9):
            if (logN + 1) // 2 > tl:
                continue
            e = rel(fourstep(x, tl), ref)
            bad = e > 1e-10
            ok &= not bad
            if bad or logN in (12, 20):
                print(f"N=2^{logN:<2d} fourstep tile=2^{tl:<2d} rel_err={e:.2e} {'FAIL' if bad else 'ok'}")
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
