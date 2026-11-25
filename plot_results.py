#!/usr/bin/env python3
"""Turn the benchmark CSVs into slide-ready PNGs.

  results_single.csv   -> fig_time.png (time vs N), fig_vs_cufft.png (fraction of cuFFT speed),
                          fig_bandwidth.png (effective GB/s vs copy ceiling)
  results_batched.csv  -> fig_batched.png
  results_nopad.csv    -> fig_padding.png (only if present)

Usage: python3 plot_results.py [dir_with_csvs]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# Fixed colour per kernel (colour follows the entity, never its rank). Okabe-Ito-derived.
COL = {
    "v0_dft_naive": "#7A7A7A",
    "v1_radix2_global": "#D55E00",
    "v2_radix2_tiled": "#0072B2",
    "v3_stockham_batch": "#009E73",
    "cufft": "#000000",
    "cufft_batched": "#000000",
}
LABEL = {
    "v0_dft_naive": "v0 naive DFT",
    "v1_radix2_global": "v1 radix-2 (global)",
    "v2_radix2_tiled": "v2 tiled + fused",
    "v3_stockham_batch": "v3 Stockham (batched)",
    "cufft": "cuFFT",
    "cufft_batched": "cuFFT (batched)",
}


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)


def load(name):
    p = os.path.join(D, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ceiling(df):
    c = df[df.kernel == "d2d_copy_ceiling"]
    return float(c.eff_gbps.iloc[0]) if len(c) else None


def lines(ax, df, col, logx=True, logy=False):
    for k, g in df[df.kernel != "d2d_copy_ceiling"].groupby("kernel", sort=False):
        g = g.sort_values("N")
        ax.plot(g.N, g[col], marker="o", ms=4, lw=2, color=COL.get(k, "#444"), label=LABEL.get(k, k))
        ax.annotate(LABEL.get(k, k), (g.N.iloc[-1], g[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=COL.get(k, "#444"), va="center")
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")


single = load("results_single.csv")
if single is not None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lines(ax, single, "ms", logy=True)
    style(ax, "FFT size N (complex float)", "time per FFT (ms, log)", "From O(N^2) to cuFFT: time per transform")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_time.png"), dpi=200); plt.close(fig)

    cu = single[single.kernel == "cufft"].set_index("N").ms
    rel = single[single.kernel.isin(["v1_radix2_global", "v2_radix2_tiled"])].copy()
    rel["frac"] = rel.apply(lambda r: cu[r.N] / r.ms if r.N in cu.index else float("nan"), axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lines(ax, rel, "frac")
    ax.axhline(1.0, color="#000", lw=1, ls="--")
    ax.set_ylim(0, 1.1)
    style(ax, "FFT size N", "speed relative to cuFFT (1.0 = parity)", "How close do hand-written kernels get?")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_vs_cufft.png"), dpi=200); plt.close(fig)

    big = single[single.kernel.isin(["v1_radix2_global", "v2_radix2_tiled", "cufft"])]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lines(ax, big, "eff_gbps")
    c = ceiling(single)
    if c:
        ax.axhline(c, color="#CC79A7", lw=1.5, ls="--")
        ax.annotate(f"D2D copy ceiling {c:.0f} GB/s", (single.N.min(), c), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="#CC79A7")
    style(ax, "FFT size N", "effective GB/s (16*N bytes / time)",
          "Memory-bound view: one read + one write is the floor")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_bandwidth.png"), dpi=200); plt.close(fig)

b = load("results_batched.csv")
if b is not None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lines(ax, b, "eff_gbps")
    c = ceiling(b)
    if c:
        ax.axhline(c, color="#CC79A7", lw=1.5, ls="--")
    style(ax, "FFT size N (batch = 2^22 / N)", "effective GB/s", "Batched small FFTs: Stockham in shared memory vs cuFFT")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_batched.png"), dpi=200); plt.close(fig)

np_ = load("results_nopad.csv")
if single is not None and np_ is not None:
    a = single[single.kernel == "v2_radix2_tiled"].set_index("N").ms
    c = np_[np_.kernel == "v2_radix2_tiled"].set_index("N").ms
    both = pd.DataFrame({"padded": a, "no_pad": c}).dropna().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(both.N, both.padded, marker="o", lw=2, color="#0072B2", label="padded")
    ax.plot(both.N, both.no_pad, marker="o", lw=2, color="#D55E00", label="no padding")
    ax.set_xscale("log", base=2)
    style(ax, "FFT size N", "time (ms)", "Shared-memory padding on the tiled kernel")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig_padding.png"), dpi=200); plt.close(fig)

print("figures written to", D)
