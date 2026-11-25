# FFT talk benchmark.  Usage:
#   make                # build ./fft_bench          (smem padding ON)
#   make nopad          # build ./fft_bench_nopad    (smem padding OFF, bank-conflict experiment)
#   make run            # single-FFT ladder  -> results_single.csv
#   make run-batched    # batched small FFTs -> results_batched.csv
#
# ARCH: "native" needs nvcc >= 11.6 and a GPU visible at build time. On a login node without a GPU
# set it explicitly, e.g.  make ARCH=sm_80   (check your node with: nvidia-smi --query-gpu=compute_cap --format=csv)

NVCC     ?= nvcc
ARCH     ?= native
NVFLAGS  ?= -O3 -std=c++14 -arch=$(ARCH) -lineinfo
LIBS     := -lcufft

all: fft_bench

fft_bench: fft_bench.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

nopad: fft_bench_nopad
fft_bench_nopad: fft_bench.cu
	$(NVCC) $(NVFLAGS) -DNO_PAD -o $@ $< $(LIBS)

run: fft_bench
	./fft_bench --min-log 10 --max-log 24 --csv results_single.csv

run-batched: fft_bench
	./fft_bench --batched --csv results_batched.csv

clean:
	rm -f fft_bench fft_bench_nopad *.csv

.PHONY: all nopad run run-batched clean
