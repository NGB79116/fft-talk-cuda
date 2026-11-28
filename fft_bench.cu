// fft_bench.cu -- "From O(N^2) to cuFFT": a four-rung optimization ladder for 1D complex FFT.
//
//   v0  dft_naive        O(N^2) DFT, one thread per output bin
//   v1  radix2_global    bit-reversal pass + one kernel launch per stage (global memory)
//   v2  radix2_tiled     bit-reversal fused into the load, first TILE_LOG2 stages in shared
//                        memory (one global round trip), remaining stages in global memory
//   v3  stockham_batched many small FFTs (N <= 2048): one block per FFT, ping-pong shared
//                        buffers, no bit reversal at all (batched mode only)
//   v4  fourstep         two-pass "four-step" FFT, N = N1*N2 (N1,N2 <= 4096): pass 1 does N2 column
//                        FFTs of length N1, then a twiddle multiply; pass 2 does N1 row FFTs of
//                        length N2 and stores the result transposed. 2 global passes instead of
//                        v2's 13, using 4096-point shared-memory tiles (several FFTs per block so
//                        strided accesses still move whole 32-byte sectors). Single mode, N >= 2^12.
//   ref cuFFT            the ceiling
//
// Build:  make           (see Makefile; ARCH defaults to native)
// Run:    ./fft_bench --min-log 10 --max-log 24 --csv results_single.csv
//         ./fft_bench --batched --csv results_batched.csv
//
// All FFTs are forward, complex float, power-of-two sizes, out-of-place (d_in -> d_out).
// Timing = GPU time of the whole pipeline of launches, data already resident on the device.
// "gflops" uses the standard 5*N*log2(N) convention for every kernel (naive included) so rows
// are comparable; it is NOT the real flop count of the naive DFT.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <complex>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n  at %s:%d  (%s)\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1);} } while (0)
#define CUFFT_CHECK(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
    fprintf(stderr, "cuFFT error %d at %s:%d  (%s)\n", (int)r_, __FILE__, __LINE__, #x); exit(1);} } while (0)

#ifndef TILE_LOG2
#define TILE_LOG2 10          // v2 fuses this many stages in shared memory (tile = 2^TILE_LOG2 points)
#endif

// Shared-memory index. Padding one float2 per 32 elements is the "bank-conflict" experiment:
// build with -DNO_PAD (make nopad) and compare v2 timings / Nsight bank-conflict counters.
#ifdef NO_PAD
#define SIDX(i) (i)
#define SPAD(B) (B)
#else
#define SIDX(i) ((i) + ((i) >> 5))
#define SPAD(B) ((B) + ((B) >> 5))
#endif

// ----------------------------------------------------------------------------- device helpers
__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
// exp(-i * pi * num / den)  (so exp(-2*pi*i * k/N) == twiddle(k, N/2))
__device__ __forceinline__ float2 twiddle(int num, int den) {
    float s, c;
    sincospif(-(float)num / (float)den, &s, &c);   // s = -sin(pi x), c = cos(pi x)
    return make_float2(c, s);
}
__device__ __forceinline__ unsigned bitrev(unsigned v, int bits) { return __brev(v) >> (32 - bits); }

// ----------------------------------------------------------------------------- v0: naive DFT
// X[k] = sum_n x[n] * exp(-2*pi*i*k*n/N).  Every thread reads the same x[n] (broadcast), so the
// kernel is bound by sincos throughput, not memory. k*n is reduced mod N exactly (N = 2^m).
__global__ void dft_naive_kernel(const float2* __restrict__ x, float2* __restrict__ X, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    float2 acc = make_float2(0.f, 0.f);
    for (int n = 0; n < N; ++n) {
        int r = (int)(((unsigned long long)k * (unsigned long long)n) & (unsigned long long)(N - 1));
        float s, c;
        sincospif(-2.0f * (float)r / (float)N, &s, &c);
        acc = cadd(acc, cmul(x[n], make_float2(c, s)));
    }
    X[k] = acc;
}

// ----------------------------------------------------------------------------- v1: radix-2, global
__global__ void bitrev_kernel(const float2* __restrict__ in, float2* __restrict__ out, int N, int logN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[bitrev((unsigned)i, logN)];
}

// One DIT stage with butterfly span h = 2^s. Thread t (0..N/2-1) owns one butterfly.
__global__ void radix2_stage_kernel(float2* __restrict__ x, int N, int s) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N / 2) return;
    int h = 1 << s;
    int j = t & (h - 1);
    int i = ((t >> s) << (s + 1)) + j;
    float2 w = twiddle(j, h);
    float2 a = x[i];
    float2 b = cmul(x[i + h], w);
    x[i]     = cadd(a, b);
    x[i + h] = csub(a, b);
}

// ----------------------------------------------------------------------------- v2: tiled / fused
// After the bit-reversal permutation, stages 0..K-1 only combine elements inside aligned chunks of
// 2^K points. So one block loads a chunk (gathering from bit-reversed positions), runs K stages in
// shared memory, and writes the chunk back: K global passes become 1 (and the bit-reversal pass is
// fused into the load). Remaining stages K..logN-1 run as in v1.
__global__ void fft_tile_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                int logN, int K) {
    extern __shared__ float2 sm[];
    const int B = 1 << K;
    const int base = blockIdx.x * B;
    const int t = threadIdx.x;                       // blockDim.x == B/2
    for (int e = t; e < B; e += blockDim.x)
        sm[SIDX(e)] = in[bitrev((unsigned)(base + e), logN)];
    __syncthreads();
    for (int s = 0; s < K; ++s) {
        int h = 1 << s;
        int j = t & (h - 1);
        int i = ((t >> s) << (s + 1)) + j;
        float2 w = twiddle(j, h);
        float2 a = sm[SIDX(i)];
        float2 b = cmul(sm[SIDX(i + h)], w);
        sm[SIDX(i)]     = cadd(a, b);
        sm[SIDX(i + h)] = csub(a, b);
        __syncthreads();
    }
    for (int e = t; e < B; e += blockDim.x)
        out[base + e] = sm[SIDX(e)];
}

// ----------------------------------------------------------------------------- v3: Stockham batched
// Stockham autosort: ping-pong between two shared buffers, output lands in natural order, so no
// bit reversal. One block per FFT, N/2 threads, one butterfly per thread per stage.
// At stage `stage`: n = N >> stage (current length), s = 1 << stage (stride), m = n/2.
__global__ void stockham_batched_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                        int N, int logN) {
    extern __shared__ float2 smem[];
    float2* src = smem;
    float2* dst = smem + N;
    const float2* gin = in  + (size_t)blockIdx.x * N;
    float2*       gout = out + (size_t)blockIdx.x * N;
    const int t = threadIdx.x;                       // blockDim.x == N/2
    src[t]         = gin[t];
    src[t + N / 2] = gin[t + N / 2];
    __syncthreads();
    int n = N, s = 1;
    for (int stage = 0; stage < logN; ++stage) {
        const int m = n >> 1;
        const int q = t & (s - 1);
        const int p = t >> stage;                    // t / s
        float2 w = twiddle(p, m);                    // exp(-2*pi*i*p/n)
        float2 a = src[q + s * p];
        float2 b = src[q + s * (p + m)];
        dst[q + s * (2 * p)]     = cadd(a, b);
        dst[q + s * (2 * p + 1)] = cmul(csub(a, b), w);
        __syncthreads();
        float2* tmp = src; src = dst; dst = tmp;
        n = m; s <<= 1;
    }
    gout[t]         = src[t];
    gout[t + N / 2] = src[t + N / 2];
}

// ----------------------------------------------------------------------------- v4: four-step (two-pass)
// N = N1*N2 with logN1 = ceil(logN/2), logN2 = floor(logN/2), view x as an N1 x N2 row-major matrix:
//   x[n1*N2 + n2]  ->  (1) N2 column FFTs of length N1 (stride N2), (2) multiply by W_N^(n2*k1),
//   (3) N1 row FFTs of length N2,  X[k1 + N1*k2] = Z[k1][k2]  (so the final store is a transpose).
// Each block holds FOUR_TILE = 4096 points in shared memory: G = 4096/L independent FFTs of length L,
// stored with a row stride of L+1 float2 so the transposed reads/writes are bank-conflict free.
// Adjacent FFTs in a block are adjacent columns (pass 1) / adjacent rows (pass 2), so every global
// access moves G consecutive float2 (G=4 -> a full 32-byte sector; G=1 only for N = 2^23, 2^24).
#define FOUR_LOG2 12
#define FOUR_TILE (1 << FOUR_LOG2)
#define FOUR_THREADS 1024

// All G FFTs of the block advance one radix-2 stage at a time (same index math as v2).
__device__ __forceinline__ void smem_stages(float2* sm, int logL, int G, int ld) {
    const int half = 1 << (logL - 1);                // butterflies per FFT per stage
    const int nb = G << (logL - 1);                  // butterflies per block per stage
    for (int s = 0; s < logL; ++s) {
        const int h = 1 << s;
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            const int g = b >> (logL - 1);
            const int t = b & (half - 1);
            const int j = t & (h - 1);
            const int i = ((t >> s) << (s + 1)) + j;
            float2* p = sm + g * ld;
            float2 w = twiddle(j, h);
            float2 a = p[i];
            float2 c = cmul(p[i + h], w);
            p[i]     = cadd(a, c);
            p[i + h] = csub(a, c);
        }
        __syncthreads();
    }
}

// Pass 1: column FFTs (length N1) + inter-stage twiddle. in -> out, both laid out as N1 rows x N2 cols.
__global__ void __launch_bounds__(FOUR_THREADS)
fourstep_cols_kernel(const float2* __restrict__ in, float2* __restrict__ out, int logN1, int logN2) {
    extern __shared__ float2 sm[];
    const int N1 = 1 << logN1, N2 = 1 << logN2, N = N1 << logN2;
    const int logG = FOUR_LOG2 - logN1, G = 1 << logG, ld = N1 + 1;
    const int c0 = blockIdx.x * G;                   // first column of this block
    for (int f = threadIdx.x; f < FOUR_TILE; f += blockDim.x) {
        const int g = f & (G - 1), m = f >> logG;    // column c0+g, row m (g fastest: adjacent addresses)
        sm[g * ld + (int)bitrev((unsigned)m, logN1)] = in[(size_t)m * N2 + c0 + g];
    }
    __syncthreads();
    smem_stages(sm, logN1, G, ld);
    for (int f = threadIdx.x; f < FOUR_TILE; f += blockDim.x) {
        const int g = f & (G - 1), k1 = f >> logG, c = c0 + g;
        float2 v = cmul(sm[g * ld + k1], twiddle(c * k1, N >> 1));   // exp(-2*pi*i*c*k1/N); c*k1 < N
        out[(size_t)k1 * N2 + c] = v;
    }
}

// Pass 2: row FFTs (length N2), stored transposed: out[k2*N1 + k1] = Z[k1][k2].
__global__ void __launch_bounds__(FOUR_THREADS)
fourstep_rows_kernel(const float2* __restrict__ in, float2* __restrict__ out, int logN1, int logN2) {
    extern __shared__ float2 sm[];
    const int N1 = 1 << logN1, N2 = 1 << logN2;
    const int logG = FOUR_LOG2 - logN2, G = 1 << logG, ld = N2 + 1;
    const int r0 = blockIdx.x * G;                   // first row (k1) of this block
    for (int f = threadIdx.x; f < FOUR_TILE; f += blockDim.x) {
        const int g = f >> logN2, n2 = f & (N2 - 1);                   // row r0+g, contiguous in n2
        sm[g * ld + (int)bitrev((unsigned)n2, logN2)] = in[(size_t)(r0 + g) * N2 + n2];
    }
    __syncthreads();
    smem_stages(sm, logN2, G, ld);
    for (int f = threadIdx.x; f < FOUR_TILE; f += blockDim.x) {
        const int g = f & (G - 1), k2 = f >> logG;                      // g fastest: adjacent k1 -> adjacent addresses
        out[(size_t)k2 * N1 + r0 + g] = sm[g * ld + k2];
    }
}

// ----------------------------------------------------------------------------- host: launchers
static inline int ilog2i(int v) { int l = 0; while ((1 << l) < v) ++l; return l; }

// v4 needs one N-point scratch buffer between the two passes. It is (re)allocated on first use for
// each size, i.e. during the untimed correctness launch, never inside the timed loop.
static float2* g_scratch = nullptr;
static size_t  g_scratch_n = 0;
static void ensure_scratch(size_t n) {
    if (n <= g_scratch_n) return;
    if (g_scratch) cudaFree(g_scratch);
    CUDA_CHECK(cudaMalloc(&g_scratch, n * sizeof(float2)));
    g_scratch_n = n;
}
static void launch_fourstep(const float2* d_in, float2* d_out, int N) {
    const int logN = ilog2i(N), logN1 = (logN + 1) / 2, logN2 = logN / 2;
    const int G1 = 1 << (FOUR_LOG2 - logN1), G2 = 1 << (FOUR_LOG2 - logN2);
    ensure_scratch((size_t)N);
    fourstep_cols_kernel<<<(1 << logN2) / G1, FOUR_THREADS, (size_t)G1 * ((1 << logN1) + 1) * sizeof(float2)>>>(
        d_in, g_scratch, logN1, logN2);
    fourstep_rows_kernel<<<(1 << logN1) / G2, FOUR_THREADS, (size_t)G2 * ((1 << logN2) + 1) * sizeof(float2)>>>(
        g_scratch, d_out, logN1, logN2);
}

static void launch_naive(const float2* d_in, float2* d_out, int N) {
    int block = 128;
    dft_naive_kernel<<<(N + block - 1) / block, block>>>(d_in, d_out, N);
}
static void launch_radix2_global(const float2* d_in, float2* d_out, int N) {
    int logN = ilog2i(N);
    int block = 256;
    bitrev_kernel<<<(N + block - 1) / block, block>>>(d_in, d_out, N, logN);
    int nb = (N / 2 + block - 1) / block;
    for (int s = 0; s < logN; ++s) radix2_stage_kernel<<<nb, block>>>(d_out, N, s);
}
static void launch_radix2_tiled(const float2* d_in, float2* d_out, int N) {
    int logN = ilog2i(N);
    int K = std::min(TILE_LOG2, logN);
    int B = 1 << K;
    size_t smem = (size_t)SPAD(B) * sizeof(float2);
    fft_tile_kernel<<<N / B, std::max(B / 2, 1), smem>>>(d_in, d_out, logN, K);
    int block = 256;
    int nb = (N / 2 + block - 1) / block;
    for (int s = K; s < logN; ++s) radix2_stage_kernel<<<nb, block>>>(d_out, N, s);
}
static void launch_stockham_batched(const float2* d_in, float2* d_out, int N, int batch) {
    size_t smem = (size_t)2 * N * sizeof(float2);    // N <= 2048 -> <= 32 KB (under the 48 KB default)
    stockham_batched_kernel<<<batch, N / 2, smem>>>(d_in, d_out, N, ilog2i(N));
}

// ----------------------------------------------------------------------------- host: reference + timing
// Double-precision iterative radix-2 FFT with a precomputed twiddle table.
static void cpu_fft_ref(const std::complex<float>* in, std::complex<double>* out, int N) {
    int logN = ilog2i(N);
    std::vector<std::complex<double>> w(N / 2);
    const double two_pi = 6.283185307179586476925286766559;
    for (int k = 0; k < N / 2; ++k) w[k] = std::complex<double>(cos(two_pi * k / N), -sin(two_pi * k / N));
    for (int i = 0; i < N; ++i) {
        unsigned r = 0, v = (unsigned)i;
        for (int b = 0; b < logN; ++b) { r = (r << 1) | (v & 1u); v >>= 1; }
        out[r] = std::complex<double>(in[i].real(), in[i].imag());
    }
    for (int h = 1; h < N; h <<= 1) {
        int step = N / (2 * h);
        for (int base = 0; base < N; base += 2 * h)
            for (int j = 0; j < h; ++j) {
                std::complex<double> a = out[base + j], b = out[base + j + h] * w[(size_t)j * step];
                out[base + j] = a + b;
                out[base + j + h] = a - b;
            }
    }
}

// max |gpu - ref| / max |ref|
static double rel_err(const std::vector<float2>& gpu, const std::vector<std::complex<double>>& ref, size_t off, int N) {
    double maxd = 0.0, maxr = 1e-30;
    for (int i = 0; i < N; ++i) {
        double dx = gpu[off + i].x - ref[i].real(), dy = gpu[off + i].y - ref[i].imag();
        maxd = std::max(maxd, std::sqrt(dx * dx + dy * dy));
        maxr = std::max(maxr, std::abs(ref[i]));
    }
    return maxd / maxr;
}

template <typename F>
static float time_ms(F&& f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

// Achievable device bandwidth: device-to-device memcpy counts read + write.
static double measure_copy_gbps() {
    size_t bytes = (size_t)128 << 20;
    void *p, *q;
    CUDA_CHECK(cudaMalloc(&p, bytes)); CUDA_CHECK(cudaMalloc(&q, bytes));
    CUDA_CHECK(cudaMemset(p, 1, bytes));
    float ms = time_ms([&] { cudaMemcpy(q, p, bytes, cudaMemcpyDeviceToDevice); }, 3, 20);
    cudaFree(p); cudaFree(q);
    return 2.0 * bytes / (ms * 1e6);
}

struct Row { std::string kernel; int N; int batch; float ms; double err; };

static void report(std::vector<Row>& rows, const char* name, int N, int batch, float ms, double err) {
    double gflops = 5.0 * N * ilog2i(N) * batch / (ms * 1e6);
    double gbps   = 16.0 * N * (double)batch / (ms * 1e6);
    printf("  %-18s N=2^%-2d batch=%-6d %10.4f ms  %9.1f GFLOP/s*  %8.1f GB/s(ideal traffic)  err=%.2e %s\n",
           name, ilog2i(N), batch, ms, gflops, gbps, err, err < 1e-3 ? "" : "  <-- FAIL");
    rows.push_back({name, N, batch, ms, err});
}

static void write_csv(const std::string& path, const std::vector<Row>& rows, double ceiling_gbps) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { perror("csv"); return; }
    fprintf(f, "kernel,N,batch,ms,gflops,eff_gbps,rel_err\n");
    fprintf(f, "d2d_copy_ceiling,0,1,0,0,%.3f,0\n", ceiling_gbps);
    for (auto& r : rows) {
        double gflops = 5.0 * r.N * ilog2i(r.N) * r.batch / (r.ms * 1e6);
        double gbps   = 16.0 * r.N * (double)r.batch / (r.ms * 1e6);
        fprintf(f, "%s,%d,%d,%.6f,%.3f,%.3f,%.3e\n", r.kernel.c_str(), r.N, r.batch, r.ms, gflops, gbps, r.err);
    }
    fclose(f);
    printf("wrote %s\n", path.c_str());
}

static std::vector<float2> make_input(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float2> v(n);
    for (auto& e : v) e = make_float2(U(rng), U(rng));
    return v;
}

// ----------------------------------------------------------------------------- modes
static void run_single(int minLog, int maxLog, int naiveMaxLog, const std::string& csv, double ceiling) {
    std::vector<Row> rows;
    for (int logN = minLog; logN <= maxLog; ++logN) {
        const int N = 1 << logN;
        printf("N = 2^%d = %d\n", logN, N);
        auto h_in = make_input(N, 12345u + logN);
        std::vector<float2> h_out(N);
        std::vector<std::complex<double>> ref(N);
        cpu_fft_ref(reinterpret_cast<const std::complex<float>*>(h_in.data()), ref.data(), N);

        float2 *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), (size_t)N * sizeof(float2), cudaMemcpyHostToDevice));

        int iters = std::max(5, std::min(200, (1 << 27) / N));

        auto check_and_time = [&](const char* name, auto&& launch, int it) {
            CUDA_CHECK(cudaMemset(d_out, 0, (size_t)N * sizeof(float2)));
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)N * sizeof(float2), cudaMemcpyDeviceToHost));
            double err = rel_err(h_out, ref, 0, N);
            float ms = time_ms(launch, 2, it);
            report(rows, name, N, 1, ms, err);
        };

        if (logN <= naiveMaxLog)
            check_and_time("v0_dft_naive", [&] { launch_naive(d_in, d_out, N); }, std::min(iters, 5));
        check_and_time("v1_radix2_global", [&] { launch_radix2_global(d_in, d_out, N); }, iters);
        check_and_time("v2_radix2_tiled",  [&] { launch_radix2_tiled(d_in, d_out, N); }, iters);
        if (logN >= 12 && logN <= 24)                // N1, N2 <= 4096 and >= 1 FFT per block
            check_and_time("v4_fourstep", [&] { launch_fourstep(d_in, d_out, N); }, iters);

        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
        check_and_time("cufft", [&] { cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD); }, iters);
        cufftDestroy(plan);

        cudaFree(d_in); cudaFree(d_out);
    }
    if (!csv.empty()) write_csv(csv, rows, ceiling);
}

static void run_batched(const std::string& csv, double ceiling) {
    std::vector<Row> rows;
    const int sizes[] = {128, 256, 512, 1024, 2048};
    for (int N : sizes) {
        const int batch = (1 << 22) / N;                 // 4M points total (32 MB per buffer)
        const size_t total = (size_t)N * batch;
        printf("batched: N = %d, batch = %d\n", N, batch);
        auto h_in = make_input(total, 777u + N);
        std::vector<float2> h_out(total);
        float2 *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float2)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total * sizeof(float2), cudaMemcpyHostToDevice));

        // reference for first / middle / last FFT of the batch
        const int picks[3] = {0, batch / 2, batch - 1};
        std::vector<std::vector<std::complex<double>>> refs(3, std::vector<std::complex<double>>(N));
        for (int k = 0; k < 3; ++k)
            cpu_fft_ref(reinterpret_cast<const std::complex<float>*>(h_in.data()) + (size_t)picks[k] * N, refs[k].data(), N);

        auto check_and_time = [&](const char* name, auto&& launch) {
            CUDA_CHECK(cudaMemset(d_out, 0, total * sizeof(float2)));
            launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total * sizeof(float2), cudaMemcpyDeviceToHost));
            double err = 0;
            for (int k = 0; k < 3; ++k) err = std::max(err, rel_err(h_out, refs[k], (size_t)picks[k] * N, N));
            float ms = time_ms(launch, 3, 50);
            report(rows, name, N, batch, ms, err);
        };
        check_and_time("v3_stockham_batch", [&] { launch_stockham_batched(d_in, d_out, N, batch); });

        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, batch));
        check_and_time("cufft_batched", [&] { cufftExecC2C(plan, (cufftComplex*)d_in, (cufftComplex*)d_out, CUFFT_FORWARD); });
        cufftDestroy(plan);

        cudaFree(d_in); cudaFree(d_out);
    }
    if (!csv.empty()) write_csv(csv, rows, ceiling);
}

int main(int argc, char** argv) {
    int minLog = 10, maxLog = 24, naiveMaxLog = 15;
    bool batched = false;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1);} return argv[++i]; };
        if (a == "--min-log") minLog = atoi(next());
        else if (a == "--max-log") maxLog = atoi(next());
        else if (a == "--naive-max-log") naiveMaxLog = atoi(next());
        else if (a == "--csv") csv = next();
        else if (a == "--batched") batched = true;
        else { fprintf(stderr, "usage: %s [--min-log L] [--max-log L] [--naive-max-log L] [--batched] [--csv file]\n", argv[0]); return 1; }
    }
    if (minLog < 4) minLog = 4;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0; cudaRuntimeGetVersion(&rt);
    printf("GPU: %s  (sm_%d%d, %d SMs)  CUDA runtime %d.%d  TILE_LOG2=%d  %s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, rt / 1000, (rt % 1000) / 10, TILE_LOG2,
#ifdef NO_PAD
           "smem padding OFF"
#else
           "smem padding ON"
#endif
    );
    double ceiling = measure_copy_gbps();
    printf("device-to-device copy bandwidth (read+write): %.1f GB/s\n\n", ceiling);

    if (batched) run_batched(csv, ceiling);
    else run_single(minLog, maxLog, naiveMaxLog, csv, ceiling);
    printf("\n* GFLOP/s uses 5*N*log2(N) for every kernel (naive DFT included).\n");
    return 0;
}
