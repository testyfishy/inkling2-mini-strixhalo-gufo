// A micro-benchmark of the FP16 matrix instruction on this GPU, to learn what
// the kernels that use it could reach: how many multiply-adds per second with
// the operands in registers, with operands read from shared memory at the rate
// real kernels need, with more or fewer independent accumulators, with a small
// or a large shared-memory footprint, and with few or many blocks.
#include "bench.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace inkling2::gpu {
namespace {

using Half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

struct BenchArgs {
  const std::uint32_t* src;  // 16,384 words of small FP16 values
  float* out;                // 8 per block
  int iters;
};

// MODE 0: both operands stay in registers.
// MODE 1: one operand is read from shared memory per group of CHAINS products.
// MODE 2: 1 + CHAINS operands are read per CHAINS products (what the
//         attention kernel does: 4 reads per 3 products, 7 per 6).
// KB: kilobytes of shared memory the block owns and reads across.
template <int CHAINS, int MODE, int KB>
__launch_bounds__(256) __global__ void WmmaBenchKernel(BenchArgs a) {
  __shared__ std::uint32_t s[KB * 256];
  constexpr int kWords = KB * 256, kEntries = kWords / 8;
  const int tid = threadIdx.x, lane = tid & 31;
  for (int i = tid; i < kWords; i += 256)
    s[i] = a.src[i & 16383];
  __syncthreads();
  Half16 xb, wa;
  __builtin_memcpy(&xb, s + lane * 8, 32);
  __builtin_memcpy(&wa, s + (32 + lane) * 8, 32);
  Float8 acc[CHAINS] = {};
  for (int it = 0; it < a.iters; ++it) {
    if (MODE >= 1) {
      const int e = (it * 7) % (kEntries - 64);
      __builtin_memcpy(&xb, s + (e + lane) * 8, 32);
    }
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
      if (MODE == 2) {
        const int e = (it * 5 + c * 13 + 3) % (kEntries - 64);
        __builtin_memcpy(&wa, s + (e + lane) * 8, 32);
      }
      acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa, acc[c]);
    }
  }
  float sum = 0.0f;
  for (int c = 0; c < CHAINS; ++c)
    for (int e = 0; e < 8; ++e)
      sum += acc[c][e];
  if (lane == 0)
    a.out[blockIdx.x * 8 + (tid >> 5)] = sum;
}

// A group of XT * WT products per iteration from XT + WT operand reads, each
// operand used by several products (register blocking). With AHEAD the
// operands of the next iteration are requested before this iteration's
// products run, so no product waits for its own read. With GLOBAL the
// operands come straight from device memory instead of shared memory.
template <int XT, int WT, bool AHEAD, bool GLOBAL>
__launch_bounds__(256) __global__ void WmmaBlockedKernel(BenchArgs a) {
  constexpr int kWords = 16 * 256, kEntries = kWords / 8;
  __shared__ std::uint32_t s[kWords];
  const int tid = threadIdx.x, lane = tid & 31;
  for (int i = tid; i < kWords; i += 256)
    s[i] = a.src[i & 16383];
  __syncthreads();
  const std::uint32_t* base = GLOBAL ? a.src : s;
  const auto fetch = [&](int it, int k) {
    Half16 v;
    const int e = (it * 5 + k * 13 + 3) % (kEntries - 64);
    __builtin_memcpy(&v, base + (e + lane) * 8, 32);
    return v;
  };
  Half16 x[XT], w[WT], xn[XT], wn[WT];
  Float8 acc[XT * WT] = {};
#pragma unroll
  for (int i = 0; i < XT; ++i)
    xn[i] = fetch(0, i);
#pragma unroll
  for (int j = 0; j < WT; ++j)
    wn[j] = fetch(0, XT + j);
  for (int it = 0; it < a.iters; ++it) {
    if (AHEAD) {
#pragma unroll
      for (int i = 0; i < XT; ++i)
        x[i] = xn[i];
#pragma unroll
      for (int j = 0; j < WT; ++j)
        w[j] = wn[j];
#pragma unroll
      for (int i = 0; i < XT; ++i)
        xn[i] = fetch(it + 1, i);
#pragma unroll
      for (int j = 0; j < WT; ++j)
        wn[j] = fetch(it + 1, XT + j);
    } else {
#pragma unroll
      for (int i = 0; i < XT; ++i)
        x[i] = fetch(it, i);
#pragma unroll
      for (int j = 0; j < WT; ++j)
        w[j] = fetch(it, XT + j);
    }
#pragma unroll
    for (int i = 0; i < XT; ++i)
#pragma unroll
      for (int j = 0; j < WT; ++j)
        acc[i * WT + j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            x[i], w[j], acc[i * WT + j]);
  }
  float sum = 0.0f;
  for (int c = 0; c < XT * WT; ++c)
    for (int e = 0; e < 8; ++e)
      sum += acc[c][e];
  if (lane == 0)
    a.out[blockIdx.x * 8 + (tid >> 5)] = sum;
}

void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}

template <int CHAINS, int MODE, int KB>
void Run(const std::uint32_t* src, float* out, hipStream_t stream) {
  for (const int blocks : {40, 320, 4096}) {
    BenchArgs args{src, out, 0};
    // About 4e12 multiply-adds per run.
    const double per_iter = double(blocks) * 8 * CHAINS * 4096;
    args.iters = static_cast<int>(4e12 / per_iter);
    if (args.iters < 64)
      args.iters = 64;
    double best = 0;
    for (int rep = 0; rep < 2; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      WmmaBenchKernel<CHAINS, MODE, KB><<<blocks, 256, 0, stream>>>(args);
      Check(hipStreamSynchronize(stream));
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      const double rate = per_iter * args.iters / s;
      if (rate > best)
        best = rate;
    }
    std::printf("  accumulators %d  %-34s  shared %2d KB  blocks %5d  %7.2fe12 "
                "multiply-adds/s%c",
                CHAINS,
                MODE == 0   ? "operands in registers"
                : MODE == 1 ? "1 read per group of products"
                            : "1 + n reads per n products",
                KB, blocks, best / 1e12, 10);
    std::fflush(stdout);
  }
}

template <int XT, int WT, bool AHEAD, bool GLOBAL>
void RunBlocked(const std::uint32_t* src, float* out, hipStream_t stream) {
  for (const int blocks : {320, 4096}) {
    BenchArgs args{src, out, 0};
    const double per_iter = double(blocks) * 8 * XT * WT * 4096;
    args.iters = static_cast<int>(4e12 / per_iter);
    double best = 0;
    for (int rep = 0; rep < 2; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      WmmaBlockedKernel<XT, WT, AHEAD, GLOBAL><<<blocks, 256, 0, stream>>>(args);
      Check(hipStreamSynchronize(stream));
      const double s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      best = std::max(best, per_iter * args.iters / s);
    }
    std::printf("  %d x %d tiles: %2d products per %d reads, %s, from %s  "
                "blocks %5d  %7.2fe12 multiply-adds/s%c",
                XT, WT, XT * WT, XT + WT,
                AHEAD ? "read ahead" : "read when needed",
                GLOBAL ? "device memory" : "shared memory", blocks,
                best / 1e12, 10);
    std::fflush(stdout);
  }
}

}  // namespace

// What a launch costs: a kernel that does next to nothing, in a chain.
__global__ void TinyKernel(float* x, int n) {
  const int t = threadIdx.x;
  if (t < n)
    x[blockIdx.x * 1024 + t] += 1.0f;
}
__global__ void TinyBarrierKernel(float* x, int n, int barriers) {
  __shared__ float sh[1024];
  const int t = threadIdx.x;
  sh[t] = x[t % n];
  for (int i = 0; i < barriers; ++i) {
    __syncthreads();
    if (t < 512)
      sh[t] += sh[t + 512] * 0.0f;
  }
  __syncthreads();
  if (t < n)
    x[t] = sh[t];
}

// A norm of 2,304 values in several layouts. Strided: thread t takes values
// t, t + T, ...; sums through shared memory.
template <int T>
__launch_bounds__(T) __global__ void NormStridedBench(float* x, float* y) {
  __shared__ float sh[T];
  const int t = threadIdx.x;
  float mine = 0.0f;
  for (int i = t; i < 2304; i += T)
    mine += x[i] * x[i];
  sh[t] = mine;
  __syncthreads();
  for (int step = T / 2; step > 0; step >>= 1) {
    if (t < step)
      sh[t] += sh[t + step];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(sh[0] / 2304 + 1e-6f);
  for (int i = t; i < 2304; i += T)
    y[i] = x[i] * inv;
}
// Contiguous: thread t owns 2304 / T values in a row, kept in registers; lanes
// add by exchange, waves through one shared word each.
template <int T>
__launch_bounds__(T) __global__ void NormLaneBench(float* x, float* y) {
  constexpr int kMine = 2304 / T;
  __shared__ float sh[32];
  const int t = threadIdx.x;
  float v[kMine];
  __builtin_memcpy(v, x + t * kMine, sizeof v);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kMine; ++i)
    sum += v[i] * v[i];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor(sum, offset, 32);
  if (T > 32) {
    if ((t & 31) == 0)
      sh[t >> 5] = sum;
    __syncthreads();
    sum = 0.0f;
#pragma unroll
    for (int w = 0; w < T / 32; ++w)
      sum += sh[w];
  }
  const float inv = 1.0f / sqrtf(sum / 2304 + 1e-6f);
#pragma unroll
  for (int i = 0; i < kMine; ++i)
    v[i] *= inv;
  __builtin_memcpy(y + t * kMine, v, sizeof v);
}
// 128 rows of 2,304 F32 weights times one vector. Strided: a block of 64 per
// row. Lane: a wave per row, 72 values per lane.
__launch_bounds__(64) __global__ void DotStridedBench(float* w, float* x,
                                                      float* y) {
  __shared__ float sh[64];
  const int t = threadIdx.x;
  const float* row = w + blockIdx.x * 2304;
  float acc = 0.0f;
  for (int i = t; i < 2304; i += 64)
    acc += row[i] * x[i];
  sh[t] = acc;
  __syncthreads();
  for (int step = 32; step > 0; step >>= 1) {
    if (t < step)
      sh[t] += sh[t + step];
    __syncthreads();
  }
  if (t == 0)
    y[blockIdx.x] = sh[0];
}
template <int Waves>
__launch_bounds__(Waves * 32) __global__ void DotLaneBench(float* w, float* x,
                                                           float* y) {
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x * Waves + (threadIdx.x >> 5);
  const float* row = w + r * 2304 + lane * 72;
  const float* in = x + lane * 72;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int c = 0; c < 18; ++c) {
    float a[4], b[4];
    __builtin_memcpy(a, row + c * 4, 16);
    __builtin_memcpy(b, in + c * 4, 16);
#pragma unroll
    for (int e = 0; e < 4; ++e)
      acc[e] += a[e] * b[e];
  }
  float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_xor(sum, offset, 32);
  if (lane == 0)
    y[r] = sum;
}

int RunLaunchBench() {
  try {
    hipStream_t stream;
    Check(hipStreamCreate(&stream));
    float* x = nullptr;
    Check(hipMalloc(&x, 64 * 1024 * 4));
    Check(hipMemset(x, 0, 64 * 1024 * 4));
    const auto time = [&](const char* what, int count, auto&& launch,
                          bool sync_each) {
      for (int i = 0; i < 200; ++i)
        launch();
      Check(hipStreamSynchronize(stream));
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < count; ++i) {
        launch();
        if (sync_each)
          Check(hipStreamSynchronize(stream));
      }
      const double queued = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count();
      Check(hipStreamSynchronize(stream));
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
      std::printf("launch bench: %-46s %7.2f us each (%.2f us to queue)%c", what,
                  1e6 * seconds / count, 1e6 * queued / count, 10);
    };
    const int n = 20000;
    time("1 block of 64 threads", n,
         [&] { TinyKernel<<<1, 64, 0, stream>>>(x, 64); }, false);
    time("1 block of 1,024 threads", n,
         [&] { TinyKernel<<<1, 1024, 0, stream>>>(x, 1024); }, false);
    time("9 blocks of 256 threads", n,
         [&] { TinyKernel<<<9, 256, 0, stream>>>(x, 256); }, false);
    time("1 block of 1,024 threads, 11 barriers", n,
         [&] { TinyBarrierKernel<<<1, 1024, 0, stream>>>(x, 1024, 11); }, false);
    time("1 block of 1,024 threads, 44 barriers", n,
         [&] { TinyBarrierKernel<<<1, 1024, 0, stream>>>(x, 1024, 44); }, false);
    time("1 block of 1,024 threads, wait after each", 5000,
         [&] { TinyKernel<<<1, 1024, 0, stream>>>(x, 1024); }, true);
    float host[16];
    time("the same, and 64 bytes copied back", 5000,
         [&] {
           TinyKernel<<<1, 1024, 0, stream>>>(x, 1024);
           Check(hipMemcpyAsync(host, x, 64, hipMemcpyDeviceToHost, stream));
         },
         true);
    float* y = x + 4096;
    time("norm of 2,304: 1,024 threads, strided", n,
         [&] { NormStridedBench<1024><<<1, 1024, 0, stream>>>(x, y); }, false);
    time("norm of 2,304: 256 threads, strided", n,
         [&] { NormStridedBench<256><<<1, 256, 0, stream>>>(x, y); }, false);
    time("norm of 2,304: 256 threads, 9 each in a row", n,
         [&] { NormLaneBench<256><<<1, 256, 0, stream>>>(x, y); }, false);
    time("norm of 2,304: 128 threads, 18 each", n,
         [&] { NormLaneBench<128><<<1, 128, 0, stream>>>(x, y); }, false);
    time("norm of 2,304: 64 threads, 36 each", n,
         [&] { NormLaneBench<64><<<1, 64, 0, stream>>>(x, y); }, false);
    time("norm of 2,304: 32 threads, 72 each", n,
         [&] { NormLaneBench<32><<<1, 32, 0, stream>>>(x, y); }, false);
    float* w = nullptr;
    Check(hipMalloc(&w, 128 * 2304 * 4));
    Check(hipMemset(w, 0, 128 * 2304 * 4));
    time("128 dots of 2,304: a block of 64 per row", n,
         [&] { DotStridedBench<<<128, 64, 0, stream>>>(w, x, y); }, false);
    time("128 dots of 2,304: a wave per row, 1 per block", n,
         [&] { DotLaneBench<1><<<128, 32, 0, stream>>>(w, x, y); }, false);
    time("128 dots of 2,304: a wave per row, 4 per block", n,
         [&] { DotLaneBench<4><<<32, 128, 0, stream>>>(w, x, y); }, false);
    time("128 dots of 2,304: a wave per row, 8 per block", n,
         [&] { DotLaneBench<8><<<16, 256, 0, stream>>>(w, x, y); }, false);
    Check(hipFree(w));
    Check(hipFree(x));
    Check(hipStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s%c", e.what(), 10);
    return 1;
  }
  return 0;
}

int RunWmmaBench() {
  try {
    hipStream_t stream{};
    Check(hipStreamCreate(&stream));
    std::vector<std::uint32_t> host(16384, 0x14001400u);  // 2^-10 twice
    std::uint32_t* src{};
    float* out{};
    Check(hipMalloc(&src, host.size() * 4));
    Check(hipMalloc(&out, 4096 * 8 * 4));
    Check(hipMemcpy(src, host.data(), host.size() * 4, hipMemcpyHostToDevice));
    std::printf("matrix instruction benchmark (8 waves per block)%c", 10);
    if (std::getenv("INKLING2_WMMA_BENCH_ALL") != nullptr) {
    Run<1, 0, 4>(src, out, stream);
    Run<3, 0, 4>(src, out, stream);
    Run<6, 0, 4>(src, out, stream);
    Run<8, 0, 4>(src, out, stream);
    Run<3, 1, 16>(src, out, stream);
    Run<8, 1, 16>(src, out, stream);
    Run<3, 2, 16>(src, out, stream);
    Run<6, 2, 16>(src, out, stream);
    Run<8, 2, 16>(src, out, stream);
    Run<3, 2, 60>(src, out, stream);
    Run<6, 2, 60>(src, out, stream);
    Run<8, 2, 60>(src, out, stream);
    }
    Run<6, 0, 4>(src, out, stream);
    RunBlocked<1, 3, false, false>(src, out, stream);
    RunBlocked<1, 3, true, false>(src, out, stream);
    RunBlocked<1, 3, true, true>(src, out, stream);
    RunBlocked<1, 4, true, false>(src, out, stream);
    RunBlocked<2, 3, false, false>(src, out, stream);
    RunBlocked<2, 3, true, false>(src, out, stream);
    RunBlocked<2, 3, true, true>(src, out, stream);
    RunBlocked<2, 4, false, false>(src, out, stream);
    RunBlocked<2, 4, false, true>(src, out, stream);
    RunBlocked<3, 4, false, false>(src, out, stream);
    Check(hipFree(src));
    Check(hipFree(out));
    Check(hipStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s%c", e.what(), 10);
    return 1;
  }
  return 0;
}

}  // namespace inkling2::gpu
