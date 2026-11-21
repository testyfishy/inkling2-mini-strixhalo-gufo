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
