// HIP kernels for the heavy matrix products of Inkling 2 Mini decode.
// First version: straightforward and exact in F32 accumulation, not yet tuned.

#include "ops.hpp"

#include <cstddef>

namespace inkling2::gpu {
namespace {

constexpr int kNvfp4Values = 64;
constexpr int kNvfp4Bytes = 36;

__device__ inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  __builtin_memcpy(&value, &wide, sizeof value);
  return value;
}

// Unsigned E4M3 sub-block scale, halved because the E2M1 table is doubled
// (ggml_ue4m3_to_fp32).
__device__ inline float Ue4m3(std::uint32_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0f;
  const std::uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
  if (exp == 0)
    return static_cast<float>(man) * (1.0f / 1024.0f);
  // (1 + man / 8) * 2^(exp - 7) * 0.5
  const std::uint32_t bits = ((exp + 119u) << 23) | (man << 20);
  float value;
  __builtin_memcpy(&value, &bits, sizeof value);
  return value;
}

// E2M1 value, doubled: 0, 1, 2, 3, 4, 6, 8, 12 and their negatives.
__device__ inline float Fp4(std::uint32_t nibble) {
  const std::uint32_t m = nibble & 7u;
  const float mag = m < 4u ? static_cast<float>(m)
                           : static_cast<float>((2u + (m & 1u)) << ((m >> 1) - 1u));
  return (nibble & 8u) ? -mag : mag;
}

// Dot product of one 64-value NVFP4 block with x.
__device__ inline float DotNvfp4Block(const std::uint8_t* block,
                                      const float* x) {
  float sum = 0.0f;
  for (int s = 0; s < 4; ++s) {
    const float d = Ue4m3(block[s]);
    float part = 0.0f;
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t q = block[4 + s * 8 + j];
      part += Fp4(q & 15u) * x[s * 16 + j] + Fp4(q >> 4) * x[s * 16 + j + 8];
    }
    sum += part * d;
  }
  return sum;
}

// Sums `shared[0..N)` into shared[0]; every thread of the block must call it.
template<int N>
__device__ inline void Reduce(float* shared) {
  __syncthreads();
  for (int step = N / 2; step > 0; step >>= 1) {
    if (static_cast<int>(threadIdx.x) < step)
      shared[threadIdx.x] += shared[threadIdx.x + step];
    __syncthreads();
  }
}

__device__ inline float Situ(float gate, float up, float beta, float lb) {
  const float sig = 1.0f / (1.0f + __expf(-gate));
  return beta * tanhf(gate / beta) * sig * lb * tanhf(up / lb);
}

int gemv_threads = 128;  // best of the layouts measured so far, by a small margin

// One block of `Threads` threads per output row.
template<int Threads>
__global__ void GemvBf16Kernel(const std::uint16_t* __restrict__ w,
                               const float* __restrict__ x,
                               float* __restrict__ y, int cols) {
  __shared__ float shared[Threads];
  const std::uint16_t* row = w + static_cast<std::size_t>(blockIdx.x) * cols;
  float acc = 0.0f;
  for (int c = threadIdx.x; c < cols; c += Threads)
    acc += Bf16(row[c]) * x[c];
  shared[threadIdx.x] = acc;
  Reduce<Threads>(shared);
  if (threadIdx.x == 0)
    y[blockIdx.x] = shared[0];
}

// `Rows` output rows per block of 32 * Rows threads: each wave of 32 threads
// owns one row and reads it in contiguous 32-value steps.
template<int Rows>
__global__ void GemvBf16WaveKernel(const std::uint16_t* __restrict__ w,
                                   const float* __restrict__ x,
                                   float* __restrict__ y, int rows, int cols) {
  __shared__ float shared[32 * Rows];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r = blockIdx.x * Rows + wave;
  float acc = 0.0f;
  if (r < rows) {
    const std::uint16_t* row = w + static_cast<std::size_t>(r) * cols;
    for (int c = lane; c < cols; c += 32)
      acc += Bf16(row[c]) * x[c];
  }
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int step = 16; step > 0; step >>= 1) {
    if (lane < step)
      shared[threadIdx.x] += shared[threadIdx.x + step];
    __syncthreads();
  }
  if (lane == 0 && r < rows)
    y[r] = shared[threadIdx.x];
}

// One block per (expert, hidden unit): the gate row and the up row, then situ.
__global__ void ExpertUpNvfp4Kernel(ExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float gate[kThreads];
  __shared__ float up[kThreads];
  const int e = blockIdx.x / a.ff, k = blockIdx.x % a.ff;
  const int blocks = a.n_embd / kNvfp4Values;
  const std::size_t row_bytes = static_cast<std::size_t>(blocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(a.ids[e]) * 2 * a.ff * row_bytes;
  float g = 0.0f, u = 0.0f;
  for (int b = threadIdx.x; b < blocks; b += kThreads) {
    const float* xb = a.x + b * kNvfp4Values;
    g += DotNvfp4Block(base + k * row_bytes + b * kNvfp4Bytes, xb);
    u += DotNvfp4Block(base + (a.ff + k) * row_bytes + b * kNvfp4Bytes, xb);
  }
  gate[threadIdx.x] = g;
  up[threadIdx.x] = u;
  Reduce<kThreads>(gate);
  Reduce<kThreads>(up);
  if (threadIdx.x == 0)
    a.act[e * a.ff + k] = Situ(gate[0] * a.up_scale[e], up[0] * a.up_scale[e],
                               a.situ_beta, a.situ_linear_beta);
}

// One block per output value: every selected expert's down row, weighted.
__global__ void ExpertDownNvfp4Kernel(ExpertArgs a) {
  constexpr int kThreads = 32;
  __shared__ float shared[kThreads];
  const int blocks = a.ff / kNvfp4Values;
  const std::size_t row_bytes = static_cast<std::size_t>(blocks) * kNvfp4Bytes;
  float acc = 0.0f;
  for (int b = threadIdx.x; b < blocks; b += kThreads) {
    for (int e = 0; e < a.count; ++e) {
      const std::uint8_t* row =
          a.down + (static_cast<std::size_t>(a.ids[e]) * a.n_embd + blockIdx.x) *
                       row_bytes;
      acc += a.coef[e] * DotNvfp4Block(row + b * kNvfp4Bytes,
                                       a.act + e * a.ff + b * kNvfp4Values);
    }
  }
  shared[threadIdx.x] = acc;
  Reduce<kThreads>(shared);
  if (threadIdx.x == 0)
    a.out[blockIdx.x] = shared[0];
}

__global__ void ExpertUpBf16Kernel(ExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float gate[kThreads];
  __shared__ float up[kThreads];
  const int e = blockIdx.x / a.ff, k = blockIdx.x % a.ff;
  const std::uint16_t* base =
      reinterpret_cast<const std::uint16_t*>(a.gate_up) +
      static_cast<std::size_t>(a.ids[e]) * 2 * a.ff * a.n_embd;
  const std::uint16_t* gate_row = base + static_cast<std::size_t>(k) * a.n_embd;
  const std::uint16_t* up_row =
      base + static_cast<std::size_t>(a.ff + k) * a.n_embd;
  float g = 0.0f, u = 0.0f;
  for (int c = threadIdx.x; c < a.n_embd; c += kThreads) {
    g += Bf16(gate_row[c]) * a.x[c];
    u += Bf16(up_row[c]) * a.x[c];
  }
  gate[threadIdx.x] = g;
  up[threadIdx.x] = u;
  Reduce<kThreads>(gate);
  Reduce<kThreads>(up);
  if (threadIdx.x == 0)
    a.act[e * a.ff + k] = Situ(gate[0] * a.up_scale[e], up[0] * a.up_scale[e],
                               a.situ_beta, a.situ_linear_beta);
}

__global__ void ExpertDownBf16Kernel(ExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float shared[kThreads];
  const std::uint16_t* down = reinterpret_cast<const std::uint16_t*>(a.down);
  float acc = 0.0f;
  for (int e = 0; e < a.count; ++e) {
    const std::uint16_t* row =
        down + (static_cast<std::size_t>(a.ids[e]) * a.n_embd + blockIdx.x) *
                   a.ff;
    const float* act = a.act + e * a.ff;
    float part = 0.0f;
    for (int c = threadIdx.x; c < a.ff; c += kThreads)
      part += Bf16(row[c]) * act[c];
    acc += a.coef[e] * part;
  }
  shared[threadIdx.x] = acc;
  Reduce<kThreads>(shared);
  if (threadIdx.x == 0)
    a.out[blockIdx.x] = shared[0];
}

}  // namespace

void GemvBf16(const std::uint16_t* w, const float* x, float* y, int rows,
              int cols, hipStream_t stream) {
  switch (gemv_threads) {
    case 32:
      GemvBf16Kernel<32><<<rows, 32, 0, stream>>>(w, x, y, cols);
      break;
    case 128:
      GemvBf16Kernel<128><<<rows, 128, 0, stream>>>(w, x, y, cols);
      break;
    case 256:
      GemvBf16Kernel<256><<<rows, 256, 0, stream>>>(w, x, y, cols);
      break;
    case 1004:  // four rows per block, one wave each
      GemvBf16WaveKernel<4><<<(rows + 3) / 4, 128, 0, stream>>>(w, x, y, rows,
                                                               cols);
      break;
    case 1008:
      GemvBf16WaveKernel<8><<<(rows + 7) / 8, 256, 0, stream>>>(w, x, y, rows,
                                                               cols);
      break;
    default:
      GemvBf16Kernel<64><<<rows, 64, 0, stream>>>(w, x, y, cols);
  }
}

void SetGemvLayout(int layout) {
  gemv_threads = layout;
}

void Experts(const ExpertArgs& args, hipStream_t stream) {
  if (args.nvfp4) {
    ExpertUpNvfp4Kernel<<<args.count * args.ff, 64, 0, stream>>>(args);
    ExpertDownNvfp4Kernel<<<args.n_embd, 32, 0, stream>>>(args);
  } else {
    ExpertUpBf16Kernel<<<args.count * args.ff, 64, 0, stream>>>(args);
    ExpertDownBf16Kernel<<<args.n_embd, 64, 0, stream>>>(args);
  }
}

}  // namespace inkling2::gpu
