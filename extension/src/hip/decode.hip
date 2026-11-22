// Kernels for layer visits that stay on the device, over a batch of rows.
// See decode.hpp. The single-block "glue" kernels run one block per row and
// their phases in sequence with __syncthreads between them; every thread of a
// block executes every phase.

#include "decode.hpp"

#include <cstddef>

namespace inkling2::gpu {
namespace {

// Slices of 16 columns the staged kernels hold in shared memory at once.
#ifndef INKLING2_STAGE_ROW
#define INKLING2_STAGE_ROW 8
#endif
constexpr int kStageRow = INKLING2_STAGE_ROW;
#ifndef INKLING2_FLASH_PAD
#define INKLING2_FLASH_PAD 8
#endif
#ifndef INKLING2_BK
#define INKLING2_BK 4
#endif

constexpr int kT = 1024;  // threads of a glue block
constexpr int kSums = 5;  // values summed together by BlockSum
constexpr int kNvfp4Values = 64;
constexpr int kNvfp4Bytes = 36;

__device__ inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  __builtin_memcpy(&value, &wide, sizeof value);
  return value;
}

__device__ inline float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Unsigned E4M3 sub-block scale, halved because the E2M1 table is doubled
// (ggml_ue4m3_to_fp32).
__device__ inline float Ue4m3(std::uint32_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0f;
  const std::uint32_t exp = (x >> 3) & 0xF, man = x & 0x7;
  if (exp == 0)
    return static_cast<float>(man) * (1.0f / 1024.0f);
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

// E2M1 values, doubled, indexed by the 4-bit code.
constexpr float kFp4Table[16] = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  6.0f,
                                 8.0f,  12.0f, -0.0f, -1.0f, -2.0f, -3.0f,
                                 -4.0f, -6.0f, -8.0f, -12.0f};

// Dot product of one 64-value NVFP4 block with x, reading the block as nine
// 32-bit words (blocks are 36 bytes, so they stay 4-byte aligned) and
// decoding the codes through the table.
__device__ inline float DotNvfp4Words(const std::uint8_t* block,
                                      const float* x) {
  const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(block);
  const std::uint32_t scales = w[0];
  float sum = 0.0f;
  for (int s = 0; s < 4; ++s) {
    const float d = Ue4m3((scales >> (8 * s)) & 0xFFu);
    const float* xs = x + s * 16;
    float part = 0.0f;
    for (int half = 0; half < 2; ++half) {
      const std::uint32_t word = w[1 + 2 * s + half];
      for (int j = 0; j < 4; ++j) {
        const std::uint32_t q = (word >> (8 * j)) & 0xFFu;
        part += kFp4Table[q & 15u] * xs[half * 4 + j] +
                kFp4Table[q >> 4] * xs[half * 4 + j + 8];
      }
    }
    sum += part * d;
  }
  return sum;
}

__device__ inline float Situ(float gate, float up, float beta, float lb) {
  return beta * tanhf(gate / beta) * Sigmoid(gate) * lb * tanhf(up / lb);
}

// Sums `shared[0..n)` into shared[0]; every thread of the block calls it.
__device__ inline void Reduce(float* shared, int n) {
  const int t = threadIdx.x;
  __syncthreads();
  for (int step = n / 2; step > 0; step >>= 1) {
    if (t < step)
      shared[t] += shared[t + step];
    __syncthreads();
  }
}

// Sums each of the first `count` entries of `mine` over a glue block's
// threads. Every thread calls it and gets the same totals.
__device__ inline void BlockSum(float (*sh)[kT], const float* mine, int count,
                                float* total) {
  const int t = threadIdx.x;
  for (int a = 0; a < count; ++a)
    sh[a][t] = mine[a];
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int a = 0; a < count; ++a)
        sh[a][t] += sh[a][t + step];
    __syncthreads();
  }
  for (int a = 0; a < count; ++a)
    total[a] = sh[a][0];
  __syncthreads();
}

// Attention over the depth: each state is RMS-normalized, scored against each
// query, and the normalized states are averaged with the softmax weights.
// `out` (nq rows of kE) must not alias a state.
__device__ inline void Mix(float (*sh)[kT], const float* const* states, int n,
                           const float* queries, int nq, float* out) {
  const int t = threadIdx.x;
  float inv[kMaxStates];
  float score[4][kMaxStates];
  for (int s = 0; s < n; ++s) {
    float mine[kSums] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t; i < kE; i += kT) {
      const float x = states[s][i];
      mine[0] += x * x;
      for (int q = 0; q < nq; ++q)
        mine[1 + q] += x * queries[q * kE + i];
    }
    float total[kSums];
    BlockSum(sh, mine, 1 + nq, total);
    inv[s] = 1.0f / sqrtf(total[0] / kE + 1e-6f);
    for (int q = 0; q < nq; ++q)
      score[q][s] = total[1 + q] * inv[s] * (1.0f / 48.0f);  // 1 / sqrt(kE)
  }
  for (int q = 0; q < nq; ++q) {
    float max = score[q][0];
    for (int s = 1; s < n; ++s)
      max = fmaxf(max, score[q][s]);
    float sum = 0.0f;
    for (int s = 0; s < n; ++s) {
      score[q][s] = __expf(score[q][s] - max);
      sum += score[q][s];
    }
    for (int s = 0; s < n; ++s)
      score[q][s] = score[q][s] / sum * inv[s];
  }
  for (int i = t; i < kE; i += kT)
    for (int q = 0; q < nq; ++q) {
      float acc = 0.0f;
      for (int s = 0; s < n; ++s)
        acc += score[q][s] * states[s][i];
      out[q * kE + i] = acc;
    }
  __syncthreads();
}

// x = rms_norm(x) * weight over kE values, in place, inside a glue block.
__device__ inline void NormW(float (*sh)[kT], float* x, const float* weight,
                             float eps) {
  const int t = threadIdx.x;
  float mine = 0.0f;
  for (int i = t; i < kE; i += kT)
    mine += x[i] * x[i];
  float total;
  BlockSum(sh, &mine, 1, &total);
  const float inv = 1.0f / sqrtf(total / kE + eps);
  for (int i = t; i < kE; i += kT)
    x[i] = x[i] * inv * weight[i];
  __syncthreads();
}

// One channel of the causal short convolution with its residual, for batch row
// `b`. The window's earlier inputs come from the previous rows of the same
// session's run when they are in this batch, otherwise from the history. The
// run's last row stores the new history.
__device__ inline float ConvRow(const float* values, int stride, int b,
                                const RowState& row, int which,
                                const float* weight, int channels, int c,
                                const float* taps, int per_head) {
  float x[kNC];
  for (int i = 0; i < kNC; ++i) {
    const int offset = row.run - (kNC - 1 - i);  // within the run; < 0: history
    x[i] = offset >= 0
               ? values[static_cast<std::size_t>(b - row.run + offset) * stride + c]
               : row.hist_in[which][(kNC - 1 + offset) * channels + c];
  }
  float out = 0.0f;
  for (int i = 0; i < kNC; ++i)
    out += x[i] * weight[c * kNC + i];
  if (taps) {
    const float* tap = taps + kNC * (c / per_head);
    for (int lag = 0; lag < kNC; ++lag)
      out += x[kNC - 1 - lag] * tap[lag];
  }
  if (row.last)
    for (int i = 0; i < kNC - 1; ++i)
      row.hist_out[which][i * channels + c] = x[i + 1];
  // 1 / sqrt(n_conv + 1) = 1 / sqrt(5), or 1 / sqrt(2 * n_conv + 1) = 1 / 3
  return (x[kNC - 1] + out) * (taps ? (1.0f / 3.0f) : 0.4472135955f);
}

__device__ inline float ClipFactor(float sum_sq, float threshold) {
  const float factor = threshold / (sqrtf(sum_sq) + 1e-6f);
  return fminf(fmaxf(factor, 0.0f), 1.0f);
}

// ---- kernels -------------------------------------------------------------

__launch_bounds__(kT) __global__ void EmbedKernel(EmbedArgs a) {
  __shared__ float sh[kSums][kT];
  const int t = threadIdx.x, b = blockIdx.x;
  const std::uint16_t* row =
      a.table + static_cast<std::size_t>(a.tokens[b]) * kE;
  float* cur = a.cur + static_cast<std::size_t>(b) * kE;
  float mine = 0.0f;
  for (int i = t; i < kE; i += kT) {
    const float x = Bf16(row[i]);
    mine += x * x;
  }
  float total;
  BlockSum(sh, &mine, 1, &total);
  const float inv = 1.0f / sqrtf(total / kE + a.eps);
  for (int i = t; i < kE; i += kT)
    cur[i] = Bf16(row[i]) * inv * a.norm[i];
}

__launch_bounds__(kT) __global__ void GlueAKernel(GlueAArgs a) {
  __shared__ float sh[kSums][kT];
  const int t = threadIdx.x;
  const std::size_t b = blockIdx.x;
  const float* cur = a.cur + b * kE;
  float* input = a.input + b * kE;
  float* in = a.in + b * 3 * kE;
  for (int i = t; i < kE; i += kT)
    input[i] = cur[i];

  if (a.n_res > 0) {
    const float* states[kMaxStates];
    for (int s = 0; s < a.n_blocks; ++s)
      states[s] = a.stack + (b * kMaxStates + s) * kE;
    states[a.n_blocks] = cur;
    Mix(sh, states, a.n_blocks + 1, a.res_query, a.n_res, in);
    if (a.n_res == 1) {
      for (int i = t; i < kE; i += kT) {
        in[kE + i] = in[i];
        in[2 * kE + i] = in[i];
      }
      __syncthreads();
    }
  } else {
    for (int i = t; i < kE; i += kT) {
      const float x = cur[i];
      in[i] = x;
      in[kE + i] = x;
      in[2 * kE + i] = x;
    }
    __syncthreads();
  }

  if (a.looped) {
    const float* bank[2] = {a.bank0 + b * kE, a.bank_out + b * kE};
    Mix(sh, bank, 2, a.msg_query, 4, a.msgs + b * 4 * kE);
  }
}

// The loop controllers' rank-64 projection of [rms(cur), rms(msg)]: one block
// per (row, controller, rank). Each block also sums the squares it needs for
// the two norms, which costs little next to its 2 * kE-wide weight row.
constexpr int kCtrlThreads = 128;
__launch_bounds__(kCtrlThreads) __global__ void CtrlDownKernel(CtrlArgs a) {
  __shared__ float sh[4][kCtrlThreads];
  const int t = threadIdx.x;
  const int d = blockIdx.x % kRank;
  const int c = (blockIdx.x / kRank) % a.count;
  const std::size_t b = blockIdx.x / (kRank * a.count);
  const float* cur = a.cur + b * a.cur_stride + c * kE;
  const float* msg = a.msg + b * a.msg_stride + c * kE;
  const float* row = a.down[c] + static_cast<std::size_t>(d) * 2 * kE;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = t; i < kE; i += kCtrlThreads) {
    const float x = cur[i], y = msg[i];
    v[0] += x * x;
    v[1] += y * y;
    v[2] += row[i] * x;
    v[3] += row[kE + i] * y;
  }
  for (int k = 0; k < 4; ++k)
    sh[k][t] = v[k];
  __syncthreads();
  for (int step = kCtrlThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int k = 0; k < 4; ++k)
        sh[k][t] += sh[k][t + step];
    __syncthreads();
  }
  if (t == 0) {
    const float inv_cur = 1.0f / sqrtf(sh[0][0] / kE + 1e-6f);
    const float inv_msg = 1.0f / sqrtf(sh[1][0] / kE + 1e-6f);
    a.low[(b * 3 + c) * kRank + d] = sh[2][0] * inv_cur + sh[3][0] * inv_msg;
  }
}

// One block per (row, controller): the two norms, and low from the products.
__launch_bounds__(256) __global__ void CtrlCombineKernel(CtrlArgs a) {
  __shared__ float sh[2][256];
  const int t = threadIdx.x;
  const int c = blockIdx.x % a.count;
  const std::size_t b = blockIdx.x / a.count;
  const float* cur = a.cur + b * a.cur_stride + c * kE;
  const float* msg = a.msg + b * a.msg_stride + c * kE;
  float v[2] = {0.0f, 0.0f};
  for (int i = t; i < kE; i += 256) {
    const float x = cur[i], y = msg[i];
    v[0] += x * x;
    v[1] += y * y;
  }
  sh[0][t] = v[0];
  sh[1][t] = v[1];
  __syncthreads();
  for (int step = 128; step > 0; step >>= 1) {
    if (t < step) {
      sh[0][t] += sh[0][t + step];
      sh[1][t] += sh[1][t + step];
    }
    __syncthreads();
  }
  if (t < kRank) {
    const float inv_cur = 1.0f / sqrtf(sh[0][0] / kE + 1e-6f);
    const float inv_msg = 1.0f / sqrtf(sh[1][0] / kE + 1e-6f);
    const std::size_t at = (b * 3 + c) * kRank + t;
    a.low[at] = a.yc[at] * inv_cur + a.ym[at] * inv_msg;
  }
}

// cur = (1 - tanh(forget)) * cur + tanh(gain) * msg, one thread per value.
__global__ void CtrlApplyKernel(CtrlArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = idx % kE;
  const int c = (idx / kE) % a.count;
  const std::size_t b = idx / (static_cast<std::size_t>(kE) * a.count);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  float gain = 0.0f, forget = 0.0f;
  if (a.yg) {
    // tanh(v) = expm1(2 v) / (expm1(2 v) + 2), exact to rounding and several
    // times cheaper than tanhf here; beyond |v| = 10 it is 1 in F32.
    const float g = fminf(fmaxf(a.yg[(b * 3 + c) * kE + i], -10.0f), 10.0f);
    const float f = fminf(fmaxf(a.yf[(b * 3 + c) * kE + i], -10.0f), 10.0f);
    const float eg = expm1f(2.0f * g), ef = expm1f(2.0f * f);
    float* cur = a.cur + b * a.cur_stride + c * kE;
    const float msg = a.msg[b * a.msg_stride + c * kE + i];
    const float x = cur[i];
    cur[i] = x - ef / (ef + 2.0f) * x + eg / (eg + 2.0f) * msg;
    return;
  } else {
    const float* low = a.low + (b * 3 + c) * kRank;
    const float* gain_row = a.gain[c] + static_cast<std::size_t>(i) * kRank;
    const float* forget_row = a.forget[c] + static_cast<std::size_t>(i) * kRank;
    for (int r = 0; r < kRank; ++r) {
      gain += gain_row[r] * low[r];
      forget += forget_row[r] * low[r];
    }
  }
  float* cur = a.cur + b * a.cur_stride + c * kE;
  const float msg = a.msg[b * a.msg_stride + c * kE + i];
  const float x = cur[i];
  cur[i] = x - tanhf(forget) * x + tanhf(gain) * msg;
}

// One block per (row, vector).
constexpr int kNormThreads = 256;
__launch_bounds__(kNormThreads) __global__ void NormKernel(NormArgs a) {
  __shared__ float sh[kNormThreads];
  const int t = threadIdx.x;
  const int j = blockIdx.x % a.count;
  const std::size_t b = blockIdx.x / a.count;
  float* x = a.x + b * a.stride + j * kE;
  const float* w = a.w[j];
  float mine = 0.0f;
  for (int i = t; i < kE; i += kNormThreads)
    mine += x[i] * x[i];
  sh[t] = mine;
  Reduce(sh, kNormThreads);
  const float inv = 1.0f / sqrtf(sh[0] / kE + a.eps);
  for (int i = t; i < kE; i += kNormThreads)
    x[i] = x[i] * inv * w[i];
}

// One block per (output row, group of batch rows): a weight row is read once
// for up to kGemmGroup batch rows.
constexpr int kGemmThreads = 128;
__launch_bounds__(kGemmThreads) __global__ void GemmKernel(GemmArgs a) {
  __shared__ float sh[kGemmGroup][kGemmThreads];
  const int t = threadIdx.x;
  const int groups = (a.batch + kGemmGroup - 1) / kGemmGroup;
  const int group = blockIdx.x % groups;
  int r = blockIdx.x / groups, j = 0;
  while (r >= a.jobs[j].rows) {
    r -= a.jobs[j].rows;
    ++j;
  }
  const GemmJob job = a.jobs[j];
  const int b0 = group * kGemmGroup;
  const int nb = a.batch - b0 < kGemmGroup ? a.batch - b0 : kGemmGroup;
  const std::uint16_t* row = job.w + static_cast<std::size_t>(r) * job.cols;
  float acc[kGemmGroup];
  for (int b = 0; b < nb; ++b)
    acc[b] = 0.0f;
  if (job.xt) {
    // Column-major inputs: one contiguous read per column for all rows.
    const float* xt = job.xt + static_cast<std::size_t>(group) * job.xt_group_stride;
    for (int c = t; c < job.cols; c += kGemmThreads) {
      const float w = Bf16(row[c]);
      const float* column = xt + static_cast<std::size_t>(c) * kGemmGroup;
      for (int b = 0; b < nb; ++b)
        acc[b] += w * column[b];
    }
  } else {
    const float* x = job.x + static_cast<std::size_t>(b0) * job.x_stride;
    for (int c = t; c < job.cols; c += kGemmThreads) {
      const float w = Bf16(row[c]);
      for (int b = 0; b < nb; ++b)
        acc[b] += w * x[static_cast<std::size_t>(b) * job.x_stride + c];
    }
  }
  for (int b = 0; b < nb; ++b)
    sh[b][t] = acc[b];
  __syncthreads();
  for (int step = kGemmThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int b = 0; b < nb; ++b)
        sh[b][t] += sh[b][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int b = 0; b < nb; ++b)
      job.y[static_cast<std::size_t>(b0 + b) * job.y_stride + r] = sh[b][0];
}

// One thread per (row, column).
__global__ void ToHalfKernel(ToHalfArgs a) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t b = i / a.cols, c = i % a.cols;
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  float v = a.x[b * a.x_stride + c];
  v = fminf(fmaxf(v, -65504.0f), 65504.0f);
  const _Float16 h = static_cast<_Float16>(v);
  reinterpret_cast<_Float16*>(a.out)[i] = h;
  if (a.out_lo)
    reinterpret_cast<_Float16*>(a.out_lo)[i] =
        static_cast<_Float16>(v - static_cast<float>(h));
}

// The projections with the 16x16x16 FP16 matrix instruction, FP32 sums.
// A block is 8 waves; each wave owns 16 output rows of one job and WTS tiles
// of 16 batch rows, and walks the inputs 16 columns at a time. Lane l (and
// lane l + 16, which mirrors it) holds weight row l and batch row l of each
// tile; in the result, lane l holds output row l, and element e of lane
// l + 16 * h is batch row 2 * e + h.
using Half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

template <int WTS>
__launch_bounds__(256) __global__ void WmmaGemmKernel(GemmArgs a) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int sl = lane & 15, hi = lane >> 4;
  int tile = blockIdx.x * 8 + wave, j = 0;
  for (;;) {
    const int tiles = (a.jobs[j].rows + 15) >> 4;
    if (tile < tiles)
      break;
    tile -= tiles;
    if (++j == a.count)
      return;
  }
  const GemmJob job = a.jobs[j];
  const int r = tile * 16 + sl;
  const _Float16* w = reinterpret_cast<const _Float16*>(job.wh) +
                      static_cast<std::size_t>(r < job.rows ? r : job.rows - 1) *
                          job.cols;
  const int t0 = blockIdx.y * 16 * WTS;
  const _Float16* x[WTS];
#pragma unroll
  for (int g = 0; g < WTS; ++g) {
    const int t = t0 + g * 16 + sl;
    x[g] = reinterpret_cast<const _Float16*>(job.xh) +
           static_cast<std::size_t>(t < a.batch ? t : a.batch - 1) *
               job.xh_stride;
  }
  Float8 acc[WTS] = {};
  for (int k = 0; k < job.cols; k += 16) {
    Half16 wa;
    __builtin_memcpy(&wa, w + k, 32);
#pragma unroll
    for (int g = 0; g < WTS; ++g) {
      Half16 xb;
      __builtin_memcpy(&xb, x[g] + k, 32);
      acc[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa, acc[g]);
    }
  }
  if (r >= job.rows)
    return;
#pragma unroll
  for (int g = 0; g < WTS; ++g)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int t = t0 + g * 16 + 2 * e + hi;
      if (t < a.batch) {
        float* y = job.swap
                       ? job.y + static_cast<std::size_t>(r) * job.y_stride + t
                       : job.y + static_cast<std::size_t>(t) * job.y_stride + r;
        *y = job.add ? *y + acc[g][e] : acc[g][e];
      }
    }
}

// Which kernels read their operands straight from device memory instead of
// staging them in shared memory: bit 1 the projections' block kernel, bit 2
// the experts' inputs. INKLING2_WMMA_DIRECT, default 0: both were slower
// (see docs/BENCHMARKS.md).
int g_direct = 0;

// The same product for larger steps: a block owns 128 output rows of one job
// and 32 * WTS batch rows. It walks the inputs 64 columns at a time: all
// threads first copy that slice of the weights and of the inputs into shared
// memory, then each of the 8 waves multiplies 2 row tiles by WTS batch tiles
// out of shared memory. Every weight and input value is read from global
// memory once per block.
//
// Direct: the same tiles, read by each wave straight from device memory. No
// shared memory and no barrier, so many more blocks run at once; the cache
// in front of memory serves the waves that read the same tile.
template <int WTS, bool Direct>
__launch_bounds__(256) __global__ void WmmaGemmBlockKernel(GemmArgs a) {
  constexpr int BM = 128, BN = 32 * WTS, BK = INKLING2_BK;
  // Rows of 8 words padded to kStageRow, so that a wave's lanes read
  // different banks of shared memory.
  __shared__ std::uint32_t s_a[Direct ? 1 : BK][Direct ? 1 : BM][kStageRow];
  __shared__ std::uint32_t s_b[Direct ? 1 : BK][Direct ? 1 : BN][kStageRow];
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int sl = lane & 15, hi = lane >> 4;
  const int wr = wave >> 1, wt = wave & 1;
  int block = blockIdx.x, j = 0;
  for (;;) {
    const int blocks = (a.jobs[j].rows + BM - 1) / BM;
    if (block < blocks)
      break;
    block -= blocks;
    if (++j == a.count)
      return;
  }
  const GemmJob job = a.jobs[j];
  const int r_block = block * BM, t_block = blockIdx.y * BN;
  // The slices this thread copies: units of 16 values, 4 per row.
  constexpr int UA = BM * BK / 256;
  constexpr int UB = (BN * BK + 255) / 256;
  const std::uint32_t* w_src[UA];
  const std::uint32_t* x_src[UB];
#pragma unroll
  for (int p = 0; p < UA; ++p) {
    const int unit = p * 256 + tid, r = r_block + unit / BK;
    w_src[p] = reinterpret_cast<const std::uint32_t*>(
        job.wh + static_cast<std::size_t>(r < job.rows ? r : job.rows - 1) *
                     job.cols +
        (unit % BK) * 16);
  }
#pragma unroll
  for (int p = 0; p < UB; ++p) {
    const int unit = p * 256 + tid, t = t_block + unit / BK;
    x_src[p] = reinterpret_cast<const std::uint32_t*>(
        job.xh + static_cast<std::size_t>(t < a.batch ? t : a.batch - 1) *
                     job.xh_stride +
        (unit % BK) * 16);
  }
  Float8 acc[2][WTS] = {};
  if constexpr (Direct) {
    const std::uint16_t* w_lane[2];
    const std::uint16_t* x_lane[WTS];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = r_block + (wr * 2 + i) * 16 + sl;
      w_lane[i] = job.wh +
                  static_cast<std::size_t>(r < job.rows ? r : job.rows - 1) *
                      job.cols;
    }
#pragma unroll
    for (int g = 0; g < WTS; ++g) {
      const int t = t_block + (wt * WTS + g) * 16 + sl;
      x_lane[g] = job.xh +
                  static_cast<std::size_t>(t < a.batch ? t : a.batch - 1) *
                      job.xh_stride;
    }
    for (int k = 0; k < job.cols; k += 16) {
      Half16 wa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        __builtin_memcpy(&wa[i], w_lane[i] + k, 32);
#pragma unroll
      for (int g = 0; g < WTS; ++g) {
        Half16 xb;
        __builtin_memcpy(&xb, x_lane[g] + k, 32);
#pragma unroll
        for (int i = 0; i < 2; ++i)
          acc[i][g] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa[i], acc[i][g]);
      }
    }
  } else
  for (int k = 0; k < job.cols; k += 16 * BK) {
#pragma unroll
    for (int p = 0; p < UA; ++p) {
      const int unit = p * 256 + tid;
      const std::uint32_t* src = w_src[p] + k / 2;
      std::uint32_t* dst = s_a[unit % BK][unit / BK];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        dst[i] = src[i];
    }
#pragma unroll
    for (int p = 0; p < UB; ++p) {
      const int unit = p * 256 + tid;
      if (unit < BN * BK) {
        const std::uint32_t* src = x_src[p] + k / 2;
        std::uint32_t* dst = s_b[unit % BK][unit / BK];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = src[i];
      }
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < BK; ++ks) {
      Half16 wa[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        __builtin_memcpy(&wa[i], s_a[ks][(wr * 2 + i) * 16 + sl], 32);
#pragma unroll
      for (int g = 0; g < WTS; ++g) {
        Half16 xb;
        __builtin_memcpy(&xb, s_b[ks][(wt * WTS + g) * 16 + sl], 32);
#pragma unroll
        for (int i = 0; i < 2; ++i)
          acc[i][g] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa[i], acc[i][g]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = r_block + (wr * 2 + i) * 16 + sl;
    if (r >= job.rows)
      continue;
#pragma unroll
    for (int g = 0; g < WTS; ++g)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int t = t_block + (wt * WTS + g) * 16 + 2 * e + hi;
        if (t < a.batch) {
          float* y = job.swap
                         ? job.y + static_cast<std::size_t>(r) * job.y_stride + t
                         : job.y + static_cast<std::size_t>(t) * job.y_stride + r;
          *y = job.add ? *y + acc[i][g][e] : acc[i][g][e];
        }
      }
  }
}

__global__ void TransposeKernel(TransposeArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t b = idx / a.cols;
  const int c = static_cast<int>(idx % a.cols);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  const std::size_t group = b / kGemmGroup;
  a.xt[(group * a.cols + c) * kGemmGroup + b % kGemmGroup] =
      a.x[b * a.x_stride + c];
}

// The single-row case: one block of `Threads` threads per output row, no
// per-row bookkeeping.
template<int Threads>
__launch_bounds__(Threads) __global__ void GemvKernel(GemmArgs a) {
  __shared__ float shared[Threads];
  const int t = threadIdx.x;
  int r = blockIdx.x, j = 0;
  while (r >= a.jobs[j].rows) {
    r -= a.jobs[j].rows;
    ++j;
  }
  const int cols = a.jobs[j].cols;
  const std::uint16_t* row = a.jobs[j].w + static_cast<std::size_t>(r) * cols;
  const float* x = a.jobs[j].x;
  float acc = 0.0f;
  for (int c = t; c < cols; c += Threads)
    acc += Bf16(row[c]) * x[c];
  shared[t] = acc;
  Reduce(shared, Threads);
  if (t == 0)
    a.jobs[j].y[r] = shared[0];
}

__device__ __forceinline__ _Float16 ClampHalf(float v) {
  return static_cast<_Float16>(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// One block per row.
__launch_bounds__(kT) __global__ void GlueBKernel(GlueBArgs a) {
  // Group factors: Q heads 0..23, K 24..31, V 32..39, R 40..63.
  __shared__ float factor[64];
  __shared__ float k_new[kHK];
  __shared__ float v_new[kHK];
  const int t = threadIdx.x;
  const int b = blockIdx.x;
  const RowState row = a.rows[b];
  float* q = a.q + static_cast<std::size_t>(b) * kHQ;
  float* r = a.r + static_cast<std::size_t>(b) * kNR;
  const float* k_taps =
      a.k_taps ? a.k_taps + static_cast<std::size_t>(b) * kNC * kNKV : nullptr;
  const float* v_taps =
      a.v_taps ? a.v_taps + static_cast<std::size_t>(b) * kNC * kNKV : nullptr;
  for (int c = t; c < kHK; c += kT) {
    k_new[c] = ConvRow(a.k_raw, kHK, b, row, 0, a.k_conv, kHK, c, k_taps, kHD);
    v_new[c] = ConvRow(a.v_raw, kHK, b, row, 1, a.v_conv, kHK, c, v_taps, kHD);
  }
  __syncthreads();

  if (t < kNH) {
    float sum = 0.0f;
    for (int i = 0; i < kHD; ++i) {
      const float x = q[t * kHD + i] * a.q_clip[t * kHD + i];
      sum += x * x;
    }
    factor[t] = ClipFactor(sum, a.qk_clip) * row.tau;
  } else if (t < kNH + kNKV) {
    const int h = t - kNH;
    float sum = 0.0f;
    for (int i = 0; i < kHD; ++i)
      sum += k_new[h * kHD + i] * k_new[h * kHD + i];
    factor[t] = ClipFactor(sum, a.qk_clip);
  } else if (t < kNH + 2 * kNKV) {
    const int h = t - kNH - kNKV;
    float sum = 0.0f;
    for (int i = 0; i < kHD; ++i)
      sum += v_new[h * kHD + i] * v_new[h * kHD + i];
    factor[t] = 1.0f / sqrtf(sum / kHD + a.eps);
  } else if (t < 2 * kNH + 2 * kNKV) {
    const int h = t - kNH - 2 * kNKV;
    float sum = 0.0f;
    for (int i = 0; i < kNRel; ++i) {
      const float x = r[h * kNRel + i] * a.rel_clip[h * kNRel + i];
      sum += x * x;
    }
    factor[t] = ClipFactor(sum, a.rel_clip_threshold) * row.tau;
  }
  __syncthreads();

  for (int i = t; i < kHQ; i += kT)
    q[i] = q[i] * a.q_clip[i] * factor[i / kHD];
  if (row.layout == 1) {
    const int chunk = row.pos / kT, within = row.pos % kT;
    for (int c = t; c < kHK; c += kT) {
      const std::size_t head = c / kHD, dim = c % kHD;
      if (row.kv_k) {
        row.kv_k[((head * row.chunk_cap + chunk) * kHD + dim) * kT + within] =
            k_new[c] * factor[kNH + c / kHD];
        row.kv_v[(head * row.cap + row.pos) * kHD + dim] =
            v_new[c] * factor[kNH + kNKV + c / kHD];
      }
    }
    if (row.kh) {
      _Float16* kh = reinterpret_cast<_Float16*>(row.kh);
      _Float16* vh = reinterpret_cast<_Float16*>(row.vh);
      for (int c = t; c < kHK; c += kT) {
        const std::size_t head = c / kHD, dim = c % kHD;
        kh[(head * row.cap + row.pos) * kHD + dim] =
            ClampHalf(k_new[c] * factor[kNH + c / kHD]);
        vh[((head * row.chunk_cap + chunk) * kHD + dim) * kT + within] =
            ClampHalf(v_new[c] * factor[kNH + kNKV + c / kHD]);
      }
    }
  } else {
    const std::size_t ring = row.pos % row.cap;
    if (row.kv_k) {
      float* k_row = row.kv_k + ring * kHK;
      float* v_row = row.kv_v + ring * kHK;
      for (int c = t; c < kHK; c += kT) {
        k_row[c] = k_new[c] * factor[kNH + c / kHD];
        v_row[c] = v_new[c] * factor[kNH + kNKV + c / kHD];
      }
    }
    if (row.kh) {
      _Float16* kh = reinterpret_cast<_Float16*>(row.kh) + ring * kHK;
      _Float16* vh = reinterpret_cast<_Float16*>(row.vh) + ring;
      for (int c = t; c < kHK; c += kT) {
        kh[c] = ClampHalf(k_new[c] * factor[kNH + c / kHD]);
        vh[static_cast<std::size_t>(c) * row.cap] =
            ClampHalf(v_new[c] * factor[kNH + kNKV + c / kHD]);
      }
    }
  }
  for (int i = t; i < kNR; i += kT)
    r[i] = r[i] * a.rel_clip[i] * factor[kNH + 2 * kNKV + i / kNRel];
}

// One block per (row, query head).
constexpr int kAttnThreads = 256;
__launch_bounds__(kAttnThreads) __global__ void AttentionKernel(AttnArgs a) {
  __shared__ float shared[kAttnThreads];
  const int t = threadIdx.x;
  const int h = blockIdx.x % kNH;
  const std::size_t b = blockIdx.x / kNH;
  const RowState row = a.rows[b];
  const int kv_head = h / (kNH / kNKV);
  const int n = row.pos - row.first + 1;
  const float* q = a.q + b * kHQ + h * kHD;
  const float* r = a.r + b * kNR + h * kNRel;
  float* logit = a.scratch + (b * kNH + h) * a.max_n;
  const float sink = a.sink[b * kNH + h];

  float max = 0.0f;  // the zero sink's logit
  for (int idx = t; idx < n; idx += kAttnThreads) {
    const int j = row.first + idx;
    const float* k =
        row.kv_k + static_cast<std::size_t>(j % row.cap) * kHK + kv_head * kHD;
    float dot = 0.0f;
    for (int i = 0; i < kHD; ++i)
      dot += q[i] * k[i];
    const int distance = row.pos - j;
    float bias = 0.0f;
    if (distance < a.extent) {
      const float* table =
          a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
      for (int x = 0; x < kNRel; ++x)
        bias += table[x] * r[x];
    }
    const float value = dot * a.scale + bias - sink;
    logit[idx] = value;
    max = fmaxf(max, value);
  }
  shared[t] = max;
  __syncthreads();
  for (int step = kAttnThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      shared[t] = fmaxf(shared[t], shared[t + step]);
    __syncthreads();
  }
  max = shared[0];
  __syncthreads();

  float sum = 0.0f;
  for (int idx = t; idx < n; idx += kAttnThreads) {
    const float e = __expf(logit[idx] - max);
    logit[idx] = e;
    sum += e;
  }
  shared[t] = sum;
  Reduce(shared, kAttnThreads);
  const float total = shared[0] + __expf(0.0f - max);

  if (t < kHD) {
    float acc = 0.0f;
    for (int idx = 0; idx < n; ++idx) {
      const int j = row.first + idx;
      acc += logit[idx] * row.kv_v[static_cast<std::size_t>(j % row.cap) * kHK +
                                   kv_head * kHD + t];
    }
    a.out[b * kHQ + h * kHD + t] =
        acc / total * Sigmoid(a.gate[b * kHQ + h * kHD + t]);
  }
}

// The same attention for rows that read at most kT positions, which is every
// sliding-window layer and the full-attention layers of short contexts. One
// block of kT threads per (row, query head): a thread per position for the
// scores, then the value sum split eight ways per output dimension, so no
// thread walks the whole context by itself.
__launch_bounds__(kT) __global__ void AttentionWindowKernel(AttnArgs a) {
  __shared__ float prob[kT];
  __shared__ float red[kT];
  const int t = threadIdx.x;
  const int h = blockIdx.x % kNH;
  const std::size_t b = blockIdx.x / kNH;
  const RowState row = a.rows[b];
  const int kv_head = h / (kNH / kNKV);
  const int n = row.pos - row.first + 1;
  // Cache rows of positions first, first + 1, ...: the ring wraps once at most.
  const int base = row.first % row.cap;

  float value = 0.0f;  // the zero sink's logit bounds the maximum from below
  if (t < n) {
    int kr = base + t;
    if (kr >= row.cap)
      kr -= row.cap;
    const float* q = a.q + b * kHQ + h * kHD;
    const float* k = row.kv_k + static_cast<std::size_t>(kr) * kHK + kv_head * kHD;
    float dot = 0.0f;
    for (int i = 0; i < kHD; ++i)
      dot += q[i] * k[i];
    const int distance = n - 1 - t;
    float bias = 0.0f;
    if (distance < a.extent) {
      const float* r = a.r + b * kNR + h * kNRel;
      const float* table =
          a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
      for (int x = 0; x < kNRel; ++x)
        bias += table[x] * r[x];
    }
    value = dot * a.scale + bias - a.sink[b * kNH + h];
    prob[t] = value;
  }
  red[t] = value > 0.0f ? value : 0.0f;
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      red[t] = fmaxf(red[t], red[t + step]);
    __syncthreads();
  }
  const float max = red[0];
  __syncthreads();

  float e = 0.0f;
  if (t < n) {
    e = __expf(prob[t] - max);
    prob[t] = e;
  }
  red[t] = e;
  Reduce(red, kT);
  const float total = red[0] + __expf(0.0f - max);
  __syncthreads();

  constexpr int kParts = kT / kHD;
  const int dim = t % kHD, part = t / kHD;
  const float* v = row.kv_v + kv_head * kHD + dim;
  float acc = 0.0f;
  for (int idx = part; idx < n; idx += kParts) {
    int kr = base + idx;
    if (kr >= row.cap)
      kr -= row.cap;
    acc += prob[idx] * v[static_cast<std::size_t>(kr) * kHK];
  }
  red[t] = acc;
  __syncthreads();
  if (t < kHD) {
    float sum = 0.0f;
    for (int p = 0; p < kParts; ++p)
      sum += red[t + p * kHD];
    a.out[b * kHQ + h * kHD + t] =
        sum / total * Sigmoid(a.gate[b * kHQ + h * kHD + t]);
  }
}

// ---- attention over long contexts ----------------------------------------

constexpr int kShare = kNH / kNKV;  // query heads per key/value head
constexpr float kLowest = -3.0e38f;

// One block per (row, key head, chunk of kT positions). For each of the query
// heads that share this key head: the scores of the chunk's positions, their
// maximum, the sum of exp(score - maximum), and the values weighted by
// exp(score - maximum). Keys and values are read once for all of them.
__launch_bounds__(kT) __global__ void AttentionChunkKernel(AttnArgs a) {
  __shared__ float prob[kShare][kT];
  __shared__ float red[kShare][kT];
  const int t = threadIdx.x;
  const int chunk = blockIdx.x % a.max_chunks;
  const int kv_head = (blockIdx.x / a.max_chunks) % kNKV;
  const std::size_t b = blockIdx.x / (a.max_chunks * kNKV);
  const RowState row = a.rows[b];
  const int n = row.pos - row.first + 1;
  if (chunk * kT >= n)
    return;  // this row has no positions here; the combine skips the chunk
  const int idx = chunk * kT + t;
  const bool live = idx < n;
  // Full-attention layers: positions start at 0 and the cache is laid out per
  // head (RowState::layout 1).
  const float* keys =
      row.kv_k +
      (static_cast<std::size_t>(kv_head) * row.chunk_cap + chunk) * kHD * kT;
  const float* values =
      row.kv_v + (static_cast<std::size_t>(kv_head) * row.cap + chunk * kT) * kHD;

  float value[kShare];
  for (int s = 0; s < kShare; ++s)
    value[s] = kLowest;
  if (live) {
    const float* q = a.q + b * kHQ + kv_head * kShare * kHD;
    float dot[kShare];
    for (int s = 0; s < kShare; ++s)
      dot[s] = 0.0f;
    for (int i = 0; i < kHD; ++i) {
      // Neighbouring threads read neighbouring positions of dimension i.
      const float key = keys[static_cast<std::size_t>(i) * kT + t];
      for (int s = 0; s < kShare; ++s)
        dot[s] += q[s * kHD + i] * key;
    }
    const int distance = n - 1 - idx;
    for (int s = 0; s < kShare; ++s) {
      const int h = kv_head * kShare + s;
      float bias = 0.0f;
      if (distance < a.extent) {
        const float* r = a.r + b * kNR + h * kNRel;
        const float* table =
            a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
        for (int x = 0; x < kNRel; ++x)
          bias += table[x] * r[x];
      }
      value[s] = dot[s] * a.scale + bias - a.sink[b * kNH + h];
    }
  }
  for (int s = 0; s < kShare; ++s) {
    prob[s][t] = value[s];
    red[s][t] = value[s];
  }
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int s = 0; s < kShare; ++s)
        red[s][t] = fmaxf(red[s][t], red[s][t + step]);
    __syncthreads();
  }
  float max[kShare];
  for (int s = 0; s < kShare; ++s)
    max[s] = red[s][0];
  __syncthreads();

  for (int s = 0; s < kShare; ++s) {
    const float e = live ? __expf(prob[s][t] - max[s]) : 0.0f;
    prob[s][t] = e;
    red[s][t] = e;
  }
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int s = 0; s < kShare; ++s)
        red[s][t] += red[s][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int s = 0; s < kShare; ++s) {
      const std::size_t slot =
          (b * kNH + kv_head * kShare + s) * a.max_chunks + chunk;
      a.chunk_max[slot] = max[s];
      a.chunk_sum[slot] = red[s][0];
    }
  __syncthreads();

  // The weighted values: eight threads per output dimension.
  constexpr int kParts = kT / kHD;
  const int dim = t % kHD, part = t / kHD;
  const int count = n - chunk * kT < kT ? n - chunk * kT : kT;
  float acc[kShare];
  for (int s = 0; s < kShare; ++s)
    acc[s] = 0.0f;
  for (int j = part; j < count; j += kParts) {
    const float v = values[static_cast<std::size_t>(j) * kHD + dim];
    for (int s = 0; s < kShare; ++s)
      acc[s] += prob[s][j] * v;
  }
  for (int s = 0; s < kShare; ++s)
    red[s][t] = acc[s];
  __syncthreads();
  if (t < kHD)
    for (int s = 0; s < kShare; ++s) {
      float sum = 0.0f;
      for (int p = 0; p < kParts; ++p)
        sum += red[s][t + p * kHD];
      a.chunk_out[((b * kNH + kv_head * kShare + s) * a.max_chunks + chunk) *
                      kHD +
                  t] = sum;
    }
}

// Eight consecutive FP16 values at a 16-byte boundary, in one load.
__device__ __forceinline__ void LoadHalf8(const _Float16* p, float* out) {
  using Half8 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
  Half8 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 16), 16);
#pragma unroll
  for (int e = 0; e < 8; ++e)
    out[e] = static_cast<float>(v[e]);
}

// AttentionChunkKernel on the FP16 cache, for both layouts: half the bytes
// per position. The queries, the scores, the weights and the sums are F32;
// only the cached keys and values are FP16. On a sliding-window layer the
// one chunk is the row's window, read from the ring.
__launch_bounds__(kT) __global__ void AttentionChunkHalfKernel(AttnArgs a) {
  __shared__ float prob[kShare][kT];
  __shared__ float red[kShare][kT];
  const int t = threadIdx.x;
  const int chunk = blockIdx.x % a.max_chunks;
  const int kv_head = (blockIdx.x / a.max_chunks) % kNKV;
  const std::size_t b = blockIdx.x / (a.max_chunks * kNKV);
  const RowState row = a.rows[b];
  const int n = row.pos - row.first + 1;
  if (chunk * kT >= n)
    return;
  const bool ring = row.layout == 0;
  const int idx = chunk * kT + t;
  const bool live = idx < n;

  float value[kShare];
  for (int s = 0; s < kShare; ++s)
    value[s] = kLowest;
  if (live) {
    const std::size_t entry = ring ? (row.first + idx) % row.cap : idx;
    const _Float16* key = reinterpret_cast<const _Float16*>(
        ring ? row.kh + (entry * kNKV + kv_head) * kHD
             : row.kh + (static_cast<std::size_t>(kv_head) * row.cap + entry) * kHD);
    const float* q = a.q + b * kHQ + kv_head * kShare * kHD;
    float dot[kShare];
    for (int s = 0; s < kShare; ++s)
      dot[s] = 0.0f;
    // A position's 128 key values are contiguous: 16 bytes per load.
    for (int i = 0; i < (a.skip == 2 ? 0 : kHD); i += 8) {
      float k[8];
      LoadHalf8(key + i, k);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        for (int s = 0; s < kShare; ++s)
          dot[s] += q[s * kHD + i + e] * k[e];
    }
    const int distance = n - 1 - idx;
    for (int s = 0; s < kShare; ++s) {
      const int h = kv_head * kShare + s;
      float bias = 0.0f;
      if (distance < a.extent) {
        const float* r = a.r + b * kNR + h * kNRel;
        const float* table =
            a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
        for (int x = 0; x < kNRel; ++x)
          bias += table[x] * r[x];
      }
      value[s] = dot[s] * a.scale + bias - a.sink[b * kNH + h];
    }
  }
  for (int s = 0; s < kShare; ++s) {
    prob[s][t] = value[s];
    red[s][t] = value[s];
  }
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int s = 0; s < kShare; ++s)
        red[s][t] = fmaxf(red[s][t], red[s][t + step]);
    __syncthreads();
  }
  float max[kShare];
  for (int s = 0; s < kShare; ++s)
    max[s] = red[s][0];
  __syncthreads();

  for (int s = 0; s < kShare; ++s) {
    const float e = live ? __expf(prob[s][t] - max[s]) : 0.0f;
    prob[s][t] = e;
    red[s][t] = e;
  }
  __syncthreads();
  for (int step = kT / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int s = 0; s < kShare; ++s)
        red[s][t] += red[s][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int s = 0; s < kShare; ++s) {
      const std::size_t slot =
          (b * kNH + kv_head * kShare + s) * a.max_chunks + chunk;
      a.chunk_max[slot] = max[s];
      a.chunk_sum[slot] = red[s][0];
    }
  __syncthreads();

  // The weighted values: eight threads per output dimension, each over an
  // eighth of the chunk's positions, which are contiguous for a dimension
  // (16 bytes per load where the entries are aligned).
  constexpr int kParts = kT / kHD, kSpan = kT / kParts;
  const int dim = t % kHD, part = t / kHD;
  const int count = n - chunk * kT < kT ? n - chunk * kT : kT;
  const _Float16* values = reinterpret_cast<const _Float16*>(
      ring ? row.vh + (static_cast<std::size_t>(kv_head) * kHD + dim) * row.cap
           : row.vh + ((static_cast<std::size_t>(kv_head) * row.chunk_cap + chunk) *
                           kHD +
                       dim) *
                          kT);
  const int end = (part + 1) * kSpan < count ? (part + 1) * kSpan : count;
  float acc[kShare];
  for (int s = 0; s < kShare; ++s)
    acc[s] = 0.0f;
  for (int j = part * kSpan; j < (a.skip == 1 ? 0 : end);) {
    const std::size_t entry = ring ? (row.first + j) % row.cap : j;
    if ((entry & 7) == 0 && j + 8 <= end) {
      float v[8];
      LoadHalf8(values + entry, v);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        for (int s = 0; s < kShare; ++s)
          acc[s] += prob[s][j + e] * v[e];
      j += 8;
    } else {
      const float v = static_cast<float>(values[entry]);
      for (int s = 0; s < kShare; ++s)
        acc[s] += prob[s][j] * v;
      ++j;
    }
  }
  for (int s = 0; s < kShare; ++s)
    red[s][t] = acc[s];
  __syncthreads();
  if (t < kHD)
    for (int s = 0; s < kShare; ++s) {
      float sum = 0.0f;
      for (int p = 0; p < kParts; ++p)
        sum += red[s][t + p * kHD];
      a.chunk_out[((b * kNH + kv_head * kShare + s) * a.max_chunks + chunk) *
                      kHD +
                  t] = sum;
    }
}

// ---- the same per chunk, for many rows of one session at once ---------------
//
// One block per (group of up to 32 consecutive rows, key head, chunk of 1,024
// positions): 96 queries. The block walks the chunk 64 positions at a time:
//   1. all threads copy the slice's keys into shared memory;
//   2. scores = Q K^T with the matrix instruction (a wave owns one tile of 16
//      positions and three tiles of 16 queries), then the relative bias, the
//      sink and each row's causal limit;
//   3. the softmax is kept online: per query a running maximum M; the slice's
//      weights exp(score - M) go to shared memory as FP16, and what was summed
//      before is rescaled by exp(M_old - M). Meanwhile all threads copy the
//      slice's values into shared memory;
//   4. out += P V with the matrix instruction (a wave owns 16 output
//      dimensions for all six query tiles).
// It leaves the same per-chunk maximum, sum and weighted values as
// AttentionChunkKernel, for AttentionCombineKernel.
//
// On sliding-window layers the chunks start at the group's first window
// position rounded down to 64 (two chunks cover any group), a slice is 64
// consecutive ring entries, and each row also masks what is before its window.
//
// Weights are stored times 2^15 so that FP16 keeps 11 significant bits down to
// 2e-9 of the running maximum and only drops what is below 1e-12 of it.
constexpr int kAQ = kAttnRows * kShare;  // queries of a block
constexpr int kAP = 64;                  // positions of a slice
constexpr float kProbScale = 32768.0f;

//
// FarOnly is the same kernel for the parts of a context that lie wholly far
// before the group's rows (with a.fast & 4 the general kernel leaves those
// to it). It keeps neither queries nor keys and values in shared memory, so
// its blocks are a third of the size and several run on a compute unit at
// once; the general kernel's 60 KB let only one run.
// QDirect reads the queries from device memory too, which brings the general
// kernel under half of a compute unit's shared memory (a.fast & 8).
template <bool FarOnly, bool QDirect>
__launch_bounds__(256) __global__ void AttentionBatchKernel(AttnArgs a) {
  __shared__ std::uint32_t s_q[QDirect ? 1 : kAQ][kHD / 2];
  __shared__ std::uint32_t s_kv[FarOnly ? 8 : kAP * kHD / 2];
  __shared__ _Float16 s_p[kAQ][kAP];
  __shared__ float s_red[8][kAQ];
    __shared__ float s_f[kAQ];
  __shared__ int s_flag[2];
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int sl = lane & 15, hi = lane >> 4;
  // Blocks that run at the same time should read the same keys and values,
  // so that the cache is fetched from memory once for all of them: the group
  // is the fastest-changing part of the block index (a.fast & 2; otherwise
  // the order of before, group-major).
  const bool shared_order = (a.fast & 2) != 0;
  const int part = shared_order ? blockIdx.x / (a.group_count * kNKV)
                                : blockIdx.x % a.max_chunks;
  const int span = a.span > 0 ? a.span : 1;
  const int kv_head = shared_order ? (blockIdx.x / a.group_count) % kNKV
                                   : (blockIdx.x / a.max_chunks) % kNKV;
  const AttnGroup group =
      a.groups[shared_order ? blockIdx.x % a.group_count
                            : blockIdx.x / (a.max_chunks * kNKV)];
  const RowState row0 = a.rows[group.row0];
  // The group's rows are at consecutive positions pos0, pos0 + 1, ...
  const int pos0 = row0.pos, pos_last = pos0 + group.count - 1;
  const bool ring = row0.layout == 0;
  const int origin = ring ? row0.first / kAP * kAP : 0;
  if (!ring && part * span * kT > pos_last)
    return;  // the combine does not look at this part
  const bool part_far =
      !ring && (a.fast & 1) != 0 &&
      pos0 - ((part + 1) * span * kT - 1) >= a.extent;
  if (FarOnly ? !part_far : (part_far && (a.fast & 4) != 0))
    return;  // the other kernel's part

  // Queries: index s * 32 + row for query head kv_head * 3 + s.
  if constexpr (!QDirect)
  for (int p = 0; p < kAQ * (kHD / 2) / 256; ++p) {
    const int unit = p * 256 + tid, qi = unit / (kHD / 2), dw = unit % (kHD / 2);
    const int s = qi / kAttnRows, row = qi % kAttnRows;
    std::uint32_t v = 0;
    if (row < group.count)
      v = reinterpret_cast<const std::uint32_t*>(
          a.qh + static_cast<std::size_t>(group.row0 + row) * kHQ +
          (kv_head * kShare + s) * kHD)[dw];
    s_q[qi][dw] = v;
  }

  // Score role: position tile pt, query tiles qt0 .. qt0 + 2; this lane's
  // queries are entry sl of each.
  const int pt = wave & 3, qt0 = (wave >> 2) * 3;
  const int red_slot = pt * 2 + hi;
  int q_index[3], q_pos[3];
  float q_sink[3], run_max[3];
  const float* q_rel[3];
  const std::uint16_t* q_ptr[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int qt = qt0 + i, s = qt >> 1, row = (qt & 1) * 16 + sl;
    const int h = kv_head * kShare + s;
    q_ptr[i] = a.qh +
               static_cast<std::size_t>(group.row0 +
                                        (row < group.count ? row : 0)) *
                   kHQ +
               h * kHD;
    q_index[i] = qt * 16 + sl;
    q_pos[i] = row < group.count ? pos0 + row : -1;  // -1: no such row
    const std::size_t b = group.row0 + (row < group.count ? row : 0);
    q_sink[i] = a.sink[b * kNH + h];
    q_rel[i] = a.r + b * kNR + h * kNRel;
    run_max[i] = kLowest;
  }
  // Keys of entry i (a position, or a ring entry): keys + i * key_stride.
  // Values of dimension d at entry i: values + d * value_stride + i.
  const std::uint16_t* keys =
      ring ? row0.kh + kv_head * kHD
           : row0.kh + static_cast<std::size_t>(kv_head) * row0.cap * kHD;
  const std::size_t key_stride = ring ? kHK : kHD;
  const std::size_t value_stride = ring ? row0.cap : kT;
  Float8 out[2 * kShare] = {};
  // Each lane's share of its queries' weight sums; combined at the end.
  float lane_sum[3] = {0.0f, 0.0f, 0.0f};
  if (tid == 0) {
    s_flag[0] = 0;
    s_flag[1] = 0;
  }
  __syncthreads();

  int slice = 0;
  for (int chunk = part * span; chunk < (part + 1) * span; ++chunk) {
  const int start = origin + chunk * kT;
  if (start > pos_last)
    break;
  const std::uint16_t* values =
      ring ? row0.vh + static_cast<std::size_t>(kv_head) * kHD * row0.cap
           : row0.vh + (static_cast<std::size_t>(kv_head) * row0.chunk_cap + chunk) *
                           kHD * kT;
  for (int base = 0; base < kT && start + base <= pos_last;
       base += kAP, ++slice) {
    // The slice's first entry in the key and in the value arrays.
    const std::size_t key_entry = ring ? (start + base) % row0.cap : start + base;
    const std::size_t value_entry = ring ? key_entry : base;
    // A far slice lies at least `extent` positions before the group's first
    // row: no row's causal limit and no relative bias reaches it. Almost
    // every slice of a deep context is far, and for those the block does
    // only what the arithmetic needs: keys and values go from the cache
    // straight into the matrix instruction, and the running maximum is only
    // recomputed when some score exceeds it (s_flag), which after the first
    // slices is rare.
    const bool far = FarOnly || ((a.fast & 1) != 0 && !ring &&
                                 pos0 - (start + base + kAP - 1) >= a.extent);
    const int flag_slot = slice & 1;
    if (!far) {
      // 1. Keys: s_kv[(position * 8 + step) * 8 ..] = 16 dimensions.
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int unit = p * 256 + tid;
        const std::uint32_t* src = reinterpret_cast<const std::uint32_t*>(
            keys + (key_entry + unit / 8) * key_stride + (unit % 8) * 16);
        std::uint32_t* dst = s_kv + unit * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = src[i];
      }
      __syncthreads();
    }

    // 2. Scores.
    Float8 acc[3] = {};
    const std::uint16_t* far_keys =
        keys + (key_entry + pt * 16 + sl) * key_stride;
#pragma unroll
    for (int ks = 0; ks < kHD / 16; ++ks) {
      // a.skip bits 4, 8, 16, 32, 64 are timing experiments on far slices:
      // no score products, no value products, no exponentials, no key
      // reads, no value reads.
      Half16 xb = {};
      if (far) {
        if (!(a.skip & 32))
          __builtin_memcpy(&xb, far_keys + ks * 16, 32);
      } else {
        __builtin_memcpy(&xb, s_kv + ((pt * 16 + sl) * 8 + ks) * 8, 32);
      }
      if (far && (a.skip & 4))
        continue;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        Half16 wa = {};
        if constexpr (QDirect) {
          if (!(far && (a.skip & 128)))  // timing: no query reads
            __builtin_memcpy(&wa, q_ptr[i] + ks * 16, 32);
        }
        else
          __builtin_memcpy(&wa, &s_q[q_index[i]][ks * 8], 32);
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa, acc[i]);
      }
    }
    float val[3][8];
    bool exceeds = false;
    if (far) {
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float local = kLowest;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float v = acc[i][e] * a.scale - q_sink[i];
          val[i][e] = v;
          local = fmaxf(local, v);
        }
        s_red[red_slot][q_index[i]] = local;
        exceeds = exceeds || local > run_max[i];
      }
      if (exceeds)
        s_flag[flag_slot] = 1;
    } else {
      const int j0 = start + base + pt * 16 + hi;  // position of element 0
      // Whether any row of the group is within the bias extent of this slice.
      const bool near = pos0 - (start + base + kAP - 1) < a.extent;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        float local = kLowest;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = j0 + 2 * e;
          float v = kLowest;
          // A window is the last `extent` positions.
          if (j <= q_pos[i] && (!ring || q_pos[i] - j < a.extent)) {
            v = acc[i][e] * a.scale - q_sink[i];
            const int distance = q_pos[i] - j;
            if (near && distance < a.extent) {
              const float* table =
                  a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
              float bias = 0.0f;
              for (int x = 0; x < kNRel; ++x)
                bias += table[x] * q_rel[i][x];
              v += bias;
            }
          }
          val[i][e] = v;
          local = fmaxf(local, v);
        }
        s_red[red_slot][q_index[i]] = local;
      }
    }
    __syncthreads();

    // 3. Weights (and, for a slice that is not far, the values:
    // s_kv[(tile * 128 + dim) * 8 ..] = 16 positions).
    const bool update = !far || s_flag[flag_slot] != 0;
    if (tid == 0)
      s_flag[flag_slot ^ 1] = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      float m = run_max[i];
      if (update) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
          m = fmaxf(m, s_red[k][q_index[i]]);
        const float factor = __expf(run_max[i] - m);
        run_max[i] = m;
        if (red_slot == 0)
          s_f[q_index[i]] = factor;
        lane_sum[i] *= factor;
      }
      float part = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p =
            far && (a.skip & 16)
                ? val[i][e] - m + kProbScale
                : val[i][e] > 0.5f * kLowest
                      ? __expf(val[i][e] - m) * kProbScale
                      : 0.0f;
        const _Float16 h = static_cast<_Float16>(p);
        s_p[q_index[i]][pt * 16 + 2 * e + hi] = h;
        part += static_cast<float>(h);
      }
      lane_sum[i] += part;
    }
    if (!far) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int unit = p * 256 + tid, dim = unit / 4, tile = unit % 4;
        const std::uint32_t* src = reinterpret_cast<const std::uint32_t*>(
            values + dim * value_stride + value_entry + tile * 16);
        std::uint32_t* dst = s_kv + (tile * kHD + dim) * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = src[i];
      }
    }
    __syncthreads();

    // 4. Weighted values; this wave owns dimensions wave * 16 .. + 15.
    if (update) {
#pragma unroll
      for (int qt = 0; qt < 2 * kShare; ++qt)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          out[qt][e] *= s_f[qt * 16 + 2 * e + hi];
    }
    const std::uint16_t* far_values =
        values + (wave * 16 + sl) * value_stride + value_entry;
#pragma unroll
    for (int tile = 0; tile < kAP / 16; ++tile) {
      Half16 wa = {};
      if (far) {
        if (!(a.skip & 64))
          __builtin_memcpy(&wa, far_values + tile * 16, 32);
      } else {
        __builtin_memcpy(&wa, s_kv + (tile * kHD + wave * 16 + sl) * 8, 32);
      }
      if (far && (a.skip & 8))
        continue;
#pragma unroll
      for (int qt = 0; qt < 2 * kShare; ++qt) {
        Half16 xb = {};
        if (!(far && (a.skip & 256)))  // timing: no weight reads
          __builtin_memcpy(&xb, &s_p[qt * 16 + sl][tile * 16], 32);
        out[qt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa, out[qt]);
      }
    }
    // A slice that is not far reuses s_kv; a far one only needs the two
    // barriers above (the next slice's weights wait for its first one).
    if (!far)
      __syncthreads();
  }
  }

  constexpr float kUnscale = 1.0f / kProbScale;
#pragma unroll
  for (int qt = 0; qt < 2 * kShare; ++qt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = (qt & 1) * 16 + 2 * e + hi;
      if (row >= group.count)
        continue;
      const std::size_t slot =
          (static_cast<std::size_t>(group.row0 + row) * kNH + kv_head * kShare +
           (qt >> 1)) *
              a.max_chunks +
          part;
      a.chunk_out[slot * kHD + wave * 16 + sl] = out[qt][e] * kUnscale;
    }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 3; ++i)
    s_red[red_slot][q_index[i]] = lane_sum[i];
  __syncthreads();
  if (red_slot == 0)
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (q_pos[i] < 0)
        continue;
      float sum = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        sum += s_red[k][q_index[i]];
      const int qt = qt0 + i, row = (qt & 1) * 16 + sl;
      const std::size_t slot =
          (static_cast<std::size_t>(group.row0 + row) * kNH + kv_head * kShare +
           (qt >> 1)) *
              a.max_chunks +
          part;
      a.chunk_max[slot] = run_max[i];
      a.chunk_sum[slot] = sum * kUnscale;
    }
}

// ---- one pass over the keys, softmax and weights in registers ----------------
//
// The design of the prefill attention kernel in headbouyJB/inkling2-gufo
// (lab/attn/attn_kernels.h, MIT), written for this engine's caches and
// arithmetic. One block per (group of up to 64 consecutive rows, key head);
// each of its 12 waves owns 16 rows of one query head and keeps, in
// registers, those queries, their running softmax and their output. The
// block walks the keys 64 at a time (keys and values staged once in shared
// memory); per 16 keys a wave computes scores with the matrix instruction,
// updates its maximum and sum, exchanges the weights between its lane pairs
// to form the operand of the second product, and accumulates the weighted
// values. No weights go through shared memory, nothing is left for a second
// kernel, and a row's whole context is one stream.
//
// Arithmetic as in the kernels above: scores from FP16 queries and keys with
// FP32 sums, times `scale`, minus the row's sink; the relative bias in FP32
// within `extent`; weights as FP16 times 2^15; the zero sink as the initial
// state (maximum 0, sum 1); the output gate at the end.
constexpr int kFlashPad = INKLING2_FLASH_PAD;  // see s_k below
constexpr int kFlashKeys = 64;  // keys staged at once

__launch_bounds__(384) __global__ void AttentionFlashKernel(AttnArgs a) {
  // Rows are padded by 16 bytes: with a stride that is a power of two, the
  // 16 lanes of a wave would read the same bank of shared memory at once.
  __shared__ __attribute__((aligned(16))) std::uint16_t s_k[kFlashKeys][kHD + kFlashPad];
  __shared__ __attribute__((aligned(16))) std::uint16_t s_v[kHD][kFlashKeys + kFlashPad];
  constexpr int kThreads = 384;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int q16 = lane & 15, half = lane >> 4;
  const int hh = wave % kShare, tile_of_rows = wave / kShare;
  const int kv_head = blockIdx.x / a.group_count;
  const AttnGroup group = a.groups[blockIdx.x % a.group_count];
  const RowState row0 = a.rows[group.row0];
  const bool ring = row0.layout == 0;
  const int h = kv_head * kShare + hh;
  const int row = tile_of_rows * 16 + q16;
  const bool valid = row < group.count;
  const std::size_t b = group.row0 + (valid ? row : 0);
  const int pos0 = row0.pos, pos_last = pos0 + group.count - 1;
  const int q_pos = valid ? pos0 + row : -1;
  // This wave's rows span positions wave_min .. wave_max.
  const int wave_min = pos0 + tile_of_rows * 16, wave_max = wave_min + 15;

  Half16 q[kHD / 16];
#pragma unroll
  for (int s = 0; s < kHD / 16; ++s)
    __builtin_memcpy(&q[s], a.qh + b * kHQ + h * kHD + s * 16, 32);
  const float* q_rel = a.r + b * kNR + h * kNRel;
  const float sink = a.sink[b * kNH + h];
  float run_max = 0.0f;                           // the zero sink's logit
  float run_sum = half == 0 ? kProbScale : 0.0f;  // and its weight, once
  Float8 out[kHD / 16] = {};

  const int first = ring ? (row0.first / kFlashKeys) * kFlashKeys : 0;
  const int tiles = (pos_last - first) / kFlashKeys + 1;
  for (int tile = 0; tile < tiles; ++tile) {
    const int base = first + tile * kFlashKeys;
    __syncthreads();
    // Keys: 16 bytes at a time, 16 pieces per key.
    for (int unit = tid; unit < kFlashKeys * 16; unit += kThreads) {
      const int key = unit >> 4, piece = unit & 15;
      const std::size_t entry = ring ? (base + key) % row0.cap : base + key;
      const std::uint16_t* src =
          ring ? row0.kh + (entry * kNKV + kv_head) * kHD
               : row0.kh + (static_cast<std::size_t>(kv_head) * row0.cap + entry) * kHD;
      __builtin_memcpy(&s_k[key][piece * 8], src + piece * 8, 16);
    }
    // Values: a dimension's 64 positions are contiguous in the cache.
    {
      const std::size_t entry = ring ? base % row0.cap : base % kT;
      const std::uint16_t* values =
          ring ? row0.vh + static_cast<std::size_t>(kv_head) * kHD * row0.cap
               : row0.vh + (static_cast<std::size_t>(kv_head) * row0.chunk_cap +
                            base / kT) *
                               kHD * kT;
      const std::size_t stride = ring ? row0.cap : kT;
      for (int unit = tid; unit < kHD * 8; unit += kThreads) {
        const int dim = unit >> 3, piece = unit & 7;
        __builtin_memcpy(&s_v[dim][piece * 8],
                         values + dim * stride + entry + piece * 8, 16);
      }
    }
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < kFlashKeys / 16; ++kt) {
      const int key0 = base + kt * 16;
      if (key0 > wave_max)
        continue;  // after every row of this wave
      if (ring && wave_min - (key0 + 15) >= a.extent)
        continue;  // before every row's window
      Float8 c = {};
#pragma unroll
      for (int s = 0; s < kHD / 16; ++s) {
        Half16 kf;
        __builtin_memcpy(&kf, &s_k[kt * 16 + q16][s * 16], 32);
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, q[s], c);
      }
      // Lane (q16, half) holds its row's scores for keys key0 + 2 i + half.
      const bool near = valid && q_pos - (key0 + 15) < a.extent;
      const bool any_near = __any(near);
      float score[8], local = kLowest;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int j = key0 + 2 * i + half, distance = q_pos - j;
        float v = kLowest;
        if (distance >= 0 && (!ring || distance < a.extent)) {
          v = c[i] * a.scale - sink;
          if (any_near && distance < a.extent) {
            const float* table =
                a.rel_bias + static_cast<std::size_t>(distance) * kNRel;
            float bias = 0.0f;
            for (int x = 0; x < kNRel; ++x)
              bias += table[x] * q_rel[x];
            v += bias;
          }
        }
        score[i] = v;
        local = fmaxf(local, v);
      }
      local = fmaxf(local, __shfl_xor(local, 16, 32));
      const float m = fmaxf(run_max, local);
      const float factor = __expf(run_max - m);
      run_max = m;
      // The weights, as FP16 times 2^15, packed two to a word.
      std::uint32_t mine[4], theirs[4];
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        union {
          _Float16 f[2];
          std::uint32_t u;
        } pair;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float v = score[2 * i + e];
          pair.f[e] = static_cast<_Float16>(
              v > 0.5f * kLowest ? __expf(v - m) * kProbScale : 0.0f);
          part += static_cast<float>(pair.f[e]);
        }
        mine[i] = pair.u;
        theirs[i] = __shfl_xor(mine[i], 16, 32);
      }
      run_sum = run_sum * factor + part;
      // As an operand a lane needs its row's 16 keys in order; it has the
      // even ones (half 0) or the odd ones (half 1) and its pair the others.
      union {
        Half16 f;
        std::uint16_t u[16];
      } p;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint16_t m0 = mine[i] & 0xFFFFu, m1 = mine[i] >> 16;
        const std::uint16_t t0 = theirs[i] & 0xFFFFu, t1 = theirs[i] >> 16;
        p.u[4 * i + 0] = half ? t0 : m0;
        p.u[4 * i + 1] = half ? m0 : t0;
        p.u[4 * i + 2] = half ? t1 : m1;
        p.u[4 * i + 3] = half ? m1 : t1;
      }
#pragma unroll
      for (int dt = 0; dt < kHD / 16; ++dt) {
        out[dt] *= factor;
        Half16 vf;
        __builtin_memcpy(&vf, &s_v[dt * 16 + q16][kt * 16], 32);
        out[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf, p.f, out[dt]);
      }
    }
  }
  if (!valid)
    return;
  const float inverse = 1.0f / (run_sum + __shfl_xor(run_sum, 16, 32));
#pragma unroll
  for (int dt = 0; dt < kHD / 16; ++dt)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int dim = dt * 16 + 2 * i + half;
      const std::size_t at = b * kHQ + h * kHD + dim;
      a.out[at] = out[dt][i] * inverse * Sigmoid(a.gate[at]);
    }
}

// One block per (row, query head): the chunks combined against the overall
// maximum, with the zero sink and the output gate.
__launch_bounds__(kHD) __global__ void AttentionCombineKernel(AttnArgs a) {
  const int t = threadIdx.x;
  const int h = blockIdx.x % kNH;
  const std::size_t b = blockIdx.x / kNH;
  const RowState row = a.rows[b];
  const int n = row.pos - row.first + 1;
  // Sliding-window layers: the batched kernel wrote every chunk, with the
  // lowest maximum where a row has no position.
  const int per_part = kT * (a.span > 0 ? a.span : 1);
  const int chunks =
      row.layout == 0 ? a.max_chunks : (n + per_part - 1) / per_part;
  const std::size_t base = (b * kNH + h) * a.max_chunks;
  float max = 0.0f;  // the zero sink's logit
  for (int c = 0; c < chunks; ++c)
    max = fmaxf(max, a.chunk_max[base + c]);
  float total = __expf(0.0f - max);
  float acc = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    if (a.chunk_max[base + c] < 0.5f * kLowest)
      continue;
    const float weight = __expf(a.chunk_max[base + c] - max);
    total += a.chunk_sum[base + c] * weight;
    acc += a.chunk_out[(base + c) * kHD + t] * weight;
  }
  a.out[b * kHQ + h * kHD + t] =
      acc / total * Sigmoid(a.gate[b * kHQ + h * kHD + t]);
}

// One block per row.
__launch_bounds__(kT) __global__ void GlueCKernel(GlueCArgs a) {
  __shared__ float sh[kSums][kT];
  const int t = threadIdx.x;
  const int b = blockIdx.x;
  const std::size_t bz = b;
  const RowState row = a.rows[b];
  const float* input = a.input + bz * kE;
  float* cur = a.cur + bz * kE;
  float* stack = a.stack + bz * kMaxStates * kE;
  for (int i = t; i < kE; i += kT) {
    const float conv =
        ConvRow(a.proj, kE, b, row, 2, a.post_conv, kE, i, nullptr, 1);
    if (a.block_start) {
      // A block's first layer starts a new residual stream; the earlier one
      // joins the mixed blocks.
      stack[static_cast<std::size_t>(a.n_blocks) * kE + i] = input[i];
      cur[i] = conv;
    } else {
      cur[i] = input[i] + conv;
    }
  }
  __syncthreads();

  const int n = a.n_blocks + (a.block_start ? 1 : 0);
  const float* states[kMaxStates];
  for (int s = 0; s < n; ++s)
    states[s] = stack + static_cast<std::size_t>(s) * kE;
  states[n] = cur;
  Mix(sh, states, n + 1, a.ffn_res_query, 1, a.xn + bz * kE);
}

// The router's logits: one block per (row, expert).
constexpr int kRouterThreads = 64;
__launch_bounds__(kRouterThreads) __global__ void RouterLogitsKernel(
    RouterArgs a) {
  __shared__ float sh[kRouterThreads];
  const int t = threadIdx.x;
  const int expert = blockIdx.x % kNX;
  const std::size_t b = blockIdx.x / kNX;
  const float* row = a.gate_inp + static_cast<std::size_t>(expert) * kE;
  const float* x = a.x + b * kE;
  float acc = 0.0f;
  for (int i = t; i < kE; i += kRouterThreads)
    acc += row[i] * x[i];
  sh[t] = acc;
  Reduce(sh, kRouterThreads);
  if (t == 0)
    a.logits[b * kNX + expert] = sh[0];
}

// Top-8 of the 128 experts by sigmoid(logit) + bias, with the router weights.
// One block per row.
__launch_bounds__(kNX) __global__ void RouterSelectKernel(RouterArgs a) {
  __shared__ float prob[kNX];
  __shared__ float key[kNX];
  __shared__ float top_prob[kTopK];
  __shared__ int top_id[kTopK];
  const int t = threadIdx.x;
  const std::size_t b = blockIdx.x;
  prob[t] = Sigmoid(a.logits[b * kNX + t]);
  key[t] = prob[t] + (a.exp_probs_b ? a.exp_probs_b[t] : 0.0f);
  __syncthreads();
  // Rank by selection key, highest first, ties to the lower index.
  int rank = 0;
  for (int j = 0; j < kNX; ++j)
    rank += (key[j] > key[t] || (key[j] == key[t] && j < t)) ? 1 : 0;
  if (rank < kTopK) {
    top_prob[rank] = prob[t];
    top_id[rank] = t;
  }
  __syncthreads();
  if (t == 0) {
    float sum = 1e-20f;
    for (int i = 0; i < kTopK; ++i)
      sum += top_prob[i];
    for (int i = 0; i < kTopK; ++i) {
      const int id = top_id[i];
      a.ids[b * kTopK + i] = static_cast<std::uint32_t>(id);
      a.up_scale[b * kTopK + i] = a.up_scale_all ? a.up_scale_all[id] : 1.0f;
      a.coef[b * kTopK + i] = top_prob[i] / sum * a.weights_scale *
                              (a.down_scale_all ? a.down_scale_all[id] : 1.0f);
    }
  }
}

// One block per (row, expert, hidden unit): the gate row and the up row, then
// situ.
__launch_bounds__(64) __global__ void ExpertUpNvfp4Kernel(BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float gate[kThreads];
  __shared__ float up[kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = (blockIdx.x / kFF) % kTopK;
  const std::size_t b = blockIdx.x / (kFF * kTopK);
  constexpr int kBlocks = kE / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(a.ids[b * kTopK + e]) * 2 * kFF * kRowBytes;
  const float* x = a.x + b * kE;
  float g = 0.0f, u = 0.0f;
  for (int blk = t; blk < kBlocks; blk += kThreads) {
    const float* xb = x + blk * kNvfp4Values;
    g += DotNvfp4Block(base + k * kRowBytes + blk * kNvfp4Bytes, xb);
    u += DotNvfp4Block(base + (kFF + k) * kRowBytes + blk * kNvfp4Bytes, xb);
  }
  gate[t] = g;
  up[t] = u;
  Reduce(gate, kThreads);
  Reduce(up, kThreads);
  if (t == 0) {
    const float s = a.up_scale[b * kTopK + e];
    a.act[(b * kTopK + e) * kFF + k] =
        Situ(gate[0] * s, up[0] * s, a.situ_beta, a.situ_linear_beta);
  }
}

// One block per (row, output value): every selected expert's down row,
// weighted.
__launch_bounds__(32) __global__ void ExpertDownNvfp4Kernel(BatchExpertArgs a) {
  constexpr int kThreads = 32;
  __shared__ float shared[kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const std::size_t b = blockIdx.x / kE;
  constexpr int kBlocks = kFF / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  float acc = 0.0f;
  for (int blk = t; blk < kBlocks; blk += kThreads) {
    for (int e = 0; e < kTopK; ++e) {
      const std::uint8_t* row =
          a.down +
          (static_cast<std::size_t>(a.ids[b * kTopK + e]) * kE + i) * kRowBytes;
      acc += a.coef[b * kTopK + e] *
             DotNvfp4Block(row + blk * kNvfp4Bytes,
                           a.act + (b * kTopK + e) * kFF + blk * kNvfp4Values);
    }
  }
  shared[t] = acc;
  Reduce(shared, kThreads);
  if (t == 0)
    a.out[b * kE + i] = shared[0];
}

__launch_bounds__(64) __global__ void ExpertUpBf16Kernel(BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float gate[kThreads];
  __shared__ float up[kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = (blockIdx.x / kFF) % kTopK;
  const std::size_t b = blockIdx.x / (kFF * kTopK);
  const std::uint16_t* base =
      reinterpret_cast<const std::uint16_t*>(a.gate_up) +
      static_cast<std::size_t>(a.ids[b * kTopK + e]) * 2 * kFF * kE;
  const std::uint16_t* gate_row = base + static_cast<std::size_t>(k) * kE;
  const std::uint16_t* up_row = base + static_cast<std::size_t>(kFF + k) * kE;
  const float* x = a.x + b * kE;
  float g = 0.0f, u = 0.0f;
  for (int c = t; c < kE; c += kThreads) {
    g += Bf16(gate_row[c]) * x[c];
    u += Bf16(up_row[c]) * x[c];
  }
  gate[t] = g;
  up[t] = u;
  Reduce(gate, kThreads);
  Reduce(up, kThreads);
  if (t == 0) {
    const float s = a.up_scale[b * kTopK + e];
    a.act[(b * kTopK + e) * kFF + k] =
        Situ(gate[0] * s, up[0] * s, a.situ_beta, a.situ_linear_beta);
  }
}

__launch_bounds__(64) __global__ void ExpertDownBf16Kernel(BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float shared[kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const std::size_t b = blockIdx.x / kE;
  const std::uint16_t* down = reinterpret_cast<const std::uint16_t*>(a.down);
  float acc = 0.0f;
  for (int e = 0; e < kTopK; ++e) {
    const std::uint16_t* row =
        down + (static_cast<std::size_t>(a.ids[b * kTopK + e]) * kE + i) * kFF;
    const float* act = a.act + (b * kTopK + e) * kFF;
    float part = 0.0f;
    for (int c = t; c < kFF; c += kThreads)
      part += Bf16(row[c]) * act[c];
    acc += a.coef[b * kTopK + e] * part;
  }
  shared[t] = acc;
  Reduce(shared, kThreads);
  if (t == 0)
    a.out[b * kE + i] = shared[0];
}

// ---- expert-major variant --------------------------------------------------

// One 64-value NVFP4 block as F32, each value already times its sub-block
// scale.
__device__ inline void DecodeNvfp4(const std::uint8_t* block, float* out) {
  const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(block);
  const std::uint32_t scales = w[0];
  for (int s = 0; s < 4; ++s) {
    const float d = Ue4m3((scales >> (8 * s)) & 0xFFu);
    for (int half = 0; half < 2; ++half) {
      const std::uint32_t word = w[1 + 2 * s + half];
      for (int j = 0; j < 4; ++j) {
        const std::uint32_t q = (word >> (8 * j)) & 0xFFu;
        out[s * 16 + half * 4 + j] = Fp4(q & 15u) * d;
        out[s * 16 + half * 4 + j + 8] = Fp4(q >> 4) * d;
      }
    }
  }
}

// For each expert, the pairs (row * kTopK + slot) that selected it. One block,
// one thread per expert.
__launch_bounds__(kNX) __global__ void ExpertGroupKernel(BatchExpertArgs a) {
  const int e = threadIdx.x;
  const int pairs = a.batch * kTopK;
  std::int32_t* list = a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  int n = 0;
  for (int p = 0; p < pairs; ++p)
    if (a.ids[p] == static_cast<std::uint32_t>(e))
      list[n++] = p;
  a.group_count[e] = n;
}

constexpr int kMajorRows = 16;  // rows sharing one decode per pass

// One block per (expert, hidden unit): the gate row and the up row are decoded
// once, then applied to every row that selected the expert.
__launch_bounds__(64) __global__ void ExpertUpNvfp4MajorKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float sg[kMajorRows][kThreads];
  __shared__ float su[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = blockIdx.x / kFF;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  constexpr int kBlocks = kE / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(e) * 2 * kFF * kRowBytes;
  const bool active = t < kBlocks;
  float wg[kNvfp4Values], wu[kNvfp4Values];
  if (active) {
    DecodeNvfp4(base + k * kRowBytes + t * kNvfp4Bytes, wg);
    DecodeNvfp4(base + (kFF + k) * kRowBytes + t * kNvfp4Bytes, wu);
  }
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      float g = 0.0f, u = 0.0f;
      if (active) {
        const float* x = a.x +
                         static_cast<std::size_t>(list[r0 + r] / kTopK) * kE +
                         t * kNvfp4Values;
        for (int i = 0; i < kNvfp4Values; ++i) {
          g += wg[i] * x[i];
          u += wu[i] * x[i];
        }
      }
      sg[r][t] = g;
      su[r][t] = u;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r) {
          sg[r][t] += sg[r][t + step];
          su[r][t] += su[r][t + step];
        }
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        const float scale = a.up_scale[pair];
        a.act[static_cast<std::size_t>(pair) * kFF + k] = Situ(
            sg[r][0] * scale, su[r][0] * scale, a.situ_beta, a.situ_linear_beta);
      }
    __syncthreads();
  }
}

// One block per (expert, output value): the down row is decoded once, then
// applied to every row that selected the expert.
__launch_bounds__(32) __global__ void ExpertDownNvfp4MajorKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 32;
  __shared__ float sh[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const int e = blockIdx.x / kE;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  constexpr int kBlocks = kFF / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const bool active = t < kBlocks;
  float w[kNvfp4Values];
  if (active)
    DecodeNvfp4(a.down + (static_cast<std::size_t>(e) * kE + i) * kRowBytes +
                    t * kNvfp4Bytes,
                w);
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      float acc = 0.0f;
      if (active) {
        const float* act = a.act +
                           static_cast<std::size_t>(list[r0 + r]) * kFF +
                           t * kNvfp4Values;
        for (int j = 0; j < kNvfp4Values; ++j)
          acc += w[j] * act[j];
      }
      sh[r][t] = acc;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r)
          sh[r][t] += sh[r][t + step];
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        a.contrib[static_cast<std::size_t>(pair) * kE + i] =
            a.coef[pair] * sh[r][0];
      }
    __syncthreads();
  }
}

// ---- 8-bit integer variant of the NVFP4 expert-major kernels ---------------

// Four E2M1 codes (one per byte, 0..15) to four signed 8-bit values, doubled:
// magnitudes 0, 1, 2, 3, 4, 6, 8, 12 with bit 3 as the sign. No table: with
// a = bit 2 and b = the low two bits, the magnitude is b + a * (4 + b + 2t),
// where t is 1 when b is 3.
__device__ inline std::uint32_t DecodeCodes(std::uint32_t codes) {
  const std::uint32_t b = codes & 0x03030303u;
  const std::uint32_t a = (codes >> 2) & 0x01010101u;
  const std::uint32_t t = (b & (b >> 1)) & 0x01010101u;
  const std::uint32_t mag = b + ((0x04040404u + b + 2u * t) & (a * 0xFFu));
  const std::uint32_t sign = (codes >> 3) & 0x01010101u;
  // Per-byte negation, (mag ^ 0xFF) + 1, without carries between bytes.
  const std::uint32_t x = mag ^ (sign * 0xFFu);
  return ((x & 0x7F7F7F7Fu) + sign) ^ (x & 0x80808080u);
}

// a . b over four signed 8-bit values each, added to c.
__device__ inline int Dot4(std::uint32_t a, std::uint32_t b, int c) {
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true,
                                 static_cast<int>(b), c, false);
}

// One 64-value NVFP4 block as four sub-blocks of 16 signed 8-bit values (four
// per word, in element order) and each sub-block's scale. Exact.
__device__ inline void DecodeNvfp4Int(const std::uint8_t* block,
                                      std::uint32_t* q, float* d) {
  const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(block);
  const std::uint32_t scales = w[0];
  for (int s = 0; s < 4; ++s) {
    d[s] = Ue4m3((scales >> (8 * s)) & 0xFFu);
    for (int half = 0; half < 2; ++half) {
      const std::uint32_t word = w[1 + 2 * s + half];
      // Elements 0..7 are the low codes of the eight bytes, 8..15 the high.
      q[s * 4 + half] = DecodeCodes(word & 0x0F0F0F0Fu);
      q[s * 4 + 2 + half] = DecodeCodes((word >> 4) & 0x0F0F0F0Fu);
    }
  }
}

// Rounds vectors to 8 bits per block of 16 values: q = round(x * 127 / max|x|)
// and the scale max|x| / 127. One thread per block of 16.
struct QuantizeArgs {
  const float* x;
  int x_stride;
  int vectors;
  int length;  // values per vector, a multiple of 16
  std::uint32_t* q;  // length / 4 words per vector
  float* s;          // length / 16 scales per vector
};
__global__ void QuantizeKernel(QuantizeArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int per_vector = a.length / 16;
  const std::size_t v = idx / per_vector;
  const int blk = static_cast<int>(idx % per_vector);
  if (v >= static_cast<std::size_t>(a.vectors))
    return;
  const float* x = a.x + v * a.x_stride + blk * 16;
  float amax = 0.0f;
  for (int i = 0; i < 16; ++i)
    amax = fmaxf(amax, fabsf(x[i]));
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  std::uint32_t* q = a.q + (v * per_vector + blk) * 4;
  for (int w = 0; w < 4; ++w) {
    std::uint32_t word = 0;
    for (int j = 0; j < 4; ++j) {
      const int value = static_cast<int>(rintf(x[w * 4 + j] * inv));
      word |= (static_cast<std::uint32_t>(value) & 0xFFu) << (8 * j);
    }
    q[w] = word;
  }
  a.s[v * per_vector + blk] = amax * (1.0f / 127.0f);
}

__launch_bounds__(64) __global__ void ExpertUpNvfp4Int8Kernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float sg[kMajorRows][kThreads];
  __shared__ float su[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = blockIdx.x / kFF;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  constexpr int kBlocks = kE / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(e) * 2 * kFF * kRowBytes;
  const bool active = t < kBlocks;
  std::uint32_t qg[16], qu[16];
  float dg[4], du[4];
  if (active) {
    DecodeNvfp4Int(base + k * kRowBytes + t * kNvfp4Bytes, qg, dg);
    DecodeNvfp4Int(base + (kFF + k) * kRowBytes + t * kNvfp4Bytes, qu, du);
  }
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      float g = 0.0f, u = 0.0f;
      if (active) {
        const std::size_t b = list[r0 + r] / kTopK;
        // This thread's 64 inputs: 16 words and 4 scales.
        const std::uint32_t* xq = a.xq + (b * kE + t * kNvfp4Values) / 4;
        const float* xs = a.xs + (b * kE + t * kNvfp4Values) / 16;
        for (int s = 0; s < 4; ++s) {
          int ig = 0, iu = 0;
          for (int w = 0; w < 4; ++w) {
            ig = Dot4(qg[s * 4 + w], xq[s * 4 + w], ig);
            iu = Dot4(qu[s * 4 + w], xq[s * 4 + w], iu);
          }
          g += static_cast<float>(ig) * dg[s] * xs[s];
          u += static_cast<float>(iu) * du[s] * xs[s];
        }
      }
      sg[r][t] = g;
      su[r][t] = u;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r) {
          sg[r][t] += sg[r][t + step];
          su[r][t] += su[r][t + step];
        }
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        const float scale = a.up_scale[pair];
        a.act[static_cast<std::size_t>(pair) * kFF + k] = Situ(
            sg[r][0] * scale, su[r][0] * scale, a.situ_beta, a.situ_linear_beta);
      }
    __syncthreads();
  }
}

__launch_bounds__(32) __global__ void ExpertDownNvfp4Int8Kernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 32;
  __shared__ float sh[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const int e = blockIdx.x / kE;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  constexpr int kBlocks = kFF / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const bool active = t < kBlocks;
  std::uint32_t q[16];
  float d[4];
  if (active)
    DecodeNvfp4Int(a.down + (static_cast<std::size_t>(e) * kE + i) * kRowBytes +
                       t * kNvfp4Bytes,
                   q, d);
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      float acc = 0.0f;
      if (active) {
        const std::size_t pair = list[r0 + r];
        const std::uint32_t* aq = a.aq + (pair * kFF + t * kNvfp4Values) / 4;
        const float* as = a.as + (pair * kFF + t * kNvfp4Values) / 16;
        for (int s = 0; s < 4; ++s) {
          int sum = 0;
          for (int w = 0; w < 4; ++w)
            sum = Dot4(q[s * 4 + w], aq[s * 4 + w], sum);
          acc += static_cast<float>(sum) * d[s] * as[s];
        }
      }
      sh[r][t] = acc;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r)
          sh[r][t] += sh[r][t + step];
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        a.contrib[static_cast<std::size_t>(pair) * kE + i] =
            a.coef[pair] * sh[r][0];
      }
    __syncthreads();
  }
}

// The integer dot product of one 64-value NVFP4 block with 8-bit inputs: 16
// words `xq` and 4 scales `xs`.
__device__ inline float DotNvfp4Int(const std::uint8_t* block,
                                    const std::uint32_t* xq, const float* xs) {
  const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(block);
  const std::uint32_t scales = w[0];
  float sum = 0.0f;
  for (int s = 0; s < 4; ++s) {
    const std::uint32_t w0 = w[1 + 2 * s], w1 = w[2 + 2 * s];
    int acc = Dot4(DecodeCodes(w0 & 0x0F0F0F0Fu), xq[s * 4], 0);
    acc = Dot4(DecodeCodes(w1 & 0x0F0F0F0Fu), xq[s * 4 + 1], acc);
    acc = Dot4(DecodeCodes((w0 >> 4) & 0x0F0F0F0Fu), xq[s * 4 + 2], acc);
    acc = Dot4(DecodeCodes((w1 >> 4) & 0x0F0F0F0Fu), xq[s * 4 + 3], acc);
    sum += static_cast<float>(acc) * Ue4m3((scales >> (8 * s)) & 0xFFu) * xs[s];
  }
  return sum;
}

// One block per (row, expert, hidden unit), integer products.
__launch_bounds__(64) __global__ void ExpertUpNvfp4Int8RowKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 64;
  __shared__ float gate[kThreads];
  __shared__ float up[kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = (blockIdx.x / kFF) % kTopK;
  const std::size_t b = blockIdx.x / (kFF * kTopK);
  constexpr int kBlocks = kE / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(a.ids[b * kTopK + e]) * 2 * kFF * kRowBytes;
  float g = 0.0f, u = 0.0f;
  for (int blk = t; blk < kBlocks; blk += kThreads) {
    const std::uint32_t* xq = a.xq + (b * kE + blk * kNvfp4Values) / 4;
    const float* xs = a.xs + (b * kE + blk * kNvfp4Values) / 16;
    g += DotNvfp4Int(base + k * kRowBytes + blk * kNvfp4Bytes, xq, xs);
    u += DotNvfp4Int(base + (kFF + k) * kRowBytes + blk * kNvfp4Bytes, xq, xs);
  }
  gate[t] = g;
  up[t] = u;
  Reduce(gate, kThreads);
  Reduce(up, kThreads);
  if (t == 0) {
    const float scale = a.up_scale[b * kTopK + e];
    a.act[(b * kTopK + e) * kFF + k] =
        Situ(gate[0] * scale, up[0] * scale, a.situ_beta, a.situ_linear_beta);
  }
}

// One block per (row, output value), integer products.
__launch_bounds__(32) __global__ void ExpertDownNvfp4Int8RowKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 32;
  __shared__ float shared[kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const std::size_t b = blockIdx.x / kE;
  constexpr int kBlocks = kFF / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  float acc = 0.0f;
  for (int blk = t; blk < kBlocks; blk += kThreads) {
    for (int e = 0; e < kTopK; ++e) {
      const std::size_t pair = b * kTopK + e;
      const std::uint8_t* row =
          a.down + (static_cast<std::size_t>(a.ids[pair]) * kE + i) * kRowBytes;
      acc += a.coef[pair] *
             DotNvfp4Int(row + blk * kNvfp4Bytes,
                         a.aq + (pair * kFF + blk * kNvfp4Values) / 4,
                         a.as + (pair * kFF + blk * kNvfp4Values) / 16);
    }
  }
  shared[t] = acc;
  Reduce(shared, kThreads);
  if (t == 0)
    a.out[b * kE + i] = shared[0];
}

// The BF16 counterparts. A thread owns a contiguous run of columns, so the
// weights it converts and each row's inputs are both contiguous reads.
__launch_bounds__(64) __global__ void ExpertUpBf16MajorKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 64;
  constexpr int kSpan = kE / kThreads;
  __shared__ float sg[kMajorRows][kThreads];
  __shared__ float su[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int k = blockIdx.x % kFF;
  const int e = blockIdx.x / kFF;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  const std::uint16_t* base =
      reinterpret_cast<const std::uint16_t*>(a.gate_up) +
      static_cast<std::size_t>(e) * 2 * kFF * kE;
  const std::uint16_t* gate_row =
      base + static_cast<std::size_t>(k) * kE + t * kSpan;
  const std::uint16_t* up_row =
      base + static_cast<std::size_t>(kFF + k) * kE + t * kSpan;
  float wg[kSpan], wu[kSpan];
  for (int i = 0; i < kSpan; ++i) {
    wg[i] = Bf16(gate_row[i]);
    wu[i] = Bf16(up_row[i]);
  }
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      const float* x = a.x +
                       static_cast<std::size_t>(list[r0 + r] / kTopK) * kE +
                       t * kSpan;
      float g = 0.0f, u = 0.0f;
      for (int i = 0; i < kSpan; ++i) {
        g += wg[i] * x[i];
        u += wu[i] * x[i];
      }
      sg[r][t] = g;
      su[r][t] = u;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r) {
          sg[r][t] += sg[r][t + step];
          su[r][t] += su[r][t + step];
        }
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        const float scale = a.up_scale[pair];
        a.act[static_cast<std::size_t>(pair) * kFF + k] = Situ(
            sg[r][0] * scale, su[r][0] * scale, a.situ_beta, a.situ_linear_beta);
      }
    __syncthreads();
  }
}

__launch_bounds__(64) __global__ void ExpertDownBf16MajorKernel(
    BatchExpertArgs a) {
  constexpr int kThreads = 64;
  constexpr int kSpan = kFF / kThreads;
  __shared__ float sh[kMajorRows][kThreads];
  const int t = threadIdx.x;
  const int i = blockIdx.x % kE;
  const int e = blockIdx.x / kE;
  const int n = a.group_count[e];
  if (n == 0)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  const std::uint16_t* row =
      reinterpret_cast<const std::uint16_t*>(a.down) +
      (static_cast<std::size_t>(e) * kE + i) * kFF + t * kSpan;
  float w[kSpan];
  for (int j = 0; j < kSpan; ++j)
    w[j] = Bf16(row[j]);
  for (int r0 = 0; r0 < n; r0 += kMajorRows) {
    const int nr = n - r0 < kMajorRows ? n - r0 : kMajorRows;
    for (int r = 0; r < nr; ++r) {
      const float* act =
          a.act + static_cast<std::size_t>(list[r0 + r]) * kFF + t * kSpan;
      float acc = 0.0f;
      for (int j = 0; j < kSpan; ++j)
        acc += w[j] * act[j];
      sh[r][t] = acc;
    }
    __syncthreads();
    for (int step = kThreads / 2; step > 0; step >>= 1) {
      if (t < step)
        for (int r = 0; r < nr; ++r)
          sh[r][t] += sh[r][t + step];
      __syncthreads();
    }
    if (t == 0)
      for (int r = 0; r < nr; ++r) {
        const int pair = list[r0 + r];
        a.contrib[static_cast<std::size_t>(pair) * kE + i] =
            a.coef[pair] * sh[r][0];
      }
    __syncthreads();
  }
}

// ---- expert-major with the FP16 matrix instruction --------------------------

// Sixteen consecutive NVFP4 weights of a row (sub-block `kk` of 16) as FP16.
// A code times its sub-block scale has at most 6 significant bits, so the
// conversion is exact.
__device__ __forceinline__ void LoadHalfNvfp4(const std::uint8_t* row, int kk,
                                              std::uint32_t* out) {
  const std::uint8_t* block = row + (kk >> 2) * kNvfp4Bytes;
  const int sub = kk & 3;
  const float d = Ue4m3(block[sub]);
  std::uint32_t w[2];
  __builtin_memcpy(w, block + 4 + sub * 8, 8);
  union {
    Half16 f;
    std::uint32_t u[8];
  } v;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const std::uint32_t lo = DecodeCodes(w[half] & 0x0F0F0F0Fu);
    const std::uint32_t hi = DecodeCodes((w[half] >> 4) & 0x0F0F0F0Fu);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v.f[half * 4 + j] = static_cast<_Float16>(
          d * static_cast<float>(static_cast<signed char>(lo >> (8 * j))));
      v.f[8 + half * 4 + j] = static_cast<_Float16>(
          d * static_cast<float>(static_cast<signed char>(hi >> (8 * j))));
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[i] = v.u[i];
}

// The same for BF16 weights.
__device__ __forceinline__ void LoadHalfBf16(const std::uint16_t* w,
                                             std::uint32_t* out) {
  union {
    Half16 f;
    std::uint32_t u[8];
  } v;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    v.f[i] = static_cast<_Float16>(Bf16(w[i]));
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[i] = v.u[i];
}

// A block owns one expert, 128 weight rows and up to 32 * WTS of the pairs
// that selected the expert; it works like WmmaGemmBlockKernel, with the
// weights converted to FP16 as they are copied into shared memory.
//   Up:   the rows are 64 hidden units' gate rows and up rows, the inputs are
//         the pairs' x; a wave holds gate and up of the same units and stores
//         the activation.
//   Down: the rows are 128 outputs, the inputs are the pairs' activations.
// DirectX: the inputs are read by each wave straight from device memory, and
// only the weights, which need converting, go through shared memory: half
// the footprint, so twice the blocks at once.
template <bool Up, bool Nvfp4, int WTS, bool DirectX>
__launch_bounds__(256) __global__ void ExpertWmmaKernel(BatchExpertArgs a) {
  constexpr int BM = 128, BN = 32 * WTS, BK = INKLING2_BK;
  constexpr int kCols = Up ? kE : kFF;
  constexpr int kPerExpert = Up ? kFF / 64 : kE / BM;
  __shared__ std::uint32_t s_a[BK][BM][kStageRow];
  __shared__ std::uint32_t s_b[DirectX ? 1 : BK][DirectX ? 1 : BN][kStageRow];
  const int tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int sl = lane & 15, hi = lane >> 4;
  const int wr = wave >> 1, wt = wave & 1;
  const int e = blockIdx.x / kPerExpert, rb = blockIdx.x % kPerExpert;
  const int n = a.group_count[e];
  const int t_block = blockIdx.y * BN;
  if (t_block >= n)
    return;
  const std::int32_t* list =
      a.group_list + static_cast<std::size_t>(e) * a.max_batch;
  constexpr std::size_t kRowBytes =
      Nvfp4 ? static_cast<std::size_t>(kCols / kNvfp4Values) * kNvfp4Bytes
            : static_cast<std::size_t>(kCols) * 2;
  const std::uint8_t* base = (Up ? a.gate_up : a.down) +
                             static_cast<std::size_t>(e) * (Up ? 2 * kFF : kE) *
                                 kRowBytes;
  constexpr int UA = BM * BK / 256;
  constexpr int UB = (BN * BK + 255) / 256;
  const std::uint8_t* w_row[UA];
#pragma unroll
  for (int p = 0; p < UA; ++p) {
    const int pr = (p * 256 + tid) / BK;
    const int row = Up ? ((pr & 16) ? kFF : 0) + rb * 64 + (pr >> 5) * 16 + (pr & 15)
                       : rb * BM + pr;
    w_row[p] = base + static_cast<std::size_t>(row) * kRowBytes;
  }
  const std::uint32_t* x_src[UB];
#pragma unroll
  for (int p = 0; p < UB; ++p) {
    const int unit = p * 256 + tid;
    const int t = t_block + unit / BK;
    const int pair = list[t < n ? t : n - 1];
    const std::uint16_t* x =
        Up ? a.xh + static_cast<std::size_t>(pair / kTopK) * kE
           : a.acth + static_cast<std::size_t>(pair) * kFF;
    x_src[p] = reinterpret_cast<const std::uint32_t*>(x + (unit % BK) * 16);
  }
  // How many of this wave's batch tiles hold pairs; a wave with none only
  // helps copying.
  int live = 0;
#pragma unroll
  for (int g = 0; g < WTS; ++g)
    live += t_block + (wt * WTS + g) * 16 < n ? 1 : 0;
  const std::uint16_t* x_lane[WTS];
#pragma unroll
  for (int g = 0; g < WTS; ++g) {
    const int t = t_block + (wt * WTS + g) * 16 + sl;
    const int pair = list[t < n ? t : n - 1];
    x_lane[g] = Up ? a.xh + static_cast<std::size_t>(pair / kTopK) * kE
                   : a.acth + static_cast<std::size_t>(pair) * kFF;
  }
  Float8 acc[2][WTS] = {};
  for (int k = 0; k < kCols; k += 16 * BK) {
#pragma unroll
    for (int p = 0; p < UA; ++p) {
      const int unit = p * 256 + tid, ks = unit % BK;
      std::uint32_t* dst = s_a[ks][unit / BK];
      if (a.timing & 1)  // timing: no weight conversion
        continue;
      if constexpr (Nvfp4)
        LoadHalfNvfp4(w_row[p], k / 16 + ks, dst);
      else
        LoadHalfBf16(reinterpret_cast<const std::uint16_t*>(w_row[p]) + k +
                         ks * 16,
                     dst);
    }
    if constexpr (!DirectX) {
#pragma unroll
    for (int p = 0; p < UB; ++p) {
      const int unit = p * 256 + tid;
      if (unit < BN * BK) {
        const std::uint32_t* src = x_src[p] + k / 2;
        std::uint32_t* dst = s_b[unit % BK][unit / BK];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          dst[i] = src[i];
      }
    }
    }
    __syncthreads();
    if (live > 0 && !(a.timing & 2)) {  // timing: no products
#pragma unroll
      for (int ks = 0; ks < BK; ++ks) {
        Half16 wa[2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
          __builtin_memcpy(&wa[i], s_a[ks][(wr * 2 + i) * 16 + sl], 32);
#pragma unroll
        for (int g = 0; g < WTS; ++g) {
          if (g >= live)
            break;
          Half16 xb;
          if constexpr (DirectX)
            __builtin_memcpy(&xb, x_lane[g] + k + ks * 16, 32);
          else
            __builtin_memcpy(&xb, s_b[ks][(wt * WTS + g) * 16 + sl], 32);
#pragma unroll
          for (int i = 0; i < 2; ++i)
            acc[i][g] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xb, wa[i], acc[i][g]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int g = 0; g < WTS; ++g)
#pragma unroll
    for (int el = 0; el < 8; ++el) {
      const int t = t_block + (wt * WTS + g) * 16 + 2 * el + hi;
      if (t >= n)
        continue;
      const int pair = list[t];
      if constexpr (Up) {
        const float scale = a.up_scale[pair];
        const float v = Situ(acc[0][g][el] * scale, acc[1][g][el] * scale,
                             a.situ_beta, a.situ_linear_beta);
        reinterpret_cast<_Float16*>(a.acth)[static_cast<std::size_t>(pair) * kFF +
                                            rb * 64 + wr * 16 + sl] =
            static_cast<_Float16>(v);
      } else {
        const float coef = a.coef[pair];
#pragma unroll
        for (int i = 0; i < 2; ++i)
          a.contrib[static_cast<std::size_t>(pair) * kE + rb * BM +
                    (wr * 2 + i) * 16 + sl] = coef * acc[i][g][el];
      }
    }
}

template <int WTS>
void LaunchExpertWmma(const BatchExpertArgs& args, hipStream_t stream) {
  const dim3 grid(kNX * 18, (args.batch + 32 * WTS - 1) / (32 * WTS));
  if (args.nvfp4) {
    if (g_direct & 2)
      ExpertWmmaKernel<true, true, WTS, true><<<grid, 256, 0, stream>>>(args);
    else
      ExpertWmmaKernel<true, true, WTS, false><<<grid, 256, 0, stream>>>(args);
    if (g_direct & 2)
      ExpertWmmaKernel<false, true, WTS, true><<<grid, 256, 0, stream>>>(args);
    else
      ExpertWmmaKernel<false, true, WTS, false><<<grid, 256, 0, stream>>>(args);
  } else {
    if (g_direct & 2)
      ExpertWmmaKernel<true, false, WTS, true><<<grid, 256, 0, stream>>>(args);
    else
      ExpertWmmaKernel<true, false, WTS, false><<<grid, 256, 0, stream>>>(args);
    if (g_direct & 2)
      ExpertWmmaKernel<false, false, WTS, true><<<grid, 256, 0, stream>>>(args);
    else
      ExpertWmmaKernel<false, false, WTS, false><<<grid, 256, 0, stream>>>(args);
  }
}

// out = the sum of a row's kTopK contributions. One thread per (row, value).
__global__ void ExpertSumKernel(BatchExpertArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(idx % kE);
  const std::size_t b = idx / kE;
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  float acc = 0.0f;
  for (int j = 0; j < kTopK; ++j)
    acc += a.contrib[(b * kTopK + j) * kE + i];
  a.out[idx] = acc;
}

// ---- thread-per-output variants: no shared memory, no reductions ----------

// One thread per (row, expert, hidden unit).
__global__ void ExpertUpNvfp4RowKernel(BatchExpertArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int k = static_cast<int>(idx % kFF);
  const int e = static_cast<int>((idx / kFF) % kTopK);
  const std::size_t b = idx / (kFF * kTopK);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  constexpr int kBlocks = kE / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  const std::uint8_t* base =
      a.gate_up + static_cast<std::size_t>(a.ids[b * kTopK + e]) * 2 * kFF * kRowBytes;
  const std::uint8_t* gate_row = base + k * kRowBytes;
  const std::uint8_t* up_row = base + (kFF + k) * kRowBytes;
  const float* x = a.x + b * kE;
  float g = 0.0f, u = 0.0f;
  for (int blk = 0; blk < kBlocks; ++blk) {
    const float* xb = x + blk * kNvfp4Values;
    g += DotNvfp4Words(gate_row + blk * kNvfp4Bytes, xb);
    u += DotNvfp4Words(up_row + blk * kNvfp4Bytes, xb);
  }
  const float scale = a.up_scale[b * kTopK + e];
  a.act[(b * kTopK + e) * kFF + k] =
      Situ(g * scale, u * scale, a.situ_beta, a.situ_linear_beta);
}

// One thread per (row, output value).
__global__ void ExpertDownNvfp4RowKernel(BatchExpertArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(idx % kE);
  const std::size_t b = idx / kE;
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  constexpr int kBlocks = kFF / kNvfp4Values;
  constexpr std::size_t kRowBytes = static_cast<std::size_t>(kBlocks) * kNvfp4Bytes;
  float acc = 0.0f;
  for (int e = 0; e < kTopK; ++e) {
    const std::uint8_t* row =
        a.down +
        (static_cast<std::size_t>(a.ids[b * kTopK + e]) * kE + i) * kRowBytes;
    const float* act = a.act + (b * kTopK + e) * kFF;
    float part = 0.0f;
    for (int blk = 0; blk < kBlocks; ++blk)
      part += DotNvfp4Words(row + blk * kNvfp4Bytes, act + blk * kNvfp4Values);
    acc += a.coef[b * kTopK + e] * part;
  }
  a.out[idx] = acc;
}

__global__ void ExpertUpBf16RowKernel(BatchExpertArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int k = static_cast<int>(idx % kFF);
  const int e = static_cast<int>((idx / kFF) % kTopK);
  const std::size_t b = idx / (kFF * kTopK);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  const std::uint16_t* base =
      reinterpret_cast<const std::uint16_t*>(a.gate_up) +
      static_cast<std::size_t>(a.ids[b * kTopK + e]) * 2 * kFF * kE;
  const std::uint16_t* gate_row = base + static_cast<std::size_t>(k) * kE;
  const std::uint16_t* up_row = base + static_cast<std::size_t>(kFF + k) * kE;
  const float* x = a.x + b * kE;
  float g = 0.0f, u = 0.0f;
  for (int c = 0; c < kE; ++c) {
    g += Bf16(gate_row[c]) * x[c];
    u += Bf16(up_row[c]) * x[c];
  }
  const float scale = a.up_scale[b * kTopK + e];
  a.act[(b * kTopK + e) * kFF + k] =
      Situ(g * scale, u * scale, a.situ_beta, a.situ_linear_beta);
}

__global__ void ExpertDownBf16RowKernel(BatchExpertArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(idx % kE);
  const std::size_t b = idx / kE;
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  const std::uint16_t* down = reinterpret_cast<const std::uint16_t*>(a.down);
  float acc = 0.0f;
  for (int e = 0; e < kTopK; ++e) {
    const std::uint16_t* row =
        down + (static_cast<std::size_t>(a.ids[b * kTopK + e]) * kE + i) * kFF;
    const float* act = a.act + (b * kTopK + e) * kFF;
    float part = 0.0f;
    for (int c = 0; c < kFF; ++c)
      part += Bf16(row[c]) * act[c];
    acc += a.coef[b * kTopK + e] * part;
  }
  a.out[idx] = acc;
}

// One thread per (output row, group of batch rows).
__global__ void GemmRowKernel(GemmArgs a, int total_rows) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int groups = (a.batch + kGemmGroup - 1) / kGemmGroup;
  const int group = static_cast<int>(idx % groups);
  int r = static_cast<int>(idx / groups), j = 0;
  if (r >= total_rows)
    return;
  while (r >= a.jobs[j].rows) {
    r -= a.jobs[j].rows;
    ++j;
  }
  const GemmJob job = a.jobs[j];
  const int b0 = group * kGemmGroup;
  const int nb = a.batch - b0 < kGemmGroup ? a.batch - b0 : kGemmGroup;
  const std::uint16_t* row = job.w + static_cast<std::size_t>(r) * job.cols;
  float acc[kGemmGroup];
  for (int b = 0; b < nb; ++b)
    acc[b] = 0.0f;
  if (job.xt) {
    const float* xt = job.xt + static_cast<std::size_t>(group) * job.xt_group_stride;
    for (int c = 0; c < job.cols; ++c) {
      const float w = Bf16(row[c]);
      const float* column = xt + static_cast<std::size_t>(c) * kGemmGroup;
      for (int b = 0; b < nb; ++b)
        acc[b] += w * column[b];
    }
  } else {
    const float* x = job.x + static_cast<std::size_t>(b0) * job.x_stride;
    for (int c = 0; c < job.cols; ++c) {
      const float w = Bf16(row[c]);
      for (int b = 0; b < nb; ++b)
        acc[b] += w * x[static_cast<std::size_t>(b) * job.x_stride + c];
    }
  }
  for (int b = 0; b < nb; ++b)
    job.y[static_cast<std::size_t>(b0 + b) * job.y_stride + r] = acc[b];
}

// One thread per (row, channel).
__global__ void GlueDKernel(GlueDArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int b = static_cast<int>(idx / kE);
  const int i = static_cast<int>(idx % kE);
  if (b >= a.batch)
    return;
  a.cur[idx] +=
      ConvRow(a.moe, kE, b, a.rows[b], 3, a.post_conv, kE, i, nullptr, 1);
}

// One block per listed row.
__launch_bounds__(kT) __global__ void FinalKernel(FinalArgs a) {
  __shared__ float sh[kSums][kT];
  const std::size_t b = a.head_rows[blockIdx.x];
  const float* states[kMaxStates];
  for (int s = 0; s < a.n_blocks; ++s)
    states[s] = a.stack + (b * kMaxStates + s) * kE;
  states[a.n_blocks] = a.cur + b * kE;
  float* out = a.out + static_cast<std::size_t>(blockIdx.x) * kE;
  Mix(sh, states, a.n_blocks + 1, a.res_query, 1, out);
  NormW(sh, out, a.norm, a.eps);
}

// One thread per (row, value).
__global__ void CopyKernel(CopyArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t b = idx / kE;
  const int i = static_cast<int>(idx % kE);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  float value[3];
  for (int k = 0; k < a.count; ++k)
    value[k] = a.src[k][b * a.src_stride[k] + i];
  for (int k = 0; k < a.count; ++k)
    a.dst[k][b * a.dst_stride[k] + i] = value[k];
}

// acc + a.lo * b.lo + a.hi * b.hi, where a and b each hold two BF16 values.
__device__ inline float Dot2Bf16(float acc, std::uint32_t a, std::uint32_t b) {
  asm volatile("v_dot2_f32_bf16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
  return acc;
}

constexpr int kBenchThreads = 128;
__launch_bounds__(kBenchThreads) __global__ void Dot4BenchKernel(
    const std::uint32_t* __restrict__ w, const std::uint32_t* __restrict__ xt,
    int words, int batch, int* __restrict__ y) {
  __shared__ int sh[kGemmGroup][kBenchThreads];
  const int t = threadIdx.x;
  const std::uint32_t* row = w + static_cast<std::size_t>(blockIdx.x) * words;
  int acc[kGemmGroup];
  for (int b = 0; b < batch; ++b)
    acc[b] = 0;
  for (int c = t; c < words; c += kBenchThreads) {
    const int wv = static_cast<int>(row[c]);
    const std::uint32_t* column = xt + static_cast<std::size_t>(c) * kGemmGroup;
    for (int b = 0; b < batch; ++b)
      acc[b] = __builtin_amdgcn_sudot4(true, wv, true,
                                       static_cast<int>(column[b]), acc[b],
                                       false);
  }
  for (int b = 0; b < batch; ++b)
    sh[b][t] = acc[b];
  __syncthreads();
  for (int step = kBenchThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int b = 0; b < batch; ++b)
        sh[b][t] += sh[b][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int b = 0; b < batch; ++b)
      y[static_cast<std::size_t>(blockIdx.x) * kGemmGroup + b] = sh[b][0];
}

__launch_bounds__(kBenchThreads) __global__ void Dot2BenchKernel(
    const std::uint32_t* __restrict__ w, const std::uint32_t* __restrict__ xt,
    int words, int batch, float* __restrict__ y) {
  __shared__ float sh[kGemmGroup][kBenchThreads];
  const int t = threadIdx.x;
  const std::uint32_t* row = w + static_cast<std::size_t>(blockIdx.x) * words;
  float acc[kGemmGroup];
  for (int b = 0; b < batch; ++b)
    acc[b] = 0.0f;
  for (int c = t; c < words; c += kBenchThreads) {
    const std::uint32_t wv = row[c];
    const std::uint32_t* column = xt + static_cast<std::size_t>(c) * kGemmGroup;
    for (int b = 0; b < batch; ++b)
      acc[b] = Dot2Bf16(acc[b], wv, column[b]);
  }
  for (int b = 0; b < batch; ++b)
    sh[b][t] = acc[b];
  __syncthreads();
  for (int step = kBenchThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int b = 0; b < batch; ++b)
        sh[b][t] += sh[b][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int b = 0; b < batch; ++b)
      y[static_cast<std::size_t>(blockIdx.x) * kGemmGroup + b] = sh[b][0];
}

__launch_bounds__(kBenchThreads) __global__ void FloatBenchKernel(
    const float* __restrict__ w, const float* __restrict__ xt, int cols,
    int batch, float* __restrict__ y) {
  __shared__ float sh[kGemmGroup][kBenchThreads];
  const int t = threadIdx.x;
  const float* row = w + static_cast<std::size_t>(blockIdx.x) * cols;
  float acc[kGemmGroup];
  for (int b = 0; b < batch; ++b)
    acc[b] = 0.0f;
  for (int c = t; c < cols; c += kBenchThreads) {
    const float wv = row[c];
    const float* column = xt + static_cast<std::size_t>(c) * kGemmGroup;
    for (int b = 0; b < batch; ++b)
      acc[b] += wv * column[b];
  }
  for (int b = 0; b < batch; ++b)
    sh[b][t] = acc[b];
  __syncthreads();
  for (int step = kBenchThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      for (int b = 0; b < batch; ++b)
        sh[b][t] += sh[b][t + step];
    __syncthreads();
  }
  if (t == 0)
    for (int b = 0; b < batch; ++b)
      y[static_cast<std::size_t>(blockIdx.x) * kGemmGroup + b] = sh[b][0];
}

inline int Blocks(std::size_t items, int per_block) {
  return static_cast<int>((items + per_block - 1) / per_block);
}

}  // namespace

void Dot4Bench(const std::uint32_t* w, const std::uint32_t* xt, int rows,
               int words, int batch, int* y, hipStream_t stream) {
  Dot4BenchKernel<<<rows, kBenchThreads, 0, stream>>>(w, xt, words, batch, y);
}

void Dot2Bench(const std::uint32_t* w, const std::uint32_t* xt, int rows,
               int words, int batch, float* y, hipStream_t stream) {
  Dot2BenchKernel<<<rows, kBenchThreads, 0, stream>>>(w, xt, words, batch, y);
}

void FloatBench(const float* w, const float* xt, int rows, int cols, int batch,
                float* y, hipStream_t stream) {
  FloatBenchKernel<<<rows, kBenchThreads, 0, stream>>>(w, xt, cols, batch, y);
}

void Embed(const EmbedArgs& args, hipStream_t stream) {
  EmbedKernel<<<args.batch, kT, 0, stream>>>(args);
}

void GlueA(const GlueAArgs& args, hipStream_t stream) {
  GlueAKernel<<<args.batch, kT, 0, stream>>>(args);
}

void CtrlDown(const CtrlArgs& args, hipStream_t stream) {
  CtrlDownKernel<<<args.batch * args.count * kRank, kCtrlThreads, 0, stream>>>(
      args);
}

void CtrlApply(const CtrlArgs& args, hipStream_t stream) {
  CtrlApplyKernel<<<Blocks(std::size_t(args.batch) * args.count * kE, 256), 256,
                    0, stream>>>(args);
}

void Norm(const NormArgs& args, hipStream_t stream) {
  NormKernel<<<args.batch * args.count, kNormThreads, 0, stream>>>(args);
}

void ToHalf(const ToHalfArgs& args, hipStream_t stream) {
  ToHalfKernel<<<Blocks(std::size_t(args.batch) * args.cols, 256), 256, 0,
                 stream>>>(args);
}

void Gemm(const GemmArgs& args, hipStream_t stream) {
  int rows = 0;
  for (int j = 0; j < args.count; ++j)
    rows += args.jobs[j].rows;
  if (args.wmma) {
    // Up to 16 rows: one tile of batch rows per wave, straight from global
    // memory (the weights are the traffic). Above: blocks staged in shared
    // memory, 32, 64 or 128 batch rows wide.
    const int wide = args.wmma_tiles > 0 ? args.wmma_tiles
                     : args.batch <= 16  ? 0
                     : args.batch <= 32  ? 1
                     : args.batch <= 64  ? 2
                                         : 4;
    int blocks = 0;
    for (int j = 0; j < args.count; ++j)
      blocks += (args.jobs[j].rows + 127) / 128;
    // A job of few rows makes too few staged blocks to fill the GPU; up to 64
    // batch rows the direct kernel, with up to 4 batch tiles per wave, is
    // faster there.
    const bool few = args.wmma_tiles == 0 && args.batch <= 64 && blocks < 32;
    if (wide == 0 || wide > 4 || few) {
      int tiles = 0;
      for (int j = 0; j < args.count; ++j)
        tiles += (args.jobs[j].rows + 15) / 16;
      const int wts = args.batch <= 16 ? 1 : args.batch <= 32 ? 2 : 4;
      const dim3 grid((tiles + 7) / 8, (args.batch + 16 * wts - 1) / (16 * wts));
      if (wts == 1)
        WmmaGemmKernel<1><<<grid, 256, 0, stream>>>(args);
      else if (wts == 2)
        WmmaGemmKernel<2><<<grid, 256, 0, stream>>>(args);
      else
        WmmaGemmKernel<4><<<grid, 256, 0, stream>>>(args);
      return;
    }
    const int bn = 32 * wide;
    const dim3 grid(blocks, (args.batch + bn - 1) / bn);
    if (g_direct & 1) {
      if (wide == 1)
        WmmaGemmBlockKernel<1, true><<<grid, 256, 0, stream>>>(args);
      else if (wide == 2)
        WmmaGemmBlockKernel<2, true><<<grid, 256, 0, stream>>>(args);
      else
        WmmaGemmBlockKernel<4, true><<<grid, 256, 0, stream>>>(args);
      return;
    }
    if (wide == 1)
      WmmaGemmBlockKernel<1, false><<<grid, 256, 0, stream>>>(args);
    else if (wide == 2)
      WmmaGemmBlockKernel<2, false><<<grid, 256, 0, stream>>>(args);
    else
      WmmaGemmBlockKernel<4, false><<<grid, 256, 0, stream>>>(args);
    return;
  }
  if (args.batch == 1 && args.variant == 0) {
    if (args.single_threads == 64)
      GemvKernel<64><<<rows, 64, 0, stream>>>(args);
    else
      GemvKernel<128><<<rows, 128, 0, stream>>>(args);
    return;
  }
  const int groups = (args.batch + kGemmGroup - 1) / kGemmGroup;
  if (args.variant == 1) {
    GemmRowKernel<<<Blocks(std::size_t(rows) * groups, 256), 256, 0, stream>>>(
        args, rows);
    return;
  }
  GemmKernel<<<rows * groups, kGemmThreads, 0, stream>>>(args);
}

void Transpose(const TransposeArgs& args, hipStream_t stream) {
  TransposeKernel<<<Blocks(std::size_t(args.batch) * args.cols, 256), 256, 0,
                    stream>>>(args);
}

void GlueB(const GlueBArgs& args, hipStream_t stream) {
  GlueBKernel<<<args.batch, kT, 0, stream>>>(args);
}

void Attention(const AttnArgs& args, hipStream_t stream) {
  if (args.batched && (args.fast & 16)) {
    AttentionFlashKernel<<<args.group_count * kNKV, 384, 0, stream>>>(args);
    return;
  }
  if (args.batched) {
    const int grid = args.group_count * kNKV * args.max_chunks;
    if (args.fast & 8)
      AttentionBatchKernel<false, true><<<grid, 256, 0, stream>>>(args);
    else
      AttentionBatchKernel<false, false><<<grid, 256, 0, stream>>>(args);
    if ((args.fast & 5) == 5 && args.span > 0)
      AttentionBatchKernel<true, true><<<grid, 256, 0, stream>>>(args);
    AttentionCombineKernel<<<args.batch * kNH, kHD, 0, stream>>>(args);
    return;
  }
  if (args.half) {
    AttentionChunkHalfKernel<<<args.batch * kNKV * args.max_chunks, kT, 0,
                               stream>>>(args);
    AttentionCombineKernel<<<args.batch * kNH, kHD, 0, stream>>>(args);
    return;
  }
  if (args.chunked) {
    AttentionChunkKernel<<<args.batch * kNKV * args.max_chunks, kT, 0,
                           stream>>>(args);
    AttentionCombineKernel<<<args.batch * kNH, kHD, 0, stream>>>(args);
    return;
  }
  if (args.window) {
    AttentionWindowKernel<<<args.batch * kNH, kT, 0, stream>>>(args);
    return;
  }
  AttentionKernel<<<args.batch * kNH, kAttnThreads, 0, stream>>>(args);
}

void GlueC(const GlueCArgs& args, hipStream_t stream) {
  GlueCKernel<<<args.batch, kT, 0, stream>>>(args);
}

void CtrlCombine(const CtrlArgs& args, hipStream_t stream) {
  CtrlCombineKernel<<<args.batch * args.count, 256, 0, stream>>>(args);
}

void Router(const RouterArgs& args, hipStream_t stream) {
  if (!args.logits_ready)
    RouterLogitsKernel<<<args.batch * kNX, kRouterThreads, 0, stream>>>(args);
  RouterSelectKernel<<<args.batch, kNX, 0, stream>>>(args);
}

void BatchExperts(const BatchExpertArgs& args, hipStream_t stream) {
  if (args.variant == 5) {
    ExpertGroupKernel<<<1, kNX, 0, stream>>>(args);
    ToHalfArgs h{args.x, kE, args.xh, kE, args.batch};
    ToHalfKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0, stream>>>(h);
    // A row picks 8 of 128 experts: batch / 16 pairs per expert on average.
    // The tile width follows that, so that an expert's weights are read
    // about once.
    const int wide = args.wmma_tiles > 0 ? args.wmma_tiles
                     : args.batch <= 384 ? 1
                     : args.batch <= 768 ? 2
                                         : 4;
    if (wide == 1)
      LaunchExpertWmma<1>(args, stream);
    else if (wide == 2)
      LaunchExpertWmma<2>(args, stream);
    else
      LaunchExpertWmma<4>(args, stream);
    ExpertSumKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0,
                      stream>>>(args);
    return;
  }
  if (args.variant == 4 && args.nvfp4) {
    QuantizeArgs qx{args.x, kE, args.batch, kE, args.xq, args.xs};
    QuantizeKernel<<<Blocks(std::size_t(args.batch) * kE / 16, 256), 256, 0,
                     stream>>>(qx);
    ExpertUpNvfp4Int8RowKernel<<<args.batch * kTopK * kFF, 64, 0, stream>>>(
        args);
    const int pairs = args.batch * kTopK;
    QuantizeArgs qa{args.act, kFF, pairs, kFF, args.aq, args.as};
    QuantizeKernel<<<Blocks(std::size_t(pairs) * kFF / 16, 256), 256, 0,
                     stream>>>(qa);
    ExpertDownNvfp4Int8RowKernel<<<args.batch * kE, 32, 0, stream>>>(args);
    return;
  }
  if (args.variant == 3 && args.nvfp4) {
    ExpertGroupKernel<<<1, kNX, 0, stream>>>(args);
    QuantizeArgs qx{args.x, kE, args.batch, kE, args.xq, args.xs};
    QuantizeKernel<<<Blocks(std::size_t(args.batch) * kE / 16, 256), 256, 0,
                     stream>>>(qx);
    ExpertUpNvfp4Int8Kernel<<<kNX * kFF, 64, 0, stream>>>(args);
    const int pairs = args.batch * kTopK;
    QuantizeArgs qa{args.act, kFF, pairs, kFF, args.aq, args.as};
    QuantizeKernel<<<Blocks(std::size_t(pairs) * kFF / 16, 256), 256, 0,
                     stream>>>(qa);
    ExpertDownNvfp4Int8Kernel<<<kNX * kE, 32, 0, stream>>>(args);
    ExpertSumKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0,
                      stream>>>(args);
    return;
  }
  if (args.variant == 2 || args.variant == 3) {
    ExpertGroupKernel<<<1, kNX, 0, stream>>>(args);
    if (args.nvfp4) {
      ExpertUpNvfp4MajorKernel<<<kNX * kFF, 64, 0, stream>>>(args);
      ExpertDownNvfp4MajorKernel<<<kNX * kE, 32, 0, stream>>>(args);
    } else {
      ExpertUpBf16MajorKernel<<<kNX * kFF, 64, 0, stream>>>(args);
      ExpertDownBf16MajorKernel<<<kNX * kE, 64, 0, stream>>>(args);
    }
    ExpertSumKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0,
                      stream>>>(args);
    return;
  }
  if (args.variant == 1) {
    const int up = Blocks(std::size_t(args.batch) * kTopK * kFF, 256);
    const int down = Blocks(std::size_t(args.batch) * kE, 256);
    if (args.nvfp4) {
      ExpertUpNvfp4RowKernel<<<up, 256, 0, stream>>>(args);
      ExpertDownNvfp4RowKernel<<<down, 256, 0, stream>>>(args);
    } else {
      ExpertUpBf16RowKernel<<<up, 256, 0, stream>>>(args);
      ExpertDownBf16RowKernel<<<down, 256, 0, stream>>>(args);
    }
    return;
  }
  // Variants 0 and 4 (on BF16 layers) end here.
  if (args.nvfp4) {
    ExpertUpNvfp4Kernel<<<args.batch * kTopK * kFF, 64, 0, stream>>>(args);
    ExpertDownNvfp4Kernel<<<args.batch * kE, 32, 0, stream>>>(args);
  } else {
    ExpertUpBf16Kernel<<<args.batch * kTopK * kFF, 64, 0, stream>>>(args);
    ExpertDownBf16Kernel<<<args.batch * kE, 64, 0, stream>>>(args);
  }
}

void GlueD(const GlueDArgs& args, hipStream_t stream) {
  GlueDKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0, stream>>>(
      args);
}

void Final(const FinalArgs& args, hipStream_t stream) {
  FinalKernel<<<args.count, kT, 0, stream>>>(args);
}

void Copy(const CopyArgs& args, hipStream_t stream) {
  CopyKernel<<<Blocks(std::size_t(args.batch) * kE, 256), 256, 0, stream>>>(
      args);
}

void SetWmmaDirect(int bits) {
  g_direct = bits;
}

}  // namespace inkling2::gpu
