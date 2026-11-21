// Kernels for layer visits that stay on the device, over a batch of rows.
// See decode.hpp. The single-block "glue" kernels run one block per row and
// their phases in sequence with __syncthreads between them; every thread of a
// block executes every phase.

#include "decode.hpp"

#include <cstddef>

namespace inkling2::gpu {
namespace {

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

// cur = (1 - tanh(forget)) * cur + tanh(gain) * msg, one thread per value.
__global__ void CtrlApplyKernel(CtrlArgs a) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = idx % kE;
  const int c = (idx / kE) % a.count;
  const std::size_t b = idx / (static_cast<std::size_t>(kE) * a.count);
  if (b >= static_cast<std::size_t>(a.batch))
    return;
  const float* low = a.low + (b * 3 + c) * kRank;
  const float* gain_row = a.gain[c] + static_cast<std::size_t>(i) * kRank;
  const float* forget_row = a.forget[c] + static_cast<std::size_t>(i) * kRank;
  float gain = 0.0f, forget = 0.0f;
  for (int r = 0; r < kRank; ++r) {
    gain += gain_row[r] * low[r];
    forget += forget_row[r] * low[r];
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

  float* k_row = row.kv_k + static_cast<std::size_t>(row.pos % row.cap) * kHK;
  float* v_row = row.kv_v + static_cast<std::size_t>(row.pos % row.cap) * kHK;
  for (int i = t; i < kHQ; i += kT)
    q[i] = q[i] * a.q_clip[i] * factor[i / kHD];
  for (int c = t; c < kHK; c += kT) {
    k_row[c] = k_new[c] * factor[kNH + c / kHD];
    v_row[c] = v_new[c] * factor[kNH + kNKV + c / kHD];
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

void Gemm(const GemmArgs& args, hipStream_t stream) {
  int rows = 0;
  for (int j = 0; j < args.count; ++j)
    rows += args.jobs[j].rows;
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
  AttentionKernel<<<args.batch * kNH, kAttnThreads, 0, stream>>>(args);
}

void GlueC(const GlueCArgs& args, hipStream_t stream) {
  GlueCKernel<<<args.batch, kT, 0, stream>>>(args);
}

void Router(const RouterArgs& args, hipStream_t stream) {
  RouterLogitsKernel<<<args.batch * kNX, kRouterThreads, 0, stream>>>(args);
  RouterSelectKernel<<<args.batch, kNX, 0, stream>>>(args);
}

void BatchExperts(const BatchExpertArgs& args, hipStream_t stream) {
  if (args.variant == 2) {
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

}  // namespace inkling2::gpu
