// Fused kernels for one decode step that stays on the device. See decode.hpp.
// The single-block "glue" kernels run their phases in sequence with
// __syncthreads between them; every thread of the block executes every phase.

#include "decode.hpp"

#include <cstddef>

namespace inkling2::gpu {
namespace {

constexpr int kT = 1024;  // threads of a glue block
constexpr int kSums = 5;  // values summed together by BlockSum

__device__ inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  __builtin_memcpy(&value, &wide, sizeof value);
  return value;
}

__device__ inline float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Sums each of the first `count` entries of `mine` over the block's threads.
// Every thread calls it and gets the same totals.
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
// `out` (nq rows of E) must not alias a state.
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
      score[q][s] = total[1 + q] * inv[s] * (1.0f / 48.0f);  // 1 / sqrt(E)
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

// cur = (1 - tanh(forget)) * cur + tanh(gain) * msg, both controllers reading
// the rank-64 projection of [rms(cur), rms(msg)]. In place on `cur`.
__device__ inline void Control(float (*sh)[kT], float* low, float* cur,
                               const float* msg, Ctrl c) {
  const int t = threadIdx.x;
  float mine[2] = {0.0f, 0.0f};
  for (int i = t; i < kE; i += kT) {
    mine[0] += cur[i] * cur[i];
    mine[1] += msg[i] * msg[i];
  }
  float total[2];
  BlockSum(sh, mine, 2, total);
  const float inv_cur = 1.0f / sqrtf(total[0] / kE + 1e-6f);
  const float inv_msg = 1.0f / sqrtf(total[1] / kE + 1e-6f);

  // 64 dot products of 2E values, each split across kT / 64 threads.
  constexpr int kParts = kT / kRank;
  constexpr int kSpan = 2 * kE / kParts;
  const int d = t % kRank, part = t / kRank;
  const float* row = c.down + static_cast<std::size_t>(d) * 2 * kE;
  float acc = 0.0f;
  for (int j = part * kSpan; j < (part + 1) * kSpan; ++j) {
    const float g = j < kE ? cur[j] * inv_cur : msg[j - kE] * inv_msg;
    acc += row[j] * g;
  }
  sh[0][t] = acc;
  __syncthreads();
  if (t < kRank) {
    float sum = 0.0f;
    for (int p = 0; p < kParts; ++p)
      sum += sh[0][t + kRank * p];
    low[t] = sum;
  }
  __syncthreads();
  for (int i = t; i < kE; i += kT) {
    float gain = 0.0f, forget = 0.0f;
    for (int r = 0; r < kRank; ++r) {
      gain += c.gain[i * kRank + r] * low[r];
      forget += c.forget[i * kRank + r] * low[r];
    }
    cur[i] = cur[i] - tanhf(forget) * cur[i] + tanhf(gain) * msg[i];
  }
  __syncthreads();
}

// x = rms_norm(x) * weight, in place.
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

// One channel of the causal short convolution with its residual: reads the
// last kNC - 1 inputs from `hist` (rows of `channels`), shifts `cur` in, and
// returns (cur + taps) / sqrt(n).
__device__ inline float ConvStep(float cur, const float* weight, float* hist,
                                 int channels, int c, const float* taps,
                                 int per_head) {
  float x[kNC];
  for (int i = 0; i < kNC - 1; ++i)
    x[i] = hist[i * channels + c];
  x[kNC - 1] = cur;
  float out = 0.0f;
  for (int i = 0; i < kNC; ++i)
    out += x[i] * weight[c * kNC + i];
  if (taps) {
    const float* tap = taps + kNC * (c / per_head);
    for (int lag = 0; lag < kNC; ++lag)
      out += x[kNC - 1 - lag] * tap[lag];
  }
  for (int i = 0; i < kNC - 2; ++i)
    hist[i * channels + c] = x[i + 1];
  hist[(kNC - 2) * channels + c] = cur;
  // 1 / sqrt(n_conv + 1) = 1 / sqrt(5), or 1 / sqrt(2 * n_conv + 1) = 1 / 3
  return (cur + out) * (taps ? (1.0f / 3.0f) : 0.4472135955f);
}

__device__ inline float ClipFactor(float sum_sq, float threshold) {
  const float factor = threshold / (sqrtf(sum_sq) + 1e-6f);
  return fminf(fmaxf(factor, 0.0f), 1.0f);
}

__launch_bounds__(kT) __global__ void GlueAKernel(GlueAArgs a) {
  __shared__ float sh[kSums][kT];
  __shared__ float low[kRank];
  const int t = threadIdx.x;
  for (int i = t; i < kE; i += kT)
    a.input[i] = a.cur[i];

  if (a.n_res > 0) {
    const float* states[kMaxStates];
    for (int s = 0; s < a.n_blocks; ++s)
      states[s] = a.stack + static_cast<std::size_t>(s) * kE;
    states[a.n_blocks] = a.cur;
    Mix(sh, states, a.n_blocks + 1, a.res_query, a.n_res, a.in);
    if (a.n_res == 1) {
      for (int i = t; i < kE; i += kT) {
        a.in[kE + i] = a.in[i];
        a.in[2 * kE + i] = a.in[i];
      }
      __syncthreads();
    }
  } else {
    for (int i = t; i < kE; i += kT) {
      const float x = a.cur[i];
      a.in[i] = x;
      a.in[kE + i] = x;
      a.in[2 * kE + i] = x;
    }
    __syncthreads();
  }

  if (a.looped) {
    const float* bank[2] = {a.bank0, a.bank_out};
    Mix(sh, bank, 2, a.msg_query, 4, a.msgs);
    for (int j = 0; j < 3; ++j)
      Control(sh, low, a.in + j * kE, a.msgs + j * kE, a.ctrl[j]);
  }
  for (int j = 0; j < 3; ++j)
    NormW(sh, a.in + j * kE, a.norm[j], a.eps);
}

constexpr int kGemvThreads = 128;

// One block per output row; a block finds its job from the row prefix.
__launch_bounds__(kGemvThreads) __global__ void GemvMultiKernel(GemvArgs a) {
  __shared__ float shared[kGemvThreads];
  int b = blockIdx.x, j = 0;
  while (b >= a.jobs[j].rows) {
    b -= a.jobs[j].rows;
    ++j;
  }
  const int cols = a.jobs[j].cols;
  const std::uint16_t* row = a.jobs[j].w + static_cast<std::size_t>(b) * cols;
  const float* x = a.jobs[j].x;
  float acc = 0.0f;
  for (int c = threadIdx.x; c < cols; c += kGemvThreads)
    acc += Bf16(row[c]) * x[c];
  shared[threadIdx.x] = acc;
  __syncthreads();
  for (int step = kGemvThreads / 2; step > 0; step >>= 1) {
    if (static_cast<int>(threadIdx.x) < step)
      shared[threadIdx.x] += shared[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    a.jobs[j].y[b] = shared[0];
}

__launch_bounds__(kT) __global__ void GlueBKernel(GlueBArgs a) {
  __shared__ float factor[64];
  // Heads of the group factors: Q 0..23, K 24..31, V 32..39, R 40..63.
  const int t = threadIdx.x;
  __shared__ float k_new[kHK];
  __shared__ float v_new[kHK];
  for (int c = t; c < kHK; c += kT) {
    k_new[c] = ConvStep(a.k_raw[c], a.k_conv, a.hist_k, kHK, c, a.k_taps, kHD);
    v_new[c] = ConvStep(a.v_raw[c], a.v_conv, a.hist_v, kHK, c, a.v_taps, kHD);
  }
  __syncthreads();

  if (t < kNH) {
    float sum = 0.0f;
    for (int i = 0; i < kHD; ++i) {
      const float x = a.q[t * kHD + i] * a.q_clip[t * kHD + i];
      sum += x * x;
    }
    factor[t] = ClipFactor(sum, a.qk_clip) * a.tau;
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
      const float x = a.r[h * kNRel + i] * a.rel_clip[h * kNRel + i];
      sum += x * x;
    }
    factor[t] = ClipFactor(sum, a.rel_clip_threshold) * a.tau;
  }
  __syncthreads();

  for (int i = t; i < kHQ; i += kT)
    a.q[i] = a.q[i] * a.q_clip[i] * factor[i / kHD];
  for (int c = t; c < kHK; c += kT) {
    a.k_row[c] = k_new[c] * factor[kNH + c / kHD];
    a.v_row[c] = v_new[c] * factor[kNH + kNKV + c / kHD];
  }
  for (int i = t; i < kNR; i += kT)
    a.r[i] = a.r[i] * a.rel_clip[i] * factor[kNH + 2 * kNKV + i / kNRel];
}

constexpr int kAttnThreads = 256;

// One block per query head.
__launch_bounds__(kAttnThreads) __global__ void AttentionKernel(AttnArgs a) {
  __shared__ float shared[kAttnThreads];
  const int t = threadIdx.x, h = blockIdx.x;
  const int kv_head = h / (kNH / kNKV);
  const int n = a.pos - a.first + 1;
  const float* q = a.q + h * kHD;
  const float* r = a.r + h * kNRel;
  float* logit = a.scratch + static_cast<std::size_t>(h) * n;
  const float sink = a.sink[h];

  float max = 0.0f;  // the zero sink's logit
  for (int idx = t; idx < n; idx += kAttnThreads) {
    const int j = a.first + idx;
    const float* k =
        a.kv_k + static_cast<std::size_t>(j % a.cap) * kHK + kv_head * kHD;
    float dot = 0.0f;
    for (int i = 0; i < kHD; ++i)
      dot += q[i] * k[i];
    const int distance = a.pos - j;
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
  __syncthreads();
  for (int step = kAttnThreads / 2; step > 0; step >>= 1) {
    if (t < step)
      shared[t] += shared[t + step];
    __syncthreads();
  }
  const float total = shared[0] + __expf(0.0f - max);

  if (t < kHD) {
    float acc = 0.0f;
    for (int idx = 0; idx < n; ++idx) {
      const int j = a.first + idx;
      acc += logit[idx] *
             a.kv_v[static_cast<std::size_t>(j % a.cap) * kHK + kv_head * kHD + t];
    }
    a.out[h * kHD + t] = acc / total * Sigmoid(a.gate[h * kHD + t]);
  }
}

__launch_bounds__(kT) __global__ void GlueCKernel(GlueCArgs a) {
  __shared__ float sh[kSums][kT];
  __shared__ float low[kRank];
  const int t = threadIdx.x;
  for (int i = t; i < kE; i += kT) {
    const float conv =
        ConvStep(a.proj[i], a.post_conv, a.hist, kE, i, nullptr, 1);
    if (a.block_start) {
      // A block's first layer starts a new residual stream; the earlier one
      // joins the mixed blocks.
      a.stack[static_cast<std::size_t>(a.n_blocks) * kE + i] = a.input[i];
      a.cur[i] = conv;
    } else {
      a.cur[i] = a.input[i] + conv;
    }
  }
  __syncthreads();

  const int n = a.n_blocks + (a.block_start ? 1 : 0);
  const float* states[kMaxStates];
  for (int s = 0; s < n; ++s)
    states[s] = a.stack + static_cast<std::size_t>(s) * kE;
  states[n] = a.cur;
  Mix(sh, states, n + 1, a.ffn_res_query, 1, a.xn);
  if (a.looped)
    Control(sh, low, a.xn, a.msg_ffn, a.ctrl);
  NormW(sh, a.xn, a.ffn_norm, a.eps);

  // Router: 128 logits of E values, each split across kT / 128 threads.
  constexpr int kParts = kT / kNX;
  constexpr int kSpan = kE / kParts;
  const int expert = t % kNX, part = t / kNX;
  const float* row = a.gate_inp + static_cast<std::size_t>(expert) * kE;
  float acc = 0.0f;
  for (int j = part * kSpan; j < (part + 1) * kSpan; ++j)
    acc += row[j] * a.xn[j];
  sh[0][t] = acc;
  __syncthreads();
  if (t < kNX) {
    float logit = 0.0f;
    for (int p = 0; p < kParts; ++p)
      logit += sh[0][t + kNX * p];
    const float prob = Sigmoid(logit);
    sh[1][t] = prob;
    sh[2][t] = prob + (a.exp_probs_b ? a.exp_probs_b[t] : 0.0f);
  }
  __syncthreads();
  if (t < kNX) {
    // Rank by selection key, highest first, ties to the lower index.
    const float key = sh[2][t];
    int rank = 0;
    for (int j = 0; j < kNX; ++j)
      rank += (sh[2][j] > key || (sh[2][j] == key && j < t)) ? 1 : 0;
    if (rank < kTopK) {
      sh[3][rank] = sh[1][t];
      sh[4][rank] = static_cast<float>(t);
    }
  }
  __syncthreads();
  if (t == 0) {
    float sum = 1e-20f;
    for (int i = 0; i < kTopK; ++i)
      sum += sh[3][i];
    for (int i = 0; i < kTopK; ++i) {
      const int id = static_cast<int>(sh[4][i]);
      a.ids[i] = static_cast<std::uint32_t>(id);
      a.up_scale[i] = a.up_scale_all ? a.up_scale_all[id] : 1.0f;
      a.coef[i] = sh[3][i] / sum * a.weights_scale *
                  (a.down_scale_all ? a.down_scale_all[id] : 1.0f);
    }
  }
}

__global__ void GlueDKernel(GlueDArgs a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kE)
    a.cur[i] += ConvStep(a.moe[i], a.post_conv, a.hist, kE, i, nullptr, 1);
}

__launch_bounds__(kT) __global__ void EmbedKernel(EmbedArgs a) {
  __shared__ float sh[kSums][kT];
  const int t = threadIdx.x;
  float mine = 0.0f;
  for (int i = t; i < kE; i += kT) {
    const float x = Bf16(a.row[i]);
    mine += x * x;
  }
  float total;
  BlockSum(sh, &mine, 1, &total);
  const float inv = 1.0f / sqrtf(total / kE + a.eps);
  for (int i = t; i < kE; i += kT)
    a.cur[i] = Bf16(a.row[i]) * inv * a.norm[i];
}

__launch_bounds__(kT) __global__ void FinalKernel(FinalArgs a) {
  __shared__ float sh[kSums][kT];
  const float* states[kMaxStates];
  for (int s = 0; s < a.n_blocks; ++s)
    states[s] = a.stack + static_cast<std::size_t>(s) * kE;
  states[a.n_blocks] = a.cur;
  Mix(sh, states, a.n_blocks + 1, a.res_query, 1, a.out);
  NormW(sh, a.out, a.norm, a.eps);
}

__global__ void CopyKernel(CopyArgs a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kE)
    return;
  float value[3];
  for (int k = 0; k < a.count; ++k)
    value[k] = a.src[k][i];
  for (int k = 0; k < a.count; ++k)
    a.dst[k][i] = value[k];
}

}  // namespace

void GlueA(const GlueAArgs& args, hipStream_t stream) {
  GlueAKernel<<<1, kT, 0, stream>>>(args);
}

void GemvMulti(const GemvArgs& args, hipStream_t stream) {
  int rows = 0;
  for (int j = 0; j < args.count; ++j)
    rows += args.jobs[j].rows;
  GemvMultiKernel<<<rows, kGemvThreads, 0, stream>>>(args);
}

void GlueB(const GlueBArgs& args, hipStream_t stream) {
  GlueBKernel<<<1, kT, 0, stream>>>(args);
}

void Attention(const AttnArgs& args, hipStream_t stream) {
  AttentionKernel<<<kNH, kAttnThreads, 0, stream>>>(args);
}

void GlueC(const GlueCArgs& args, hipStream_t stream) {
  GlueCKernel<<<1, kT, 0, stream>>>(args);
}

void GlueD(const GlueDArgs& args, hipStream_t stream) {
  GlueDKernel<<<(kE + 255) / 256, 256, 0, stream>>>(args);
}

void Embed(const EmbedArgs& args, hipStream_t stream) {
  EmbedKernel<<<1, kT, 0, stream>>>(args);
}

void Final(const FinalArgs& args, hipStream_t stream) {
  FinalKernel<<<1, kT, 0, stream>>>(args);
}

void Copy(const CopyArgs& args, hipStream_t stream) {
  CopyKernel<<<(kE + 255) / 256, 256, 0, stream>>>(args);
}

}  // namespace inkling2::gpu
