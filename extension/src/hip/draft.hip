// Kernels of the DSpark drafter. See draft.hpp.

#include "draft.hpp"

#include <cstddef>

namespace inkling2::gpu {
namespace {

constexpr int kThreads = 256;

__device__ inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  __builtin_memcpy(&value, &wide, sizeof value);
  return value;
}

__device__ inline float Half(std::uint16_t bits) {
  _Float16 h;
  __builtin_memcpy(&h, &bits, sizeof h);
  return static_cast<float>(h);
}

__device__ inline std::uint16_t ToHalfBits(float v) {
  v = fminf(fmaxf(v, -65504.0f), 65504.0f);
  const _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits;
  __builtin_memcpy(&bits, &h, sizeof bits);
  return bits;
}

// Sums shared[0..n) into shared[0]; n a power of two; every thread calls it.
__device__ inline void Sum(float* shared, int n) {
  const int t = threadIdx.x;
  __syncthreads();
  for (int step = n / 2; step > 0; step >>= 1) {
    if (t < step)
      shared[t] += shared[t + step];
    __syncthreads();
  }
}

// The largest of shared[0..n) into shared[0].
__device__ inline void Max(float* shared, int n) {
  const int t = threadIdx.x;
  __syncthreads();
  for (int step = n / 2; step > 0; step >>= 1) {
    if (t < step)
      shared[t] = fmaxf(shared[t], shared[t + step]);
    __syncthreads();
  }
}

// Rotation angle of pair j at position pos: pos * 1e6^(-2j / kHD).
__device__ inline void Angle(int pos, int j, float* cs, float* sn) {
  const double theta =
      static_cast<double>(pos) *
      exp(-13.815510557964274 * (2.0 * j / static_cast<double>(kHD)));
  *cs = static_cast<float>(cos(theta));
  *sn = static_cast<float>(sin(theta));
}

// One block per (row, r): r < 3 a layer's context input, r = 3 the head's.
__launch_bounds__(kThreads) __global__ void DraftMixKernel(DraftMixArgs a) {
  __shared__ float sh[kThreads];
  const int t = threadIdx.x;
  const int r = blockIdx.x % (kDraftLayers + 1);
  const std::size_t b = blockIdx.x / (kDraftLayers + 1);
  const float* f0 = a.feat + b * kE;
  const float* f1 = f0 + a.feat_block;
  const float* f2 = f1 + a.feat_block;
  const float* f3 = a.feat_final + b * kE;
  float* out = a.out + (b * (kDraftLayers + 1) + r) * kE;
  const float w0 = a.fuse[r][0], w1 = a.fuse[r][1], w2 = a.fuse[r][2],
              w3 = a.fuse[r][3];
  float mine = 0.0f;
  for (int i = t; i < kE; i += kThreads) {
    const float v = w0 * f0[i] + w1 * f1[i] + w2 * f2[i] + w3 * f3[i];
    out[i] = v;
    mine += v * v;
  }
  sh[t] = mine;
  Sum(sh, kThreads);
  const float inv = 1.0f / sqrtf(sh[0] / kE + a.eps);
  const float* w = a.norm[r];
  for (int i = t; i < kE; i += kThreads)
    out[i] = out[i] * inv * w[i];
}

// One block per (row, layer, key/value head), a thread per dimension.
__launch_bounds__(kHD) __global__ void DraftStoreKernel(DraftStoreArgs a) {
  __shared__ float sh[kHD];
  __shared__ float val[kHD];
  const int t = threadIdx.x;
  const int g = blockIdx.x % kNKV;
  const int l = (blockIdx.x / kNKV) % kDraftLayers;
  const std::size_t b = blockIdx.x / (kNKV * kDraftLayers);
  const DraftRow row = a.rows[b];
  if (!row.store[l])
    return;
  const float* k = a.k_raw + (b * kDraftLayers + l) * kHK + g * kHD;
  const float* v = a.v_raw + (b * kDraftLayers + l) * kHK + g * kHD;
  const std::size_t at =
      (static_cast<std::size_t>(row.pos % row.cap[l]) * kNKV + g) * kHD;
  const float x = k[t];
  sh[t] = x * x;
  Sum(sh, kHD);
  const float inv = 1.0f / sqrtf(sh[0] / kHD + a.eps);
  val[t] = x * inv * a.k_norm[l][t];
  __syncthreads();
  const int j = t % (kHD / 2);
  float cs, sn;
  Angle(row.pos, j, &cs, &sn);
  const float lo = val[j], hi = val[j + kHD / 2];
  const float rotated = t < kHD / 2 ? lo * cs - hi * sn : lo * sn + hi * cs;
  row.k[l][at + t] = ToHalfBits(rotated);
  row.v[l][at + t] = ToHalfBits(v[t]);
}

// One block per block row.
__launch_bounds__(kThreads) __global__ void DraftInitKernel(DraftInitArgs a) {
  __shared__ float sh[kThreads];
  const int t = threadIdx.x, b = blockIdx.x;
  float* x = a.x + static_cast<std::size_t>(b) * kE;
  if (b > 0) {
    for (int i = t; i < kE; i += kThreads)
      x[i] = a.mask[i];
    if (b == 1 && t < kDraftRank)
      a.state[t] = 0.0f;
    if (b == 2 && t == 0) {
      a.tokens[0] = a.anchor;
      *a.alive = 1;
    }
    return;
  }
  const std::uint16_t* row =
      a.tok_embd + static_cast<std::size_t>(a.anchor) * kE;
  float mine = 0.0f;
  for (int i = t; i < kE; i += kThreads) {
    const float v = Bf16(row[i]);
    mine += v * v;
  }
  sh[t] = mine;
  Sum(sh, kThreads);
  const float inv = 1.0f / sqrtf(sh[0] / kE + a.tok_eps);
  for (int i = t; i < kE; i += kThreads)
    x[i] = Bf16(row[i]) * inv * a.tok_norm[i];
}

// One block per block row.
__launch_bounds__(kThreads) __global__ void DraftResNormKernel(
    DraftResNormArgs a) {
  __shared__ float sh[kThreads];
  const int t = threadIdx.x;
  const std::size_t b = blockIdx.x;
  float* x = a.x + b * kE;
  float* dst = a.dst + b * kE;
  float mine = 0.0f;
  for (int i = t; i < kE; i += kThreads) {
    float v = x[i];
    if (a.add) {
      v += a.add[b * kE + i];
      x[i] = v;
    }
    mine += v * v;
  }
  sh[t] = mine;
  Sum(sh, kThreads);
  const float inv = 1.0f / sqrtf(sh[0] / kE + a.eps);
  for (int i = t; i < kE; i += kThreads)
    dst[i] = x[i] * inv * a.w[i];
}

// One block per (block row, head): the 24 query heads, then the 8 key heads.
__launch_bounds__(kHD) __global__ void DraftQkKernel(DraftQkArgs a) {
  __shared__ float sh[kHD];
  __shared__ float val[kHD];
  const int t = threadIdx.x;
  const int head = blockIdx.x % (kNH + kNKV);
  const int b = blockIdx.x / (kNH + kNKV);
  const bool query = head < kNH;
  float* x = query ? a.q + static_cast<std::size_t>(b) * kHQ + head * kHD
                   : a.k + static_cast<std::size_t>(b) * kHK + (head - kNH) * kHD;
  const float* w = query ? a.q_norm : a.k_norm;
  const float v = x[t];
  sh[t] = v * v;
  Sum(sh, kHD);
  const float inv = 1.0f / sqrtf(sh[0] / kHD + a.eps);
  val[t] = v * inv * w[t];
  __syncthreads();
  const int j = t % (kHD / 2);
  float cs, sn;
  Angle(a.pos0 + b, j, &cs, &sn);
  const float lo = val[j], hi = val[j + kHD / 2];
  x[t] = t < kHD / 2 ? lo * cs - hi * sn : lo * sn + hi * cs;
}

// One block per (block row, query head).
__launch_bounds__(kThreads) __global__ void DraftAttnKernel(DraftAttnArgs a) {
  __shared__ float sq[kHD];
  __shared__ float sh[kThreads];
  const int t = threadIdx.x;
  const int h = blockIdx.x % kNH;
  const int b = blockIdx.x / kNH;
  const int g = h / (kNH / kNKV);
  const int n_ctx = a.pos0 - a.first;
  const int n = n_ctx + kDraftBlock;
  const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
  float* score = a.scratch + static_cast<std::size_t>(blockIdx.x) * n;
  if (t < kHD)
    sq[t] = a.q[static_cast<std::size_t>(b) * kHQ + h * kHD + t];
  __syncthreads();
  float best = -3.0e38f;
  for (int i = t; i < n; i += kThreads) {
    float s = 0.0f;
    if (i < n_ctx) {
      const std::uint16_t* key =
          a.k_ring +
          (static_cast<std::size_t>((a.first + i) % a.cap) * kNKV + g) * kHD;
      for (int d = 0; d < kHD; ++d)
        s += sq[d] * Half(key[d]);
    } else {
      const float* key =
          a.k_blk + static_cast<std::size_t>(i - n_ctx) * kHK + g * kHD;
      for (int d = 0; d < kHD; ++d)
        s += sq[d] * key[d];
    }
    s *= scale;
    score[i] = s;
    best = fmaxf(best, s);
  }
  sh[t] = best;
  Max(sh, kThreads);
  best = sh[0];
  __syncthreads();
  float mine = 0.0f;
  for (int i = t; i < n; i += kThreads) {
    const float p = __expf(score[i] - best);
    score[i] = p;
    mine += p;
  }
  sh[t] = mine;
  Sum(sh, kThreads);
  const float inv = 1.0f / sh[0];
  __syncthreads();
  // Two threads per dimension, taking the even and the odd positions.
  const int d = t % kHD, part = t / kHD;
  float acc = 0.0f;
  for (int i = part; i < n; i += 2) {
    const float p = score[i];
    if (i < n_ctx) {
      const std::uint16_t* value =
          a.v_ring +
          (static_cast<std::size_t>((a.first + i) % a.cap) * kNKV + g) * kHD;
      acc += p * Half(value[d]);
    } else {
      acc += p *
             a.v_blk[static_cast<std::size_t>(i - n_ctx) * kHK + g * kHD + d];
    }
  }
  sh[t] = acc;
  __syncthreads();
  if (part == 0)
    a.out[static_cast<std::size_t>(b) * kHQ + h * kHD + d] =
        (sh[d] + sh[d + kHD]) * inv;
}

__global__ void DraftSiluKernel(float* gate, const float* up) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(kDraftBlock) * kDraftFF)
    return;
  const float g = gate[i];
  gate[i] = g / (1.0f + __expf(-g)) * up[i];
}

// One block per row of the joint projection.
constexpr int kJointThreads = 128;
__launch_bounds__(kJointThreads) __global__ void DraftJointKernel(
    DraftSlotArgs a) {
  __shared__ float sh[kJointThreads];
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  const float* w =
      a.joint + static_cast<std::size_t>(blockIdx.x) * kDraftJointIn;
  const std::uint16_t* e =
      a.markov + static_cast<std::size_t>(a.tokens[a.slot - 1]) * kDraftRank;
  const float* h = a.h + static_cast<std::size_t>(a.slot) * kE;
  float acc = 0.0f;
  for (int c = t; c < kDraftJointIn; c += kJointThreads) {
    const float u = c < kDraftRank           ? a.state[c]
                    : c < 2 * kDraftRank     ? Bf16(e[c - kDraftRank])
                    : c < 2 * kDraftRank + kE ? h[c - 2 * kDraftRank]
                                              : a.f_rnn[c - 2 * kDraftRank - kE];
    acc += w[c] * u;
  }
  sh[t] = acc;
  Sum(sh, kJointThreads);
  if (t == 0)
    a.j_raw[blockIdx.x] = sh[0] + a.joint_bias[blockIdx.x];
}

// One block: the gate, the candidate state and the latent.
__launch_bounds__(kDraftRank) __global__ void DraftGateKernel(DraftSlotArgs a) {
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  const float g = 1.0f / (1.0f + __expf(-a.j_raw[t]));
  const float cand = tanhf(a.j_raw[kDraftRank + t]);
  a.latent[t] = tanhf(a.j_raw[2 * kDraftRank + t]);
  a.state[t] = cand + g * (a.state[t] - cand);
}

// A thread per vocabulary entry: logit = base + m . latent + m_bias.
__launch_bounds__(kThreads) __global__ void DraftLogitsKernel(DraftSlotArgs a) {
  __shared__ float lat[kDraftRank];
  __shared__ float sh[kThreads];
  __shared__ int arg[kThreads];
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  lat[t] = a.latent[t];
  __syncthreads();
  const int v = blockIdx.x * kThreads + t;
  float logit = -3.0e38f;
  if (v < a.vocab) {
    // Column c of m is contiguous over the vocabulary, so a wave reads
    // neighbouring values at every step.
    const std::uint16_t* column = a.m + v;
    float acc = a.base[static_cast<std::size_t>(a.slot - 1) * a.vocab_stride + v] +
                a.m_bias[v];
    for (int c = 0; c < kDraftRank; ++c)
      acc += Half(column[static_cast<std::size_t>(c) * a.vocab_stride]) * lat[c];
    logit = acc;
    a.logits[v] = acc;
  }
  sh[t] = logit;
  arg[t] = v;
  __syncthreads();
  for (int step = kThreads / 2; step > 0; step >>= 1) {
    if (t < step && sh[t + step] > sh[t]) {
      sh[t] = sh[t + step];
      arg[t] = arg[t + step];
    }
    __syncthreads();
  }
  if (t == 0) {
    a.block_max[blockIdx.x] = sh[0];
    a.block_arg[blockIdx.x] = arg[0];
  }
}

// One block: the best of the block maxima is the slot's token.
constexpr int kPickThreads = 1024;
__launch_bounds__(kPickThreads) __global__ void DraftPickKernel(
    DraftSlotArgs a, int blocks) {
  __shared__ float sh[kPickThreads];
  __shared__ int arg[kPickThreads];
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  float best = -3.0e38f;
  int at = 0;
  for (int i = t; i < blocks; i += kPickThreads)
    if (a.block_max[i] > best) {
      best = a.block_max[i];
      at = a.block_arg[i];
    }
  sh[t] = best;
  arg[t] = at;
  __syncthreads();
  for (int step = kPickThreads / 2; step > 0; step >>= 1) {
    if (t < step && sh[t + step] > sh[t]) {
      sh[t] = sh[t + step];
      arg[t] = arg[t + step];
    }
    __syncthreads();
  }
  if (t == 0) {
    a.tokens[a.slot] =
        a.ids ? a.ids[arg[0]] : static_cast<std::uint32_t>(arg[0]);
    *a.top = sh[0];
  }
}

__device__ inline float Cap(float x, float cap) {
  return cap > 0.0f ? cap * tanhf(x / cap) : x;
}

// Per 256 vocabulary entries: the sum of exp(capped logit - capped top).
__launch_bounds__(kThreads) __global__ void DraftConfPartKernel(
    DraftSlotArgs a) {
  __shared__ float sh[kThreads];
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  const int v = blockIdx.x * kThreads + t;
  const float top = Cap(*a.top, a.softcap);
  sh[t] = v < a.vocab ? __expf(Cap(a.logits[v], a.softcap) - top) : 0.0f;
  Sum(sh, kThreads);
  if (t == 0)
    a.block_sum[blockIdx.x] = sh[0];
}

// One block: the token's probability; a low one ends the draft.
__launch_bounds__(kPickThreads) __global__ void DraftConfKernel(DraftSlotArgs a,
                                                               int blocks) {
  __shared__ float sh[kPickThreads];
  if (!*a.alive)
    return;
  const int t = threadIdx.x;
  float mine = 0.0f;
  for (int i = t; i < blocks; i += kPickThreads)
    mine += a.block_sum[i];
  sh[t] = mine;
  Sum(sh, kPickThreads);
  if (t == 0) {
    const float confidence = 1.0f / sh[0];
    a.confidence[a.slot] = confidence;
    if (confidence < a.min_confidence)
      *a.alive = 0;
  }
}

// One block per entry.
__launch_bounds__(kThreads) __global__ void DraftListFillKernel(
    DraftListArgs a) {
  const int t = threadIdx.x;
  const DraftListEntry entry = a.entries[blockIdx.x];
  const std::uint16_t* from =
      a.output + static_cast<std::size_t>(entry.token) * kE;
  std::uint16_t* to = a.list_output + static_cast<std::size_t>(entry.slot) * kE;
  for (int i = t; i < kE; i += kThreads)
    to[i] = from[i];
  a.list_m[static_cast<std::size_t>(t) * a.slots + entry.slot] =
      a.m[static_cast<std::size_t>(t) * a.vocab_stride + entry.token];
  if (t == 0) {
    a.list_bias[entry.slot] = a.m_bias[entry.token];
    a.list_ids[entry.slot] = entry.token;
  }
}

inline int Blocks(std::size_t items, int per_block) {
  return static_cast<int>((items + per_block - 1) / per_block);
}

}  // namespace

void DraftMix(const DraftMixArgs& args, hipStream_t stream) {
  DraftMixKernel<<<args.batch * (kDraftLayers + 1), kThreads, 0, stream>>>(args);
}

void DraftStore(const DraftStoreArgs& args, hipStream_t stream) {
  DraftStoreKernel<<<args.batch * kDraftLayers * kNKV, kHD, 0, stream>>>(args);
}

void DraftInit(const DraftInitArgs& args, hipStream_t stream) {
  DraftInitKernel<<<kDraftBlock, kThreads, 0, stream>>>(args);
}

void DraftResNorm(const DraftResNormArgs& args, hipStream_t stream) {
  DraftResNormKernel<<<kDraftBlock, kThreads, 0, stream>>>(args);
}

void DraftQk(const DraftQkArgs& args, hipStream_t stream) {
  DraftQkKernel<<<kDraftBlock * (kNH + kNKV), kHD, 0, stream>>>(args);
}

void DraftAttn(const DraftAttnArgs& args, hipStream_t stream) {
  DraftAttnKernel<<<kDraftBlock * kNH, kThreads, 0, stream>>>(args);
}

void DraftSilu(float* gate, const float* up, hipStream_t stream) {
  DraftSiluKernel<<<Blocks(std::size_t(kDraftBlock) * kDraftFF, 256), 256, 0,
                    stream>>>(gate, up);
}

void DraftSlot(const DraftSlotArgs& args, hipStream_t stream) {
  const int blocks = Blocks(static_cast<std::size_t>(args.vocab), kThreads);
  DraftJointKernel<<<kDraftJoint, kJointThreads, 0, stream>>>(args);
  DraftGateKernel<<<1, kDraftRank, 0, stream>>>(args);
  DraftLogitsKernel<<<blocks, kThreads, 0, stream>>>(args);
  DraftPickKernel<<<1, kPickThreads, 0, stream>>>(args, blocks);
  DraftConfPartKernel<<<blocks, kThreads, 0, stream>>>(args);
  DraftConfKernel<<<1, kPickThreads, 0, stream>>>(args, blocks);
}

void DraftListFill(const DraftListArgs& args, hipStream_t stream) {
  DraftListFillKernel<<<args.count, kThreads, 0, stream>>>(args);
}

}  // namespace inkling2::gpu
