#pragma once

// Kernels for one Inkling 2 Mini decode step that stays on the device. A layer
// visit is: GlueA, GemvMulti, GlueB, Attention, GemvMulti (output projection),
// GlueC, the expert kernels (ops.hpp) and GlueD.
//
// The dimensions are Inkling 2 Mini's; the host refuses any other file.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace inkling2::gpu {

inline constexpr int kE = 2304;     // model width
inline constexpr int kHD = 128;     // head dimension
inline constexpr int kNH = 24;      // query heads
inline constexpr int kNKV = 8;      // key/value heads
inline constexpr int kHQ = kNH * kHD;
inline constexpr int kHK = kNKV * kHD;
inline constexpr int kNRel = 16;    // relative-position features per head
inline constexpr int kNR = kNH * kNRel;
inline constexpr int kNC = 4;       // short-convolution taps
inline constexpr int kNX = 128;     // experts
inline constexpr int kTopK = 8;     // experts used
inline constexpr int kRank = 64;    // loop controller rank
inline constexpr int kMaxStates = 8;

/// A loop controller: down {2E -> rank}, gain and forget {rank -> E}, F32.
struct Ctrl {
  const float* down;
  const float* gain;
  const float* forget;
};

/// Before the projections: keeps the layer input, mixes the earlier block
/// outputs into the attention inputs, applies the cross-pass messages, and
/// normalizes. Writes the three inputs (Q, K, V) to `in` (3 * E).
struct GlueAArgs {
  const float* cur;
  float* input;         // copy of cur (the residual branch)
  float* in;            // 3 * E
  float* msgs;          // 4 * E, written when looped
  const float* stack;   // n_blocks rows of E
  int n_blocks;
  const float* res_query;  // n_res rows of E; n_res is 0, 1 or 3
  int n_res;
  bool looped;
  const float* bank0;      // the previous pass's block after the first core block
  const float* bank_out;   // the previous pass's output
  const float* msg_query;  // 4 * E
  Ctrl ctrl[3];
  const float* norm[3];    // norm weight of each input
  float eps;
};
void GlueA(const GlueAArgs& args, hipStream_t stream);

struct GemvJob {
  const std::uint16_t* w;  // BF16, rows of cols values
  const float* x;
  float* y;
  int rows, cols;
};
struct GemvArgs {
  GemvJob jobs[9];
  int count;
};
/// Every job's y = W x in one launch.
void GemvMulti(const GemvArgs& args, hipStream_t stream);

/// After the projections: K and V short convolutions, the per-head clips and
/// norms, and the KV cache store.
struct GlueBArgs {
  const float* k_raw;   // kHK
  const float* v_raw;   // kHK
  float* q;             // kHQ, clipped in place
  float* r;             // kNR, clipped in place
  const float* k_taps;  // kNC * kNKV, null on sliding-window layers
  const float* v_taps;
  const float* k_conv;  // kHK * kNC
  const float* v_conv;
  float* hist_k;        // (kNC - 1) * kHK
  float* hist_v;
  const float* q_clip;    // kHQ
  const float* rel_clip;  // kNR
  float tau, qk_clip, rel_clip_threshold, eps;
  float* k_row;  // this position's KV cache rows, kHK each
  float* v_row;
};
void GlueB(const GlueBArgs& args, hipStream_t stream);

/// Attention of one query row over the cached rows [first, pos], with the
/// relative-position bias, the zero sink and the output gate.
struct AttnArgs {
  const float* q;         // kHQ
  const float* r;         // kNR
  const float* sink;      // kNH
  const float* gate;      // kHQ logits
  const float* rel_bias;  // extent * kNRel
  const float* kv_k;      // cap rows of kHK
  const float* kv_v;
  float* scratch;         // kNH * (pos - first + 1)
  float* out;             // kHQ
  int extent, cap, first, pos;
  float scale;
};
void Attention(const AttnArgs& args, hipStream_t stream);

/// After the output projection: the post-attention convolution, the residual
/// or block push, the feed-forward input (mix, message, norm) and the router.
struct GlueCArgs {
  const float* proj;       // kE
  const float* post_conv;  // kE * kNC
  float* hist;             // (kNC - 1) * kE
  const float* input;      // kE
  bool block_start;
  float* stack;
  int n_blocks;            // before this layer's push
  float* cur;              // written: the feed-forward residual input
  float* xn;               // written: the experts' input
  const float* ffn_res_query;
  bool looped;
  const float* msg_ffn;
  Ctrl ctrl;
  const float* ffn_norm;
  float eps;
  const float* gate_inp;      // kNX * kE
  const float* exp_probs_b;   // kNX or null
  const float* up_scale_all;  // kNX or null
  const float* down_scale_all;
  float weights_scale;
  std::uint32_t* ids;  // kTopK, written
  float* up_scale;     // kTopK, written
  float* coef;         // kTopK, written
};
void GlueC(const GlueCArgs& args, hipStream_t stream);

/// After the experts: the post-feed-forward convolution and the residual add.
struct GlueDArgs {
  const float* moe;        // kE
  const float* post_conv;  // kE * kNC
  float* hist;
  float* cur;              // += conv(moe)
};
void GlueD(const GlueDArgs& args, hipStream_t stream);

/// Token embedding row, normalized.
struct EmbedArgs {
  const std::uint16_t* row;  // kE, BF16
  const float* norm;
  float eps;
  float* cur;
};
void Embed(const EmbedArgs& args, hipStream_t stream);

/// The output mix over the blocks and the final norm.
struct FinalArgs {
  const float* stack;
  int n_blocks;
  const float* cur;
  const float* res_query;
  const float* norm;
  float eps;
  float* out;
};
void Final(const FinalArgs& args, hipStream_t stream);

/// Up to three E-wide copies in one launch. Every source is read before any
/// destination is written, element by element.
struct CopyArgs {
  float* dst[3];
  const float* src[3];
  int count;
};
void Copy(const CopyArgs& args, hipStream_t stream);

}  // namespace inkling2::gpu
