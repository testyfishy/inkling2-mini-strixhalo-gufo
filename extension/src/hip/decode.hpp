#pragma once

// Kernels for Inkling 2 Mini layer visits that stay on the device, over a
// batch of rows. A row is one token of one session; rows of the same session
// are adjacent and in position order, so one batch can hold a token from each
// of several sessions (parallel decode), several consecutive tokens of one
// session (prompt processing), or a mix.
//
// A layer visit is: GlueA, [CtrlDown, CtrlApply], Norm, Gemm, GlueB, Attention,
// Gemm (output projection), GlueC, [CtrlDown, CtrlApply], Norm, Router, Experts
// and GlueD. Per-row buffers are contiguous with the strides noted below.
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
inline constexpr int kFF = 1152;    // expert hidden width
inline constexpr int kMaxStates = 8;
inline constexpr int kGemmGroup = 16;  // batch rows sharing one read of a weight row

/// What one row needs from its session for one layer visit.
struct RowState {
  float* kv_k;  // cap rows of kHK
  float* kv_v;
  // Convolution histories (K, V, attention output, feed-forward output): the
  // last kNC - 1 inputs before this row's run, and where the run's last row
  // stores the new ones. The two never alias.
  const float* hist_in[4];
  float* hist_out[4];
  int cap;    // rows of the KV cache (a ring on sliding-window layers)
  int first;  // first position attention reads
  int pos;    // this row's position
  int run;    // index of this row within its session's run in the batch
  int last;   // 1 on the last row of its session's run
  float tau;  // log-scaled position factor (1 on sliding-window layers)
  // How kv_k and kv_v are laid out.
  // 0 (sliding-window layers): one row of kHK values per position, a ring of
  //   `cap` rows.
  // 1 (full-attention layers): per key/value head, so that a block of threads
  //   reads contiguous memory over a long context. Keys are dimension-major
  //   inside chunks of 1,024 positions:
  //     kv_k[((head * chunk_cap + pos / 1024) * kHD + dim) * 1024 + pos % 1024]
  //   and values are position-major per head:
  //     kv_v[(head * cap + pos) * kHD + dim]
  int layout;
  int chunk_cap;  // cap / 1024, for layout 1
  // Optional FP16 copies of a layout 1 cache for the batched attention
  // kernel, laid out the other way round, as the matrix instruction wants
  // them: keys position-major per head,
  //     kh[(head * cap + pos) * kHD + dim]
  // and values dimension-major inside chunks of 1,024 positions,
  //     vh[((head * chunk_cap + pos / 1024) * kHD + dim) * 1024 + pos % 1024]
  std::uint16_t* kh;
  std::uint16_t* vh;
};

/// Token embedding rows, normalized, into cur (stride kE).
struct EmbedArgs {
  const std::uint16_t* table;  // BF16, rows of kE
  const std::uint32_t* tokens;
  const float* norm;
  float eps;
  float* cur;
  int batch;
};
void Embed(const EmbedArgs& args, hipStream_t stream);

/// Before the projections: keeps the layer input, mixes the earlier block
/// outputs into the three attention inputs (Q, K, V) and, on a looped visit,
/// mixes the cross-pass messages.
struct GlueAArgs {
  const float* cur;        // stride kE
  float* input;            // stride kE, copy of cur
  float* in;               // stride 3 * kE
  float* msgs;             // stride 4 * kE, written when looped
  const float* stack;      // stride kMaxStates * kE; n_blocks rows used
  int n_blocks;
  const float* res_query;  // n_res rows of kE; n_res is 0, 1 or 3
  int n_res;
  bool looped;
  const float* bank0;      // stride kE
  const float* bank_out;   // stride kE
  const float* msg_query;  // 4 * kE
  int batch;
};
void GlueA(const GlueAArgs& args, hipStream_t stream);

/// The loop controllers of `count` vectors per row: low = down [rms(cur),
/// rms(msg)], then cur = (1 - tanh(forget low)) * cur + tanh(gain low) * msg.
/// Vector c of row b is at cur + b * cur_stride + c * kE (msg likewise); its
/// projection is at low + (b * 3 + c) * kRank.
struct CtrlArgs {
  float* cur;
  int cur_stride;
  const float* msg;
  int msg_stride;
  const float* down[3];
  const float* gain[3];
  const float* forget[3];
  float* low;
  int count;
  int batch;
};
void CtrlDown(const CtrlArgs& args, hipStream_t stream);
void CtrlApply(const CtrlArgs& args, hipStream_t stream);

/// x = rms_norm(x) * w in place, for `count` kE-wide vectors per row at
/// x + b * stride + j * kE.
struct NormArgs {
  float* x;
  int stride;
  const float* w[3];
  int count;
  int batch;
  float eps;
};
void Norm(const NormArgs& args, hipStream_t stream);

struct GemmJob {
  const std::uint16_t* w;  // BF16, rows of cols values
  const float* x;
  int x_stride;
  float* y;
  int y_stride;
  int rows, cols;
  // Optional column-major copy of the inputs (see Transpose): when set, the
  // kernel reads it instead of x. `xt` points at this job's first column of
  // group 0; groups are xt_group_stride floats apart.
  const float* xt;
  int xt_group_stride;
  // Optional FP16 copies for the matrix-instruction kernel (GemmArgs::wmma):
  // the weights (BF16 converts to FP16 exactly except for values too small to
  // matter) and the inputs, row-major, xh_stride values per batch row.
  const std::uint16_t* wh;
  const std::uint16_t* xh;
  int xh_stride;
};
struct GemmArgs {
  GemmJob jobs[9];
  int count;
  int batch;
  int variant;  // 0: a block per output row; 1: a thread per output row
  // Threads per output row of the single-row kernel: 64 or 128 (0 = 128).
  int single_threads;
  // Use the 16x16x16 FP16 matrix instruction with FP32 accumulation. Every
  // job needs wh and xh, and cols a multiple of 16.
  bool wmma;
  // Kernel choice, for experiments: 0 picks by batch; 1, 2, 4 force the
  // staged kernel 32, 64 or 128 batch rows wide; 8 forces the direct kernel.
  int wmma_tiles;
};
/// Every job's y[b] = W x[b] for every row, in one launch.
void Gemm(const GemmArgs& args, hipStream_t stream);

/// Column-major copy of `cols` values per row for Gemm: the batch is cut into
/// groups of kGemmGroup rows, and group g holds its rows' values of column c
/// together at xt[(g * cols + c) * kGemmGroup + (b - g * kGemmGroup)].
struct TransposeArgs {
  const float* x;
  int x_stride;
  float* xt;
  int cols;
  int batch;
};
void Transpose(const TransposeArgs& args, hipStream_t stream);

/// FP16 copy of `cols` values per row, row-major, for the matrix-instruction
/// kernel. Values are rounded to nearest and clamped to the FP16 range.
struct ToHalfArgs {
  const float* x;
  int x_stride;
  std::uint16_t* out;
  int cols;
  int batch;
};
void ToHalf(const ToHalfArgs& args, hipStream_t stream);

/// After the projections: K and V short convolutions, the per-head clips and
/// norms, and the KV cache store.
struct GlueBArgs {
  const float* k_raw;   // stride kHK
  const float* v_raw;   // stride kHK
  float* q;             // stride kHQ, clipped in place
  float* r;             // stride kNR, clipped in place
  const float* k_taps;  // stride kNC * kNKV, null on sliding-window layers
  const float* v_taps;
  const float* k_conv;  // kHK * kNC
  const float* v_conv;
  const float* q_clip;    // kHQ
  const float* rel_clip;  // kNR
  float qk_clip, rel_clip_threshold, eps;
  const RowState* rows;
  int batch;
};
void GlueB(const GlueBArgs& args, hipStream_t stream);

/// Up to kAttnRows consecutive batch rows of one session, for the batched
/// attention kernel.
inline constexpr int kAttnRows = 32;
struct AttnGroup {
  std::int32_t row0, count;
};

/// Attention of each row over its session's cached rows [first, pos], with the
/// relative-position bias, the zero sink and the output gate.
struct AttnArgs {
  const float* q;         // stride kHQ
  const float* r;         // stride kNR
  const float* sink;      // stride kNH
  const float* gate;      // stride kHQ, logits
  const float* rel_bias;  // extent * kNRel
  float* scratch;         // batch * kNH * max_n
  float* out;             // stride kHQ
  int extent;
  int max_n;
  float scale;
  const RowState* rows;
  int batch;
  // Set when no row reads more than 1,024 positions: the kernel then gives
  // every position its own thread and keeps the probabilities in fast local
  // memory.
  bool window;
  // Set for long contexts: the positions are cut into chunks of 1,024, each
  // (row, key head, chunk) is one block that reads its keys and values once
  // for the three query heads sharing them, and a second kernel combines the
  // chunks. Needs the chunk scratch below.
  bool chunked;
  int max_chunks;
  float* chunk_max;  // batch * kNH * max_chunks
  float* chunk_sum;  // batch * kNH * max_chunks
  float* chunk_out;  // batch * kNH * max_chunks * kHD
  // With `chunked`: one block per (group of rows, key head, chunk) instead of
  // per (row, key head, chunk). It reads the chunk's FP16 keys and values
  // once for all of the group's rows and multiplies with the 16x16x16 FP16
  // matrix instruction (FP32 sums). Needs RowState::kh / vh, an FP16 copy of
  // q, and the groups.
  bool batched;
  const std::uint16_t* qh;  // stride kHQ
  const AttnGroup* groups;
  int group_count;
};
void Attention(const AttnArgs& args, hipStream_t stream);

/// After the output projection: the post-attention convolution, the residual
/// or block push, and the feed-forward input mix into xn.
struct GlueCArgs {
  const float* proj;       // stride kE
  const float* post_conv;  // kE * kNC
  const float* input;      // stride kE
  bool block_start;
  float* stack;
  int n_blocks;            // before this layer's push
  float* cur;              // written: the feed-forward residual input
  float* xn;               // written: the mixed feed-forward input
  const float* ffn_res_query;
  const RowState* rows;
  int batch;
};
void GlueC(const GlueCArgs& args, hipStream_t stream);

/// The MoE router: logits = gate_inp x, then the top experts by
/// sigmoid(logit) + bias with their normalized weights.
struct RouterArgs {
  const float* gate_inp;      // kNX * kE
  const float* x;             // stride kE
  float* logits;              // stride kNX, scratch
  const float* exp_probs_b;   // kNX or null
  const float* up_scale_all;  // kNX or null
  const float* down_scale_all;
  float weights_scale;
  std::uint32_t* ids;  // stride kTopK, written
  float* up_scale;     // stride kTopK, written
  float* coef;         // stride kTopK, written
  int batch;
};
void Router(const RouterArgs& args, hipStream_t stream);

/// The selected experts of each row: act = situ(gate x, up x) per expert,
/// then out = sum of coef * (down act).
struct BatchExpertArgs {
  const std::uint8_t* gate_up;  // per expert: 2 * kFF rows of kE values
  const std::uint8_t* down;     // per expert: kE rows of kFF values
  const float* x;               // stride kE
  const std::uint32_t* ids;     // stride kTopK
  const float* up_scale;        // stride kTopK
  const float* coef;            // stride kTopK
  float* act;                   // stride kTopK * kFF, scratch
  float* out;                   // stride kE
  float situ_beta, situ_linear_beta;
  bool nvfp4;                   // NVFP4 storage, otherwise BF16
  int batch;
  // 0: a block per (row, output value); 1: a thread per output value;
  // 2: expert-major, where each expert's weights are decoded once and applied
  //    to every row that selected it;
  // 3: as 2, but the NVFP4 products use 8-bit integer dot products: the
  //    weights stay exact (they are small integers times a scale) while the
  //    inputs are rounded to 8 bits per block of 16. BF16 layers use 2;
  // 4: the 8-bit integer products with a block per (row, output value), for
  //    steps of few rows. BF16 layers use 0;
  // 5: expert-major with the 16x16x16 FP16 matrix instruction: the weights
  //    are exact as FP16 (NVFP4 always, BF16 except for values too small to
  //    matter), the inputs are rounded to FP16, sums are FP32. For steps of
  //    many rows, where experts are shared by many rows.
  int variant;
  // Expert-major scratch: for each expert, the (row * kTopK + slot) pairs that
  // selected it (group_list, max_batch entries per expert) and how many
  // (group_count), and each pair's weighted down projection (contrib, kE per
  // pair).
  std::int32_t* group_list;
  std::int32_t* group_count;
  float* contrib;
  int max_batch;
  // Variant 3 scratch: the 8-bit inputs (four values per word) and their
  // per-16 scales, for x (kE per row) and for act (kFF per pair).
  std::uint32_t* xq;
  float* xs;
  std::uint32_t* aq;
  float* as;
  // Variant 5 scratch: FP16 copies of x (kE per row) and act (kFF per pair).
  std::uint16_t* xh;
  std::uint16_t* acth;
};
void BatchExperts(const BatchExpertArgs& args, hipStream_t stream);

/// After the experts: the post-feed-forward convolution and the residual add.
struct GlueDArgs {
  const float* moe;        // stride kE
  const float* post_conv;  // kE * kNC
  float* cur;              // stride kE, += conv(moe)
  const RowState* rows;
  int batch;
};
void GlueD(const GlueDArgs& args, hipStream_t stream);

/// The output mix over the blocks and the final norm, for the listed rows.
struct FinalArgs {
  const float* stack;
  int n_blocks;
  const float* cur;
  const float* res_query;
  const float* norm;
  float eps;
  const std::int32_t* head_rows;  // count batch-row indices
  float* out;                     // stride kE, one per listed row
  int count;
};
void Final(const FinalArgs& args, hipStream_t stream);

/// Up to three kE-wide copies per row in one launch. Every source is read
/// before any destination is written, element by element.
struct CopyArgs {
  float* dst[3];
  int dst_stride[3];
  const float* src[3];
  int src_stride[3];
  int count;
  int batch;
};
void Copy(const CopyArgs& args, hipStream_t stream);

/// Tuning aid: y[r][b] = sum over 4 * words bytes of w[r] * x[b] with 8-bit
/// integer dot products, the inputs column-major as for Gemm (one 32-bit word
/// of four values per row, kGemmGroup rows per column).
void Dot4Bench(const std::uint32_t* w, const std::uint32_t* xt, int rows,
               int words, int batch, int* y, hipStream_t stream);
/// The same with BF16 pairs: each 32-bit word of w and of xt holds two BF16
/// values, multiplied and accumulated by one instruction.
void Dot2Bench(const std::uint32_t* w, const std::uint32_t* xt, int rows,
               int words, int batch, float* y, hipStream_t stream);
/// The same shape in F32: y[r][b] = sum of w[r][c] * x[b][c].
void FloatBench(const float* w, const float* xt, int rows, int cols, int batch,
                float* y, hipStream_t stream);

}  // namespace inkling2::gpu
