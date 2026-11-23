#include "decoder.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hip/decode.hpp"
#include "hip/draft.hpp"

namespace inkling2 {
namespace {

using gpu::kE;
using gpu::kHK;
using gpu::kHQ;
using gpu::kMaxStates;
using gpu::kNC;
using gpu::kNH;
using gpu::kNKV;
using gpu::kNR;
using gpu::kNX;
using gpu::kRank;
using gpu::kTopK;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("inkling2 gpu: ") + what + ": " +
                             hipGetErrorString(status));
}

int EnvOr(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoi(v) : fallback;
}

// FP32 to FP16 bits, round to nearest even; `exact` is cleared when the value
// changed.
std::uint16_t HalfBits(float f, bool* exact) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  x &= 0x7fffffffu;
  if (x >= 0x47800000u) {  // 65536 and above, infinities, NaN
    *exact = false;
    return static_cast<std::uint16_t>(sign | (x > 0x7f800000u ? 0x7e00u : 0x7c00u));
  }
  if (x < 0x38800000u) {  // below 2^-14: subnormal in FP16
    if (x < 0x33000000u) {
      *exact = x == 0;
      return static_cast<std::uint16_t>(sign);
    }
    const int shift = 126 - static_cast<int>(x >> 23);
    const std::uint32_t m = (x & 0x7fffffu) | 0x800000u;
    std::uint32_t h = m >> shift;
    const std::uint32_t rem = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (h & 1)))
      ++h;
    *exact = rem == 0;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = (((x >> 23) - 112) << 10) | ((x >> 13) & 0x3ffu);
  const std::uint32_t rem = x & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1)))
    ++h;
  *exact = rem == 0;
  return static_cast<std::uint16_t>(sign | h);
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
  const std::uint32_t exponent = (h >> 10) & 0x1fu, mantissa = h & 0x3ffu;
  float magnitude;
  if (exponent == 0)
    magnitude = std::ldexp(static_cast<float>(mantissa), -24);
  else
    magnitude = std::ldexp(static_cast<float>(mantissa | 0x400u),
                           static_cast<int>(exponent) - 25);
  return sign ? -magnitude : magnitude;
}

}  // namespace

struct GpuEngine::Impl {
  // A small F32 matrix as hi + lo, both FP16 (lo null when it is all zero).
  struct Split {
    const std::uint16_t* hi{};
    const std::uint16_t* lo{};
  };
  struct SplitJob {
    Split w;
    const std::uint16_t *xh, *xl;
    int x_stride;
    float* y;
    int y_stride, rows, cols;
  };
  static bool attention_window_env() {
    const char* v = std::getenv("INKLING2_ATTN_WINDOW");
    return !(v && *v == '0');
  }
  // Device copies of one physical layer's weights.
  struct Layer {
    const std::uint16_t *wq, *wk, *wv, *wo, *gate, *sink, *rel, *kcp, *vcp;
    const std::uint8_t *gate_up, *down;
    bool nvfp4;
    const float *q_clip, *rel_bias, *rel_clip, *k_conv, *v_conv, *attn_post,
        *ffn_post, *norm[3], *res_query, *ffn_res_query, *ffn_norm, *gate_inp,
        *exp_probs_b, *up_scale, *down_scale, *msg_query;
    const float *ctrl_down[4], *ctrl_gain[4], *ctrl_forget[4];
    // The router and the controllers' down matrices (cur columns, msg
    // columns) as two FP16 parts each, for SplitGemm.
    Split gate_split, down_cur[4], down_msg[4], gain_split[4], forget_split[4];
  };
  // One session's state for one layer visit. The convolution histories are
  // kept twice: a step reads one copy and writes the other.
  struct Slot {
    // The F32 cache is only kept with INKLING2_F32_CACHE=1.
    float *k{}, *v{};
    int cap;
    std::uint16_t *kh{}, *vh{};
  };
  struct Session {
    std::vector<Slot> slots;
    std::uint32_t pos{0};
    // The convolution histories of every layer visit, contiguous: per visit
    // the last kNC - 1 inputs of K, V, attention output and feed-forward
    // output (kHistFloats values).
    float* hist{};
    // The run of the last tentative step: its first batch row and length.
    int pending_first{-1}, pending_count{0};
    // With a drafter: per layer the key ring and the value ring, then the
    // head's feature of the last kept row (kE floats). One allocation.
    std::uint8_t* draft_state{};
    bool draft_ready{false};
    // The windows of the sliding-window visits, packed for a cache: per such
    // visit, in order, 2 * kRingWindowBytes.
    std::uint16_t* ring_stage{};
    // The shortlist of the vocabulary the drafter's head scores for this
    // session. Host side: where each token is, what each slot holds, which
    // slots were asked for again since the clock hand passed, and the
    // entries not yet copied to the device. Device side: the head's matrices
    // for the listed tokens.
    struct Shortlist {
      int used{0}, hand{0};
      std::vector<std::int32_t> slot_of;
      std::vector<std::uint32_t> token_of;
      std::vector<std::uint8_t> touched;
      std::vector<gpu::DraftListEntry> pending;
      std::uint16_t *output{}, *m{};
      float* bias{};
      std::uint32_t* ids{};
    } list;
  };
  static constexpr std::size_t kRingWindowBytes = std::size_t{1024} * kHK * 2;
  // Positions zeroed after the last row of a step of several rows: the tile
  // of 64 the attention kernels stage.
  static constexpr int kClearAhead = 64;
  std::vector<int> ring_index;  // per visit: its place among the ring visits

  void RingWindow(std::uint32_t index, std::uint32_t visit,
                  std::uint32_t position, bool unpack) {
    Session& session = sessions.at(index);
    const Slot& slot = session.slots.at(visit);
    if (ring_index.at(visit) < 0)
      throw std::runtime_error("inkling2 gpu: not a sliding-window visit");
    gpu::RingWindowArgs a{};
    a.kh = slot.kh;
    a.vh = slot.vh;
    a.cap = slot.cap;
    a.position = static_cast<int>(position);
    a.window = 1024;
    a.stage_k = session.ring_stage +
                std::size_t(ring_index[visit]) * (kRingWindowBytes / 2) * 2;
    a.stage_v = a.stage_k + kRingWindowBytes / 2;
    a.unpack = unpack;
    gpu::RingWindow(a, stream);
  }

  // The drafter on the device.
  struct DraftLayer {
    const std::uint16_t *wq, *wk, *wv, *wo, *wk_ctx, *wv_ctx, *gate, *up, *down;
    const float *ctx_norm, *attn_norm, *ffn_norm, *q_norm, *k_norm, *k_ctx_norm;
  };
  struct Draft {
    DraftLayer layers[gpu::kDraftLayers];
    const float *head_norm, *mask, *output_norm, *joint, *joint_bias;
    const std::uint16_t* markov;
    std::uint16_t* m;
    float* m_bias;
    int cap[gpu::kDraftLayers];
    std::size_t ring_offset[gpu::kDraftLayers];  // of the key ring; values follow
    std::size_t f_rnn_offset, state_bytes;
    // Per batch row of a step.
    float *mix, *k_raw, *v_raw;
    std::uint16_t* mix_h;
    // One draft block.
    float *x, *a, *q, *k, *v, *o, *proj, *gate, *up, *h, *base, *logits,
        *state, *j_raw, *latent, *block_max, *block_sum, *top, *confidence,
        *scratch;
    std::int32_t *block_arg, *alive;
    std::uint32_t* tokens;
    // FP16 copies of the inputs of a block's products.
    std::uint16_t *a_h, *o_h, *gate_h, *h_h;
    // Entries of a session's shortlist of the vocabulary (0: none), the
    // staging for new entries, and the bias of an empty list.
    int slots{0};
    gpu::DraftListEntry* list_entries{};
    float* list_empty_bias{};
    std::size_t listed{0}, evicted{0};
    // INKLING2_DRAFT_PROFILE: seconds in the layers, the output matrix and
    // the slots, and calls.
    double seconds[3] = {};
    std::size_t calls{0};
  };
  bool draft_profile{EnvOr("INKLING2_DRAFT_PROFILE", 0) != 0};
  // Experiments: which of a draft's products use the matrix instruction
  // (1 the layers, 2 the output matrix) and which of its kernels (see
  // GemmArgs::wmma_tiles).
  int draft_wmma{EnvOr("INKLING2_DRAFT_WMMA", 3)};
  int draft_tiles{EnvOr("INKLING2_DRAFT_TILES", 1)};
  int draft_head_tiles{EnvOr("INKLING2_DRAFT_HEAD_TILES", 1)};
  int head_tiles{EnvOr("INKLING2_WMMA_HEAD_TILES", -1)};
  void DraftWmma(gpu::GemmArgs& g, const float* x, int cols, std::uint16_t* xh,
                 bool head) {
    if (!(draft_wmma & (head ? 2 : 1)))
      return;
    if (Wmma(g, x, cols, xh))
      g.wmma_tiles = head ? draft_head_tiles : draft_tiles;
  }
  std::unique_ptr<Draft> draft;
  const gpu::DraftRow* dev_draft_rows{};
  std::size_t off_draft{0}, off_ends{0};
  bool need_features{false};

  [[nodiscard]] std::uint16_t* DraftRing(const Session& session, int layer,
                                         bool values) const {
    const std::size_t bytes = std::size_t(draft->cap[layer]) * kHK * 2;
    return reinterpret_cast<std::uint16_t*>(
        session.draft_state + draft->ring_offset[layer] + (values ? bytes : 0));
  }
  [[nodiscard]] float* DraftFeature(const Session& session) const {
    return reinterpret_cast<float*>(session.draft_state + draft->f_rnn_offset);
  }

  static constexpr std::size_t kHistFloats = (kNC - 1) * (2 * kHK + 2 * kE);
  static constexpr std::size_t kHistOffset[4] = {
      0, (kNC - 1) * kHK, 2 * (kNC - 1) * kHK,
      2 * (kNC - 1) * kHK + (kNC - 1) * kE};
  // Each batch row's candidate histories for every visit, written by the rows
  // that may end a run; committing a row copies its candidate to its session.
  float* hist_candidates{};
  std::size_t hist_slots{0};
  std::vector<std::uint32_t> hist_slot_of;  // per batch row of the step

  const Model& m;
  const Config& c;
  Options opt;
  hipStream_t stream{};
  std::vector<void*> owned;
  std::vector<Layer> layers;
  std::vector<Session> sessions;
  const std::uint16_t *tok_embd{}, *output{};
  const float *tok_norm{}, *output_norm{}, *output_res_query{};
  // Per-row working buffers.
  float *cur{}, *input{}, *in{}, *msgs{}, *stack{}, *prelude{}, *bank0{},
      *bank_out{}, *q{}, *r{}, *gate{}, *sink{}, *k_raw{}, *v_raw{}, *k_taps{},
      *v_taps{}, *attn{}, *proj{}, *xn{}, *sel_up{}, *sel_coef{}, *act{},
      *moe{}, *scratch{}, *final_x{}, *logits{}, *low{}, *router_logits{},
      *xt_in{}, *xt_attn{}, *contrib{};
  std::int32_t *group_list{}, *group_count{}, *expert_items{},
      *expert_items_full{};
  std::uint32_t *xq{}, *aq{};
  float *xs{}, *as{};
  int major_min_batch{4};
  bool transpose{true};
  // Steps of at least this many rows run the projections through the FP16
  // matrix-instruction kernel (0 = never). INKLING2_WMMA_MIN_BATCH.
  int wmma_min_batch{EnvOr("INKLING2_WMMA_MIN_BATCH", 2)};
  int wmma_tiles{EnvOr("INKLING2_WMMA_TILES", 0)};
  std::unordered_map<const void*, const std::uint16_t*> half_of;
  std::size_t half_values{0}, half_inexact{0};
  std::uint16_t *xh_in{}, *xh_attn{}, *xh_head{}, *xh_x{}, *acth{};
  // Steps of at least this many rows run the experts through the FP16
  // matrix-instruction kernels (0 = never). INKLING2_WMMA_EXPERT_MIN_BATCH.
  int wmma_expert_min_batch{EnvOr("INKLING2_WMMA_EXPERT_MIN_BATCH", 16)};
  int gemm_variant{0}, expert_variant{0};
  int single_threads{128}, head_threads{64};
  bool attention_window{true};
  // The key and value caches are FP16. INKLING2_F32_CACHE=1 keeps F32 caches
  // beside them (three times the memory), which the F32 attention kernels
  // need: they are then used by steps the two switches below leave to them,
  // and by INKLING2_ATTN_AB as the reference.
  bool keep_f32{EnvOr("INKLING2_F32_CACHE", 0) != 0};
  // INKLING2_ATTN_HALF=0 (with the F32 cache): per-row attention on F32.
  bool attn_half{EnvOr("INKLING2_ATTN_HALF", 1) != 0};
  // Steps with at least this many rows per session on average run attention
  // through the batched FP16 kernel, which reads the cache once per group of
  // rows (0 = never). Steps of one row per session are faster with the
  // per-row kernel. INKLING2_WMMA_ATTN_MIN_ROWS.
  int attn_batched_min{EnvOr("INKLING2_WMMA_ATTN_MIN_ROWS", 4)};
  // INKLING2_WMMA_ATTN_WINDOW=0 keeps sliding-window layers on the per-row
  // kernel.
  bool attn_batched_window{EnvOr("INKLING2_WMMA_ATTN_WINDOW", 1) != 0};
  std::uint16_t* qh{};
  const gpu::AttnGroup* dev_groups{};
  std::size_t off_groups{0};
  int n_groups{0};
  // INKLING2_ATTN_AB=1 (with the F32 cache): also run the per-row F32 kernels
  // on every visit and report how far the outputs are from theirs.
  bool attn_ab{EnvOr("INKLING2_ATTN_AB", 0) != 0};
  std::vector<float> ab_old, ab_new;
  // INKLING2_ATTN_STATS=N: each time a session's position crosses a multiple
  // of N, measure on the host how concentrated the full-attention layers'
  // weights are for the step's last row (a diagnostic; it is slow).
  int attn_stats_every{EnvOr("INKLING2_ATTN_STATS", 0)};
  bool attn_stats_now{false};
  int attn_skip{EnvOr("INKLING2_ATTN_SKIP", 0)};
  int expert_skip{EnvOr("INKLING2_EXPERT_SKIP", 0)};
  bool expert_stats{EnvOr("INKLING2_EXPERT_STATS", 0) != 0};
  std::size_t stat_experts{0}, stat_empty{0}, stat_pairs{0}, stat_tiles128{0},
      stat_tiles256{0}, stat_over128{0}, stat_over256{0}, stat_over512{0},
      stat_max{0};
  // Bit 16: the one-pass kernel (AttentionFlashKernel), whose blocks take
  // groups of up to 64 rows; the older batched kernels take 32.
  int attn_fast{EnvOr("INKLING2_ATTN_FAST", 63)};
  // Bit 32 of INKLING2_ATTN_FAST: steps of at least this many rows, and
  // fewer than the one-pass kernel takes whole, run full attention through
  // the one-pass kernel with the context in parts.
  int attn_few_min{EnvOr("INKLING2_ATTN_FEW_MIN", 2)};
  int attn_few_parts{EnvOr("INKLING2_ATTN_FEW_PARTS", 32)};
  bool few_now{false};
  // Steps of fewer rows (draft verification) keep the kernels that split a
  // context into parts: one block walking a whole context for a handful of
  // rows was slower.
  int attn_flash_min{EnvOr("INKLING2_ATTN_FLASH_MIN", 64)};
  bool flash_now{false};
  int wmma_direct{(gpu::SetWmmaDirect(EnvOr("INKLING2_WMMA_DIRECT", 0)), 0)};
  std::uint32_t stats_pos{0};
  struct HeadStats {
    double sink, last_1024, top_2048, pos_999, pos_9999, slice_999, slice_9999,
        slice_99999, chunk_9999;
  };
  std::vector<HeadStats> stats;
  // The host's per-chunk maximum and sum of the visit just measured, to be
  // compared with what the attention kernel leaves in its chunk scratch.
  std::vector<double> stats_chunk_max, stats_chunk_sum;
  double stats_max_diff{0}, stats_sum_diff{0};
  std::size_t stats_chunks{0};
  double ab_max_diff{0}, ab_max_abs{0}, ab_sq_diff{0}, ab_sq{0};
  std::size_t ab_bad{0}, ab_visits{0};
  int max_chunks{1};
  std::size_t partial_slots{0};
  int head_wmma_min{EnvOr("INKLING2_WMMA_HEAD_MIN_BATCH", 16)};
  // Steps of at least this many rows compute the router and the loop
  // controllers' projections with SplitGemm (0 = never).
  // INKLING2_SPLIT_GEMM_MIN_BATCH.
  int split_min_batch{EnvOr("INKLING2_SPLIT_GEMM_MIN_BATCH", 2)};
  std::size_t split_count{0}, split_with_lo{0};
  // The loop controllers' products take their inputs as one FP16 value each,
  // like every other projection; INKLING2_CTRL_LOW_INPUTS=1 adds the part
  // that rounding lost, as the router always does (it picks the experts).
  bool ctrl_low{EnvOr("INKLING2_CTRL_LOW_INPUTS", 0) != 0};
  std::uint16_t *sh_in{}, *sl_in{}, *sh_m{}, *sl_m{}, *sl_x{};
  float *low_c{}, *low_m{}, *ctrl_g{}, *ctrl_f{};
  std::uint16_t *low_h{}, *low_l{};
  std::size_t max_heads{0};
  int expert_tiles{EnvOr("INKLING2_WMMA_EXPERT_TILES", 0)};
  float *chunk_max{}, *chunk_sum{}, *chunk_out{};
  std::uint32_t max_pos{0};  // the highest position in the current step
  std::uint32_t* sel_ids{};
  // Per-step staging: the row states of every visit, the tokens, the head rows.
  void* staging{};
  std::vector<std::uint8_t> host_staging;
  std::size_t off_tokens{0}, off_heads{0};
  const gpu::RowState* row_states{};
  const std::uint32_t* dev_tokens{};
  const std::int32_t* dev_heads{};
  const std::int32_t* dev_identity{};
  std::size_t off_identity{0};
  // Feature capture: the residual stream entering executed visits 5, 23, 41
  // (three blocks of max_batch rows) and the final hidden state of every row.
  float *feat{}, *feat_final{};
  std::vector<float> host_feat, host_feat_final;
  int executed{0};
  int max_n{0};

  int batch{0};
  int n_blocks{0};
  std::size_t n_heads{0};
  std::vector<float> host_logits;
  std::size_t n_launches{0};

  static constexpr int kPhases = 18;
  static constexpr const char* kPhaseNames[kPhases] = {
      "glue A (mix, messages, norms)", "input projections",
      "glue B (conv, clips, KV store)", "attention", "output projection",
      "glue C (conv, mix, norm, router)", "experts NVFP4", "experts BF16",
      "glue D (conv, residual)", "embed and copies", "output head",
      "loop controllers: apply", "router", "norms",
      "loop controllers: FP16 copies", "loop controllers: down",
      "loop controllers: norms", "loop controllers: gain and forget"};
  bool profile{false};
  double phase_seconds[kPhases] = {};
  std::chrono::steady_clock::time_point mark{};

  // While profiling, a round trip after each kernel group, charged to `phase`.
  void Mark(int phase) {
    if (!profile)
      return;
    Check(hipStreamSynchronize(stream), "synchronize");
    const auto now = std::chrono::steady_clock::now();
    if (phase >= 0)
      phase_seconds[phase] += std::chrono::duration<double>(now - mark).count();
    mark = now;
  }

  void* Alloc(std::size_t bytes, bool zero = false) {
    void* p = nullptr;
    Check(hipMalloc(&p, bytes), "allocate");
    owned.push_back(p);
    if (zero)
      Check(hipMemset(p, 0, bytes), "clear");
    return p;
  }
  float* Floats(std::size_t count, bool zero = false) {
    return static_cast<float*>(Alloc(count * 4, zero));
  }
  void* Upload(const void* host, std::size_t bytes) {
    void* p = Alloc(bytes);
    Check(hipMemcpy(p, host, bytes, hipMemcpyHostToDevice), "upload");
    return p;
  }
  const float* Up(const Vec& v) {
    return v.empty() ? nullptr
                     : static_cast<const float*>(Upload(v.data(), v.size() * 4));
  }
  // `half` also uploads an FP16 copy for the matrix-instruction kernel.
  const std::uint16_t* UpBf16(const Mat& mat, bool half = false) {
    if (mat.type != kGgmlTypeBf16)
      throw std::runtime_error("inkling2 gpu: projection is not BF16");
    const auto* device = static_cast<const std::uint16_t*>(
        Upload(mat.data, mat.rows * mat.RowBytes()));
    if (half && wmma_min_batch > 0) {
      const std::size_t n = mat.rows * mat.cols;
      const auto* bits = reinterpret_cast<const std::uint16_t*>(mat.data);
      std::vector<std::uint16_t> h(n);
      const std::size_t threads = 16, step = (n + threads - 1) / threads;
      std::vector<std::size_t> inexact(threads, 0);
      std::vector<std::thread> pool;
      for (std::size_t t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
          const std::size_t end = std::min(n, (t + 1) * step);
          for (std::size_t i = t * step; i < end; ++i) {
            const std::uint32_t wide = std::uint32_t{bits[i]} << 16;
            float f;
            std::memcpy(&f, &wide, 4);
            bool exact = true;
            h[i] = HalfBits(f, &exact);
            inexact[t] += exact ? 0 : 1;
          }
        });
      for (auto& th : pool)
        th.join();
      for (const std::size_t v : inexact)
        half_inexact += v;
      half_values += n;
      half_of[device] =
          static_cast<const std::uint16_t*>(Upload(h.data(), n * 2));
    }
    return device;
  }

  // Routes a projection call to the matrix-instruction kernel when the step
  // is large enough and every weight has an FP16 copy: converts the inputs
  // (`cols` values per row at x) into `xh` and fills in the jobs.
  bool Wmma(gpu::GemmArgs& g, const float* x, int cols, std::uint16_t* xh) {
    if (wmma_min_batch <= 0 || g.batch < wmma_min_batch)
      return false;
    for (int j = 0; j < g.count; ++j)
      if (half_of.find(g.jobs[j].w) == half_of.end())
        return false;
    gpu::ToHalfArgs h{};
    h.x = x;
    h.x_stride = cols;
    h.out = xh;
    h.cols = cols;
    h.batch = g.batch;
    gpu::ToHalf(h, stream);
    for (int j = 0; j < g.count; ++j) {
      g.jobs[j].wh = half_of[g.jobs[j].w];
      g.jobs[j].xh = xh + (g.jobs[j].x - x);
      g.jobs[j].xh_stride = cols;
    }
    g.wmma = true;
    g.wmma_tiles = wmma_tiles;
    return true;
  }
  // Columns [col0, col0 + cols) of a small matrix as two FP16 parts.
  Split UpSplit(const Mat& mat, std::size_t col0, std::size_t cols) {
    std::vector<std::uint16_t> hi(mat.rows * cols), lo(hi.size());
    bool any_lo = false;
    for (std::size_t r = 0; r < mat.rows; ++r)
      for (std::size_t k = 0; k < cols; ++k) {
        const std::size_t at = r * mat.cols + col0 + k;
        float v;
        if (mat.type == kGgmlTypeF32) {
          std::memcpy(&v, static_cast<const std::uint8_t*>(mat.data) + at * 4, 4);
        } else if (mat.type == kGgmlTypeBf16) {
          std::uint16_t bits;
          std::memcpy(&bits, static_cast<const std::uint8_t*>(mat.data) + at * 2, 2);
          const std::uint32_t wide = std::uint32_t{bits} << 16;
          std::memcpy(&v, &wide, 4);
        } else {
          throw std::runtime_error("inkling2 gpu: unsupported small matrix");
        }
        bool exact = true;
        const std::uint16_t h = HalfBits(v, &exact);
        hi[r * cols + k] = h;
        lo[r * cols + k] = HalfBits(v - HalfToFloat(h), &exact);
        any_lo = any_lo || (lo[r * cols + k] & 0x7fffu) != 0;
      }
    Split out;
    out.hi = static_cast<const std::uint16_t*>(Upload(hi.data(), hi.size() * 2));
    if (any_lo) {
      out.lo = static_cast<const std::uint16_t*>(Upload(lo.data(), lo.size() * 2));
      ++split_with_lo;
    }
    ++split_count;
    return out;
  }

  // y = W x for every job, with the matrix-instruction kernel and F32-like
  // accuracy: W = hi + lo and x = xh + xl in FP16, and the three products
  // that matter (hi xh, hi xl, lo xh) are summed in FP32.
  void SplitGemm(const SplitJob* jobs, int n, bool low_inputs = true) {
    for (int pass = 0; pass < 3; ++pass) {
      if (pass == 1 && !low_inputs)
        continue;
      gpu::GemmArgs g{};
      for (int j = 0; j < n; ++j) {
        const std::uint16_t* w = pass == 2 ? jobs[j].w.lo : jobs[j].w.hi;
        if (!w)
          continue;
        gpu::GemmJob& job = g.jobs[g.count++];
        job.wh = w;
        job.xh = pass == 1 ? jobs[j].xl : jobs[j].xh;
        job.xh_stride = jobs[j].x_stride;
        job.y = jobs[j].y;
        job.y_stride = jobs[j].y_stride;
        job.rows = jobs[j].rows;
        job.cols = jobs[j].cols;
        job.add = pass > 0;
      }
      if (g.count == 0)
        continue;
      g.batch = batch;
      g.wmma = true;
      g.wmma_tiles = wmma_tiles;
      gpu::Gemm(g, stream);
    }
  }
  void ToHalf2(const float* x, int stride, int cols, std::uint16_t* hi,
               std::uint16_t* lo) {
    gpu::ToHalfArgs h{};
    h.x = x;
    h.x_stride = stride;
    h.out = hi;
    h.cols = cols;
    h.batch = batch;
    h.out_lo = lo;
    gpu::ToHalf(h, stream);
  }

  // A small matrix as F32 on the device; BF16 storage is widened exactly.
  const float* UpF32(const Mat& mat) {
    if (mat.type == kGgmlTypeF32)
      return static_cast<const float*>(
          Upload(mat.data, mat.rows * mat.RowBytes()));
    if (mat.type != kGgmlTypeBf16)
      throw std::runtime_error("inkling2 gpu: unsupported small matrix");
    const auto* bits = reinterpret_cast<const std::uint16_t*>(mat.data);
    std::vector<float> wide(mat.rows * mat.cols);
    for (std::size_t i = 0; i < wide.size(); ++i) {
      const std::uint32_t u = std::uint32_t{bits[i]} << 16;
      std::memcpy(&wide[i], &u, 4);
    }
    return static_cast<const float*>(Upload(wide.data(), wide.size() * 4));
  }

  // The physical layer a visit slot runs.
  [[nodiscard]] std::uint32_t Physical(std::uint32_t slot) const {
    const std::uint32_t core = c.loop_end - c.loop_start;
    return slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % core;
  }

  Impl(const Model& model, const Options& options)
      : m(model), c(model.c), opt(options) {
    // Full-attention caches are laid out in chunks of 1,024 positions.
    opt.context = (opt.context + 1023) / 1024 * 1024;
    const bool shape_ok =
        c.n_embd == kE && c.head_dim == gpu::kHD && c.n_head == kNH &&
        c.n_head_kv == kNKV && c.n_rel == gpu::kNRel && c.n_conv == kNC &&
        c.n_expert == kNX && c.n_expert_used == kTopK &&
        c.n_ff_exp == gpu::kFF && c.loop_gate_rank == kRank &&
        c.n_layer == 24 && c.loop_start == 6 && c.loop_end == 18 &&
        c.res_block == 6 && c.n_swa == 1024 && c.n_rel_extent == 1024;
    if (!shape_ok)
      throw std::runtime_error(
          "inkling2 gpu: the kernels are built for Inkling 2 Mini's "
          "dimensions; this file differs");
    if (opt.sessions == 0 || opt.max_batch == 0)
      throw std::runtime_error("inkling2 gpu: need a session and a batch row");
    Check(hipStreamCreate(&stream), "create stream");

    tok_embd = UpBf16(m.tok_embd);
    output = UpBf16(m.output, true);
    tok_norm = Up(m.tok_norm);
    output_norm = Up(m.output_norm);
    output_res_query = Up(m.output_res_query);

    layers.resize(c.n_layer);
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
      const LayerWeights& w = m.layers[il];
      Layer& l = layers[il];
      l.wq = UpBf16(w.wq, true);
      l.wk = UpBf16(w.wk, true);
      l.wv = UpBf16(w.wv, true);
      l.wo = UpBf16(w.wo, true);
      l.gate = UpBf16(w.gate, true);
      l.sink = UpBf16(w.sink, true);
      l.rel = UpBf16(w.rel, true);
      l.kcp = w.swa ? nullptr : UpBf16(w.k_conv_proj, true);
      l.vcp = w.swa ? nullptr : UpBf16(w.v_conv_proj, true);
      l.nvfp4 = w.gate_up.type == kGgmlTypeNvfp4;
      if (w.gate_up.type != w.down.type ||
          (!l.nvfp4 && w.gate_up.type != kGgmlTypeBf16))
        throw std::runtime_error("inkling2 gpu: unsupported expert storage");
      l.gate_up = static_cast<const std::uint8_t*>(Upload(
          w.gate_up.data, w.gate_up.rows * w.gate_up.RowBytes() * c.n_expert));
      l.down = static_cast<const std::uint8_t*>(
          Upload(w.down.data, w.down.rows * w.down.RowBytes() * c.n_expert));
      l.q_clip = Up(w.q_clip);
      l.rel_bias = Up(w.rel_bias);
      l.rel_clip = Up(w.rel_clip);
      l.k_conv = Up(w.k_conv);
      l.v_conv = Up(w.v_conv);
      l.attn_post = Up(w.attn_post_conv);
      l.ffn_post = Up(w.ffn_post_conv);
      if (w.swa) {
        l.norm[0] = l.norm[1] = l.norm[2] = Up(w.attn_norm);
      } else {
        l.norm[0] = Up(w.q_norm);
        l.norm[1] = Up(w.k_norm);
        l.norm[2] = Up(w.v_norm);
      }
      l.res_query = Up(w.attn_res_query);
      l.ffn_res_query = Up(w.ffn_res_query);
      l.ffn_norm = Up(w.ffn_norm);
      l.gate_inp = UpF32(w.gate_inp);
      if (split_min_batch > 0)
        l.gate_split = UpSplit(w.gate_inp, 0, kE);
      l.exp_probs_b = Up(w.exp_probs_b);
      l.up_scale = Up(w.gate_up_scale);
      l.down_scale = Up(w.down_scale);
      l.msg_query = Up(w.msg_query);
      for (int k = 0; k < 4; ++k) {
        const bool core = il >= c.loop_start && il < c.loop_end;
        l.ctrl_down[k] = core ? UpF32(w.ctrl_down[k]) : nullptr;
        if (core && split_min_batch > 0) {
          l.down_cur[k] = UpSplit(w.ctrl_down[k], 0, kE);
          l.down_msg[k] = UpSplit(w.ctrl_down[k], kE, kE);
          l.gain_split[k] = UpSplit(w.ctrl_gain[k], 0, kRank);
          l.forget_split[k] = UpSplit(w.ctrl_forget[k], 0, kRank);
        }
        l.ctrl_gain[k] = core ? UpF32(w.ctrl_gain[k]) : nullptr;
        l.ctrl_forget[k] = core ? UpF32(w.ctrl_forget[k]) : nullptr;
      }
    }

    // A sliding-window ring must keep the window of the earliest row of a
    // batch while the latest row of the same session is stored.
    // (A multiple of 64, for the FP16 copies.)
    // And the positions a step clears after its last row (kClearAhead).
    const int swa_cap = static_cast<int>(
        (c.n_swa + opt.max_batch + kClearAhead + 63) / 64 * 64);
    sessions.resize(opt.sessions);
    int rings = 0;
    for (std::uint32_t s = 0; s < c.visits(); ++s)
      ring_index.push_back(c.is_swa[Physical(s)] ? rings++ : -1);
    for (Session& session : sessions) {
      session.ring_stage = static_cast<std::uint16_t*>(
          Alloc(std::size_t(rings) * 2 * kRingWindowBytes, true));
      session.hist = Floats(kHistFloats * c.visits(), true);
      session.slots.resize(c.visits());
      for (std::uint32_t s = 0; s < c.visits(); ++s) {
        Slot& slot = session.slots[s];
        // Full-attention caches are laid out in chunks of 1,024 positions.
        const bool ring = c.is_swa[Physical(s)];
        slot.cap = ring ? swa_cap
                        : static_cast<int>((opt.context + 1023) / 1024 * 1024);
        if (keep_f32) {
          slot.k = Floats(std::size_t(slot.cap) * kHK, true);
          slot.v = Floats(std::size_t(slot.cap) * kHK, true);
        }
        slot.kh = static_cast<std::uint16_t*>(
            Alloc(std::size_t(slot.cap) * kHK * 2, true));
        slot.vh = static_cast<std::uint16_t*>(
            Alloc(std::size_t(slot.cap) * kHK * 2, true));
      }
    }

    const std::size_t B = opt.max_batch;
    // Only a row that may end its session's run stores candidates: the last
    // row of each run, or every row of a tentative step.
    hist_slots = std::min<std::size_t>(B, std::max<std::size_t>(256, opt.sessions));
    hist_candidates = Floats(hist_slots * c.visits() * kHistFloats);
    hist_slot_of.assign(B, 0);
    cur = Floats(B * kE);
    input = Floats(B * kE);
    in = Floats(B * 3 * kE);
    msgs = Floats(B * 4 * kE);
    stack = Floats(B * kMaxStates * kE);
    prelude = Floats(B * kE);
    bank0 = Floats(B * kE);
    bank_out = Floats(B * kE);
    q = Floats(B * kHQ);
    r = Floats(B * kNR);
    gate = Floats(B * kHQ);
    sink = Floats(B * kNH);
    k_raw = Floats(B * kHK);
    v_raw = Floats(B * kHK);
    k_taps = Floats(B * kNC * kNKV);
    v_taps = Floats(B * kNC * kNKV);
    attn = Floats(B * kHQ);
    proj = Floats(B * kE);
    xn = Floats(B * kE);
    sel_ids = static_cast<std::uint32_t*>(Alloc(B * kTopK * 4));
    sel_up = Floats(B * kTopK);
    sel_coef = Floats(B * kTopK);
    act = Floats(B * kTopK * gpu::kFF);
    moe = Floats(B * kE);
    max_n = static_cast<int>(std::max<std::uint32_t>(opt.context, c.n_swa));
    if (keep_f32 && !attention_window_env())
      scratch = Floats(B * kNH * std::size_t(max_n));
    const std::size_t groups = (B + gpu::kGemmGroup - 1) / gpu::kGemmGroup;
    xt_in = Floats(groups * 3 * kE * gpu::kGemmGroup);
    xt_attn = Floats(groups * kHQ * gpu::kGemmGroup);
    xh_x = static_cast<std::uint16_t*>(Alloc(B * kE * 2));
    qh = static_cast<std::uint16_t*>(Alloc(B * kHQ * 2));
    acth = static_cast<std::uint16_t*>(Alloc(B * kTopK * gpu::kFF * 2));
    if (wmma_min_batch > 0) {
      xh_in = static_cast<std::uint16_t*>(Alloc(B * 3 * kE * 2));
      xh_attn = static_cast<std::uint16_t*>(Alloc(B * kHQ * 2));
      xh_head = static_cast<std::uint16_t*>(
          Alloc(std::max<std::size_t>(opt.sessions, B) * kE * 2));
      std::fprintf(stderr,
                   "inkling2 gpu: %zu projection weights also stored as FP16, "
                   "%zu not exactly representable\n",
                   half_values, half_inexact);
    }
    if (const char* v = std::getenv("INKLING2_NO_TRANSPOSE"); v && *v == '1')
      transpose = false;
    if (const char* v = std::getenv("INKLING2_GEMM_VARIANT"))
      gemm_variant = std::atoi(v);
    if (const char* v = std::getenv("INKLING2_EXPERT_VARIANT"))
      expert_variant = std::atoi(v);
    if (const char* v = std::getenv("INKLING2_GEMV_THREADS"))
      single_threads = std::atoi(v);
    if (const char* v = std::getenv("INKLING2_HEAD_THREADS"))
      head_threads = std::atoi(v);
    if (const char* v = std::getenv("INKLING2_ATTN_WINDOW"); v && *v == '0')
      attention_window = false;
    // Attention partials: the batched kernel leaves at most 16 per head and
    // row; the per-row kernel one per 1,024 positions, which is only
    // affordable for few rows (others then go through the batched kernel).
    max_chunks = static_cast<int>(std::max<std::size_t>(2, (opt.context + 1023) / 1024));
    partial_slots = std::max<std::size_t>(
        B * 16, std::min<std::size_t>(B, 16) * std::size_t(max_chunks));
    if (keep_f32)  // the F32 kernels leave one partial per chunk and row
      partial_slots = std::max(partial_slots, B * std::size_t(max_chunks));
    chunk_max = Floats(partial_slots * kNH);
    chunk_sum = Floats(partial_slots * kNH);
    chunk_out = Floats(partial_slots * kNH * gpu::kHD);
    contrib = Floats(B * kTopK * kE);
    if (opt.int8_experts) {
      xq = static_cast<std::uint32_t*>(Alloc(B * kE));
      xs = Floats(B * kE / 16);
      aq = static_cast<std::uint32_t*>(Alloc(B * kTopK * gpu::kFF));
      as = Floats(B * kTopK * gpu::kFF / 16);
    }
    group_list = static_cast<std::int32_t*>(Alloc(std::size_t(kNX) * B * 4));
    group_count = static_cast<std::int32_t*>(Alloc(std::size_t(kNX) * 4));
    expert_items = static_cast<std::int32_t*>(
        Alloc((std::size_t(kNX) + B * kTopK / 32 + 16) * 2 * 4));
    expert_items_full =
        static_cast<std::int32_t*>(Alloc(std::size_t(kNX) * 8 * 2 * 4));
    if (const char* v = std::getenv("INKLING2_MAJOR_MIN_BATCH"))
      major_min_batch = std::atoi(v);
    low = Floats(B * 3 * kRank);
    if (split_min_batch > 0) {
      low_c = Floats(B * 3 * kRank);
      low_m = Floats(B * 3 * kRank);
      ctrl_g = Floats(B * 3 * kE);
      ctrl_f = Floats(B * 3 * kE);
      low_h = static_cast<std::uint16_t*>(Alloc(B * 3 * kRank * 2));
      low_l = static_cast<std::uint16_t*>(Alloc(B * 3 * kRank * 2));
      sh_in = static_cast<std::uint16_t*>(Alloc(B * 3 * kE * 2));
      sl_in = static_cast<std::uint16_t*>(Alloc(B * 3 * kE * 2));
      sh_m = static_cast<std::uint16_t*>(Alloc(B * 4 * kE * 2));
      sl_m = static_cast<std::uint16_t*>(Alloc(B * 4 * kE * 2));
      sl_x = static_cast<std::uint16_t*>(Alloc(B * kE * 2));
      std::fprintf(stderr,
                   "inkling2 gpu: %zu small matrices as two FP16 parts, %zu "
                   "with a second part\n",
                   split_count, split_with_lo);
    }
    router_logits = Floats(B * kNX);
    // Rows whose logits a step may ask for.
    const std::size_t heads_cap =
        std::max<std::size_t>(opt.sessions, std::min<std::size_t>(B, 256));
    max_heads = heads_cap;
    final_x = Floats(heads_cap * kE);
    logits = Floats(heads_cap * c.n_vocab_padded);
    host_logits.resize(heads_cap * c.n_vocab_padded);

    off_tokens = sizeof(gpu::RowState) * c.visits() * B;
    off_heads = off_tokens + 4 * B;
    off_identity = off_heads + 4 * heads_cap;
    off_groups = off_identity + 4 * B;
    off_draft = (off_groups + sizeof(gpu::AttnGroup) * B + 15) / 16 * 16;
    off_ends = off_draft + sizeof(gpu::DraftRow) * B;
    host_staging.resize(off_ends + 4 * B);
    need_features = opt.capture_features || opt.drafter != nullptr;
    for (std::size_t b = 0; b < B; ++b) {
      const std::int32_t index = static_cast<std::int32_t>(b);
      std::memcpy(host_staging.data() + off_identity + 4 * b, &index, 4);
    }
    if (need_features) {
      feat = Floats(3 * B * kE);
      feat_final = Floats(B * kE);
    }
    if (opt.capture_features) {
      host_feat.resize(3 * B * kE);
      host_feat_final.resize(B * kE);
    }
    staging = Alloc(host_staging.size());
    dev_draft_rows = reinterpret_cast<const gpu::DraftRow*>(
        static_cast<const std::uint8_t*>(staging) + off_draft);
    if (opt.drafter)
      LoadDraft(*opt.drafter);
    row_states = static_cast<const gpu::RowState*>(staging);
    dev_tokens = reinterpret_cast<const std::uint32_t*>(
        static_cast<const std::uint8_t*>(staging) + off_tokens);
    dev_heads = reinterpret_cast<const std::int32_t*>(
        static_cast<const std::uint8_t*>(staging) + off_heads);
    dev_identity = reinterpret_cast<const std::int32_t*>(
        static_cast<const std::uint8_t*>(staging) + off_identity);
    dev_groups = reinterpret_cast<const gpu::AttnGroup*>(
        static_cast<const std::uint8_t*>(staging) + off_groups);
    if (std::getenv("INKLING2_DOTBENCH"))
      DotBench();
  }

  void LoadDraft(const DrafterWeights& w) {
    draft = std::make_unique<Draft>();
    Draft& d = *draft;
    const std::size_t B = opt.max_batch;
    for (int l = 0; l < gpu::kDraftLayers; ++l) {
      const DrafterWeights::Layer& from = w.layers[l];
      DraftLayer& to = d.layers[l];
      to.wq = UpBf16(from.wq, true);
      to.wk = UpBf16(from.wk, true);
      to.wv = UpBf16(from.wv, true);
      to.wo = UpBf16(from.wo, true);
      to.wk_ctx = UpBf16(from.wk_ctx, true);
      to.wv_ctx = UpBf16(from.wv_ctx, true);
      to.gate = UpBf16(from.gate, true);
      to.up = UpBf16(from.up, true);
      to.down = UpBf16(from.down, true);
      to.ctx_norm = Up(from.ctx_norm);
      to.attn_norm = Up(from.attn_norm);
      to.ffn_norm = Up(from.ffn_norm);
      to.q_norm = Up(from.q_norm);
      to.k_norm = Up(from.k_norm);
      to.k_ctx_norm = Up(from.k_ctx_norm);
    }
    d.head_norm = Up(w.head_norm);
    d.mask = Up(w.mask);
    d.output_norm = Up(w.output_norm);
    d.joint = UpF32(w.joint);
    d.joint_bias = Up(w.joint_bias);
    d.markov = UpBf16(w.markov);

    // A ring holds its window and the rows of a tentative step, so that rows
    // which are then dropped never overwrite a position still in a window.
    std::size_t at = 0;
    for (int l = 0; l < gpu::kDraftLayers; ++l) {
      const std::uint32_t window =
          l == 1 ? std::clamp<std::uint32_t>(opt.draft_window, 64, opt.context)
                 : gpu::kDraftWindow;
      d.cap[l] = static_cast<int>((window + 64 + 63) / 64 * 64);
      d.ring_offset[l] = at;
      at += std::size_t(d.cap[l]) * kHK * 2 * 2;
    }
    d.f_rnn_offset = at;
    d.state_bytes = at + kE * 4;
    for (Session& session : sessions)
      session.draft_state =
          static_cast<std::uint8_t*>(Alloc(d.state_bytes, true));

    d.mix = Floats(B * 4 * kE);
    d.mix_h = static_cast<std::uint16_t*>(Alloc(B * 4 * kE * 2));
    d.k_raw = Floats(B * gpu::kDraftLayers * kHK);
    d.v_raw = Floats(B * gpu::kDraftLayers * kHK);
    const std::size_t R = gpu::kDraftBlock;
    d.x = Floats(R * kE);
    d.a = Floats(R * kE);
    d.q = Floats(R * kHQ);
    d.k = Floats(R * kHK);
    d.v = Floats(R * kHK);
    d.o = Floats(R * kHQ);
    d.proj = Floats(R * kE);
    d.gate = Floats(R * gpu::kDraftFF);
    d.up = Floats(R * gpu::kDraftFF);
    d.h = Floats(R * kE);
    d.slots = wmma_min_batch > 0 && (draft_wmma & 2) && opt.draft_vocab > 0 &&
                      opt.draft_vocab < c.n_vocab
                  ? static_cast<int>((opt.draft_vocab + 255) / 256 * 256)
                  : 0;
    d.base = Floats((R - 1) * c.n_vocab_padded);
    d.logits = Floats(c.n_vocab_padded);
    d.state = Floats(gpu::kDraftRank, true);
    d.j_raw = Floats(gpu::kDraftJoint);
    d.latent = Floats(gpu::kDraftRank);
    const std::size_t vocab_blocks = (c.n_vocab + 255) / 256;
    d.block_max = Floats(vocab_blocks);
    d.block_sum = Floats(vocab_blocks);
    d.block_arg = static_cast<std::int32_t*>(Alloc(vocab_blocks * 4));
    d.top = Floats(1);
    d.confidence = Floats(R, true);
    d.tokens = static_cast<std::uint32_t*>(Alloc(R * 4, true));
    d.alive = static_cast<std::int32_t*>(Alloc(4, true));
    d.scratch = Floats(R * kNH * (std::size_t(d.cap[1]) + R));
    d.a_h = static_cast<std::uint16_t*>(Alloc(R * kE * 2));
    d.o_h = static_cast<std::uint16_t*>(Alloc(R * kHQ * 2));
    d.gate_h = static_cast<std::uint16_t*>(Alloc(R * gpu::kDraftFF * 2));
    d.h_h = static_cast<std::uint16_t*>(Alloc(R * kE * 2));

    // The head's correction is linear before the output matrix:
    //   output (h + hidden_out latent + bias)
    //     = output h + (output hidden_out) latent + output bias.
    // The second and third products are computed here, once, so that a slot
    // of the recurrent head reads 200K rows of 256 values instead of the
    // whole output matrix.
    const int cols = gpu::kDraftRank + 1;
    std::vector<float> columns(std::size_t(cols) * kE);
    if (w.hidden_out.type != kGgmlTypeF32 && w.hidden_out.type != kGgmlTypeBf16)
      throw std::runtime_error("inkling2 gpu: unsupported drafter head matrix");
    const auto value = [&](std::size_t row, std::size_t col) {
      const std::size_t i = row * w.hidden_out.cols + col;
      float v;
      if (w.hidden_out.type == kGgmlTypeF32) {
        std::memcpy(&v, w.hidden_out.data + i * 4, 4);
        return v;
      }
      std::uint16_t bits;
      std::memcpy(&bits, w.hidden_out.data + i * 2, 2);
      const std::uint32_t wide = std::uint32_t{bits} << 16;
      std::memcpy(&v, &wide, 4);
      return v;
    };
    for (std::size_t col = 0; col < gpu::kDraftRank; ++col)
      for (std::size_t row = 0; row < kE; ++row)
        columns[col * kE + row] = value(row, col);
    std::copy(w.hidden_out_bias.begin(), w.hidden_out_bias.end(),
              columns.begin() + std::size_t(gpu::kDraftRank) * kE);
    const float* dev_columns =
        static_cast<const float*>(Upload(columns.data(), columns.size() * 4));
    // Kept in the order the product leaves it (column c at m + c * vocab),
    // as FP16: a slot's cost is reading it.
    void* product = nullptr;
    Check(hipMalloc(&product, std::size_t(cols) * c.n_vocab_padded * 4),
          "allocate");
    gpu::GemmArgs g{};
    const int vocab = static_cast<int>(c.n_vocab_padded);
    g.jobs[0] = {output, dev_columns, kE, static_cast<float*>(product), vocab,
                 vocab, kE};
    g.count = 1;
    g.batch = cols;
    gpu::Gemm(g, stream);
    d.m = static_cast<std::uint16_t*>(
        Alloc(std::size_t(gpu::kDraftRank) * c.n_vocab_padded * 2));
    d.m_bias = Floats(c.n_vocab_padded);
    gpu::ToHalfArgs half{};
    half.x = static_cast<const float*>(product);
    half.x_stride = vocab;
    half.out = d.m;
    half.cols = vocab;
    half.batch = gpu::kDraftRank;
    gpu::ToHalf(half, stream);
    Check(hipMemcpyAsync(d.m_bias,
                         static_cast<const float*>(product) +
                             std::size_t(gpu::kDraftRank) * c.n_vocab_padded,
                         std::size_t(c.n_vocab_padded) * 4,
                         hipMemcpyDeviceToDevice, stream),
          "copy");
    Check(hipStreamSynchronize(stream), "synchronize");
    (void)hipFree(product);
    if (d.slots > 0) {
      const std::size_t S = static_cast<std::size_t>(d.slots);
      // An empty slot can never be the best.
      const std::vector<float> lowest(S, -3.0e38f);
      d.list_empty_bias =
          static_cast<float*>(Upload(lowest.data(), lowest.size() * 4));
      d.list_entries = static_cast<gpu::DraftListEntry*>(
          Alloc(kListBatch * sizeof(gpu::DraftListEntry)));
      for (Session& session : sessions) {
        Session::Shortlist& list = session.list;
        list.slot_of.assign(c.n_vocab, -1);
        list.token_of.assign(S, 0);
        list.touched.assign(S, 0);
        list.output = static_cast<std::uint16_t*>(Alloc(S * kE * 2, true));
        list.m =
            static_cast<std::uint16_t*>(Alloc(S * gpu::kDraftRank * 2, true));
        list.bias = static_cast<float*>(Upload(lowest.data(), lowest.size() * 4));
        list.ids = static_cast<std::uint32_t*>(Alloc(S * 4, true));
        // The matrix kernel finds a job's FP16 weights by its pointer.
        half_of[list.output] = list.output;
      }
    }
  }

  static constexpr std::size_t kListBatch = 4096;

  // Empties a session's shortlist.
  void ResetList(Session& session) {
    if (!draft || draft->slots == 0)
      return;
    Session::Shortlist& list = session.list;
    list.used = list.hand = 0;
    std::fill(list.slot_of.begin(), list.slot_of.end(), -1);
    std::fill(list.touched.begin(), list.touched.end(), 0);
    list.pending.clear();
    Check(hipMemcpyAsync(list.bias, draft->list_empty_bias,
                         std::size_t(draft->slots) * 4, hipMemcpyDeviceToDevice,
                         stream),
          "clear shortlist");
  }

  // Puts a token on a session's shortlist. A full list gives up the slot the
  // clock hand reaches first that was not asked for again since its last
  // pass.
  void List(Session& session, std::uint32_t token) {
    Draft& d = *draft;
    Session::Shortlist& list = session.list;
    const std::int32_t have = list.slot_of[token];
    if (have >= 0) {
      list.touched[have] = 1;
      return;
    }
    int slot;
    if (list.used < d.slots) {
      slot = list.used++;
    } else {
      while (list.touched[list.hand]) {
        list.touched[list.hand] = 0;
        list.hand = (list.hand + 1) % d.slots;
      }
      slot = list.hand;
      list.hand = (list.hand + 1) % d.slots;
      list.slot_of[list.token_of[slot]] = -1;
      ++d.evicted;
    }
    list.token_of[slot] = token;
    list.slot_of[token] = slot;
    list.pending.push_back({slot, token});
    ++d.listed;
  }

  // The tokens the model finds likely after a row: those within a margin of
  // the best (soft-capped) logit, at most kListPerRow of them.
  static constexpr std::size_t kListPerRow = 192;
  void ListLikely(Session& session, const float* raw) {
    const std::size_t n = c.n_vocab;
    float top = raw[0];
    for (std::size_t i = 1; i < n; ++i)
      top = std::max(top, raw[i]);
    const float floor = top - 7.0f;
    likely.clear();
    for (std::size_t i = 0; i < n; ++i)
      if (raw[i] > floor)
        likely.push_back(static_cast<std::uint32_t>(i));
    if (likely.size() > kListPerRow) {
      std::nth_element(likely.begin(), likely.begin() + kListPerRow,
                       likely.end(), [&](std::uint32_t a, std::uint32_t b) {
                         return raw[a] > raw[b];
                       });
      likely.resize(kListPerRow);
    }
    for (const std::uint32_t token : likely)
      List(session, token);
  }
  std::vector<std::uint32_t> likely;

  // Copies a shortlist's new entries to the device.
  void FlushList(Session& session) {
    Draft& d = *draft;
    Session::Shortlist& list = session.list;
    std::size_t at = 0;
    std::vector<gpu::DraftListEntry> batch;
    while (at < list.pending.size()) {
      batch.clear();
      for (; at < list.pending.size() && batch.size() < kListBatch; ++at) {
        const gpu::DraftListEntry& entry = list.pending[at];
        // An entry whose slot was given to another token since is stale.
        if (list.token_of[entry.slot] == entry.token)
          batch.push_back(entry);
      }
      if (batch.empty())
        continue;
      // The staging is shared: nothing may still be reading it.
      Check(hipStreamSynchronize(stream), "synchronize");
      Check(hipMemcpy(d.list_entries, batch.data(),
                      batch.size() * sizeof(gpu::DraftListEntry),
                      hipMemcpyHostToDevice),
            "copy shortlist entries");
      gpu::DraftListArgs a{};
      a.entries = d.list_entries;
      a.count = static_cast<int>(batch.size());
      a.output = half_of.at(output);
      a.m = d.m;
      a.m_bias = d.m_bias;
      a.vocab_stride = static_cast<int>(c.n_vocab_padded);
      a.list_output = list.output;
      a.list_m = list.m;
      a.list_bias = list.bias;
      a.list_ids = list.ids;
      a.slots = d.slots;
      gpu::DraftListFill(a, stream);
      Check(hipStreamSynchronize(stream), "synchronize");
    }
    list.pending.clear();
  }

  // What the drafter reads of the step's rows, into their sessions' rings.
  void StoreDraftContext() {
    const Draft& d = *draft;
    gpu::DraftMixArgs mix{};
    mix.feat = feat;
    mix.feat_block = std::size_t(opt.max_batch) * kE;
    mix.feat_final = feat_final;
    for (int r = 0; r <= gpu::kDraftLayers; ++r) {
      for (int k = 0; k < 4; ++k)
        mix.fuse[r][k] = opt.drafter->fuse[r][k];
      mix.norm[r] = r < gpu::kDraftLayers ? d.layers[r].ctx_norm : d.head_norm;
    }
    mix.eps = 1e-6f;
    mix.out = d.mix;
    mix.batch = batch;
    gpu::DraftMix(mix, stream);
    gpu::GemmArgs g{};
    for (int l = 0; l < gpu::kDraftLayers; ++l) {
      g.jobs[2 * l] = {d.layers[l].wk_ctx, d.mix + l * kE, 4 * kE,
                       d.k_raw + l * kHK, gpu::kDraftLayers * kHK, kHK, kE};
      g.jobs[2 * l + 1] = {d.layers[l].wv_ctx, d.mix + l * kE, 4 * kE,
                           d.v_raw + l * kHK, gpu::kDraftLayers * kHK, kHK, kE};
    }
    g.count = 2 * gpu::kDraftLayers;
    g.batch = batch;
    g.variant = gemm_variant;
    Wmma(g, d.mix, 4 * kE, d.mix_h);
    gpu::Gemm(g, stream);
    gpu::DraftStoreArgs store{};
    store.k_raw = d.k_raw;
    store.v_raw = d.v_raw;
    for (int l = 0; l < gpu::kDraftLayers; ++l)
      store.k_norm[l] = d.layers[l].k_ctx_norm;
    store.rows = dev_draft_rows;
    store.eps = 1e-6f;
    store.batch = batch;
    gpu::DraftStore(store, stream);
  }

  // Batch row `b` is the last row its session keeps: its head feature is
  // what the next draft reads.
  void CommitDraft(Session& session, std::size_t b) {
    if (!draft)
      return;
    Check(hipMemcpyAsync(DraftFeature(session),
                         draft->mix + (b * 4 + gpu::kDraftLayers) * kE, kE * 4,
                         hipMemcpyDeviceToDevice, stream),
          "commit drafter feature");
    session.draft_ready = true;
  }

  std::vector<DraftToken> DraftTokens(std::uint32_t index, std::uint32_t anchor,
                                      int max_tokens, float min_confidence) {
    Session& session = sessions.at(index);
    if (!draft || !session.draft_ready || session.pos == 0 ||
        session.pending_first >= 0 || max_tokens < 1 || anchor >= c.n_vocab)
      return {};
    max_tokens = std::min(max_tokens, gpu::kDraftBlock - 1);
    Draft& d = *draft;
    const int pos0 = static_cast<int>(session.pos);
    gpu::DraftInitArgs init{};
    init.tok_embd = tok_embd;
    init.tok_norm = tok_norm;
    init.tok_eps = c.norm_eps;
    init.mask = d.mask;
    init.anchor = anchor;
    init.x = d.x;
    init.state = d.state;
    init.tokens = d.tokens;
    init.alive = d.alive;
    if (d.slots > 0)
      FlushList(session);
    auto mark = std::chrono::steady_clock::now();
    const auto lap = [&](int part) {
      if (!draft_profile)
        return;
      Check(hipStreamSynchronize(stream), "synchronize");
      const auto now = std::chrono::steady_clock::now();
      d.seconds[part] += std::chrono::duration<double>(now - mark).count();
      mark = now;
    };
    gpu::DraftInit(init, stream);
    const float eps = 1e-6f;
    for (int l = 0; l < gpu::kDraftLayers; ++l) {
      const DraftLayer& w = d.layers[l];
      gpu::DraftResNorm({d.x, l > 0 ? d.proj : nullptr, w.attn_norm, eps, d.a},
                        stream);
      gpu::GemmArgs g{};
      g.jobs[0] = {w.wq, d.a, kE, d.q, kHQ, kHQ, kE};
      g.jobs[1] = {w.wk, d.a, kE, d.k, kHK, kHK, kE};
      g.jobs[2] = {w.wv, d.a, kE, d.v, kHK, kHK, kE};
      g.count = 3;
      g.batch = gpu::kDraftBlock;
      DraftWmma(g, d.a, kE, d.a_h, false);
      gpu::Gemm(g, stream);
      gpu::DraftQk({d.q, d.k, w.q_norm, w.k_norm, eps, pos0}, stream);
      gpu::DraftAttnArgs at{};
      at.q = d.q;
      at.k_blk = d.k;
      at.v_blk = d.v;
      at.k_ring = DraftRing(session, l, false);
      at.v_ring = DraftRing(session, l, true);
      at.cap = d.cap[l];
      at.first = std::max(0, pos0 - (d.cap[l] - 64));
      at.pos0 = pos0;
      at.scratch = d.scratch;
      at.out = d.o;
      gpu::DraftAttn(at, stream);
      gpu::GemmArgs o{};
      o.jobs[0] = {w.wo, d.o, kHQ, d.proj, kE, kE, kHQ};
      o.count = 1;
      o.batch = gpu::kDraftBlock;
      DraftWmma(o, d.o, kHQ, d.o_h, false);
      gpu::Gemm(o, stream);
      gpu::DraftResNorm({d.x, d.proj, w.ffn_norm, eps, d.a}, stream);
      gpu::GemmArgs f{};
      f.jobs[0] = {w.gate, d.a, kE, d.gate, gpu::kDraftFF, gpu::kDraftFF, kE};
      f.jobs[1] = {w.up, d.a, kE, d.up, gpu::kDraftFF, gpu::kDraftFF, kE};
      f.count = 2;
      f.batch = gpu::kDraftBlock;
      DraftWmma(f, d.a, kE, d.a_h, false);
      gpu::Gemm(f, stream);
      gpu::DraftSilu(d.gate, d.up, stream);
      gpu::GemmArgs down{};
      down.jobs[0] = {w.down, d.gate, gpu::kDraftFF, d.proj, kE, kE,
                      gpu::kDraftFF};
      down.count = 1;
      down.batch = gpu::kDraftBlock;
      DraftWmma(down, d.gate, gpu::kDraftFF, d.gate_h, false);
      gpu::Gemm(down, stream);
    }
    lap(0);
    gpu::DraftResNorm({d.x, d.proj, d.output_norm, eps, d.h}, stream);
    // output . h for the slots in one pass over the output matrix.
    gpu::GemmArgs head{};
    const int vocab = static_cast<int>(c.n_vocab_padded);
    head.jobs[0] = {output, d.h + kE, kE, d.base, vocab, vocab, kE};
    if (d.slots > 0)
      head.jobs[0] = {session.list.output, d.h + kE, kE,     d.base,
                      d.slots,             d.slots,  kE};
    head.count = 1;
    head.batch = max_tokens;
    head.single_threads = head_threads;
    DraftWmma(head, d.h + kE, kE, d.h_h, true);
    gpu::Gemm(head, stream);
    lap(1);
    gpu::DraftSlotArgs slot{};
    slot.joint = d.joint;
    slot.joint_bias = d.joint_bias;
    slot.markov = d.markov;
    slot.h = d.h;
    slot.f_rnn = DraftFeature(session);
    slot.state = d.state;
    slot.j_raw = d.j_raw;
    slot.latent = d.latent;
    slot.base = d.base;
    slot.m = d.m;
    slot.m_bias = d.m_bias;
    slot.logits = d.logits;
    slot.block_max = d.block_max;
    slot.block_arg = d.block_arg;
    slot.block_sum = d.block_sum;
    slot.top = d.top;
    slot.tokens = d.tokens;
    slot.confidence = d.confidence;
    slot.alive = d.alive;
    slot.softcap = c.logit_softcap;
    slot.min_confidence = min_confidence;
    slot.vocab = static_cast<int>(c.n_vocab);
    slot.vocab_stride = vocab;
    if (d.slots > 0) {
      slot.m = session.list.m;
      slot.m_bias = session.list.bias;
      slot.vocab = slot.vocab_stride = d.slots;
      slot.ids = session.list.ids;
    }
    for (int i = 1; i <= max_tokens; ++i) {
      slot.slot = i;
      gpu::DraftSlot(slot, stream);
    }
    std::uint32_t tokens[gpu::kDraftBlock];
    float confidence[gpu::kDraftBlock];
    Check(hipMemcpyAsync(tokens, d.tokens, sizeof tokens, hipMemcpyDeviceToHost,
                         stream),
          "copy drafts");
    Check(hipMemcpyAsync(confidence, d.confidence, sizeof confidence,
                         hipMemcpyDeviceToHost, stream),
          "copy drafts");
    Check(hipStreamSynchronize(stream), "synchronize");
    lap(2);
    if (draft_profile && ++d.calls % 32 == 0)
      std::fprintf(stderr,
                   "inkling2 gpu: draft, ms per call: layers %.2f, output "
                   "matrix %.2f, slots %.2f; shortlist %d of %d, %zu listed, "
                   "%zu given up\n",
                   1e3 * d.seconds[0] / d.calls, 1e3 * d.seconds[1] / d.calls,
                   1e3 * d.seconds[2] / d.calls, session.list.used, d.slots,
                   d.listed, d.evicted);
    std::vector<DraftToken> out;
    for (int i = 1; i <= max_tokens; ++i) {
      if (!(confidence[i] >= min_confidence))
        break;
      out.push_back({tokens[i], confidence[i]});
    }
    return out;
  }

  // Tuning aid: throughput of 8-bit integer dot products against F32
  // multiply-adds, in the shape of the batched matrix kernel.
  void DotBench() {
    const int rows = 32768, cols = 2304, words = cols / 4;
    auto* w8 = static_cast<std::uint32_t*>(Alloc(std::size_t(rows) * words * 4));
    auto* x8 = static_cast<std::uint32_t*>(
        Alloc(std::size_t(words) * gpu::kGemmGroup * 4));
    auto* y8 = static_cast<int*>(Alloc(std::size_t(rows) * gpu::kGemmGroup * 4));
    float* wf = Floats(std::size_t(rows) * cols);
    float* xf = Floats(std::size_t(cols) * gpu::kGemmGroup);
    float* yf = Floats(std::size_t(rows) * gpu::kGemmGroup);
    Check(hipMemset(w8, 3, std::size_t(rows) * words * 4), "fill");
    Check(hipMemset(x8, 5, std::size_t(words) * gpu::kGemmGroup * 4), "fill");
    Check(hipMemset(wf, 0, std::size_t(rows) * cols * 4), "fill");
    Check(hipMemset(xf, 0, std::size_t(cols) * gpu::kGemmGroup * 4), "fill");
    for (const int rows_in_batch : {1, 4, 16}) {
      for (int kind = 0; kind < 3; ++kind) {
        const int repeats = 20;
        Check(hipStreamSynchronize(stream), "synchronize");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
          if (kind == 0)
            gpu::Dot4Bench(w8, x8, rows, words, rows_in_batch, y8, stream);
          else if (kind == 2)
            gpu::Dot2Bench(w8, x8, rows, words, rows_in_batch, yf, stream);
          else
            gpu::FloatBench(wf, xf, rows, cols, rows_in_batch, yf, stream);
        }
        Check(hipStreamSynchronize(stream), "synchronize");
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        // The integer and BF16 runs read `words` 32-bit words per row: four
        // 8-bit values or two BF16 values each.
        const double per_row = kind == 0 ? cols : kind == 2 ? words * 2.0 : cols;
        const double macs = per_row * rows * rows_in_batch * repeats;
        const double bytes =
            double(rows) * (kind == 1 ? cols * 4.0 : words * 4.0) * repeats;
        std::fprintf(stderr,
                     "dot bench: %-5s %2d row(s): %7.1f G multiply-adds/s, "
                     "%6.1f GB/s of weights\n",
                     kind == 0 ? "int8" : kind == 2 ? "bf16" : "f32",
                     rows_in_batch,
                     macs / seconds / 1e9, bytes / seconds / 1e9);
      }
    }
  }

  ~Impl() {
    if (stream)
      (void)hipStreamSynchronize(stream);
    if (stat_experts > 0)
      std::fprintf(stderr,
                   "inkling2 gpu: experts over %zu (visit, expert): %.1f%% "
                   "unused, %.1f pairs on average, most %zu; more than 128 "
                   "pairs %.1f%%, more than 256 %.1f%%, more than 512 %.1f%%; "
                   "tiles of 128 per expert %.2f, of 256 %.2f%c",
                   stat_experts, 100.0 * stat_empty / stat_experts,
                   double(stat_pairs) / stat_experts, stat_max,
                   100.0 * stat_over128 / stat_experts,
                   100.0 * stat_over256 / stat_experts,
                   100.0 * stat_over512 / stat_experts,
                   double(stat_tiles128) / stat_experts,
                   double(stat_tiles256) / stat_experts, 10);
    if (ab_visits > 0)
      std::fprintf(stderr,
                   "inkling2 gpu: attention A/B over %zu visits: max |old - "
                   "new| %.3g (max |old| %.3g), rms difference / rms %.3g, "
                   "%zu values not finite\n",
                   ab_visits, ab_max_diff, ab_max_abs,
                   ab_sq > 0 ? std::sqrt(ab_sq_diff / ab_sq) : 0.0, ab_bad);
    for (void* p : owned)
      (void)hipFree(p);
    if (stream)
      (void)hipStreamDestroy(stream);
  }

  // Called before each layer visit, in execution order: keeps the residual
  // stream entering visits 5, 23 and 41 when features are captured.
  void Tap() {
    const int visit = executed++;
    if (!need_features)
      return;
    const int k = visit == 5 ? 0 : visit == 23 ? 1 : visit == 41 ? 2 : -1;
    if (k < 0)
      return;
    gpu::CopyArgs copy{};
    copy.dst[0] = feat + std::size_t(k) * opt.max_batch * kE;
    copy.dst_stride[0] = kE;
    copy.src[0] = cur;
    copy.src_stride[0] = kE;
    copy.count = 1;
    copy.batch = batch;
    gpu::Copy(copy, stream);
  }

  // How concentrated the attention weights of the step's last row are in
  // this full-attention visit, computed on the host from the device's q, r,
  // sink and FP16 keys: one entry per query head.
  void AttnStats(const gpu::RowState* rows, std::uint32_t il) {
    Check(hipStreamSynchronize(stream), "synchronize");
    const std::size_t b = static_cast<std::size_t>(batch) - 1;
    gpu::RowState row;
    Check(hipMemcpy(&row, rows + b, sizeof(row), hipMemcpyDeviceToHost), "row");
    const std::size_t n = static_cast<std::size_t>(row.pos) + 1;
    std::vector<float> hq(kHQ), hr(kNR), hsink(kNH);
    Check(hipMemcpy(hq.data(), q + b * kHQ, kHQ * 4, hipMemcpyDeviceToHost), "q");
    Check(hipMemcpy(hr.data(), r + b * kNR, kNR * 4, hipMemcpyDeviceToHost), "r");
    Check(hipMemcpy(hsink.data(), sink + b * kNH, kNH * 4, hipMemcpyDeviceToHost),
          "sink");
    const Vec& table = m.layers[il].rel_bias;
    const float scale = 1.0f / std::sqrt(static_cast<float>(gpu::kHD));
    std::vector<std::uint16_t> keys(n * gpu::kHD);
    std::vector<float> key(n * gpu::kHD);
    stats.resize(stats.size() + kNH);
    HeadStats* out = stats.data() + stats.size() - kNH;
    stats_chunk_max.assign(kNH * ((n + 1023) / 1024), 0.0);
    stats_chunk_sum.assign(kNH * ((n + 1023) / 1024), 0.0);
    for (std::size_t kv = 0; kv < kNKV; ++kv) {
      Check(hipMemcpy(keys.data(),
                      row.kh + kv * std::size_t(row.cap) * gpu::kHD,
                      keys.size() * 2, hipMemcpyDeviceToHost),
            "keys");
      for (std::size_t i = 0; i < keys.size(); ++i)
        key[i] = HalfToFloat(keys[i]);
      const int share = kNH / kNKV;
      std::vector<std::thread> pool;
      for (int s = 0; s < share; ++s)
        pool.emplace_back([&, s] {
          const std::size_t h = kv * share + s;
          std::vector<double> w(n);
          double max = 0.0;  // the zero sink's logit
          for (std::size_t j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (int i = 0; i < gpu::kHD; ++i)
              dot += hq[h * gpu::kHD + i] * key[j * gpu::kHD + i];
            double v = double(dot) * scale - hsink[h];
            const std::size_t distance = n - 1 - j;
            if (distance < c.n_rel_extent)
              for (int x = 0; x < gpu::kNRel; ++x)
                v += double(table[distance * gpu::kNRel + x]) * hr[h * gpu::kNRel + x];
            w[j] = v;
            max = std::max(max, v);
          }
          // Per chunk of 1,024 positions, as the kernels keep it: the
          // highest logit and the sum of exp(logit - that).
          const std::size_t n_chunks = (n + 1023) / 1024;
          for (std::size_t ch = 0; ch < n_chunks; ++ch) {
            const std::size_t lo = ch * 1024, hi = std::min(n, lo + 1024);
            double cm = w[lo];
            for (std::size_t j = lo; j < hi; ++j)
              cm = std::max(cm, w[j]);
            double cs = 0.0;
            for (std::size_t j = lo; j < hi; ++j)
              cs += std::exp(w[j] - cm);
            stats_chunk_max[h * n_chunks + ch] = cm;
            stats_chunk_sum[h * n_chunks + ch] = cs;
          }
          double total = std::exp(-max);
          for (double& v : w) {
            v = std::exp(v - max);
            total += v;
          }
          HeadStats& st = out[h];
          st.sink = std::exp(-max) / total;
          {
            std::vector<double> sorted(w);
            const std::size_t top = std::min<std::size_t>(2048, n);
            std::partial_sort(sorted.begin(), sorted.begin() + top, sorted.end(),
                              std::greater<double>());
            double held = 0.0;
            for (std::size_t j = 0; j < top; ++j)
              held += sorted[j];
            st.top_2048 = held / total;
          }
          double recent = 0.0;
          for (std::size_t j = n > 1024 ? n - 1024 : 0; j < n; ++j)
            recent += w[j];
          st.last_1024 = recent / total;
          // The fewest units (positions, slices of 64, chunks of 1,024) that
          // hold all but `miss` of the weight given to positions.
          const double given = total - std::exp(-max);
          auto fewest = [&](std::size_t unit, double miss) {
            std::vector<double> mass((n + unit - 1) / unit, 0.0);
            for (std::size_t j = 0; j < n; ++j)
              mass[j / unit] += w[j];
            std::sort(mass.begin(), mass.end(), std::greater<double>());
            double covered = 0.0;
            std::size_t count = 0;
            // `miss` is a share of everything, the sink included.
            while (count < mass.size() && given - covered > miss * total)
              covered += mass[count++];
            return double(count) / double(mass.size());
          };
          st.pos_999 = fewest(1, 1e-3);
          st.pos_9999 = fewest(1, 1e-4);
          st.slice_999 = fewest(64, 1e-3);
          st.slice_9999 = fewest(64, 1e-4);
          st.slice_99999 = fewest(64, 1e-5);
          st.chunk_9999 = fewest(1024, 1e-4);
        });
      for (auto& t : pool)
        t.join();
    }
  }

  // Prints and clears what AttnStats collected in this step.
  void ReportAttnStats(std::uint32_t pos) {
    if (stats.empty())
      return;
    auto line = [&](const char* name, double HeadStats::*field) {
      std::vector<double> v;
      double mean = 0.0;
      for (const HeadStats& st : stats) {
        v.push_back(st.*field);
        mean += st.*field;
      }
      std::sort(v.begin(), v.end());
      std::fprintf(stderr, "  %-46s mean %7.4f  median %7.4f  90%% %7.4f  max %7.4f\n",
                   name, mean / v.size(), v[v.size() / 2], v[v.size() * 9 / 10],
                   v.back());
    };
    std::fprintf(stderr,
                 "attention stats at position %u: %zu query heads over the "
                 "full-attention visits (shares of 1)\n",
                 pos, stats.size());
    line("weight on the sink", &HeadStats::sink);
    line("weight on the last 1,024 positions", &HeadStats::last_1024);
    line("weight on the 2,048 heaviest positions", &HeadStats::top_2048);
    line("positions holding all but 1e-3", &HeadStats::pos_999);
    line("positions holding all but 1e-4", &HeadStats::pos_9999);
    line("slices of 64 holding all but 1e-3", &HeadStats::slice_999);
    line("slices of 64 holding all but 1e-4", &HeadStats::slice_9999);
    line("slices of 64 holding all but 1e-5", &HeadStats::slice_99999);
    line("chunks of 1,024 holding all but 1e-4", &HeadStats::chunk_9999);
    std::fprintf(stderr,
                 "  against the attention kernel, %zu chunks: largest difference "
                 "of a chunk's highest logit %.3g, of its sum %.3g (relative)\n",
                 stats_chunks, stats_max_diff, stats_sum_diff);
    stats_max_diff = stats_sum_diff = 0.0;
    stats_chunks = 0;
    stats.clear();
  }

  void Visit(std::uint32_t slot_index, std::uint32_t il, bool looped) {
    const Layer& l = layers[il];
    const bool swa = c.is_swa[il];
    const bool block_start = il % c.res_block == 0;
    const gpu::RowState* rows =
        row_states + std::size_t(slot_index) * opt.max_batch;

    gpu::GlueAArgs a{};
    a.cur = cur;
    a.input = input;
    a.in = in;
    a.msgs = msgs;
    a.stack = stack;
    a.n_blocks = n_blocks;
    a.res_query = l.res_query;
    a.n_res = il == 0 ? 0 : (swa ? 1 : 3);
    a.looped = looped;
    a.bank0 = bank0;
    a.bank_out = bank_out;
    a.msg_query = l.msg_query;
    a.batch = batch;
    gpu::GlueA(a, stream);
    Mark(0);
    const bool split = split_min_batch > 0 && batch >= split_min_batch;
    if (looped) {
      gpu::CtrlArgs ctrl{};
      ctrl.cur = in;
      ctrl.cur_stride = 3 * kE;
      ctrl.msg = msgs;
      ctrl.msg_stride = 4 * kE;
      for (int j = 0; j < 3; ++j) {
        ctrl.down[j] = l.ctrl_down[j];
        ctrl.gain[j] = l.ctrl_gain[j];
        ctrl.forget[j] = l.ctrl_forget[j];
      }
      ctrl.low = low;
      ctrl.count = 3;
      ctrl.batch = batch;
      if (split) {
        ToHalf2(in, 3 * kE, 3 * kE, sh_in, ctrl_low ? sl_in : nullptr);
        ToHalf2(msgs, 4 * kE, 4 * kE, sh_m, ctrl_low ? sl_m : nullptr);
        Mark(14);
        SplitJob jobs[6];
        for (int j = 0; j < 3; ++j) {
          jobs[2 * j] = {l.down_cur[j], sh_in + j * kE, sl_in + j * kE, 3 * kE,
                         low_c + j * kRank, 3 * kRank, kRank, kE};
          jobs[2 * j + 1] = {l.down_msg[j], sh_m + j * kE, sl_m + j * kE, 4 * kE,
                             low_m + j * kRank, 3 * kRank, kRank, kE};
        }
        SplitGemm(jobs, 6, ctrl_low);
        Mark(15);
        ctrl.yc = low_c;
        ctrl.ym = low_m;
        gpu::CtrlCombine(ctrl, stream);
        Mark(16);
        ToHalf2(low, 3 * kRank, 3 * kRank, low_h, ctrl_low ? low_l : nullptr);
        for (int j = 0; j < 3; ++j) {
          jobs[2 * j] = {l.gain_split[j], low_h + j * kRank, low_l + j * kRank,
                         3 * kRank, ctrl_g + j * kE, 3 * kE, kE, kRank};
          jobs[2 * j + 1] = {l.forget_split[j], low_h + j * kRank,
                             low_l + j * kRank, 3 * kRank, ctrl_f + j * kE,
                             3 * kE, kE, kRank};
        }
        SplitGemm(jobs, 6, ctrl_low);
        Mark(17);
        ctrl.yg = ctrl_g;
        ctrl.yf = ctrl_f;
      } else {
        gpu::CtrlDown(ctrl, stream);
      }
      gpu::CtrlApply(ctrl, stream);
    }
    Mark(11);
    gpu::NormArgs norm{};
    norm.x = in;
    norm.stride = 3 * kE;
    for (int j = 0; j < 3; ++j)
      norm.w[j] = l.norm[j];
    norm.count = 3;
    norm.batch = batch;
    norm.eps = c.norm_eps;
    gpu::Norm(norm, stream);
    Mark(13);

    const int taps = kNC * kNKV;
    gpu::GemmArgs g{};
    g.jobs[0] = {l.wq, in, 3 * kE, q, kHQ, kHQ, kE};
    g.jobs[1] = {l.rel, in, 3 * kE, r, kNR, kNR, kE};
    g.jobs[2] = {l.gate, in, 3 * kE, gate, kHQ, kHQ, kE};
    g.jobs[3] = {l.sink, in, 3 * kE, sink, kNH, kNH, kE};
    g.jobs[4] = {l.wk, in + kE, 3 * kE, k_raw, kHK, kHK, kE};
    g.jobs[5] = {l.wv, in + 2 * kE, 3 * kE, v_raw, kHK, kHK, kE};
    g.count = 6;
    if (!swa) {
      g.jobs[6] = {l.kcp, in + kE, 3 * kE, k_taps, taps, taps, kE};
      g.jobs[7] = {l.vcp, in + 2 * kE, 3 * kE, v_taps, taps, taps, kE};
      g.count = 8;
    }
    g.batch = batch;
    g.variant = gemm_variant;
    g.single_threads = single_threads;
    // Above one row, the kernel reads a column-major copy of the inputs.
    const bool use_xt = transpose && batch > 1;
    if (Wmma(g, in, 3 * kE, xh_in)) {
    } else if (use_xt) {
      gpu::TransposeArgs tr{};
      tr.x = in;
      tr.x_stride = 3 * kE;
      tr.xt = xt_in;
      tr.cols = 3 * kE;
      tr.batch = batch;
      gpu::Transpose(tr, stream);
      for (int j = 0; j < g.count; ++j) {
        const std::size_t column = static_cast<std::size_t>(g.jobs[j].x - in);
        g.jobs[j].xt = xt_in + column * gpu::kGemmGroup;
        g.jobs[j].xt_group_stride = 3 * kE * gpu::kGemmGroup;
      }
    }
    gpu::Gemm(g, stream);
    Mark(1);

    gpu::GlueBArgs b{};
    b.k_raw = k_raw;
    b.v_raw = v_raw;
    b.q = q;
    b.r = r;
    b.k_taps = swa ? nullptr : k_taps;
    b.v_taps = swa ? nullptr : v_taps;
    b.k_conv = l.k_conv;
    b.v_conv = l.v_conv;
    b.q_clip = l.q_clip;
    b.rel_clip = l.rel_clip;
    b.qk_clip = c.qk_clip;
    b.rel_clip_threshold = c.rel_clip;
    b.eps = c.norm_eps;
    b.rows = rows;
    b.batch = batch;
    gpu::GlueB(b, stream);
    Mark(2);

    gpu::AttnArgs at{};
    at.q = q;
    at.r = r;
    at.sink = sink;
    at.gate = gate;
    at.rel_bias = l.rel_bias;
    at.scratch = scratch;
    at.out = attn;
    at.extent = static_cast<int>(swa ? c.n_swa : c.n_rel_extent);
    at.max_n = max_n;
    at.scale = 1.0f / std::sqrt(static_cast<float>(gpu::kHD));
    at.rows = rows;
    at.batch = batch;
    // Sliding-window layers read at most 1,024 positions from their ring;
    // full-attention layers always use the chunked kernel and its layout.
    at.window = swa && attention_window;
    at.chunked = !swa;
    at.max_chunks = static_cast<int>(max_pos / 1024 + 1);
    at.chunk_max = chunk_max;
    at.chunk_sum = chunk_sum;
    at.chunk_out = chunk_out;
    if (!swa && attn_stats_now)
      AttnStats(rows, il);
    // Which kernel: the batched FP16 one for steps of several rows per
    // session, else the per-row FP16 one; the F32 kernels only when the F32
    // cache is kept and a switch asks for them.
    const std::size_t chunks_now = max_pos / 1024 + 1;
    const bool few = few_now && !swa;
    const bool batched =
        !few &&
        ((attn_batched_min > 0 && batch >= attn_batched_min * n_groups &&
          (!swa || attn_batched_window)) ||
         (!swa && std::size_t(batch) * chunks_now > partial_slots));
    const bool half = !batched && !few && (attn_half || !keep_f32);
    if (batched || few) {
      gpu::ToHalfArgs h{};
      h.x = q;
      h.x_stride = kHQ;
      h.out = qh;
      h.cols = kHQ;
      h.batch = batch;
      gpu::ToHalf(h, stream);
      at.qh = qh;
      at.groups = dev_groups;
      at.group_count = n_groups;
    }
    if ((batched || half) && attn_ab && keep_f32) {
      const std::size_t n = std::size_t(batch) * kHQ;
      ab_old.resize(n);
      ab_new.resize(n);
      gpu::Attention(at, stream);
      Check(hipMemcpyAsync(ab_old.data(), attn, n * 4, hipMemcpyDeviceToHost,
                           stream),
            "copy attention");
    }
    at.batched = batched;
    at.half = half;
    at.skip = attn_skip;
    at.fast = flash_now ? attn_fast : (attn_fast & ~16);
    if (swa && (batched || half))
      at.max_chunks = batched ? 2 : 1;
    if (!swa && batched) {
      at.span = static_cast<int>((chunks_now + 15) / 16);
      at.max_chunks = static_cast<int>((chunks_now + at.span - 1) / at.span);
    }
    if (few) {
      // As many parts as the partial scratch holds for this many rows.
      const std::size_t parts = std::max<std::size_t>(
          1, std::min<std::size_t>(std::size_t(attn_few_parts),
                                   partial_slots / std::size_t(batch)));
      at.few = true;
      at.span = static_cast<int>((chunks_now + parts - 1) / parts);
      at.max_chunks = static_cast<int>((chunks_now + at.span - 1) / at.span);
    }
    gpu::Attention(at, stream);
    if (!swa && attn_stats_now && at.span <= 1) {
      // The kernel's per-chunk maximum and sum for the measured row, against
      // the host's: this checks the measurement and the kernel against each
      // other.
      Check(hipStreamSynchronize(stream), "synchronize");
      const std::size_t chunks = stats_chunk_max.size() / kNH;
      const std::size_t row_base =
          (std::size_t(batch) - 1) * kNH * at.max_chunks;
      std::vector<float> dev_max(kNH * std::size_t(at.max_chunks)),
          dev_sum(dev_max.size());
      Check(hipMemcpy(dev_max.data(), chunk_max + row_base, dev_max.size() * 4,
                      hipMemcpyDeviceToHost),
            "chunk maxima");
      Check(hipMemcpy(dev_sum.data(), chunk_sum + row_base, dev_sum.size() * 4,
                      hipMemcpyDeviceToHost),
            "chunk sums");
      for (std::size_t h = 0; h < kNH; ++h)
        for (std::size_t ch = 0; ch < chunks; ++ch) {
          const double hm = stats_chunk_max[h * chunks + ch],
                       hs = stats_chunk_sum[h * chunks + ch];
          stats_max_diff = std::max(
              stats_max_diff, std::fabs(dev_max[h * at.max_chunks + ch] - hm));
          stats_sum_diff = std::max(
              stats_sum_diff,
              std::fabs(dev_sum[h * at.max_chunks + ch] - hs) / hs);
          ++stats_chunks;
        }
    }
    if ((batched || half) && attn_ab && keep_f32) {
      const std::size_t n = std::size_t(batch) * kHQ;
      Check(hipMemcpyAsync(ab_new.data(), attn, n * 4, hipMemcpyDeviceToHost,
                           stream),
            "copy attention");
      Check(hipStreamSynchronize(stream), "synchronize");
      for (std::size_t i = 0; i < n; ++i) {
        const double o = ab_old[i], d = double(ab_new[i]) - o;
        if (!std::isfinite(ab_new[i])) {
          ++ab_bad;
          continue;
        }
        ab_max_diff = std::max(ab_max_diff, std::fabs(d));
        ab_max_abs = std::max(ab_max_abs, std::fabs(o));
        ab_sq_diff += d * d;
        ab_sq += o * o;
      }
      ++ab_visits;
    }
    Mark(3);

    gpu::GemmArgs o{};
    o.jobs[0] = {l.wo, attn, kHQ, proj, kE, kE, kHQ};
    o.count = 1;
    o.batch = batch;
    o.variant = gemm_variant;
    o.single_threads = single_threads;
    if (Wmma(o, attn, kHQ, xh_attn)) {
    } else if (use_xt) {
      gpu::TransposeArgs tr{};
      tr.x = attn;
      tr.x_stride = kHQ;
      tr.xt = xt_attn;
      tr.cols = kHQ;
      tr.batch = batch;
      gpu::Transpose(tr, stream);
      o.jobs[0].xt = xt_attn;
      o.jobs[0].xt_group_stride = kHQ * gpu::kGemmGroup;
    }
    gpu::Gemm(o, stream);
    Mark(4);

    gpu::GlueCArgs cc{};
    cc.proj = proj;
    cc.post_conv = l.attn_post;
    cc.input = input;
    cc.block_start = block_start;
    cc.stack = stack;
    cc.n_blocks = n_blocks;
    cc.cur = cur;
    cc.xn = xn;
    cc.ffn_res_query = l.ffn_res_query;
    cc.rows = rows;
    cc.batch = batch;
    gpu::GlueC(cc, stream);
    Mark(5);
    if (block_start)
      ++n_blocks;
    if (looped) {
      gpu::CtrlArgs ctrl{};
      ctrl.cur = xn;
      ctrl.cur_stride = kE;
      ctrl.msg = msgs + 3 * kE;
      ctrl.msg_stride = 4 * kE;
      ctrl.down[0] = l.ctrl_down[3];
      ctrl.gain[0] = l.ctrl_gain[3];
      ctrl.forget[0] = l.ctrl_forget[3];
      ctrl.low = low;
      ctrl.count = 1;
      ctrl.batch = batch;
      if (split) {
        // msgs has not changed since its FP16 parts were made above.
        ToHalf2(xn, kE, kE, xh_x, ctrl_low ? sl_x : nullptr);
        Mark(14);
        SplitJob jobs[2] = {
            {l.down_cur[3], xh_x, sl_x, kE, low_c, 3 * kRank, kRank, kE},
            {l.down_msg[3], sh_m + 3 * kE, sl_m + 3 * kE, 4 * kE, low_m,
             3 * kRank, kRank, kE}};
        SplitGemm(jobs, 2, ctrl_low);
        Mark(15);
        ctrl.yc = low_c;
        ctrl.ym = low_m;
        gpu::CtrlCombine(ctrl, stream);
        Mark(16);
        // (ToHalf packs its output: kRank values per row here.)
        ToHalf2(low, 3 * kRank, kRank, low_h, ctrl_low ? low_l : nullptr);
        jobs[0] = {l.gain_split[3], low_h, low_l, kRank, ctrl_g, 3 * kE, kE,
                   kRank};
        jobs[1] = {l.forget_split[3], low_h, low_l, kRank, ctrl_f, 3 * kE, kE,
                   kRank};
        SplitGemm(jobs, 2, ctrl_low);
        Mark(17);
        ctrl.yg = ctrl_g;
        ctrl.yf = ctrl_f;
      } else {
        gpu::CtrlDown(ctrl, stream);
      }
      gpu::CtrlApply(ctrl, stream);
    }
    Mark(11);
    gpu::NormArgs ffn_norm{};
    ffn_norm.x = xn;
    ffn_norm.stride = kE;
    ffn_norm.w[0] = l.ffn_norm;
    ffn_norm.count = 1;
    ffn_norm.batch = batch;
    ffn_norm.eps = c.norm_eps;
    gpu::Norm(ffn_norm, stream);
    Mark(13);
    gpu::RouterArgs router{};
    router.gate_inp = l.gate_inp;
    router.x = xn;
    router.logits = router_logits;
    router.exp_probs_b = l.exp_probs_b;
    router.up_scale_all = l.up_scale;
    router.down_scale_all = l.down_scale;
    router.weights_scale = c.expert_weights_scale;
    router.ids = sel_ids;
    router.up_scale = sel_up;
    router.coef = sel_coef;
    router.batch = batch;
    if (split) {
      ToHalf2(xn, kE, kE, xh_x, sl_x);
      const SplitJob job{l.gate_split, xh_x, sl_x, kE, router_logits,
                         gpu::kNX, gpu::kNX, kE};
      SplitGemm(&job, 1);
      router.logits_ready = true;
    }
    gpu::Router(router, stream);
    Mark(12);

    gpu::BatchExpertArgs e{};
    e.gate_up = l.gate_up;
    e.down = l.down;
    e.x = xn;
    e.ids = sel_ids;
    e.up_scale = sel_up;
    e.coef = sel_coef;
    e.act = act;
    e.out = moe;
    e.situ_beta = c.situ_beta;
    e.situ_linear_beta = c.situ_linear_beta;
    e.nvfp4 = l.nvfp4;
    e.batch = batch;
    // Expert-major pays off once rows share experts; alone, a row's eight
    // experts are cheaper to visit directly.
    e.variant = expert_variant;
    if (expert_variant == 0 && wmma_expert_min_batch > 0 &&
        batch >= wmma_expert_min_batch)
      e.variant = 5;
    else if (expert_variant == 0 && batch >= major_min_batch)
      e.variant = opt.int8_experts ? 3 : 2;
    else if (expert_variant == 0 && opt.int8_experts && opt.int8_single)
      e.variant = 4;
    e.xq = xq;
    e.xs = xs;
    e.aq = aq;
    e.as = as;
    e.group_list = group_list;
    e.group_count = group_count;
    e.items = expert_items;
    e.items_full = expert_items_full;
    e.contrib = contrib;
    e.xh = xh_x;
    e.wmma_tiles = expert_tiles;
    e.timing = expert_skip;
    e.acth = acth;
    e.max_batch = static_cast<int>(opt.max_batch);
    gpu::BatchExperts(e, stream);
    if (expert_stats && batch >= 256 && e.variant == 5) {
      // INKLING2_EXPERT_STATS: how the step's rows spread over the experts.
      Check(hipStreamSynchronize(stream), "synchronize");
      std::int32_t counts[kNX];
      Check(hipMemcpy(counts, group_count, sizeof counts, hipMemcpyDeviceToHost),
            "copy expert counts");
      for (const std::int32_t n : counts) {
        ++stat_experts;
        stat_empty += n == 0;
        stat_pairs += n;
        stat_tiles128 += (n + 127) / 128;
        stat_tiles256 += (n + 255) / 256;
        stat_over128 += n > 128;
        stat_over256 += n > 256;
        stat_over512 += n > 512;
        stat_max = std::max<std::size_t>(stat_max, n);
      }
    }
    Mark(l.nvfp4 ? 6 : 7);

    gpu::GlueDArgs d{};
    d.moe = moe;
    d.post_conv = l.ffn_post;
    d.cur = cur;
    d.rows = rows;
    d.batch = batch;
    gpu::GlueD(d, stream);
    Mark(8);
    n_launches += looped ? 17 : 13;
  }

  void Reset(std::uint32_t index) {
    Session& session = sessions.at(index);
    session.pos = 0;
    session.pending_first = -1;
    session.draft_ready = false;
    ResetList(session);
    Check(hipMemsetAsync(session.hist, 0, kHistFloats * c.visits() * 4, stream),
          "clear");
    Check(hipStreamSynchronize(stream), "synchronize");
  }

  // A session as bytes: position, the sliding-window rings whole, and of the
  // full-attention caches the positions in use. It can be restored into a
  // session of an engine with the same ring size (same max_batch).
  std::vector<std::uint8_t> Save(std::uint32_t index) {
    const Session& session = sessions.at(index);
    if (session.pending_first >= 0)
      throw std::runtime_error("inkling2 gpu: a tentative step is pending");
    Check(hipStreamSynchronize(stream), "synchronize");
    const std::size_t block = std::size_t(gpu::kHD) * 1024 * 2;  // bytes
    const std::size_t chunks = (session.pos + 1023) / 1024;
    std::vector<std::uint8_t> out(16);
    const std::uint32_t header[4] = {
        0x32494e4bu, session.pos,
        static_cast<std::uint32_t>(session.slots[0].cap),
        static_cast<std::uint32_t>(c.visits())};
    std::memcpy(out.data(), header, 16);
    auto take = [&](const void* device, std::size_t bytes) {
      const std::size_t at = out.size();
      out.resize(at + bytes);
      if (bytes > 0)
        Check(hipMemcpy(out.data() + at, device, bytes, hipMemcpyDeviceToHost),
              "save");
    };
    take(session.hist, kHistFloats * c.visits() * 4);
    for (std::uint32_t s = 0; s < c.visits(); ++s) {
      const Slot& slot = session.slots[s];
      if (c.is_swa[Physical(s)]) {
        take(slot.kh, std::size_t(slot.cap) * kHK * 2);
        take(slot.vh, std::size_t(slot.cap) * kHK * 2);
        continue;
      }
      const std::size_t per_head = std::size_t(slot.cap) * gpu::kHD;
      for (std::size_t head = 0; head < kNKV; ++head) {
        take(slot.kh + head * per_head, std::size_t(session.pos) * gpu::kHD * 2);
        take(slot.vh + head * per_head, chunks * block);
      }
    }
    return out;
  }

  void Restore(std::uint32_t index, const std::uint8_t* data, std::size_t size) {
    Session& session = sessions.at(index);
    std::uint32_t header[4];
    if (size < 16)
      throw std::runtime_error("inkling2 gpu: not a saved session");
    std::memcpy(header, data, 16);
    const std::uint32_t pos = header[1];
    if (header[0] != 0x32494e4bu || keep_f32 ||
        header[2] != static_cast<std::uint32_t>(session.slots[0].cap) ||
        header[3] != c.visits() || pos > opt.context)
      throw std::runtime_error(
          "inkling2 gpu: the saved session does not fit this engine");
    Check(hipStreamSynchronize(stream), "synchronize");
    const std::size_t block = std::size_t(gpu::kHD) * 1024 * 2;
    const std::size_t chunks = (pos + 1023) / 1024;
    std::size_t at = 16;
    auto put = [&](void* device, std::size_t bytes) {
      if (at + bytes > size)
        throw std::runtime_error("inkling2 gpu: the saved session is cut short");
      if (bytes > 0)
        Check(hipMemcpy(device, data + at, bytes, hipMemcpyHostToDevice),
              "restore");
      at += bytes;
    };
    put(session.hist, kHistFloats * c.visits() * 4);
    for (std::uint32_t s = 0; s < c.visits(); ++s) {
      Slot& slot = session.slots[s];
      if (c.is_swa[Physical(s)]) {
        put(slot.kh, std::size_t(slot.cap) * kHK * 2);
        put(slot.vh, std::size_t(slot.cap) * kHK * 2);
        continue;
      }
      const std::size_t per_head = std::size_t(slot.cap) * gpu::kHD;
      for (std::size_t head = 0; head < kNKV; ++head) {
        put(slot.kh + head * per_head, std::size_t(pos) * gpu::kHD * 2);
        put(slot.vh + head * per_head, chunks * block);
      }
    }
    session.pos = pos;
    session.pending_first = -1;
  }

  void FakeContext(std::uint32_t index, std::uint32_t position) {
    Session& session = sessions.at(index);
    const std::uint32_t have = session.pos;
    if (have == 0 || position <= have || position > opt.context)
      throw std::runtime_error("inkling2 gpu: cannot fake that context");
    // Every array of a full-attention cache holds, per key/value head, blocks
    // of 1,024 positions times 128 dimensions (in one order or the other):
    // every later block becomes a copy of the first, so that position p holds
    // what position p % 1024 holds.
    const std::size_t block = std::size_t(gpu::kHD) * 1024;
    const std::uint32_t chunks = (position + 1023) / 1024;
    for (std::uint32_t s = 0; s < c.visits(); ++s) {
      if (c.is_swa[Physical(s)])
        continue;  // sliding-window layers only keep their window
      Slot& slot = session.slots[s];
      const std::size_t per_head = std::size_t(slot.cap) * gpu::kHD;
      for (std::size_t head = 0; head < kNKV; ++head) {
        float* f32[2] = {slot.k ? slot.k + head * per_head : nullptr,
                         slot.v ? slot.v + head * per_head : nullptr};
        std::uint16_t* f16[2] = {slot.kh + head * per_head,
                                 slot.vh + head * per_head};
        for (std::uint32_t chunk = 1; chunk < chunks; ++chunk)
          for (int i = 0; i < 2; ++i) {
            if (f32[i])
              Check(hipMemcpyAsync(f32[i] + chunk * block, f32[i], block * 4,
                                   hipMemcpyDeviceToDevice, stream),
                    "copy cache");
            Check(hipMemcpyAsync(f16[i] + chunk * block, f16[i], block * 2,
                                 hipMemcpyDeviceToDevice, stream),
                  "copy cache");
          }
      }
    }
    Check(hipStreamSynchronize(stream), "synchronize");
    session.pos = position;
  }

  std::uint32_t HeadArgmax(const float* hidden) {
    Check(hipMemcpyAsync(final_x, hidden, kE * 4, hipMemcpyHostToDevice, stream),
          "copy hidden");
    gpu::GemmArgs g{};
    const int vocab = static_cast<int>(c.n_vocab_padded);
    g.jobs[0] = {output, final_x, kE, logits, vocab, vocab, kE};
    g.count = 1;
    g.batch = 1;
    g.single_threads = head_threads;
    gpu::Gemm(g, stream);
    if (c.logit_softcap != 0.0f)
      gpu::SoftCap(logits, c.n_vocab_padded, c.logit_softcap, stream);
    Check(hipMemcpyAsync(host_logits.data(), logits, std::size_t{c.n_vocab} * 4,
                         hipMemcpyDeviceToHost, stream),
          "copy logits");
    Check(hipStreamSynchronize(stream), "synchronize");
    n_heads = 1;
    const float* raw = host_logits.data();
    return static_cast<std::uint32_t>(std::max_element(raw, raw + c.n_vocab) -
                                      raw);
  }

  // Copies batch row `b`'s candidate histories into its session.
  void CommitHistory(Session& session, std::size_t b) {
    Check(hipMemcpyAsync(session.hist,
                         hist_candidates +
                             hist_slot_of[b] * c.visits() * kHistFloats,
                         kHistFloats * c.visits() * 4, hipMemcpyDeviceToDevice,
                         stream),
          "commit histories");
  }

  void Accept(std::uint32_t index, std::uint32_t rows_kept) {
    Session& session = sessions.at(index);
    if (session.pending_first < 0 ||
        rows_kept > static_cast<std::uint32_t>(session.pending_count))
      throw std::runtime_error("inkling2 gpu: nothing tentative to accept");
    if (rows_kept > 0) {
      CommitHistory(session, std::size_t(session.pending_first) + rows_kept - 1);
      CommitDraft(session, std::size_t(session.pending_first) + rows_kept - 1);
      Check(hipStreamSynchronize(stream), "synchronize");
      session.pos += rows_kept;
    }
    session.pending_first = -1;
  }

  void Step(const std::vector<Row>& rows, Head head, bool tentative,
            bool list_likely) {
    if (rows.empty() || rows.size() > opt.max_batch)
      throw std::runtime_error("inkling2 gpu: batch size out of range");
    batch = static_cast<int>(rows.size());

    // Runs: adjacent rows of one session, each session at most once.
    std::vector<int> run(rows.size()), last(rows.size(), 0);
    std::vector<std::uint32_t> seen;
    std::vector<std::int32_t> heads;
    for (std::size_t b = 0; b < rows.size(); ++b) {
      const Row& row = rows[b];
      if (row.session >= sessions.size() || row.token >= c.n_vocab)
        throw std::runtime_error("inkling2 gpu: session or token out of range");
      const bool continues = b > 0 && rows[b - 1].session == row.session;
      if (!continues) {
        if (std::find(seen.begin(), seen.end(), row.session) != seen.end())
          throw std::runtime_error(
              "inkling2 gpu: a session's rows must be adjacent");
        seen.push_back(row.session);
      }
      run[b] = continues ? run[b - 1] + 1 : 0;
      if (b + 1 == rows.size() || rows[b + 1].session != row.session)
        last[b] = 1;
      if (head == Head::kAll || (head == Head::kLast && last[b]))
        heads.push_back(static_cast<std::int32_t>(b));
    }

    std::size_t hist_next = 0;
    auto* states = reinterpret_cast<gpu::RowState*>(host_staging.data());
    auto* tokens =
        reinterpret_cast<std::uint32_t*>(host_staging.data() + off_tokens);
    auto* head_rows =
        reinterpret_cast<std::int32_t*>(host_staging.data() + off_heads);
    for (std::size_t b = 0; b < rows.size(); ++b) {
      Session& session = sessions[rows[b].session];
      const std::uint32_t pos =
          session.pos + static_cast<std::uint32_t>(run[b]);
      if (pos >= opt.context)
        throw std::runtime_error("inkling2 gpu: context is full");
      tokens[b] = rows[b].token;
      max_pos = b == 0 ? pos : std::max(max_pos, pos);
      hist_slot_of[b] = 0;
      if (last[b] || tentative) {
        if (hist_next >= hist_slots)
          throw std::runtime_error(
              "inkling2 gpu: too many rows may end a run in one step");
        hist_slot_of[b] = static_cast<std::uint32_t>(hist_next++);
      }
      const float ratio = std::max(
          1.0f, (pos + 1.0f) / static_cast<float>(c.log_scale_floor));
      const float tau = 1.0f + c.log_scale_alpha * std::log(ratio);
      for (std::uint32_t s = 0; s < c.visits(); ++s) {
        const Slot& slot = session.slots[s];
        const bool swa = c.is_swa[Physical(s)];
        gpu::RowState& state = states[std::size_t(s) * opt.max_batch + b];
        state.kv_k = slot.k;
        state.kv_v = slot.v;
        for (int k = 0; k < 4; ++k) {
          state.hist_in[k] = session.hist + s * kHistFloats + kHistOffset[k];
          state.hist_out[k] = hist_candidates +
                              (hist_slot_of[b] * c.visits() + s) * kHistFloats +
                              kHistOffset[k];
        }
        state.cap = slot.cap;
        state.first =
            swa && pos >= c.n_swa ? static_cast<int>(pos - c.n_swa + 1) : 0;
        state.pos = static_cast<int>(pos);
        state.run = run[b];
        // A tentative run may end at any of its rows.
        state.last = last[b] || tentative ? 1 : 0;
        state.tau = swa ? 1.0f : tau;
        state.layout = swa ? 0 : 1;
        state.chunk_cap = slot.cap / 1024;
        state.kh = slot.kh;
        state.vh = slot.vh;
      }
    }
    if (heads.size() > max_heads)
      throw std::runtime_error("inkling2 gpu: too many rows want logits");
    for (std::size_t i = 0; i < heads.size(); ++i)
      head_rows[i] = heads[i];
    if (draft) {
      auto* draft_rows =
          reinterpret_cast<gpu::DraftRow*>(host_staging.data() + off_draft);
      std::uint32_t run_end = 0;
      for (std::size_t b = rows.size(); b-- > 0;) {
        const Session& session = sessions[rows[b].session];
        const std::uint32_t pos =
            session.pos + static_cast<std::uint32_t>(run[b]);
        if (last[b])
          run_end = pos;
        gpu::DraftRow& row = draft_rows[b];
        for (int l = 0; l < gpu::kDraftLayers; ++l) {
          row.k[l] = DraftRing(session, l, false);
          row.v[l] = DraftRing(session, l, true);
          row.cap[l] = draft->cap[l];
          // Rows further back than a ring is long would share its slots
          // with later rows of the step.
          row.store[l] =
              run_end - pos + 48 < static_cast<std::uint32_t>(draft->cap[l]);
        }
        row.pos = static_cast<std::int32_t>(pos);
        row.pad = 0;
      }
    }
    // Groups for the batched attention kernel: runs cut into pieces of at
    // most kAttnRows rows.
    auto* groups =
        reinterpret_cast<gpu::AttnGroup*>(host_staging.data() + off_groups);
    flash_now = (attn_fast & 16) != 0 &&
                rows.size() >= static_cast<std::size_t>(attn_flash_min);
    few_now = !flash_now && (attn_fast & 32) != 0 &&
              rows.size() >= static_cast<std::size_t>(attn_few_min);
    const std::size_t attn_group_rows =
        flash_now ? std::size_t{64} : std::size_t(gpu::kAttnRows);
    n_groups = 0;
    for (std::size_t b = 0; b < rows.size();) {
      std::size_t end = b + 1;
      while (end < rows.size() && end - b < attn_group_rows &&
             rows[end].session == rows[b].session)
        ++end;
      groups[n_groups++] = {static_cast<std::int32_t>(b),
                            static_cast<std::int32_t>(end - b)};
      b = end;
    }
    // The batch row that ends each run.
    auto* ends =
        reinterpret_cast<std::int32_t*>(host_staging.data() + off_ends);
    int n_runs = 0;
    for (std::size_t b = 0; b < rows.size(); ++b)
      if (last[b])
        ends[n_runs++] = static_cast<std::int32_t>(b);
    Check(hipMemcpyAsync(staging, host_staging.data(), host_staging.size(),
                         hipMemcpyHostToDevice, stream),
          "copy step inputs");
    if (rows.size() > static_cast<std::size_t>(n_runs)) {
      gpu::ClearAheadArgs ahead{};
      ahead.rows = row_states;
      ahead.ends = reinterpret_cast<const std::int32_t*>(
          static_cast<const std::uint8_t*>(staging) + off_ends);
      ahead.runs = n_runs;
      ahead.visits = static_cast<int>(c.visits());
      ahead.max_batch = static_cast<int>(opt.max_batch);
      ahead.ahead = kClearAhead;
      gpu::ClearAhead(ahead, stream);
    }

    attn_stats_now = false;
    if (attn_stats_every > 0) {
      const Session& s0 = sessions[rows[0].session];
      const std::uint32_t before = s0.pos, after = max_pos + 1;
      attn_stats_now = before / attn_stats_every != after / attn_stats_every;
      stats_pos = max_pos;
    }
    Mark(-1);
    gpu::EmbedArgs e{};
    e.table = tok_embd;
    e.tokens = dev_tokens;
    e.norm = tok_norm;
    e.eps = c.norm_eps;
    e.cur = cur;
    e.batch = batch;
    gpu::Embed(e, stream);

    n_blocks = 0;
    executed = 0;
    for (std::uint32_t il = 0; il < c.loop_start; ++il) {
      Tap();
      Visit(il, il, false);
    }
    // Every pass starts from the prelude output.
    const int prelude_blocks = n_blocks;
    gpu::CopyArgs keep{};
    keep.dst[0] = prelude;
    keep.dst_stride[0] = kE;
    keep.src[0] = cur;
    keep.src_stride[0] = kE;
    keep.count = 1;
    keep.batch = batch;
    gpu::Copy(keep, stream);
    const std::uint32_t core = c.loop_end - c.loop_start;
    for (std::uint32_t pass = 0; pass < c.loop_count; ++pass) {
      if (pass > 0) {
        // The bank of the previous pass: its block after the first core block,
        // and its output. Then restart from the prelude.
        gpu::CopyArgs bank{};
        bank.dst[0] = bank0;
        bank.dst_stride[0] = kE;
        bank.src[0] = stack + std::size_t(prelude_blocks + 1) * kE;
        bank.src_stride[0] = kMaxStates * kE;
        bank.dst[1] = bank_out;
        bank.dst_stride[1] = kE;
        bank.src[1] = cur;
        bank.src_stride[1] = kE;
        bank.dst[2] = cur;
        bank.dst_stride[2] = kE;
        bank.src[2] = prelude;
        bank.src_stride[2] = kE;
        bank.count = 3;
        bank.batch = batch;
        gpu::Copy(bank, stream);
        n_blocks = prelude_blocks;
      }
      for (std::uint32_t ip = c.loop_start; ip < c.loop_end; ++ip) {
        const std::uint32_t slot =
            pass == 0 ? ip : c.n_layer + (pass - 1) * core + (ip - c.loop_start);
        Tap();
        Visit(slot, ip, pass > 0);
      }
    }
    for (std::uint32_t il = c.loop_end; il < c.n_layer; ++il) {
      Tap();
      Visit(il, il, false);
    }
    ReportAttnStats(stats_pos);
    if (need_features) {
      gpu::FinalArgs f{};
      f.stack = stack;
      f.n_blocks = n_blocks;
      f.cur = cur;
      f.res_query = output_res_query;
      f.norm = output_norm;
      f.eps = c.norm_eps;
      f.head_rows = dev_identity;
      f.out = feat_final;
      f.count = batch;
      gpu::Final(f, stream);
      if (draft)
        StoreDraftContext();
    }
    if (opt.capture_features) {
      for (int k = 0; k < 3; ++k)
        Check(hipMemcpyAsync(host_feat.data() + std::size_t(k) * opt.max_batch * kE,
                             feat + std::size_t(k) * opt.max_batch * kE,
                             std::size_t(batch) * kE * 4, hipMemcpyDeviceToHost,
                             stream),
              "copy features");
      Check(hipMemcpyAsync(host_feat_final.data(), feat_final,
                           std::size_t(batch) * kE * 4, hipMemcpyDeviceToHost,
                           stream),
            "copy features");
    }

    n_heads = 0;
    if (head != Head::kNone) {
      Mark(9);
      n_heads = heads.size();
      gpu::FinalArgs f{};
      f.stack = stack;
      f.n_blocks = n_blocks;
      f.cur = cur;
      f.res_query = output_res_query;
      f.norm = output_norm;
      f.eps = c.norm_eps;
      f.head_rows = dev_heads;
      f.out = final_x;
      f.count = static_cast<int>(n_heads);
      gpu::Final(f, stream);
      gpu::GemmArgs g{};
      const int vocab = static_cast<int>(c.n_vocab_padded);
      g.jobs[0] = {output, final_x, kE, logits, vocab, vocab, kE};
      g.count = 1;
      g.batch = static_cast<int>(n_heads);
      g.single_threads = head_threads;
      // Few rows: the F32 kernel, which reads a weight row once for all of
      // them, is several times faster on this 200K-row matrix.
      if (static_cast<int>(n_heads) >= head_wmma_min &&
          Wmma(g, final_x, kE, xh_head) && head_tiles >= 0)
        g.wmma_tiles = head_tiles;
      gpu::Gemm(g, stream);
      if (c.logit_softcap != 0.0f)
        gpu::SoftCap(logits, n_heads * c.n_vocab_padded, c.logit_softcap,
                     stream);
      Check(hipMemcpyAsync(host_logits.data(), logits,
                           n_heads * c.n_vocab_padded * 4,
                           hipMemcpyDeviceToHost, stream),
            "copy logits");
    }
    if (!tentative)
      for (std::size_t b = 0; b < rows.size(); ++b)
        if (last[b]) {
          CommitHistory(sessions[rows[b].session], b);
          CommitDraft(sessions[rows[b].session], b);
        }
    Check(hipStreamSynchronize(stream), "synchronize");
    Mark(10);
    n_launches += 6;

    if (draft && draft->slots > 0) {
      for (const Row& row : rows)
        List(sessions[row.session], row.token);
      if (list_likely)
        for (std::size_t i = 0; i < n_heads; ++i)
          ListLikely(sessions[rows[heads[i]].session],
                     host_logits.data() + i * c.n_vocab_padded);
    }
    for (std::size_t b = 0; b < rows.size(); ++b)
      if (last[b]) {
        Session& session = sessions[rows[b].session];
        if (tentative) {
          session.pending_first = static_cast<int>(b) - run[b];
          session.pending_count = run[b] + 1;
        } else {
          session.pos += static_cast<std::uint32_t>(run[b]) + 1;
          session.pending_first = -1;
        }
      }
  }
};

GpuEngine::GpuEngine(const Model& model, const Options& options)
    : impl_(std::make_unique<Impl>(model, options)) {}
GpuEngine::~GpuEngine() = default;

void GpuEngine::Step(const std::vector<Row>& rows, Head head, bool tentative,
                     bool list_likely) {
  impl_->Step(rows, head, tentative, list_likely);
}

void GpuEngine::Accept(std::uint32_t session, std::uint32_t rows_kept) {
  impl_->Accept(session, rows_kept);
}

void GpuEngine::Reset(std::uint32_t session) {
  impl_->Reset(session);
}

std::vector<std::uint8_t> GpuEngine::Save(std::uint32_t session) {
  return impl_->Save(session);
}

void GpuEngine::Restore(std::uint32_t session,
                        const std::vector<std::uint8_t>& saved) {
  impl_->Restore(session, saved.data(), saved.size());
}

GpuEngine::StateLayout GpuEngine::Layout() const {
  const Impl& impl = *impl_;
  StateLayout layout;
  layout.visits = impl.c.visits();
  layout.heads = kNKV;
  layout.history_bytes = Impl::kHistFloats * impl.c.visits() * 4;
  layout.key_row_bytes = std::size_t(gpu::kHD) * 2;
  layout.value_chunk_bytes = std::size_t(gpu::kHD) * 1024 * 2;
  layout.draft_bytes = impl.draft ? impl.draft->state_bytes : 0;
  const Impl::Session& session = impl.sessions.at(0);
  for (std::uint32_t s = 0; s < impl.c.visits(); ++s) {
    const bool ring = impl.c.is_swa[impl.Physical(s)];
    layout.ring.push_back(ring);
    if (ring)
      layout.ring_bytes = Impl::kRingWindowBytes;
    else
      layout.full_capacity = static_cast<std::uint32_t>(session.slots[s].cap);
  }
  return layout;
}

void* GpuEngine::DraftState(std::uint32_t session) const {
  return impl_->sessions.at(session).draft_state;
}

void GpuEngine::Scrub(std::uint32_t session) {
  Impl& impl = *impl_;
  impl.Reset(session);
  for (const Impl::Slot& slot : impl.sessions.at(session).slots) {
    const std::size_t bytes = std::size_t(slot.cap) * kHK * 2;
    Check(hipMemset(slot.kh, 0, bytes), "clear");
    Check(hipMemset(slot.vh, 0, bytes), "clear");
  }
}

void GpuEngine::ResetDraftList(std::uint32_t session) {
  impl_->ResetList(impl_->sessions.at(session));
}

void GpuEngine::ListDraftTokens(std::uint32_t session,
                                std::span<const std::uint32_t> tokens) {
  Impl& impl = *impl_;
  if (!impl.draft || impl.draft->slots == 0)
    return;
  Impl::Session& state = impl.sessions.at(session);
  for (const std::uint32_t token : tokens)
    if (token < impl.c.n_vocab)
      impl.List(state, token);
}

std::size_t GpuEngine::draft_list_size(std::uint32_t session) const {
  return static_cast<std::size_t>(impl_->sessions.at(session).list.used);
}

bool GpuEngine::has_drafter() const {
  return impl_->draft != nullptr;
}

std::vector<GpuEngine::DraftToken> GpuEngine::Draft(std::uint32_t session,
                                                    std::uint32_t anchor,
                                                    int max_tokens,
                                                    float min_confidence) {
  return impl_->DraftTokens(session, anchor, max_tokens, min_confidence);
}

void* GpuEngine::History(std::uint32_t session) const {
  return impl_->sessions.at(session).hist;
}

void* GpuEngine::RingStage(std::uint32_t session, std::uint32_t visit) const {
  const Impl& impl = *impl_;
  const int ring = impl.ring_index.at(visit);
  if (ring < 0)
    throw std::runtime_error("inkling2 gpu: not a sliding-window visit");
  return impl.sessions.at(session).ring_stage +
         std::size_t(ring) * Impl::kRingWindowBytes;
}

void GpuEngine::PackRing(std::uint32_t session, std::uint32_t visit) {
  impl_->RingWindow(session, visit, impl_->sessions.at(session).pos, false);
}

void GpuEngine::UnpackRing(std::uint32_t session, std::uint32_t visit,
                           std::uint32_t position) {
  impl_->RingWindow(session, visit, position, true);
}

void* GpuEngine::Full(std::uint32_t session, std::uint32_t visit,
                      std::uint32_t head, bool values) const {
  const auto& slot = impl_->sessions.at(session).slots.at(visit);
  // Keys: [head][position][dim]. Values: [head][block][dim][position in block].
  return values ? static_cast<void*>(slot.vh + std::size_t(head) * slot.cap *
                                                   gpu::kHD)
                : static_cast<void*>(slot.kh + std::size_t(head) * slot.cap *
                                                   gpu::kHD);
}

void GpuEngine::SetPosition(std::uint32_t session, std::uint32_t position) {
  if (impl_->keep_f32)
    throw std::runtime_error(
        "inkling2 gpu: state cannot be put in place with the F32 cache kept");
  auto& state = impl_->sessions.at(session);
  if (position > impl_->opt.context)
    throw std::runtime_error("inkling2 gpu: position beyond the context");
  state.pos = position;
  state.pending_first = -1;
  state.draft_ready = position > 0;
}

void GpuEngine::Synchronize() {
  Check(hipStreamSynchronize(impl_->stream), "synchronize");
}

std::uint32_t GpuEngine::context() const {
  return impl_->opt.context;
}

std::uint32_t GpuEngine::max_batch() const {
  return impl_->opt.max_batch;
}

std::uint32_t GpuEngine::sessions() const {
  return impl_->opt.sessions;
}

void GpuEngine::FakeContext(std::uint32_t session, std::uint32_t position) {
  impl_->FakeContext(session, position);
}

const float* GpuEngine::logits(std::size_t i) const {
  if (i >= impl_->n_heads)
    throw std::runtime_error("inkling2 gpu: no logits for that run");
  return impl_->host_logits.data() + i * impl_->c.n_vocab_padded;
}

std::vector<float> GpuEngine::capped_logits(std::size_t i) const {
  const float* capped = logits(i);
  return std::vector<float>(capped, capped + impl_->c.n_vocab);
}

std::uint32_t GpuEngine::argmax(std::size_t i) const {
  const float* raw = logits(i);
  return static_cast<std::uint32_t>(
      std::max_element(raw, raw + impl_->c.n_vocab) - raw);
}

std::uint32_t GpuEngine::position(std::uint32_t session) const {
  return impl_->sessions.at(session).pos;
}

std::vector<float> GpuEngine::features(std::size_t row) const {
  const Impl& i = *impl_;
  if (!i.opt.capture_features || row >= static_cast<std::size_t>(i.batch))
    throw std::runtime_error("inkling2 gpu: no features for that row");
  std::vector<float> out(4 * kE);
  for (int k = 0; k < 3; ++k)
    std::memcpy(out.data() + std::size_t(k) * kE,
                i.host_feat.data() + (std::size_t(k) * i.opt.max_batch + row) * kE,
                kE * 4);
  std::memcpy(out.data() + 3 * kE, i.host_feat_final.data() + row * kE, kE * 4);
  return out;
}

std::uint32_t GpuEngine::HeadArgmax(const float* hidden) {
  return impl_->HeadArgmax(hidden);
}

std::uint32_t GpuEngine::vocab() const {
  return impl_->c.n_vocab;
}

void GpuEngine::SetProfile(bool on) {
  impl_->profile = on;
}

std::vector<std::pair<std::string, double>> GpuEngine::phases() const {
  std::vector<std::pair<std::string, double>> out;
  for (int i = 0; i < Impl::kPhases; ++i)
    out.emplace_back(Impl::kPhaseNames[i], impl_->phase_seconds[i]);
  return out;
}

std::size_t GpuEngine::launches() const {
  return impl_->n_launches;
}

}  // namespace inkling2
