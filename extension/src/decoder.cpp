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
  };

  static constexpr std::size_t kHistFloats = (kNC - 1) * (2 * kHK + 2 * kE);
  static constexpr std::size_t kHistOffset[4] = {
      0, (kNC - 1) * kHK, 2 * (kNC - 1) * kHK,
      2 * (kNC - 1) * kHK + (kNC - 1) * kE};
  // Each batch row's candidate histories for every visit, written by the rows
  // that may end a run; committing a row copies its candidate to its session.
  float* hist_candidates{};

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
  std::int32_t *group_list{}, *group_count{};
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
  std::uint32_t stats_pos{0};
  struct HeadStats {
    double sink, last_1024, pos_999, pos_9999, slice_999, slice_9999,
        slice_99999, chunk_9999;
  };
  std::vector<HeadStats> stats;
  double ab_max_diff{0}, ab_max_abs{0}, ab_sq_diff{0}, ab_sq{0};
  std::size_t ab_bad{0}, ab_visits{0};
  int max_chunks{1};
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

  static constexpr int kPhases = 11;
  static constexpr const char* kPhaseNames[kPhases] = {
      "glue A (mix, messages, norms)", "input projections",
      "glue B (conv, clips, KV store)", "attention", "output projection",
      "glue C (conv, mix, norm, router)", "experts NVFP4", "experts BF16",
      "glue D (conv, residual)", "embed and copies", "output head"};
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
      l.exp_probs_b = Up(w.exp_probs_b);
      l.up_scale = Up(w.gate_up_scale);
      l.down_scale = Up(w.down_scale);
      l.msg_query = Up(w.msg_query);
      for (int k = 0; k < 4; ++k) {
        const bool core = il >= c.loop_start && il < c.loop_end;
        l.ctrl_down[k] = core ? UpF32(w.ctrl_down[k]) : nullptr;
        l.ctrl_gain[k] = core ? UpF32(w.ctrl_gain[k]) : nullptr;
        l.ctrl_forget[k] = core ? UpF32(w.ctrl_forget[k]) : nullptr;
      }
    }

    // A sliding-window ring must keep the window of the earliest row of a
    // batch while the latest row of the same session is stored.
    // (A multiple of 64, for the FP16 copies.)
    const int swa_cap =
        static_cast<int>((c.n_swa + opt.max_batch + 63) / 64 * 64);
    sessions.resize(opt.sessions);
    for (Session& session : sessions) {
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
    hist_candidates = Floats(B * c.visits() * kHistFloats);
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
    max_chunks = static_cast<int>(std::max<std::size_t>(2, (opt.context + 1023) / 1024));
    chunk_max = Floats(B * kNH * std::size_t(max_chunks));
    chunk_sum = Floats(B * kNH * std::size_t(max_chunks));
    chunk_out = Floats(B * kNH * std::size_t(max_chunks) * gpu::kHD);
    contrib = Floats(B * kTopK * kE);
    if (opt.int8_experts) {
      xq = static_cast<std::uint32_t*>(Alloc(B * kE));
      xs = Floats(B * kE / 16);
      aq = static_cast<std::uint32_t*>(Alloc(B * kTopK * gpu::kFF));
      as = Floats(B * kTopK * gpu::kFF / 16);
    }
    group_list = static_cast<std::int32_t*>(Alloc(std::size_t(kNX) * B * 4));
    group_count = static_cast<std::int32_t*>(Alloc(std::size_t(kNX) * 4));
    if (const char* v = std::getenv("INKLING2_MAJOR_MIN_BATCH"))
      major_min_batch = std::atoi(v);
    low = Floats(B * 3 * kRank);
    router_logits = Floats(B * kNX);
    const std::size_t heads_cap = std::max<std::size_t>(opt.sessions, B);
    final_x = Floats(heads_cap * kE);
    logits = Floats(heads_cap * c.n_vocab_padded);
    host_logits.resize(heads_cap * c.n_vocab_padded);

    off_tokens = sizeof(gpu::RowState) * c.visits() * B;
    off_heads = off_tokens + 4 * B;
    off_identity = off_heads + 4 * heads_cap;
    off_groups = off_identity + 4 * B;
    host_staging.resize(off_groups + sizeof(gpu::AttnGroup) * B);
    for (std::size_t b = 0; b < B; ++b) {
      const std::int32_t index = static_cast<std::int32_t>(b);
      std::memcpy(host_staging.data() + off_identity + 4 * b, &index, 4);
    }
    if (opt.capture_features) {
      feat = Floats(3 * B * kE);
      feat_final = Floats(B * kE);
      host_feat.resize(3 * B * kE);
      host_feat_final.resize(B * kE);
    }
    staging = Alloc(host_staging.size());
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
    if (!opt.capture_features)
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
          double total = std::exp(-max);
          for (double& v : w) {
            v = std::exp(v - max);
            total += v;
          }
          HeadStats& st = out[h];
          st.sink = std::exp(-max) / total;
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
    line("positions holding all but 1e-3", &HeadStats::pos_999);
    line("positions holding all but 1e-4", &HeadStats::pos_9999);
    line("slices of 64 holding all but 1e-3", &HeadStats::slice_999);
    line("slices of 64 holding all but 1e-4", &HeadStats::slice_9999);
    line("slices of 64 holding all but 1e-5", &HeadStats::slice_99999);
    line("chunks of 1,024 holding all but 1e-4", &HeadStats::chunk_9999);
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
      gpu::CtrlDown(ctrl, stream);
      gpu::CtrlApply(ctrl, stream);
    }
    gpu::NormArgs norm{};
    norm.x = in;
    norm.stride = 3 * kE;
    for (int j = 0; j < 3; ++j)
      norm.w[j] = l.norm[j];
    norm.count = 3;
    norm.batch = batch;
    norm.eps = c.norm_eps;
    gpu::Norm(norm, stream);
    Mark(0);

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
    const bool batched =
        attn_batched_min > 0 && batch >= attn_batched_min * n_groups &&
        (!swa || attn_batched_window);
    const bool half = !batched && (attn_half || !keep_f32);
    if (batched) {
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
    if (swa && (batched || half))
      at.max_chunks = batched ? 2 : 1;
    gpu::Attention(at, stream);
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
      gpu::CtrlDown(ctrl, stream);
      gpu::CtrlApply(ctrl, stream);
    }
    gpu::NormArgs ffn_norm{};
    ffn_norm.x = xn;
    ffn_norm.stride = kE;
    ffn_norm.w[0] = l.ffn_norm;
    ffn_norm.count = 1;
    ffn_norm.batch = batch;
    ffn_norm.eps = c.norm_eps;
    gpu::Norm(ffn_norm, stream);
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
    gpu::Router(router, stream);
    Mark(5);

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
    e.contrib = contrib;
    e.xh = xh_x;
    e.acth = acth;
    e.max_batch = static_cast<int>(opt.max_batch);
    gpu::BatchExperts(e, stream);
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
    Check(hipMemsetAsync(session.hist, 0, kHistFloats * c.visits() * 4, stream),
          "clear");
    Check(hipStreamSynchronize(stream), "synchronize");
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
                         hist_candidates + b * c.visits() * kHistFloats,
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
      Check(hipStreamSynchronize(stream), "synchronize");
      session.pos += rows_kept;
    }
    session.pending_first = -1;
  }

  void Step(const std::vector<Row>& rows, Head head, bool tentative) {
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
                              (b * c.visits() + s) * kHistFloats +
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
    for (std::size_t i = 0; i < heads.size(); ++i)
      head_rows[i] = heads[i];
    // Groups for the batched attention kernel: runs cut into pieces of at
    // most kAttnRows rows.
    auto* groups =
        reinterpret_cast<gpu::AttnGroup*>(host_staging.data() + off_groups);
    n_groups = 0;
    for (std::size_t b = 0; b < rows.size();) {
      std::size_t end = b + 1;
      while (end < rows.size() && end - b < std::size_t(gpu::kAttnRows) &&
             rows[end].session == rows[b].session)
        ++end;
      groups[n_groups++] = {static_cast<std::int32_t>(b),
                            static_cast<std::int32_t>(end - b)};
      b = end;
    }
    Check(hipMemcpyAsync(staging, host_staging.data(), host_staging.size(),
                         hipMemcpyHostToDevice, stream),
          "copy step inputs");

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
    if (opt.capture_features) {
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
      Wmma(g, final_x, kE, xh_head);
      gpu::Gemm(g, stream);
      Check(hipMemcpyAsync(host_logits.data(), logits,
                           n_heads * c.n_vocab_padded * 4,
                           hipMemcpyDeviceToHost, stream),
            "copy logits");
    }
    if (!tentative)
      for (std::size_t b = 0; b < rows.size(); ++b)
        if (last[b])
          CommitHistory(sessions[rows[b].session], b);
    Check(hipStreamSynchronize(stream), "synchronize");
    Mark(10);
    n_launches += 6;

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

void GpuEngine::Step(const std::vector<Row>& rows, Head head, bool tentative) {
  impl_->Step(rows, head, tentative);
}

void GpuEngine::Accept(std::uint32_t session, std::uint32_t rows_kept) {
  impl_->Accept(session, rows_kept);
}

void GpuEngine::Reset(std::uint32_t session) {
  impl_->Reset(session);
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
  const float* raw = logits(i);
  std::vector<float> out(raw, raw + impl_->c.n_vocab);
  const float cap = impl_->c.logit_softcap;
  if (cap != 0.0f)
    for (float& x : out)
      x = cap * std::tanh(x / cap);
  return out;
}

std::uint32_t GpuEngine::argmax(std::size_t i) const {
  // The soft cap is monotonic, so the raw logits give the same argmax.
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
