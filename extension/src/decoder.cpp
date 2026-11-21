#include "decoder.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hip/decode.hpp"
#include "hip/ops.hpp"

namespace inkling2 {
namespace {

using gpu::kE;
using gpu::kHK;
using gpu::kHQ;
using gpu::kNC;
using gpu::kNH;
using gpu::kNR;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("inkling2 gpu: ") + what + ": " +
                             hipGetErrorString(status));
}

}  // namespace

struct GpuDecoder::Impl {
  // Device copies of one physical layer's weights.
  struct Layer {
    const std::uint16_t *wq, *wk, *wv, *wo, *gate, *sink, *rel, *kcp, *vcp;
    const std::uint8_t *gate_up, *down;
    bool nvfp4;
    const float *q_clip, *rel_bias, *rel_clip, *k_conv, *v_conv, *attn_post,
        *ffn_post, *norm[3], *res_query, *ffn_res_query, *ffn_norm, *gate_inp,
        *exp_probs_b, *up_scale, *down_scale, *msg_query;
    gpu::Ctrl ctrl[4];
  };
  // State of one layer visit.
  struct Slot {
    float *k, *v, *hist_k, *hist_v, *hist_attn, *hist_ffn;
    int cap;
  };

  const Model& m;
  const Config& c;
  hipStream_t stream{};
  std::vector<void*> owned;
  std::vector<Layer> layers;
  std::vector<Slot> slots;
  const std::uint16_t *tok_embd{}, *output{};
  const float *tok_norm{}, *output_norm{}, *output_res_query{};
  // Working vectors.
  float *cur{}, *input{}, *in{}, *msgs{}, *stack{}, *prelude{}, *bank0{},
      *bank_out{}, *q{}, *r{}, *gate{}, *sink{}, *k_raw{}, *v_raw{}, *k_taps{},
      *v_taps{}, *attn{}, *proj{}, *xn{}, *sel{}, *act{}, *moe{}, *scratch{},
      *final_x{}, *logits{}, *low{}, *router_logits{};
  std::uint32_t context;
  std::uint32_t pos{0};
  int n_blocks{0};
  bool debug{false};
  std::vector<double> sums;
  std::vector<float> host_logits, host_vec;
  std::size_t n_launches{0};

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
  const std::uint16_t* UpBf16(const Mat& mat) {
    if (mat.type != kGgmlTypeBf16)
      throw std::runtime_error("inkling2 gpu: projection is not BF16");
    return static_cast<const std::uint16_t*>(
        Upload(mat.data, mat.rows * mat.RowBytes()));
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

  Impl(const Model& model, std::uint32_t ctx)
      : m(model), c(model.c), context(ctx) {
    const bool shape_ok =
        c.n_embd == kE && c.head_dim == gpu::kHD && c.n_head == kNH &&
        c.n_head_kv == gpu::kNKV && c.n_rel == gpu::kNRel && c.n_conv == kNC &&
        c.n_expert == gpu::kNX && c.n_expert_used == gpu::kTopK &&
        c.loop_gate_rank == gpu::kRank && c.n_layer == 24 &&
        c.loop_start == 6 && c.loop_end == 18 && c.res_block == 6 &&
        c.n_swa == 1024 && c.n_rel_extent == 1024;
    if (!shape_ok)
      throw std::runtime_error(
          "inkling2 gpu: the decoder kernels are built for Inkling 2 Mini's "
          "dimensions; this file differs");
    Check(hipStreamCreate(&stream), "create stream");

    tok_embd = UpBf16(m.tok_embd);
    output = UpBf16(m.output);
    tok_norm = Up(m.tok_norm);
    output_norm = Up(m.output_norm);
    output_res_query = Up(m.output_res_query);

    layers.resize(c.n_layer);
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
      const LayerWeights& w = m.layers[il];
      Layer& l = layers[il];
      l.wq = UpBf16(w.wq);
      l.wk = UpBf16(w.wk);
      l.wv = UpBf16(w.wv);
      l.wo = UpBf16(w.wo);
      l.gate = UpBf16(w.gate);
      l.sink = UpBf16(w.sink);
      l.rel = UpBf16(w.rel);
      l.kcp = w.swa ? nullptr : UpBf16(w.k_conv_proj);
      l.vcp = w.swa ? nullptr : UpBf16(w.v_conv_proj);
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
      if (il >= c.loop_start && il < c.loop_end)
        for (int k = 0; k < 4; ++k)
          l.ctrl[k] = {UpF32(w.ctrl_down[k]), UpF32(w.ctrl_gain[k]),
                       UpF32(w.ctrl_forget[k])};
    }

    slots.resize(c.visits());
    for (std::uint32_t s = 0; s < c.visits(); ++s) {
      Slot& slot = slots[s];
      slot.cap = static_cast<int>(c.is_swa[Physical(s)] ? c.n_swa : context);
      slot.k = Floats(std::size_t(slot.cap) * kHK, true);
      slot.v = Floats(std::size_t(slot.cap) * kHK, true);
      slot.hist_k = Floats((kNC - 1) * kHK, true);
      slot.hist_v = Floats((kNC - 1) * kHK, true);
      slot.hist_attn = Floats((kNC - 1) * kE, true);
      slot.hist_ffn = Floats((kNC - 1) * kE, true);
    }

    cur = Floats(kE);
    input = Floats(kE);
    in = Floats(3 * kE);
    msgs = Floats(4 * kE);
    stack = Floats(gpu::kMaxStates * kE);
    prelude = Floats(kE);
    bank0 = Floats(kE);
    bank_out = Floats(kE);
    q = Floats(kHQ);
    r = Floats(kNR);
    gate = Floats(kHQ);
    sink = Floats(kNH);
    k_raw = Floats(kHK);
    v_raw = Floats(kHK);
    k_taps = Floats(kNC * gpu::kNKV);
    v_taps = Floats(kNC * gpu::kNKV);
    attn = Floats(kHQ);
    proj = Floats(kE);
    xn = Floats(kE);
    sel = Floats(3 * gpu::kTopK);
    act = Floats(gpu::kTopK * c.n_ff_exp);
    moe = Floats(kE);
    scratch = Floats(std::size_t(kNH) * std::max<std::uint32_t>(context, c.n_swa));
    final_x = Floats(kE);
    low = Floats(3 * gpu::kRank);
    router_logits = Floats(gpu::kNX);
    logits = Floats(c.n_vocab_padded);
    host_logits.resize(c.n_vocab_padded);
    host_vec.resize(kE);
    sums.assign(c.visits(), 0.0);
  }

  ~Impl() {
    if (stream)
      (void)hipStreamSynchronize(stream);
    for (void* p : owned)
      (void)hipFree(p);
    if (stream)
      (void)hipStreamDestroy(stream);
  }

  // The physical layer a visit slot runs.
  [[nodiscard]] std::uint32_t Physical(std::uint32_t slot) const {
    const std::uint32_t core = c.loop_end - c.loop_start;
    return slot < c.n_layer ? slot : c.loop_start + (slot - c.n_layer) % core;
  }

  // Profiling: with `profile` on, every kernel group is followed by a round
  // trip and its wall time is added to `phase_seconds`. That adds overhead of
  // its own, so the phases show proportions, not the unprofiled cost.
  static constexpr int kPhases = 11;
  static constexpr const char* kPhaseNames[kPhases] = {
      "glue A (mix, messages, norms)", "input projections",
      "glue B (conv, clips, KV store)", "attention", "output projection",
      "glue C (conv, mix, norm, router)", "experts NVFP4", "experts BF16",
      "glue D (conv, residual)", "embed and copies", "output head"};
  bool profile{false};
  double phase_seconds[kPhases] = {};
  std::chrono::steady_clock::time_point mark{};

  void Mark(int phase) {
    if (!profile)
      return;
    Check(hipStreamSynchronize(stream), "synchronize");
    const auto now = std::chrono::steady_clock::now();
    if (phase >= 0)
      phase_seconds[phase] +=
          std::chrono::duration<double>(now - mark).count();
    mark = now;
  }

  void Visit(std::uint32_t slot_index, std::uint32_t il, bool looped) {
    const Layer& l = layers[il];
    Slot& slot = slots[slot_index];
    const bool swa = c.is_swa[il];
    const bool block_start = il % c.res_block == 0;
    if (!swa && pos >= static_cast<std::uint32_t>(slot.cap))
      throw std::runtime_error("inkling2 gpu: context is full");

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
    gpu::GlueA(a, stream);
    if (looped) {
      gpu::CtrlDownArgs down{};
      gpu::CtrlApplyArgs apply{};
      for (int j = 0; j < 3; ++j) {
        down.cur[j] = apply.cur[j] = in + j * kE;
        down.msg[j] = apply.msg[j] = msgs + j * kE;
        down.down[j] = l.ctrl[j].down;
        apply.gain[j] = l.ctrl[j].gain;
        apply.forget[j] = l.ctrl[j].forget;
      }
      down.low = low;
      apply.low = low;
      down.count = apply.count = 3;
      gpu::CtrlDown(down, stream);
      gpu::CtrlApply(apply, stream);
    }
    gpu::NormArgs norm{};
    for (int j = 0; j < 3; ++j) {
      norm.x[j] = in + j * kE;
      norm.w[j] = l.norm[j];
    }
    norm.count = 3;
    norm.eps = c.norm_eps;
    gpu::Norm(norm, stream);
    Mark(0);

    gpu::GemvArgs g{};
    g.jobs[0] = {l.wq, in, q, kHQ, kE};
    g.jobs[1] = {l.rel, in, r, kNR, kE};
    g.jobs[2] = {l.gate, in, gate, kHQ, kE};
    g.jobs[3] = {l.sink, in, sink, kNH, kE};
    g.jobs[4] = {l.wk, in + kE, k_raw, kHK, kE};
    g.jobs[5] = {l.wv, in + 2 * kE, v_raw, kHK, kE};
    g.count = 6;
    if (!swa) {
      g.jobs[6] = {l.kcp, in + kE, k_taps, kNC * gpu::kNKV, kE};
      g.jobs[7] = {l.vcp, in + 2 * kE, v_taps, kNC * gpu::kNKV, kE};
      g.count = 8;
    }
    gpu::GemvMulti(g, stream);
    Mark(1);

    const float ratio = std::max(
        1.0f, (pos + 1.0f) / static_cast<float>(c.log_scale_floor));
    gpu::GlueBArgs b{};
    b.k_raw = k_raw;
    b.v_raw = v_raw;
    b.q = q;
    b.r = r;
    b.k_taps = swa ? nullptr : k_taps;
    b.v_taps = swa ? nullptr : v_taps;
    b.k_conv = l.k_conv;
    b.v_conv = l.v_conv;
    b.hist_k = slot.hist_k;
    b.hist_v = slot.hist_v;
    b.q_clip = l.q_clip;
    b.rel_clip = l.rel_clip;
    b.tau = swa ? 1.0f : 1.0f + c.log_scale_alpha * std::log(ratio);
    b.qk_clip = c.qk_clip;
    b.rel_clip_threshold = c.rel_clip;
    b.eps = c.norm_eps;
    const std::size_t row = pos % static_cast<std::uint32_t>(slot.cap);
    b.k_row = slot.k + row * kHK;
    b.v_row = slot.v + row * kHK;
    gpu::GlueB(b, stream);
    Mark(2);

    gpu::AttnArgs at{};
    at.q = q;
    at.r = r;
    at.sink = sink;
    at.gate = gate;
    at.rel_bias = l.rel_bias;
    at.kv_k = slot.k;
    at.kv_v = slot.v;
    at.scratch = scratch;
    at.out = attn;
    at.extent = static_cast<int>(swa ? c.n_swa : c.n_rel_extent);
    at.cap = slot.cap;
    at.first = swa && pos >= c.n_swa ? static_cast<int>(pos - c.n_swa + 1) : 0;
    at.pos = static_cast<int>(pos);
    at.scale = 1.0f / std::sqrt(static_cast<float>(gpu::kHD));
    gpu::Attention(at, stream);
    Mark(3);

    gpu::GemvArgs o{};
    o.jobs[0] = {l.wo, attn, proj, kE, kHQ};
    o.count = 1;
    gpu::GemvMulti(o, stream);
    Mark(4);

    gpu::GlueCArgs cc{};
    cc.proj = proj;
    cc.post_conv = l.attn_post;
    cc.hist = slot.hist_attn;
    cc.input = input;
    cc.block_start = block_start;
    cc.stack = stack;
    cc.n_blocks = n_blocks;
    cc.cur = cur;
    cc.xn = xn;
    cc.ffn_res_query = l.ffn_res_query;
    gpu::GlueC(cc, stream);
    if (looped) {
      gpu::CtrlDownArgs down{};
      gpu::CtrlApplyArgs apply{};
      down.cur[0] = apply.cur[0] = xn;
      down.msg[0] = apply.msg[0] = msgs + 3 * kE;
      down.down[0] = l.ctrl[3].down;
      apply.gain[0] = l.ctrl[3].gain;
      apply.forget[0] = l.ctrl[3].forget;
      down.low = low;
      apply.low = low;
      down.count = apply.count = 1;
      gpu::CtrlDown(down, stream);
      gpu::CtrlApply(apply, stream);
    }
    gpu::NormArgs ffn_norm{};
    ffn_norm.x[0] = xn;
    ffn_norm.w[0] = l.ffn_norm;
    ffn_norm.count = 1;
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
    router.ids = reinterpret_cast<std::uint32_t*>(sel);
    router.up_scale = sel + gpu::kTopK;
    router.coef = sel + 2 * gpu::kTopK;
    gpu::Router(router, stream);
    Mark(5);
    if (block_start)
      ++n_blocks;

    gpu::ExpertArgs e{};
    e.gate_up = l.gate_up;
    e.down = l.down;
    e.x = xn;
    e.ids = router.ids;
    e.up_scale = router.up_scale;
    e.coef = router.coef;
    e.act = act;
    e.out = moe;
    e.count = gpu::kTopK;
    e.ff = static_cast<int>(c.n_ff_exp);
    e.n_embd = kE;
    e.situ_beta = c.situ_beta;
    e.situ_linear_beta = c.situ_linear_beta;
    e.nvfp4 = l.nvfp4;
    gpu::Experts(e, stream);
    Mark(l.nvfp4 ? 6 : 7);

    gpu::GlueDArgs d{};
    d.moe = moe;
    d.post_conv = l.ffn_post;
    d.hist = slot.hist_ffn;
    d.cur = cur;
    gpu::GlueD(d, stream);
    Mark(8);
    n_launches += looped ? 17 : 13;

    if (debug) {
      Check(hipMemcpyAsync(host_vec.data(), cur, kE * 4, hipMemcpyDeviceToHost,
                           stream),
            "copy layer output");
      Check(hipStreamSynchronize(stream), "synchronize");
      double sum = 0;
      for (const float x : host_vec)
        sum += x;
      sums[slot_index] += sum;
    }
  }

  // Queues one token's layers; `head` adds the output projection.
  void Run(std::uint32_t token, bool head) {
    if (token >= c.n_vocab)
      throw std::runtime_error("inkling2: token id out of range");
    gpu::EmbedArgs e{};
    e.row = tok_embd + std::size_t{token} * kE;
    e.norm = tok_norm;
    e.eps = c.norm_eps;
    e.cur = cur;
    Mark(-1);
    gpu::Embed(e, stream);

    n_blocks = 0;
    for (std::uint32_t il = 0; il < c.loop_start; ++il)
      Visit(il, il, false);
    // Every pass starts from the prelude output.
    const int prelude_blocks = n_blocks;
    gpu::CopyArgs keep{};
    keep.dst[0] = prelude;
    keep.src[0] = cur;
    keep.count = 1;
    gpu::Copy(keep, stream);
    const std::uint32_t core = c.loop_end - c.loop_start;
    for (std::uint32_t pass = 0; pass < c.loop_count; ++pass) {
      if (pass > 0) {
        // The bank of the previous pass: its block after the first core block,
        // and its output. Then restart from the prelude.
        gpu::CopyArgs bank{};
        bank.dst[0] = bank0;
        bank.src[0] = stack + std::size_t(prelude_blocks + 1) * kE;
        bank.dst[1] = bank_out;
        bank.src[1] = cur;
        bank.dst[2] = cur;
        bank.src[2] = prelude;
        bank.count = 3;
        gpu::Copy(bank, stream);
        n_blocks = prelude_blocks;
      }
      for (std::uint32_t ip = c.loop_start; ip < c.loop_end; ++ip) {
        const std::uint32_t slot =
            pass == 0 ? ip : c.n_layer + (pass - 1) * core + (ip - c.loop_start);
        Visit(slot, ip, pass > 0);
      }
    }
    for (std::uint32_t il = c.loop_end; il < c.n_layer; ++il)
      Visit(il, il, false);

    if (head) {
      gpu::FinalArgs f{};
      f.stack = stack;
      f.n_blocks = n_blocks;
      f.cur = cur;
      f.res_query = output_res_query;
      f.norm = output_norm;
      f.eps = c.norm_eps;
      f.out = final_x;
      Mark(9);
      gpu::Final(f, stream);
      gpu::GemvArgs g{};
      g.jobs[0] = {output, final_x, logits, static_cast<int>(c.n_vocab_padded),
                   kE};
      g.count = 1;
      gpu::GemvMulti(g, stream);
      Check(hipMemcpyAsync(host_logits.data(), logits,
                           std::size_t{c.n_vocab} * 4, hipMemcpyDeviceToHost,
                           stream),
            "copy logits");
    }
    Check(hipStreamSynchronize(stream), "synchronize");
    Mark(10);
    n_launches += 6;
    ++pos;
  }
};

GpuDecoder::GpuDecoder(const Model& model, std::uint32_t context)
    : impl_(std::make_unique<Impl>(model, context)) {}
GpuDecoder::~GpuDecoder() = default;

std::vector<float> GpuDecoder::Feed(std::uint32_t token) {
  impl_->Run(token, true);
  std::vector<float> out(impl_->host_logits.begin(),
                         impl_->host_logits.begin() + impl_->c.n_vocab);
  const float cap = impl_->c.logit_softcap;
  if (cap != 0.0f)
    for (float& x : out)
      x = cap * std::tanh(x / cap);
  return out;
}

std::uint32_t GpuDecoder::FeedGreedy(std::uint32_t token) {
  impl_->Run(token, true);
  // The soft cap is monotonic, so the raw logits give the same argmax.
  const auto begin = impl_->host_logits.begin();
  return static_cast<std::uint32_t>(
      std::max_element(begin, begin + impl_->c.n_vocab) - begin);
}

void GpuDecoder::FeedSilent(std::uint32_t token) {
  impl_->Run(token, false);
}

void GpuDecoder::SetProfile(bool on) {
  impl_->profile = on;
}

std::vector<std::pair<std::string, double>> GpuDecoder::phases() const {
  std::vector<std::pair<std::string, double>> out;
  for (int i = 0; i < Impl::kPhases; ++i)
    out.emplace_back(Impl::kPhaseNames[i], impl_->phase_seconds[i]);
  return out;
}

void GpuDecoder::SetDebug(bool on) {
  impl_->debug = on;
}
const std::vector<double>& GpuDecoder::layer_sums() const {
  return impl_->sums;
}
std::size_t GpuDecoder::launches() const {
  return impl_->n_launches;
}

}  // namespace inkling2
