#include "reference.hpp"

#include "backend.hpp"
#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace inkling2 {
namespace {

using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;

constexpr std::uint16_t kF32 = 0;
constexpr std::uint16_t kBF16 = 30;

// ---- threading ---------------------------------------------------------

void ParallelFor(std::size_t n, std::size_t work_per_item,
                 const std::function<void(std::size_t, std::size_t)>& fn) {
  const std::size_t hw = std::max(1u, std::thread::hardware_concurrency());
  const std::size_t threads =
      n * work_per_item < (1u << 20) ? 1 : std::min<std::size_t>({hw, 16, n});
  if (threads <= 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  const std::size_t step = (n + threads - 1) / threads;
  for (std::size_t begin = 0; begin < n; begin += step)
    pool.emplace_back(fn, begin, std::min(n, begin + step));
  for (auto& t : pool)
    t.join();
}

// ---- storage decoding --------------------------------------------------

inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = std::uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &wide, sizeof value);
  return value;
}

// NVFP4 as stored by ggml (block_nvfp4): 64 values per 36-byte block, in four
// sub-blocks of 16 that each carry an unsigned E4M3 scale. The E2M1 value
// table is doubled, so the scale decode halves.
constexpr int kNvfp4Block = kNvfp4Values;
constexpr float kFp4Values[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                  0, -1, -2, -3, -4, -6, -8, -12};

inline float Ue4m3(std::uint8_t x) {
  if (x == 0 || x == 0x7F)
    return 0.0f;
  const int exp = (x >> 3) & 0xF;
  const int man = x & 0x7;
  const float raw = exp == 0 ? std::ldexp(static_cast<float>(man), -9)
                             : std::ldexp(1.0f + man / 8.0f, exp - 7);
  return raw * 0.5f;
}

float DotNvfp4(const std::uint8_t* row, const float* x, std::size_t cols) {
  float sum = 0.0f;
  for (std::size_t b = 0; b < cols / kNvfp4Block; ++b) {
    const std::uint8_t* block = row + b * kNvfp4Bytes;
    const std::uint8_t* qs = block + 4;
    const float* xb = x + b * kNvfp4Block;
    for (int s = 0; s < 4; ++s) {
      const float d = Ue4m3(block[s]);
      float part = 0.0f;
      for (int j = 0; j < 8; ++j) {
        const std::uint8_t q = qs[s * 8 + j];
        part += kFp4Values[q & 0x0F] * xb[s * 16 + j] +
                kFp4Values[q >> 4] * xb[s * 16 + j + 8];
      }
      sum += part * d;
    }
  }
  return sum;
}

// ---- tensors -----------------------------------------------------------

float Dot(const Mat& m, const std::uint8_t* row, const float* x) {
  if (m.type == kGgmlTypeNvfp4)
    return DotNvfp4(row, x, m.cols);
  // Sixteen independent partial sums, so the compiler can vectorize without
  // being allowed to reorder a single running sum.
  constexpr std::size_t kLanes = 16;
  float lanes[kLanes] = {};
  const std::size_t whole = m.cols / kLanes * kLanes;
  float sum = 0.0f;
  if (m.type == kF32) {
    const float* w = reinterpret_cast<const float*>(row);
    for (std::size_t c = 0; c < whole; c += kLanes)
      for (std::size_t j = 0; j < kLanes; ++j)
        lanes[j] += w[c + j] * x[c + j];
    for (std::size_t c = whole; c < m.cols; ++c)
      sum += w[c] * x[c];
  } else {
    const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(row);
    for (std::size_t c = 0; c < whole; c += kLanes)
      for (std::size_t j = 0; j < kLanes; ++j)
        lanes[j] += Bf16(w[c + j]) * x[c + j];
    for (std::size_t c = whole; c < m.cols; ++c)
      sum += Bf16(w[c]) * x[c];
  }
  for (const float lane : lanes)
    sum += lane;
  return sum;
}

/// y = W[expert] x on the CPU.
void MulCpu(const Mat& m, const float* x, float* y, std::size_t expert = 0) {
  const std::size_t row_bytes = m.RowBytes();
  const std::uint8_t* base = m.data + expert * m.rows * row_bytes;
  ParallelFor(m.rows, m.cols, [&](std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r)
      y[r] = Dot(m, base + r * row_bytes, x);
  });
}

Vec MulCpu(const Mat& m, const Vec& x) {
  Vec y(m.rows);
  MulCpu(m, x.data(), y.data());
  return y;
}

class CpuBackend final : public Backend {
public:
  void Mul(const Job* jobs, std::size_t count) override {
    for (std::size_t i = 0; i < count; ++i)
      MulCpu(*jobs[i].mat, jobs[i].x, jobs[i].y);
  }

  void Experts(const ExpertCall& call) override {
    const std::size_t ff = call.down->cols, e = call.down->rows;
    Vec gate_up(2 * ff), act(ff), down(e);
    std::fill(call.out, call.out + e, 0.0f);
    for (std::size_t i = 0; i < call.count; ++i) {
      MulCpu(*call.gate_up, call.x, gate_up.data(), call.ids[i]);
      for (std::size_t k = 0; k < ff; ++k) {
        const float g = gate_up[k] * call.up_scale[i];
        const float u = gate_up[ff + k] * call.up_scale[i];
        // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
        act[k] = call.situ_beta * std::tanh(g / call.situ_beta) *
                 (1.0f / (1.0f + std::exp(-g))) * call.situ_linear_beta *
                 std::tanh(u / call.situ_linear_beta);
      }
      MulCpu(*call.down, act.data(), down.data(), call.ids[i]);
      for (std::size_t k = 0; k < e; ++k)
        call.out[k] += call.coef[i] * down[k];
    }
  }

  [[nodiscard]] const char* name() const override { return "cpu"; }
};

// ---- small math --------------------------------------------------------

inline float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

double Sum(const Vec& v) {
  double s = 0;
  for (const float x : v)
    s += x;
  return s;
}

// ggml_rms_norm: x / sqrt(mean(x^2) + eps), over `n` values in place.
void RmsNorm(float* x, std::size_t n, float eps) {
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(x[i]) * x[i];
  const float scale = 1.0f / std::sqrt(static_cast<float>(sum / n) + eps);
  for (std::size_t i = 0; i < n; ++i)
    x[i] *= scale;
}

Vec Normed(Vec x, const Vec& weight, float eps) {
  RmsNorm(x.data(), x.size(), eps);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] *= weight[i];
  return x;
}

// Scale each group of `width` values (optionally by `scale` first) down to an
// L2 norm of at most `threshold`.
void Clip(Vec& x, std::size_t width, const float* scale, float threshold) {
  for (std::size_t g = 0; g < x.size() / width; ++g) {
    float* v = x.data() + g * width;
    float sq = 0.0f;
    for (std::size_t i = 0; i < width; ++i) {
      if (scale)
        v[i] *= scale[g * width + i];
      sq += v[i] * v[i];
    }
    const float norm = std::sqrt(sq) + 1e-6f;
    const float factor = std::clamp(threshold / norm, 0.0f, 1.0f);
    for (std::size_t i = 0; i < width; ++i)
      v[i] *= factor;
  }
}

// Attention over the depth: softmax-weighted mix of the normalized earlier
// block outputs and `cur`, one output per learned query.
std::vector<Vec> Mix(const std::vector<Vec>& blocks, const Vec& cur,
                     const Vec& queries) {
  const std::size_t e = cur.size();
  std::vector<Vec> states(blocks);
  states.push_back(cur);
  for (Vec& s : states)
    RmsNorm(s.data(), e, 1e-6f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(e));
  std::vector<Vec> out;
  for (std::size_t q = 0; q < queries.size() / e; ++q) {
    Vec score(states.size());
    float max = -INFINITY;
    for (std::size_t s = 0; s < states.size(); ++s) {
      float dot = 0.0f;
      for (std::size_t i = 0; i < e; ++i)
        dot += states[s][i] * queries[q * e + i];
      score[s] = dot * scale;
      max = std::max(max, score[s]);
    }
    float sum = 0.0f;
    for (float& v : score) {
      v = std::exp(v - max);
      sum += v;
    }
    Vec mixed(e, 0.0f);
    for (std::size_t s = 0; s < states.size(); ++s) {
      const float p = score[s] / sum;
      for (std::size_t i = 0; i < e; ++i)
        mixed[i] += p * states[s][i];
    }
    out.push_back(std::move(mixed));
  }
  return out;
}

// ---- model -------------------------------------------------------------

using Layer = LayerWeights;

/// State of one layer visit: its own KV rows and convolution history.
struct Slot {
  std::vector<Vec> keys, values;  // one [n_head_kv * head_dim] row per position
  std::vector<Vec> history[4];    // last n_conv - 1 inputs of K, V, attn, ffn
};

}  // namespace

void CpuMul(const Mat& m, const float* x, float* y) {
  MulCpu(m, x, y);
}

struct Reference::Impl {
  Config c;
  Mat tok_embd, output;
  Vec tok_norm, output_norm, output_res_query;
  std::vector<Layer> layers;
  std::vector<Slot> slots;
  Trace trace;
  std::uint32_t pos{0};
  CpuBackend cpu;
  Backend* backend{&cpu};
  mutable bool debug_slot{false};

  // Sums of layer-0 intermediates over all tokens, for comparison with
  // llama.cpp's tensor dump.
  void Dbg(const char* name, const Vec& v) const {
    if (!debug_slot)
      return;
    auto& d = const_cast<Trace&>(trace).debug;
    for (auto& entry : d)
      if (entry.first == name) {
        entry.second += Sum(v);
        return;
      }
    d.emplace_back(name, Sum(v));
  }

  Impl(const GgufReader& reader, const Config& config, Backend* gpu)
      : c(config) {
    if (gpu)
      backend = gpu;
    Model model = LoadModel(reader, config);
    tok_embd = model.tok_embd;
    output = model.output;
    tok_norm = std::move(model.tok_norm);
    output_norm = std::move(model.output_norm);
    output_res_query = std::move(model.output_res_query);
    layers = std::move(model.layers);
    slots.resize(c.visits());
    trace.attn_out.assign(c.visits(), 0);
    trace.ffn_out.assign(c.visits(), 0);
    trace.l_out.assign(c.visits(), 0);
  }

  // Causal short convolution with a residual. `taps` adds input-dependent
  // taps per head. The last n_conv - 1 inputs are kept in `history`.
  Vec Convolve(const Vec& cur, const Vec& weight, const Vec* taps,
               std::vector<Vec>& history) const {
    const std::size_t n_conv = c.n_conv, ch = cur.size();
    if (history.empty())
      history.assign(n_conv - 1, Vec(ch, 0.0f));
    // window[i] for i in [0, n_conv): oldest first, the current input last.
    auto window = [&](std::size_t i) -> const Vec& {
      return i + 1 == n_conv ? cur : history[i];
    };
    Vec out(ch, 0.0f);
    for (std::size_t i = 0; i < n_conv; ++i) {
      const Vec& x = window(i);
      for (std::size_t k = 0; k < ch; ++k)
        out[k] += x[k] * weight[k * n_conv + i];
    }
    if (taps) {
      const std::size_t heads = taps->size() / n_conv, per_head = ch / heads;
      for (std::size_t lag = 0; lag < n_conv; ++lag) {
        const Vec& x = window(n_conv - 1 - lag);
        for (std::size_t k = 0; k < ch; ++k)
          out[k] += x[k] * (*taps)[lag + n_conv * (k / per_head)];
      }
    }
    const float scale =
        1.0f / std::sqrt(static_cast<float>((taps ? 2 : 1) * n_conv + 1));
    for (std::size_t k = 0; k < ch; ++k)
      out[k] = (cur[k] + out[k]) * scale;
    history.erase(history.begin());
    history.push_back(cur);
    return out;
  }

  // (1 - forget) * cur + gain * message, both gated on cur and message.
  Vec Control(const Vec& cur, const Vec& message, const Layer& l, int k) const {
    const std::size_t e = cur.size();
    Vec gate(2 * e);
    std::copy(cur.begin(), cur.end(), gate.begin());
    std::copy(message.begin(), message.end(), gate.begin() + e);
    RmsNorm(gate.data(), e, 1e-6f);
    RmsNorm(gate.data() + e, e, 1e-6f);
    const Vec low = MulCpu(l.ctrl_down[k], gate);
    const Vec gain = MulCpu(l.ctrl_gain[k], low);
    const Vec forget = MulCpu(l.ctrl_forget[k], low);
    Vec out(e);
    for (std::size_t i = 0; i < e; ++i)
      out[i] = cur[i] - std::tanh(forget[i]) * cur[i] +
               std::tanh(gain[i]) * message[i];
    return out;
  }

  Vec Moe(const Vec& x, const Layer& l) const {
    const std::size_t e = c.n_embd;
    Dbg("ffn_normed", x);
    Vec probs = MulCpu(l.gate_inp, x);
    Dbg("moe_logits", probs);
    for (float& p : probs)
      p = Sigmoid(p);
    std::vector<std::size_t> order(c.n_expert);
    std::iota(order.begin(), order.end(), 0);
    auto key = [&](std::size_t i) {
      return probs[i] + (l.exp_probs_b.empty() ? 0.0f : l.exp_probs_b[i]);
    };
    std::partial_sort(
        order.begin(), order.begin() + c.n_expert_used, order.end(),
        [&](std::size_t a, std::size_t b) { return key(a) > key(b); });
    float weight_sum = 0.0f;
    for (std::uint32_t i = 0; i < c.n_expert_used; ++i)
      weight_sum += probs[order[i]];
    weight_sum += 1e-20f;

    Vec out(e, 0.0f);
    Backend::ExpertCall call{};
    call.gate_up = &l.gate_up;
    call.down = &l.down;
    call.x = x.data();
    call.count = c.n_expert_used;
    call.situ_beta = c.situ_beta;
    call.situ_linear_beta = c.situ_linear_beta;
    call.out = out.data();
    for (std::uint32_t i = 0; i < c.n_expert_used; ++i) {
      const std::size_t expert = order[i];
      call.ids[i] = static_cast<std::uint32_t>(expert);
      call.up_scale[i] =
          l.gate_up_scale.empty() ? 1.0f : l.gate_up_scale[expert];
      call.coef[i] = probs[expert] / weight_sum * c.expert_weights_scale *
                     (l.down_scale.empty() ? 1.0f : l.down_scale[expert]);
    }
    backend->Experts(call);
    Dbg("moe_out", out);
    return out;
  }

  Vec Attention(const Vec in[3], const Layer& l, Slot& slot) const {
    const std::size_t hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    Dbg("attn_in_q", in[0]);
    Vec q(l.wq.rows), r(l.rel.rows), gate(l.gate.rows), sink(l.sink.rows);
    Vec k_raw(l.wk.rows), v_raw(l.wv.rows), k_taps, v_taps;
    Backend::Job jobs[8] = {
        {&l.wq, in[0].data(), q.data()},
        {&l.rel, in[0].data(), r.data()},
        {&l.gate, in[0].data(), gate.data()},
        {&l.sink, in[0].data(), sink.data()},
        {&l.wk, in[1].data(), k_raw.data()},
        {&l.wv, in[2].data(), v_raw.data()},
    };
    std::size_t n_jobs = 6;
    if (!l.swa) {
      k_taps.resize(l.k_conv_proj.rows);
      v_taps.resize(l.v_conv_proj.rows);
      jobs[n_jobs++] = {&l.k_conv_proj, in[1].data(), k_taps.data()};
      jobs[n_jobs++] = {&l.v_conv_proj, in[2].data(), v_taps.data()};
    }
    backend->Mul(jobs, n_jobs);
    Dbg("q_proj", q);
    Dbg("r_proj", r);
    Vec k = Convolve(k_raw, l.k_conv, l.swa ? nullptr : &k_taps,
                     slot.history[0]);
    Vec v = Convolve(v_raw, l.v_conv, l.swa ? nullptr : &v_taps,
                     slot.history[1]);

    Clip(q, hd, l.q_clip.data(), c.qk_clip);
    Clip(k, hd, nullptr, c.qk_clip);
    for (std::size_t h = 0; h < nkv; ++h)
      RmsNorm(v.data() + h * hd, hd, c.norm_eps);
    Clip(r, c.n_rel, l.rel_clip.data(), c.rel_clip);

    if (!l.swa) {
      // Log-scaled positions of the full-attention layers.
      const float ratio =
          std::max(1.0f, (pos + 1.0f) / static_cast<float>(c.log_scale_floor));
      const float tau = 1.0f + c.log_scale_alpha * std::log(ratio);
      for (float& x : q)
        x *= tau;
      for (float& x : r)
        x *= tau;
    }
    Dbg("Qcur", q);
    Dbg("Kcur", k);
    Dbg("Vcur", v);
    Dbg("Rcur", r);
    Dbg("sink", sink);

    slot.keys.push_back(k);
    slot.values.push_back(v);

    const std::size_t extent = l.swa ? c.n_swa : c.n_rel_extent;
    const std::size_t n_kv = slot.keys.size();
    const std::size_t first = l.swa && n_kv > c.n_swa ? n_kv - c.n_swa : 0;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    Vec out(nh * hd, 0.0f);
    ParallelFor(nh, (n_kv - first) * hd * 64,
                [&](std::size_t begin, std::size_t end) {
      Vec logit(n_kv - first);
      for (std::size_t h = begin; h < end; ++h) {
        const std::size_t kv_head = h / (nh / nkv);
        const float* qh = q.data() + h * hd;
        const float* rh = r.data() + h * c.n_rel;
        float max = 0.0f;  // the sink's logit
        for (std::size_t j = first; j < n_kv; ++j) {
          const float* kj = slot.keys[j].data() + kv_head * hd;
          float dot = 0.0f;
          for (std::size_t i = 0; i < hd; ++i)
            dot += qh[i] * kj[i];
          const std::size_t distance = n_kv - 1 - j;
          float bias = 0.0f;
          if (distance < extent)
            for (std::size_t x = 0; x < c.n_rel; ++x)
              bias += l.rel_bias[distance * c.n_rel + x] * rh[x];
          logit[j - first] = dot * scale + bias - sink[h];
          max = std::max(max, logit[j - first]);
        }
        // softmax with one extra zero-logit sink that absorbs mass
        float sum = std::exp(0.0f - max);
        for (float& x : logit) {
          x = std::exp(x - max);
          sum += x;
        }
        float* oh = out.data() + h * hd;
        for (std::size_t j = first; j < n_kv; ++j) {
          const float p = logit[j - first] / sum;
          const float* vj = slot.values[j].data() + kv_head * hd;
          for (std::size_t i = 0; i < hd; ++i)
            oh[i] += p * vj[i];
        }
      }
    });

    Dbg("kqv_out", out);
    Dbg("gate_logits", gate);
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] *= Sigmoid(gate[i]);
    Dbg("gated", out);
    Vec projected(l.wo.rows);
    const Backend::Job job{&l.wo, out.data(), projected.data()};
    backend->Mul(&job, 1);
    Dbg("wo", projected);
    return Convolve(projected, l.attn_post_conv, nullptr, slot.history[2]);
  }

  Vec RunLayer(std::size_t slot_index, std::size_t il, Vec cur,
               std::vector<Vec>& blocks, const std::vector<Vec>* messages) {
    const Layer& l = layers[il];
    Slot& slot = slots[slot_index];
    debug_slot = slot_index == 0;
    const Vec input = cur;
    Dbg("layer_in", input);

    Vec in[3] = {cur, cur, cur};
    if (!l.attn_res_query.empty()) {
      const std::vector<Vec> mixed = Mix(blocks, cur, l.attn_res_query);
      for (int j = 0; j < 3; ++j)
        in[j] = mixed[l.swa ? 0 : j];
    }
    if (messages)
      for (int j = 0; j < 3; ++j)
        in[j] = Control(in[j], (*messages)[j], l, j);
    if (l.swa) {
      for (Vec& v : in)
        v = Normed(std::move(v), l.attn_norm, c.norm_eps);
    } else {
      in[0] = Normed(std::move(in[0]), l.q_norm, c.norm_eps);
      in[1] = Normed(std::move(in[1]), l.k_norm, c.norm_eps);
      in[2] = Normed(std::move(in[2]), l.v_norm, c.norm_eps);
    }

    cur = Attention(in, l, slot);
    trace.attn_out[slot_index] += Sum(cur);

    // The first layer of each block starts a new residual stream; the earlier
    // one joins the mixed blocks.
    if (il % c.res_block == 0) {
      blocks.push_back(input);
    } else {
      for (std::size_t i = 0; i < cur.size(); ++i)
        cur[i] += input[i];
    }
    const Vec ffn_input = cur;

    Vec x = Mix(blocks, cur, l.ffn_res_query)[0];
    Dbg("ffn_mix", x);
    if (messages)
      x = Control(x, (*messages)[3], l, 3);
    x = Normed(std::move(x), l.ffn_norm, c.norm_eps);
    Vec ffn = Convolve(Moe(x, l), l.ffn_post_conv, nullptr, slot.history[3]);
    trace.ffn_out[slot_index] += Sum(ffn);

    for (std::size_t i = 0; i < ffn.size(); ++i)
      ffn[i] += ffn_input[i];
    trace.l_out[slot_index] += Sum(ffn);
    return ffn;
  }

  std::vector<float> Feed(std::uint32_t token, bool want_logits) {
    if (token >= c.n_vocab)
      throw std::runtime_error("inkling2: token id out of range");
    const std::size_t e = c.n_embd;
    Vec cur(e);
    const auto* row = reinterpret_cast<const std::uint16_t*>(
        tok_embd.data + std::size_t{token} * tok_embd.RowBytes());
    if (tok_embd.type != kBF16)
      throw std::runtime_error("inkling2: token_embd must be BF16");
    for (std::size_t i = 0; i < e; ++i)
      cur[i] = Bf16(row[i]);
    cur = Normed(std::move(cur), tok_norm, c.norm_eps);

    std::vector<Vec> blocks;
    for (std::size_t il = 0; il < c.loop_start; ++il)
      cur = RunLayer(il, il, std::move(cur), blocks, nullptr);

    // Every pass starts from the prelude output.
    const Vec prelude = cur;
    const std::vector<Vec> prelude_blocks = blocks;
    const std::size_t core = c.loop_end - c.loop_start;
    for (std::uint32_t pass = 0; pass < c.loop_count; ++pass) {
      // The bank of the previous pass: its blocks after the first core block,
      // and its output.
      std::vector<Vec> bank;
      Vec bank_out;
      if (pass > 0) {
        bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
        bank_out = cur;
        cur = prelude;
        blocks = prelude_blocks;
      }
      for (std::size_t ip = c.loop_start; ip < c.loop_end; ++ip) {
        if (pass == 0) {
          cur = RunLayer(ip, ip, std::move(cur), blocks, nullptr);
          continue;
        }
        const std::size_t slot =
            c.n_layer + (pass - 1) * core + (ip - c.loop_start);
        const std::vector<Vec> messages =
            Mix(bank, bank_out, layers[ip].msg_query);
        cur = RunLayer(slot, ip, std::move(cur), blocks, &messages);
      }
    }
    for (std::size_t il = c.loop_end; il < c.n_layer; ++il)
      cur = RunLayer(il, il, std::move(cur), blocks, nullptr);

    cur = Mix(blocks, cur, output_res_query)[0];
    cur = Normed(std::move(cur), output_norm, c.norm_eps);
    trace.result_norm += Sum(cur);
    ++pos;
    if (!want_logits)
      return {};

    Vec logits(output.rows);
    const Backend::Job job{&output, cur.data(), logits.data()};
    backend->Mul(&job, 1);
    logits.resize(c.n_vocab);  // drop the padded vocabulary
    if (c.logit_softcap != 0.0f)
      for (float& x : logits)
        x = c.logit_softcap * std::tanh(x / c.logit_softcap);
    return logits;
  }
};

Reference::Reference(const GgufReader& reader, const Config& config,
                     Backend* backend)
    : impl_(std::make_unique<Impl>(reader, config, backend)) {}
Reference::~Reference() = default;

std::vector<float> Reference::Feed(std::uint32_t token, bool want_logits) {
  return impl_->Feed(token, want_logits);
}
const Reference::Trace& Reference::trace() const {
  return impl_->trace;
}
std::uint32_t Reference::position() const {
  return impl_->pos;
}

}  // namespace inkling2
