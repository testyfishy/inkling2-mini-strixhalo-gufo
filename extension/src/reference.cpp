#include "reference.hpp"

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
using Vec = std::vector<float>;

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
constexpr int kNvfp4Block = 64;
constexpr int kNvfp4Bytes = 36;
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

/// A weight used as `y = W x`: GGUF dims {n_in, n_out[, n_expert]}, stored
/// with one row of n_in values per output.
struct Mat {
  const std::uint8_t* data{nullptr};
  std::size_t cols{0}, rows{0};
  std::uint16_t type{0};

  [[nodiscard]] std::size_t RowBytes() const {
    if (type == kF32)
      return cols * 4;
    if (type == kBF16)
      return cols * 2;
    return cols / kNvfp4Block * kNvfp4Bytes;
  }

  [[nodiscard]] float Dot(const std::uint8_t* row, const float* x) const {
    if (type == kGgmlTypeNvfp4)
      return DotNvfp4(row, x, cols);
    float sum = 0.0f;
    if (type == kF32) {
      const float* w = reinterpret_cast<const float*>(row);
      for (std::size_t c = 0; c < cols; ++c)
        sum += w[c] * x[c];
    } else {
      const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(row);
      for (std::size_t c = 0; c < cols; ++c)
        sum += Bf16(w[c]) * x[c];
    }
    return sum;
  }

  /// y = W[expert] x
  void Mul(const float* x, float* y, std::size_t expert = 0) const {
    const std::uint8_t* base = data + expert * rows * RowBytes();
    const std::size_t row_bytes = RowBytes();
    ParallelFor(rows, cols, [&](std::size_t begin, std::size_t end) {
      for (std::size_t r = begin; r < end; ++r)
        y[r] = Dot(base + r * row_bytes, x);
    });
  }

  [[nodiscard]] Vec Mul(const Vec& x, std::size_t expert = 0) const {
    Vec y(rows);
    Mul(x.data(), y.data(), expert);
    return y;
  }
};

const GgufTensorInfo& Find(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo* t = reader.FindTensor(name);
  if (!t)
    throw std::runtime_error("inkling2: missing tensor " + name);
  return *t;
}

Mat LoadMat(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo& t = Find(reader, name);
  const auto type = static_cast<std::uint16_t>(t.type);
  if (type != kF32 && type != kBF16 && type != kGgmlTypeNvfp4)
    throw std::runtime_error(
        "inkling2: the reference reads F32, BF16 and NVFP4 only; " + name +
        " is stored otherwise");
  Mat m;
  m.data = static_cast<const std::uint8_t*>(t.data);
  m.cols = t.dimensions[0];
  m.rows = t.dimensions.size() > 1 ? t.dimensions[1] : 1;
  m.type = type;
  return m;
}

/// A small tensor copied out as F32, in storage order.
Vec LoadVec(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo& t = Find(reader, name);
  const auto type = static_cast<std::uint16_t>(t.type);
  const std::size_t n = t.ElementCount();
  Vec out(n);
  if (type == kF32) {
    std::memcpy(out.data(), t.data, n * 4);
  } else if (type == kBF16) {
    const auto* w = static_cast<const std::uint16_t*>(t.data);
    for (std::size_t i = 0; i < n; ++i)
      out[i] = Bf16(w[i]);
  } else {
    throw std::runtime_error("inkling2: unexpected storage of " + name);
  }
  return out;
}

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

struct Layer {
  bool swa{false};
  Mat wq, wk, wv, wo, gate, sink, rel, k_conv_proj, v_conv_proj;
  Mat gate_inp, gate_up, down;
  Vec q_clip, rel_bias, rel_clip, k_conv, v_conv, attn_post_conv, ffn_post_conv;
  Vec attn_norm, q_norm, k_norm, v_norm, attn_res_query, ffn_res_query;
  Vec ffn_norm, exp_probs_b, gate_up_scale, down_scale;
  // loop core only
  Vec msg_query;
  Mat ctrl_down[4], ctrl_gain[4], ctrl_forget[4];
};

/// State of one layer visit: its own KV rows and convolution history.
struct Slot {
  std::vector<Vec> keys, values;  // one [n_head_kv * head_dim] row per position
  std::vector<Vec> history[4];    // last n_conv - 1 inputs of K, V, attn, ffn
};

}  // namespace

struct Reference::Impl {
  Config c;
  Mat tok_embd, output;
  Vec tok_norm, output_norm, output_res_query;
  std::vector<Layer> layers;
  std::vector<Slot> slots;
  Trace trace;
  std::uint32_t pos{0};
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

  Impl(const GgufReader& reader, const Config& config) : c(config) {
    tok_embd = LoadMat(reader, "token_embd.weight");
    output = LoadMat(reader, "output.weight");
    tok_norm = LoadVec(reader, "token_embd_norm.weight");
    output_norm = LoadVec(reader, "output_norm.weight");
    output_res_query = LoadVec(reader, "output_res_query.weight");
    layers.resize(c.n_layer);
    for (std::uint32_t il = 0; il < c.n_layer; ++il) {
      const std::string b = "blk." + std::to_string(il) + ".";
      auto mat = [&](const char* n) {
        return LoadMat(reader, b + n + ".weight");
      };
      auto vec = [&](const char* n) {
        return LoadVec(reader, b + n + ".weight");
      };
      Layer& l = layers[il];
      l.swa = c.is_swa[il];
      l.wq = mat("attn_q");
      l.wk = mat("attn_k");
      l.wv = mat("attn_v");
      l.wo = mat("attn_output");
      l.gate = mat("attn_gate");
      l.sink = mat("attn_sink_proj");
      l.rel = mat("attn_rel");
      l.q_clip = vec("attn_q_clip");
      l.rel_bias = vec("attn_rel_bias");
      l.rel_clip = vec("attn_rel_clip");
      l.k_conv = vec("attn_k_conv");
      l.v_conv = vec("attn_v_conv");
      l.attn_post_conv = vec("attn_post_conv");
      l.ffn_post_conv = vec("ffn_post_conv");
      if (l.swa) {
        l.attn_norm = vec("attn_norm");
      } else {
        l.q_norm = vec("attn_q_inp_norm");
        l.k_norm = vec("attn_k_inp_norm");
        l.v_norm = vec("attn_v_inp_norm");
        l.k_conv_proj = mat("attn_k_conv_proj");
        l.v_conv_proj = mat("attn_v_conv_proj");
      }
      if (il > 0)
        l.attn_res_query = vec("attn_res_query");
      l.ffn_res_query = vec("ffn_res_query");
      l.ffn_norm = vec("ffn_norm");
      l.gate_inp = mat("ffn_gate_inp");
      l.gate_up = mat("ffn_gate_up_exps");
      l.down = mat("ffn_down_exps");
      if (reader.HasTensor(b + "exp_probs_b.bias"))
        l.exp_probs_b = LoadVec(reader, b + "exp_probs_b.bias");
      // Per-expert scales that accompany NVFP4 experts.
      if (reader.HasTensor(b + "ffn_gate_up_exps.scale"))
        l.gate_up_scale = LoadVec(reader, b + "ffn_gate_up_exps.scale");
      if (reader.HasTensor(b + "ffn_down_exps.scale"))
        l.down_scale = LoadVec(reader, b + "ffn_down_exps.scale");
      if (il >= c.loop_start && il < c.loop_end) {
        l.msg_query = vec("loop_msg_query");
        const char* names[4] = {"q", "k", "v", "ffn"};
        for (int k = 0; k < 4; ++k) {
          const std::string p = b + "loop_" + names[k] + "_ctrl_";
          l.ctrl_down[k] = LoadMat(reader, p + "down.weight");
          l.ctrl_gain[k] = LoadMat(reader, p + "gain.weight");
          l.ctrl_forget[k] = LoadMat(reader, p + "forget.weight");
        }
      }
    }
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
    const Vec low = l.ctrl_down[k].Mul(gate);
    const Vec gain = l.ctrl_gain[k].Mul(low);
    const Vec forget = l.ctrl_forget[k].Mul(low);
    Vec out(e);
    for (std::size_t i = 0; i < e; ++i)
      out[i] = cur[i] - std::tanh(forget[i]) * cur[i] +
               std::tanh(gain[i]) * message[i];
    return out;
  }

  Vec Moe(const Vec& x, const Layer& l) const {
    const std::size_t e = c.n_embd, ff = c.n_ff_exp;
    Dbg("ffn_normed", x);
    Vec probs = l.gate_inp.Mul(x);
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

    Vec out(e, 0.0f), gate_up(2 * ff), act(ff), down(e);
    for (std::uint32_t i = 0; i < c.n_expert_used; ++i) {
      const std::size_t expert = order[i];
      const float weight =
          probs[expert] / weight_sum * c.expert_weights_scale;
      l.gate_up.Mul(x.data(), gate_up.data(), expert);
      const float s_up =
          l.gate_up_scale.empty() ? 1.0f : l.gate_up_scale[expert];
      for (std::size_t k = 0; k < ff; ++k) {
        const float g = gate_up[k] * s_up, u = gate_up[ff + k] * s_up;
        // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
        act[k] = c.situ_beta * std::tanh(g / c.situ_beta) * Sigmoid(g) *
                 c.situ_linear_beta * std::tanh(u / c.situ_linear_beta);
      }
      Dbg("gate_up_raw", gate_up);
      Dbg("situ", act);
      l.down.Mul(act.data(), down.data(), expert);
      const float s_down = l.down_scale.empty() ? 1.0f : l.down_scale[expert];
      for (std::size_t k = 0; k < e; ++k)
        out[k] += weight * s_down * down[k];
    }
    Dbg("moe_out", out);
    return out;
  }

  Vec Attention(const Vec in[3], const Layer& l, Slot& slot) const {
    const std::size_t hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    Dbg("attn_in_q", in[0]);
    Vec q = l.wq.Mul(in[0]);
    Dbg("q_proj", q);
    const Vec k_taps = l.swa ? Vec{} : l.k_conv_proj.Mul(in[1]);
    const Vec v_taps = l.swa ? Vec{} : l.v_conv_proj.Mul(in[2]);
    Vec k = Convolve(l.wk.Mul(in[1]), l.k_conv, l.swa ? nullptr : &k_taps,
                     slot.history[0]);
    Vec v = Convolve(l.wv.Mul(in[2]), l.v_conv, l.swa ? nullptr : &v_taps,
                     slot.history[1]);
    Vec r = l.rel.Mul(in[0]);
    Dbg("r_proj", r);

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
    const Vec sink = l.sink.Mul(in[0]);
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
    const Vec gate = l.gate.Mul(in[0]);
    Dbg("gate_logits", gate);
    for (std::size_t i = 0; i < out.size(); ++i)
      out[i] *= Sigmoid(gate[i]);
    Dbg("gated", out);
    Dbg("wo", l.wo.Mul(out));
    return Convolve(l.wo.Mul(out), l.attn_post_conv, nullptr, slot.history[2]);
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

    Vec logits = output.Mul(cur);
    logits.resize(c.n_vocab);  // drop the padded vocabulary
    if (c.logit_softcap != 0.0f)
      for (float& x : logits)
        x = c.logit_softcap * std::tanh(x / c.logit_softcap);
    return logits;
  }
};

Reference::Reference(const GgufReader& reader, const Config& config)
    : impl_(std::make_unique<Impl>(reader, config)) {}
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
