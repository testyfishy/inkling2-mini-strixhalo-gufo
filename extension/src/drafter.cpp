#include "drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <gufo/ext/gguf_reader.hpp>

namespace inkling2 {
namespace {

constexpr std::size_t kW = 2304;    // width
constexpr std::size_t kHd = 128;    // head dimension
constexpr std::size_t kNq = 24;     // query heads
constexpr std::size_t kNkv = 8;     // key/value heads
constexpr std::size_t kKv = kNkv * kHd;
constexpr std::size_t kFf = 9216;
constexpr std::size_t kRank = 256;  // Markov rank
constexpr int kLayers = 3;
constexpr std::size_t kWindow = 1024;
constexpr float kEps = 1e-6f;

inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = std::uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &wide, sizeof value);
  return value;
}

// x / sqrt(mean(x^2) + eps), times a weight per value.
void Rms(float* x, std::size_t n, const float* weight, float eps) {
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i)
    sum += static_cast<double>(x[i]) * x[i];
  const float scale = 1.0f / std::sqrt(static_cast<float>(sum / n) + eps);
  for (std::size_t i = 0; i < n; ++i)
    x[i] *= scale * weight[i];
}

// Rotary position encoding over all 128 dimensions of every head of `x`, the
// NeoX way: dimension j pairs with j + 64, angle pos * base^(-2j / 128).
void Rope(float* x, std::size_t heads, std::uint32_t pos, const float* freq) {
  constexpr std::size_t half = kHd / 2;
  float cs[half], sn[half];
  for (std::size_t j = 0; j < half; ++j) {
    const float theta = static_cast<float>(pos) * freq[j];
    cs[j] = std::cos(theta);
    sn[j] = std::sin(theta);
  }
  for (std::size_t h = 0; h < heads; ++h) {
    float* v = x + h * kHd;
    for (std::size_t j = 0; j < half; ++j) {
      const float a = v[j], b = v[j + half];
      v[j] = a * cs[j] - b * sn[j];
      v[j + half] = a * sn[j] + b * cs[j];
    }
  }
}

float Dot(const float* a, const float* b, std::size_t n) {
  float lanes[8] = {};
  for (std::size_t i = 0; i < n; i += 8)
    for (std::size_t j = 0; j < 8; ++j)
      lanes[j] += a[i + j] * b[i + j];
  float sum = 0;
  for (const float lane : lanes)
    sum += lane;
  return sum;
}

struct Layer {
  Mat wq, wk, wv, wo, wk_ctx, wv_ctx, gate, up, down;
  Vec ctx_norm, attn_norm, ffn_norm, q_norm, k_norm, k_ctx_norm;
  // The context cache: one row of kKv values per injected position.
  Vec keys, values;
};

}  // namespace

struct Drafter::Impl {
  std::unique_ptr<gufo::core::GgufReader> reader;
  const Model* target{};
  Layer layers[kLayers];
  float fuse[kLayers + 1][kFeatures]{};  // softmaxed mixing weights
  Vec head_norm, mask, output_norm, joint_bias, hidden_out_bias;
  Mat markov, joint, hidden_out;
  Vec f_rnn;  // of the last injected position
  std::uint32_t count{0};
  float freq[kHd / 2]{};

  Impl(const std::string& path, const Model& model) : target(&model) {
    std::string error;
    reader = gufo::core::GgufReader::OpenFile(path, &error);
    if (!reader)
      throw std::runtime_error("inkling2 drafter: cannot open " + path + ": " +
                               error);
    const auto& r = *reader;
    const Vec raw = LoadVector(r, "dspark_fuse.weight");
    if (raw.size() != (kLayers + 1) * kFeatures)
      throw std::runtime_error("inkling2 drafter: unexpected dspark_fuse");
    for (int row = 0; row <= kLayers; ++row) {
      const float* w = raw.data() + row * kFeatures;
      const float max = *std::max_element(w, w + kFeatures);
      float sum = 0;
      for (int k = 0; k < kFeatures; ++k)
        sum += fuse[row][k] = std::exp(w[k] - max);
      for (int k = 0; k < kFeatures; ++k)
        fuse[row][k] /= sum;
    }
    for (int il = 0; il < kLayers; ++il) {
      const std::string b = "blk." + std::to_string(il) + ".";
      auto mat = [&](const char* n) { return LoadMatrix(r, b + n + ".weight"); };
      auto vec = [&](const char* n) { return LoadVector(r, b + n + ".weight"); };
      Layer& l = layers[il];
      l.wq = mat("attn_q");
      l.wk = mat("attn_k");
      l.wv = mat("attn_v");
      l.wo = mat("attn_output");
      l.wk_ctx = mat("attn_k_ctx");
      l.wv_ctx = mat("attn_v_ctx");
      l.gate = mat("ffn_gate");
      l.up = mat("ffn_up");
      l.down = mat("ffn_down");
      l.ctx_norm = vec("dflare_norm");
      l.attn_norm = vec("attn_norm");
      l.ffn_norm = vec("ffn_norm");
      l.q_norm = vec("attn_q_norm");
      l.k_norm = vec("attn_k_norm");
      l.k_ctx_norm = vec("attn_k_ctx_norm");
      if (l.wq.cols != kW || l.wq.rows != kNq * kHd || l.wk.rows != kKv ||
          l.wk_ctx.rows != kKv || l.gate.rows != kFf || l.q_norm.size() != kHd)
        throw std::runtime_error("inkling2 drafter: unexpected layer shape");
    }
    head_norm = LoadVector(r, "dspark_norm.weight");
    mask = LoadVector(r, "mask_embd.weight");
    output_norm = LoadVector(r, "output_norm.weight");
    joint_bias = LoadVector(r, "joint_proj.bias");
    hidden_out_bias = LoadVector(r, "dspark_hidden_out.bias");
    markov = LoadMatrix(r, "markov_w1.weight");
    joint = LoadMatrix(r, "joint_proj.weight");
    hidden_out = LoadMatrix(r, "dspark_hidden_out.weight");
    if (markov.cols != kRank || markov.type != kGgmlTypeBf16 ||
        joint.cols != 2 * kRank + 2 * kW || joint.rows != 3 * kRank ||
        hidden_out.cols != kRank || hidden_out.rows != kW ||
        model.tok_embd.type != kGgmlTypeBf16 || model.tok_embd.cols != kW)
      throw std::runtime_error("inkling2 drafter: unexpected head shape");
    for (std::size_t j = 0; j < kHd / 2; ++j)
      freq[j] = std::pow(1e6f, -2.0f * static_cast<float>(j) / kHd);
  }

  void Inject(const float* features) {
    Vec mixed(kW), k(kKv), v(kKv);
    auto mix = [&](int row, const Vec& norm) {
      for (std::size_t i = 0; i < kW; ++i) {
        float sum = 0;
        for (int f = 0; f < kFeatures; ++f)
          sum += fuse[row][f] * features[f * kW + i];
        mixed[i] = sum;
      }
      Rms(mixed.data(), kW, norm.data(), kEps);
    };
    for (int il = 0; il < kLayers; ++il) {
      Layer& l = layers[il];
      mix(il, l.ctx_norm);
      CpuMul(l.wk_ctx, mixed.data(), k.data());
      CpuMul(l.wv_ctx, mixed.data(), v.data());
      for (std::size_t h = 0; h < kNkv; ++h)
        Rms(k.data() + h * kHd, kHd, l.k_ctx_norm.data(), kEps);
      Rope(k.data(), kNkv, count, freq);
      l.keys.insert(l.keys.end(), k.begin(), k.end());
      l.values.insert(l.values.end(), v.begin(), v.end());
    }
    mix(kLayers, head_norm);
    f_rnn = mixed;
    ++count;
  }

  std::vector<std::uint32_t> Draft(std::uint32_t anchor, int want,
                                   const Head& head) {
    if (count == 0 || want < 1 || want >= kBlock)
      throw std::runtime_error("inkling2 drafter: nothing to draft from");
    const std::uint32_t pos0 = count;
    // The block: the anchor's embedding, then seven mask rows.
    Vec x(kBlock * kW);
    {
      const auto* row = reinterpret_cast<const std::uint16_t*>(
          target->tok_embd.data + std::size_t{anchor} * kW * 2);
      for (std::size_t i = 0; i < kW; ++i)
        x[i] = Bf16(row[i]);
      Rms(x.data(), kW, target->tok_norm.data(), target->c.norm_eps);
      for (int b = 1; b < kBlock; ++b)
        std::copy(mask.begin(), mask.end(), x.begin() + b * kW);
    }
    Vec a(kW), q(kBlock * kNq * kHd), k(kBlock * kKv), v(kBlock * kKv);
    Vec o(kNq * kHd), proj(kW), gate(kFf), up(kFf);
    for (int il = 0; il < kLayers; ++il) {
      const Layer& l = layers[il];
      for (int b = 0; b < kBlock; ++b) {
        std::copy_n(x.begin() + b * kW, kW, a.begin());
        Rms(a.data(), kW, l.attn_norm.data(), kEps);
        float* qb = q.data() + b * kNq * kHd;
        float* kb = k.data() + b * kKv;
        CpuMul(l.wq, a.data(), qb);
        CpuMul(l.wk, a.data(), kb);
        CpuMul(l.wv, a.data(), v.data() + b * kKv);
        for (std::size_t h = 0; h < kNq; ++h)
          Rms(qb + h * kHd, kHd, l.q_norm.data(), kEps);
        for (std::size_t h = 0; h < kNkv; ++h)
          Rms(kb + h * kHd, kHd, l.k_norm.data(), kEps);
        Rope(qb, kNq, pos0 + b, freq);
        Rope(kb, kNkv, pos0 + b, freq);
      }
      // Layers 0 and 2 see the window before the anchor, layer 1 everything;
      // every block row sees the whole block. q, k and v above come from the
      // layer's input for all rows, so updating x row by row below is safe.
      const std::size_t first =
          il != 1 && pos0 > kWindow ? pos0 - kWindow : 0;
      const std::size_t n_ctx = pos0 - first;
      const float scale = 1.0f / std::sqrt(static_cast<float>(kHd));
      Vec score(n_ctx + kBlock);
      for (int b = 0; b < kBlock; ++b) {
        for (std::size_t h = 0; h < kNq; ++h) {
          const std::size_t g = h / (kNq / kNkv);
          const float* qh = q.data() + (b * kNq + h) * kHd;
          auto key = [&](std::size_t i) {
            return i < n_ctx ? l.keys.data() + ((first + i) * kNkv + g) * kHd
                             : k.data() + ((i - n_ctx) * kNkv + g) * kHd;
          };
          auto value = [&](std::size_t i) {
            return i < n_ctx ? l.values.data() + ((first + i) * kNkv + g) * kHd
                             : v.data() + ((i - n_ctx) * kNkv + g) * kHd;
          };
          float max = -INFINITY;
          for (std::size_t i = 0; i < score.size(); ++i) {
            score[i] = Dot(qh, key(i), kHd) * scale;
            max = std::max(max, score[i]);
          }
          float sum = 0;
          for (float& s : score)
            sum += s = std::exp(s - max);
          float* out = o.data() + h * kHd;
          std::fill_n(out, kHd, 0.0f);
          for (std::size_t i = 0; i < score.size(); ++i) {
            const float p = score[i] / sum;
            const float* vi = value(i);
            for (std::size_t d = 0; d < kHd; ++d)
              out[d] += p * vi[d];
          }
        }
        float* xb = x.data() + b * kW;
        CpuMul(l.wo, o.data(), proj.data());
        for (std::size_t i = 0; i < kW; ++i)
          xb[i] += proj[i];
        std::copy_n(xb, kW, a.begin());
        Rms(a.data(), kW, l.ffn_norm.data(), kEps);
        CpuMul(l.gate, a.data(), gate.data());
        CpuMul(l.up, a.data(), up.data());
        for (std::size_t i = 0; i < kFf; ++i)
          gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
        CpuMul(l.down, gate.data(), proj.data());
        for (std::size_t i = 0; i < kW; ++i)
          xb[i] += proj[i];
      }
    }
    // The recurrent head, one draft slot at a time.
    std::vector<std::uint32_t> out;
    Vec state(kRank, 0.0f), u(joint.cols), j(joint.rows), latent(kRank);
    Vec hidden(kW);
    std::uint32_t prev = anchor;
    for (int slot = 1; slot <= want; ++slot) {
      float* h = x.data() + slot * kW;
      Rms(h, kW, output_norm.data(), kEps);
      std::copy(state.begin(), state.end(), u.begin());
      const auto* e = reinterpret_cast<const std::uint16_t*>(
          markov.data + std::size_t{prev} * kRank * 2);
      for (std::size_t i = 0; i < kRank; ++i)
        u[kRank + i] = Bf16(e[i]);
      std::copy_n(h, kW, u.begin() + 2 * kRank);
      std::copy(f_rnn.begin(), f_rnn.end(), u.begin() + 2 * kRank + kW);
      CpuMul(joint, u.data(), j.data());
      for (std::size_t i = 0; i < kRank; ++i) {
        const float g = 1.0f / (1.0f + std::exp(-(j[i] + joint_bias[i])));
        const float cand = std::tanh(j[kRank + i] + joint_bias[kRank + i]);
        latent[i] = std::tanh(j[2 * kRank + i] + joint_bias[2 * kRank + i]);
        state[i] = cand + g * (state[i] - cand);
      }
      CpuMul(hidden_out, latent.data(), hidden.data());
      for (std::size_t i = 0; i < kW; ++i)
        hidden[i] += hidden_out_bias[i] + h[i];
      prev = head(hidden.data());
      out.push_back(prev);
    }
    return out;
  }
};

Drafter::Drafter(const std::string& path, const Model& target)
    : impl_(std::make_unique<Impl>(path, target)) {}
Drafter::~Drafter() = default;

void Drafter::Inject(const float* features) {
  impl_->Inject(features);
}

std::uint32_t Drafter::size() const {
  return impl_->count;
}

void Drafter::Reset() {
  impl_->count = 0;
  impl_->f_rnn.clear();
  for (Layer& l : impl_->layers) {
    l.keys.clear();
    l.values.clear();
  }
}

std::vector<std::uint32_t> Drafter::Draft(std::uint32_t anchor, int count,
                                          const Head& head) {
  return impl_->Draft(anchor, count, head);
}

}  // namespace inkling2
