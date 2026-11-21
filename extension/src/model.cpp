#include "model.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace inkling2 {
namespace {

using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;

inline float Bf16(std::uint16_t bits) {
  const std::uint32_t wide = std::uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &wide, sizeof value);
  return value;
}

const GgufTensorInfo& Find(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo* t = reader.FindTensor(name);
  if (!t)
    throw std::runtime_error("inkling2: missing tensor " + name);
  return *t;
}

Mat LoadMat(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo& t = Find(reader, name);
  const auto type = static_cast<std::uint16_t>(t.type);
  if (type != kGgmlTypeF32 && type != kGgmlTypeBf16 && type != kGgmlTypeNvfp4)
    throw std::runtime_error(
        "inkling2: only F32, BF16 and NVFP4 weights are supported; " + name +
        " is stored otherwise");
  Mat m;
  m.data = static_cast<const std::uint8_t*>(t.data);
  m.cols = t.dimensions[0];
  m.rows = t.dimensions.size() > 1 ? t.dimensions[1] : 1;
  m.type = type;
  return m;
}

Vec LoadVec(const GgufReader& reader, const std::string& name) {
  const GgufTensorInfo& t = Find(reader, name);
  const auto type = static_cast<std::uint16_t>(t.type);
  const std::size_t n = t.ElementCount();
  Vec out(n);
  if (type == kGgmlTypeF32) {
    std::memcpy(out.data(), t.data, n * 4);
  } else if (type == kGgmlTypeBf16) {
    const auto* w = static_cast<const std::uint16_t*>(t.data);
    for (std::size_t i = 0; i < n; ++i)
      out[i] = Bf16(w[i]);
  } else {
    throw std::runtime_error("inkling2: unexpected storage of " + name);
  }
  return out;
}

}  // namespace

Mat LoadMatrix(const GgufReader& reader, const std::string& name) {
  return LoadMat(reader, name);
}

Vec LoadVector(const GgufReader& reader, const std::string& name) {
  return LoadVec(reader, name);
}

Model LoadModel(const GgufReader& reader, const Config& config) {
  Model m;
  m.c = config;
  const Config& c = m.c;
  m.tok_embd = LoadMat(reader, "token_embd.weight");
  m.output = LoadMat(reader, "output.weight");
  m.tok_norm = LoadVec(reader, "token_embd_norm.weight");
  m.output_norm = LoadVec(reader, "output_norm.weight");
  m.output_res_query = LoadVec(reader, "output_res_query.weight");
  m.layers.resize(c.n_layer);
  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    auto mat = [&](const char* n) { return LoadMat(reader, b + n + ".weight"); };
    auto vec = [&](const char* n) { return LoadVec(reader, b + n + ".weight"); };
    LayerWeights& l = m.layers[il];
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
  return m;
}

}  // namespace inkling2
