#pragma once

#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "backend.hpp"
#include "config.hpp"

namespace inkling2 {

using Vec = std::vector<float>;

/// One physical layer's weights: the large matrices stay mapped in the GGUF
/// file (`Mat`), the small tensors are copied out as F32 in storage order.
struct LayerWeights {
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

struct Model {
  Config c;
  Mat tok_embd, output;
  Vec tok_norm, output_norm, output_res_query;
  std::vector<LayerWeights> layers;
};

/// Maps every tensor of the forward pass. Throws std::runtime_error when one
/// is missing or stored in a type other than F32, BF16 or NVFP4.
[[nodiscard]] Model LoadModel(const gufo::core::GgufReader& reader,
                              const Config& config);

/// One tensor of a GGUF file as a mapped matrix, or copied out as F32. Both
/// throw std::runtime_error when the tensor is missing or stored otherwise
/// than F32, BF16 or (matrices only) NVFP4.
[[nodiscard]] Mat LoadMatrix(const gufo::core::GgufReader& reader,
                             const std::string& name);
[[nodiscard]] Vec LoadVector(const gufo::core::GgufReader& reader,
                             const std::string& name);

}  // namespace inkling2
