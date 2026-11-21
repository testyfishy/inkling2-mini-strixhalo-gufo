#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

namespace inkling2 {

// GGML tensor type id of NVFP4, which GUFO's core reader does not size.
inline constexpr std::uint16_t kGgmlTypeNvfp4 = 40;

/// Hyperparameters of an `inkling2_loop` GGUF, as llama.cpp reads them
/// (src/models/inkling2.cpp and inkling2-loop.cpp).
struct Config {
  std::uint32_t n_layer{0};  // physical layers
  std::uint32_t n_embd{0};
  std::uint32_t n_head{0};
  std::uint32_t n_head_kv{0};
  std::uint32_t head_dim{0};
  std::uint32_t n_expert{0};
  std::uint32_t n_expert_used{0};
  std::uint32_t n_ff_exp{0};
  std::uint32_t n_vocab_padded{0};
  std::uint32_t n_vocab{0};
  std::uint32_t n_ctx{0};
  std::uint32_t n_swa{0};
  std::uint32_t n_conv{0};  // short-convolution taps
  std::uint32_t n_rel{0};
  std::uint32_t n_rel_extent{0};
  std::uint32_t res_block{0};
  std::uint32_t loop_start{0};
  std::uint32_t loop_end{0};
  std::uint32_t loop_count{0};
  std::uint32_t loop_gate_rank{0};
  std::uint32_t log_scale_floor{0};
  float log_scale_alpha{0};
  float norm_eps{0};
  float qk_clip{0};
  float rel_clip{0};
  float situ_beta{0};
  float situ_linear_beta{0};
  float expert_weights_scale{1};
  float logit_softcap{0};
  std::vector<bool> is_swa;  // per physical layer

  /// Layer visits per token: every layer once, the loop core `loop_count` times.
  [[nodiscard]] std::uint32_t visits() const {
    return n_layer + (loop_count - 1) * (loop_end - loop_start);
  }
};

/// Cheap, GPU-free check that `reader` is Inkling 2 Mini in the mainline
/// `inkling2_loop` layout. Anything else, Inkling 2 Small included, is refused.
[[nodiscard]] bool Probe(const gufo::core::GgufReader& reader);

/// Reads and validates the hyperparameters. Returns false and sets `error`.
[[nodiscard]] bool LoadConfig(const gufo::core::GgufReader& reader,
                              Config* config, std::string* error);

struct Inventory {
  std::size_t tensors{0};
  std::size_t bytes{0};
  std::map<std::string, std::pair<std::size_t, std::size_t>> by_type;
  std::size_t nvfp4_expert_tensors{0};
};

/// Checks that every tensor the forward pass needs is present with the
/// expected shape, and summarizes storage types. Returns false and sets
/// `error` on the first problem.
[[nodiscard]] bool CheckTensors(const gufo::core::GgufReader& reader,
                                const Config& config, Inventory* inventory,
                                std::string* error);

}  // namespace inkling2
