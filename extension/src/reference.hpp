#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gufo/ext/gguf_reader.hpp>

#include "config.hpp"

namespace inkling2 {

/// Plain CPU implementation of the Inkling 2 Mini forward pass, one token at a
/// time, following llama.cpp's graph (src/models/inkling2.cpp and
/// inkling2-loop.cpp) operation by operation. It is the correctness reference
/// for the GPU runner: slow, single sequence, no batching, F32 arithmetic.
class Reference {
public:
  /// Per-visit sums over every token fed so far, named like llama.cpp's
  /// debug callback (`l_out-<slot>`), to compare against its tensor dump.
  struct Trace {
    std::vector<double> attn_out, ffn_out, l_out;  // one entry per visit slot
    double result_norm{0};
    std::vector<std::pair<std::string, double>> debug;  // layer 0 stages
  };

  Reference(const gufo::core::GgufReader& reader, const Config& config);
  ~Reference();

  /// Feeds one token at the next position and returns the logits over the
  /// unpadded vocabulary when `want_logits`, else an empty vector.
  std::vector<float> Feed(std::uint32_t token, bool want_logits);

  [[nodiscard]] const Trace& trace() const;
  [[nodiscard]] std::uint32_t position() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inkling2
