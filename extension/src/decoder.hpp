#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.hpp"

namespace inkling2 {

/// One sequence decoded entirely on the GPU: weights, KV cache, convolution
/// history and every per-layer operation live on the device, and a token costs
/// one host round trip. Single sequence, one token per step.
class GpuDecoder {
public:
  /// `context` bounds the positions the full-attention layers can hold.
  /// Throws std::runtime_error when the model is not Inkling 2 Mini as the
  /// kernels expect it, or the device cannot be used.
  GpuDecoder(const Model& model, std::uint32_t context);
  ~GpuDecoder();

  /// Feeds one token and returns the logits over the vocabulary.
  std::vector<float> Feed(std::uint32_t token);
  /// Feeds one token and returns the most likely next token.
  std::uint32_t FeedGreedy(std::uint32_t token);
  /// Feeds one token without computing the output head.
  void FeedSilent(std::uint32_t token);

  /// With debugging on, every layer visit is followed by a round trip and the
  /// sum of its output is accumulated here (compare Reference::Trace::l_out).
  void SetDebug(bool on);
  [[nodiscard]] const std::vector<double>& layer_sums() const;

  /// With profiling on, every kernel group is timed with its own round trip.
  void SetProfile(bool on);
  /// Seconds accumulated per kernel group while profiling.
  [[nodiscard]] std::vector<std::pair<std::string, double>> phases() const;

  [[nodiscard]] std::size_t launches() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inkling2
