#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "model.hpp"

namespace inkling2 {

/// The DSpark drafter that ships with Inkling 2 Mini (see docs/DSPARK.md): it
/// reads four hidden states per position from the main model and proposes up
/// to seven tokens at once, which the main model then verifies.
///
/// This is the host prototype: plain CPU arithmetic, one session. It exists
/// to settle what the drafter computes and how many of its drafts the main
/// model accepts; it is not fast.
class Drafter {
public:
  static constexpr int kFeatures = 4;  // hidden states read per position
  static constexpr int kBlock = 8;     // rows per draft pass: anchor + 7

  /// `target` supplies the token embeddings the anchor row uses and must
  /// outlive the drafter. Throws std::runtime_error when the file is not the
  /// drafter this code expects.
  Drafter(const std::string& path, const Model& target);
  ~Drafter();

  /// Adds the next position's hidden states: kFeatures * 2304 values, in the
  /// order GpuEngine::features returns them. Positions are added in order,
  /// and only positions the main model has kept.
  void Inject(const float* features);

  /// Positions injected so far.
  [[nodiscard]] std::uint32_t size() const;

  /// Forgets every position.
  void Reset();

  /// Puts a final hidden state (2304 values) through the main model's LM head
  /// and returns the most likely token.
  using Head = std::function<std::uint32_t(const float*)>;

  /// Drafts the `count` (1 to 7) tokens that follow `anchor`, where `anchor`
  /// is the token about to be placed at position size().
  [[nodiscard]] std::vector<std::uint32_t> Draft(std::uint32_t anchor, int count,
                                                 const Head& head);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inkling2
