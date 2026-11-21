#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.hpp"

namespace inkling2 {

/// Inkling 2 Mini on the GPU: weights, per-session KV caches and convolution
/// histories, and every per-layer operation live on the device. One step runs
/// a batch of rows through the model with a single host round trip.
///
/// A row is one token of one session. A step may hold one token from each of
/// several sessions (parallel decode), several consecutive tokens of one
/// session (prompt processing), or both.
class GpuEngine {
public:
  struct Options {
    std::uint32_t context{8192};  // positions per session
    std::uint32_t sessions{1};
    std::uint32_t max_batch{16};  // rows per step
    /// On by default. When on, steps of several rows compute the NVFP4
    /// expert products with 8-bit integer dot products: the weights stay
    /// exact, the inputs to those products are rounded to 8 bits per block of
    /// 16. Faster, and a small change in the arithmetic. Steps of fewer rows
    /// than the expert-major threshold are never affected.
    bool int8_experts{true};
    /// Whether the 8-bit products are also used in steps of fewer rows than
    /// the expert-major threshold, which includes single-session generation.
    bool int8_single{true};
    /// Keep, for every row of a step, the hidden states the drafter reads
    /// (see `features`).
    bool capture_features{false};
  };

  /// Throws std::runtime_error when the model is not Inkling 2 Mini as the
  /// kernels expect it, or the device cannot be used.
  GpuEngine(const Model& model, const Options& options);
  ~GpuEngine();

  struct Row {
    std::uint32_t session;
    std::uint32_t token;
  };
  enum class Head {
    kNone,  // no logits
    kLast,  // after the last row of each session's run
    kAll,   // after every row
  };
  /// Runs the rows. Rows of one session must be adjacent, in order, and a
  /// session may appear once. Read the logits with `logits(i)`, where i counts
  /// the runs in order of appearance (kLast) or the rows (kAll).
  ///
  /// A `tentative` step computes everything but does not move the sessions
  /// forward: afterwards `Accept` keeps the first rows of a session's run and
  /// drops the rest, which is how drafted tokens are verified. The next step
  /// of a session overwrites whatever the dropped rows stored.
  void Step(const std::vector<Row>& rows, Head head, bool tentative = false);
  /// After a tentative step, keeps the first `rows_kept` rows of `session`'s
  /// run (0 keeps none).
  void Accept(std::uint32_t session, std::uint32_t rows_kept);
  void Step(const std::vector<Row>& rows, bool head) {
    Step(rows, head ? Head::kLast : Head::kNone);
  }

  /// Forgets a session's context: its next row is position 0 again.
  void Reset(std::uint32_t session);

  /// Timing aid, not for real use: pretends the session already holds
  /// `position` tokens by repeating the keys and values it has. The output
  /// that follows is meaningless; the cost of attending over that much
  /// context is real.
  /// A session as bytes (position, caches, histories), and back: a restored
  /// session continues exactly where it was saved, without processing its
  /// tokens again. The engine that restores must have the same max_batch as
  /// the one that saved. Not while a tentative step is pending.
  [[nodiscard]] std::vector<std::uint8_t> Save(std::uint32_t session);
  void Restore(std::uint32_t session, const std::vector<std::uint8_t>& saved);

  void FakeContext(std::uint32_t session, std::uint32_t position);

  /// Raw logits (before the soft cap) of run `i` of the last step, one per
  /// vocabulary entry.
  [[nodiscard]] const float* logits(std::size_t i) const;
  /// The soft-capped logits of run `i`.
  [[nodiscard]] std::vector<float> capped_logits(std::size_t i) const;
  /// The most likely next token of run `i`.
  [[nodiscard]] std::uint32_t argmax(std::size_t i) const;

  /// With `capture_features`: batch row `row`'s hidden states from the last
  /// step, 4 * 2304 values: the residual stream entering executed layer visits
  /// 5, 23 and 41, then the final hidden state (after the output mix and
  /// norm, before the LM head).
  [[nodiscard]] std::vector<float> features(std::size_t row) const;

  /// The most likely token for a final hidden state (2304 values) put through
  /// the LM head; ties go to the lowest id. Overwrites the last step's logits.
  std::uint32_t HeadArgmax(const float* hidden);

  [[nodiscard]] std::uint32_t position(std::uint32_t session) const;
  [[nodiscard]] std::uint32_t vocab() const;

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
