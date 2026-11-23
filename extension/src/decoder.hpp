#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "drafter.hpp"
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
    /// The DSpark drafter (docs/DSPARK.md), or null. With one, every step
    /// also keeps what the drafter reads of its rows, and `Draft` proposes
    /// tokens.
    std::shared_ptr<const DrafterWeights> drafter;
    /// How many positions back the drafter's middle layer looks. Its file
    /// lets that layer see the whole context; a window bounds its cost and
    /// memory, and only proposals depend on it.
    std::uint32_t draft_window{8192};
    /// The drafter scores a shortlist of this many tokens per session instead
    /// of the whole vocabulary: the tokens of the context and the tokens the
    /// model itself found likely at recent steps. 0 scores the whole
    /// vocabulary.
    std::uint32_t draft_vocab{32768};
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
  ///
  /// With a drafter, a step puts its rows' tokens on the session's shortlist
  /// and, unless `list_likely` is false, the tokens the model finds likely
  /// after each row that has logits.
  void Step(const std::vector<Row>& rows, Head head, bool tentative = false,
            bool list_likely = true);
  /// After a tentative step, keeps the first `rows_kept` rows of `session`'s
  /// run (0 keeps none).
  void Accept(std::uint32_t session, std::uint32_t rows_kept);
  void Step(const std::vector<Row>& rows, bool head) {
    Step(rows, head ? Head::kLast : Head::kNone);
  }

  struct DraftToken {
    std::uint32_t token;
    float confidence;   // the drafter's largest probability at this position
    float probability;  // its probability of this token
  };
  [[nodiscard]] bool has_drafter() const;
  /// Proposes the tokens that follow `anchor`, the token about to be placed
  /// at the session's position: up to `max_tokens` (at most 7), ending before
  /// the first one whose confidence is below `min_confidence`. Empty when the
  /// engine has no drafter or the session nothing to draft from.
  ///
  /// With `temperature` 0 each token is the drafter's most likely one. With a
  /// positive temperature each is drawn from the drafter's distribution at
  /// that temperature (the noise comes from `seed`), which is what
  /// speculative sampling needs: a draft x drawn from q is kept with
  /// probability min(1, p(x) / q(x)) under the model's p, and otherwise a
  /// token is drawn from the normalized max(0, p - q). `DraftDistribution`
  /// gives q for one position of the last draft.
  [[nodiscard]] std::vector<DraftToken> Draft(std::uint32_t session,
                                              std::uint32_t anchor,
                                              int max_tokens,
                                              float min_confidence,
                                              float temperature = 0.0f,
                                              std::uint32_t seed = 0);
  /// The distribution the last draft's token `index` (0-based) was drawn
  /// from: the tokens with a non-zero chance and their chances.
  void DraftDistribution(std::size_t index, std::vector<std::uint32_t>* tokens,
                         std::vector<float>* probabilities);

  /// Empties a session's shortlist, and puts tokens on it in order. What a
  /// draft proposes depends on the list, so a caller that wants a request to
  /// decode the same way every time starts it from the same list: empty,
  /// then the context's tokens.
  void ResetDraftList(std::uint32_t session);
  /// Tokens on the session's shortlist (a diagnostic).
  [[nodiscard]] std::size_t draft_list_size(std::uint32_t session) const;
  void ListDraftTokens(std::uint32_t session,
                       std::span<const std::uint32_t> tokens);

  /// Test aid: Reset, and every cached key and value of the session zeroed,
  /// as in a new engine.
  void Scrub(std::uint32_t session);

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

  /// Where a session's state lives on the device, for a cache that copies it
  /// piece by piece (the continuation cache of `gufo serve`).
  struct StateLayout {
    std::uint32_t visits{0};
    std::vector<bool> ring;  // per visit: a sliding-window visit
    std::uint32_t heads{0};  // key/value heads
    /// The convolution histories of all visits, one block per session.
    std::size_t history_bytes{0};
    /// The keys, and equally the values, of one sliding-window visit as a
    /// cache copies them: the window before the session's position, packed
    /// (`PackRing`, `RingStage`).
    std::size_t ring_bytes{0};
    /// A full-attention visit, per head: keys are one row per position,
    /// values one block per 1,024 positions.
    std::uint32_t full_capacity{0};  // positions
    std::size_t key_row_bytes{0};
    std::size_t value_chunk_bytes{0};
    /// The drafter's state of a session (0 without a drafter): copied whole.
    std::size_t draft_bytes{0};
  };
  [[nodiscard]] StateLayout Layout() const;
  [[nodiscard]] void* History(std::uint32_t session) const;
  /// A sliding-window visit's window as a contiguous buffer: `ring_bytes` of
  /// keys, then as many of values. `PackRing` fills it from the ring at the
  /// session's position; `UnpackRing` puts it back for a session that will
  /// be at `position`. Both are queued; `Synchronize` waits for them.
  [[nodiscard]] void* RingStage(std::uint32_t session,
                                std::uint32_t visit) const;
  void PackRing(std::uint32_t session, std::uint32_t visit);
  void UnpackRing(std::uint32_t session, std::uint32_t visit,
                  std::uint32_t position);
  /// A full-attention visit's keys (first row) or values (first block) of one
  /// head.
  [[nodiscard]] void* Full(std::uint32_t session, std::uint32_t visit,
                           std::uint32_t head, bool values) const;
  [[nodiscard]] void* DraftState(std::uint32_t session) const;
  /// Declares that the session's state now is that of `position` tokens: the
  /// caller has put every piece in place.
  void SetPosition(std::uint32_t session, std::uint32_t position);
  /// Waits until the device has finished everything submitted so far.
  void Synchronize();
  [[nodiscard]] std::uint32_t context() const;
  [[nodiscard]] std::uint32_t max_batch() const;
  [[nodiscard]] std::uint32_t sessions() const;

  /// The logits of run `i` of the last step, one per vocabulary entry, with
  /// the model's soft cap applied (on the device).
  [[nodiscard]] const float* logits(std::size_t i) const;
  /// A copy of them.
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
