// The runner GUFO's server drives (`gufo serve llm`, and `gufo prompt`,
// `gufo chat`, `gufo bench` through it), and the adapter through which GUFO's
// continuation cache copies a session's state.
#include "serve.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gufo/ext/adapter.hpp>
#include <gufo/ext/cache.hpp>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/prompt_lookup.hpp>
#include <gufo/ext/snapshot_copy.hpp>
#include <gufo/ext/transfer_pool.hpp>

#include "chat.hpp"
#include "config.hpp"
#include "decoder.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "tools.hpp"

namespace inkling2 {
namespace {

namespace cache = gufo::cache;
namespace ext = gufo::ext;
namespace sampling = gufo::sampling;
using cache::Rows;
using cache::TransferResult;

constexpr std::string_view kStateAbi = "inkling2-native-fp16-v1";
constexpr std::size_t kMetaBytes = 64;
constexpr Rows kValueBlock = 1024;  // positions per block of values

void CheckHip(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}

struct Meta {
  std::uint32_t magic{0x32494e4bu};
  std::uint32_t version{1};
  std::uint32_t position{0};
  std::uint32_t has_logits{0};
};
static_assert(sizeof(Meta) <= kMetaBytes);

// ---- What a session's state consists of -------------------------------------
//
//   frame    private  position, the logits of the next token, the convolution
//                     histories
//   ring     private  one per sliding-window visit: its key ring and value
//                     ring, whole
//   tail     private  one per full-attention visit: per head, the block of
//                     values that is still being filled
//   keys     rows     per full-attention visit and head: one row per position
//   values   rows     per full-attention visit and head: one row per complete
//                     block of 1,024 positions
enum class PartKind { kFrame, kRing, kTail, kKeys, kValues };

struct Part {
  cache::ComponentDescriptor descriptor;
  PartKind kind;
  std::uint32_t visit{0}, head{0};
};

std::vector<Part> BuildParts(const GpuEngine::StateLayout& layout,
                             std::uint32_t vocab) {
  std::vector<Part> parts;
  const auto add = [&](PartKind kind, std::size_t row, Rows chunk,
                       std::size_t state, std::uint32_t visit,
                       std::uint32_t head) {
    const auto id = static_cast<std::uint32_t>(parts.size() + 1);
    parts.push_back({{{id},
                      1,
                      row ? cache::ComponentKind::kAppendRows
                          : cache::ComponentKind::kPrivateState,
                      row,
                      chunk,
                      state},
                     kind,
                     visit,
                     head});
  };
  add(PartKind::kFrame, 0, 0,
      kMetaBytes + std::size_t{vocab} * 4 + layout.history_bytes, 0, 0);
  for (std::uint32_t visit = 0; visit < layout.visits; ++visit) {
    if (layout.ring[visit]) {
      add(PartKind::kRing, 0, 0, 2 * layout.ring_bytes, visit, 0);
      continue;
    }
    add(PartKind::kTail, 0, 0, layout.heads * layout.value_chunk_bytes, visit,
        0);
    for (std::uint32_t head = 0; head < layout.heads; ++head) {
      add(PartKind::kKeys, layout.key_row_bytes, 256, 0, visit, head);
      add(PartKind::kValues, layout.value_chunk_bytes, 1, 0, visit, head);
    }
  }
  return parts;
}

using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}

// ---- The adapter ------------------------------------------------------------

class Adapter final : public cache::Adapter {
public:
  class State;

  Adapter(std::shared_ptr<GpuEngine> engine, std::uint32_t vocab,
          cache::Identity identity)
      : engine_(std::move(engine)),
        layout_(engine_->Layout()),
        vocab_(vocab),
        parts_(BuildParts(layout_, vocab)),
        identity_(std::move(identity)) {
    for (const Part& part : parts_)
      descriptors_.push_back(part.descriptor);
  }

  [[nodiscard]] State& As(cache::Slot& slot) const;
  [[nodiscard]] const State& As(const cache::Slot& slot) const {
    return As(const_cast<cache::Slot&>(slot));
  }
  [[nodiscard]] GpuEngine& engine() const { return *engine_; }
  [[nodiscard]] std::uint32_t vocab() const { return vocab_; }

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram) const override {
    constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
    const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
    if (budget <= header)
      return {};
    const auto blocks = (budget - header) / (page + per_block);
    if (!blocks)
      return {};
    return {{page, blocks, header + blocks * per_block}};
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override {
    return {.continuation = true, .persistent_encoding = true};
  }
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override {
    return descriptors_;
  }
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override {
    return identity_;
  }
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard& guard) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot& slot) const override;
  [[nodiscard]] std::vector<Rows> PlanPrefill(const cache::Slot& slot,
                                              Rows first,
                                              Rows end) const override;
  void BeginRestore(cache::Slot& slot,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&,
                                              cache::ComponentId, Rows first,
                                              Rows end, std::span<std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             Rows first, Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

  [[nodiscard]] std::vector<cache::ComponentPosition> PositionsAt(
      std::uint32_t position) const {
    std::vector<cache::ComponentPosition> positions;
    positions.reserve(parts_.size());
    for (const Part& part : parts_)
      positions.push_back({part.descriptor.id,
                           part.kind == PartKind::kValues
                               ? position / kValueBlock
                               : position});
    return positions;
  }
  [[nodiscard]] Rows Capacity(const Part& part) const {
    return part.kind == PartKind::kValues
               ? layout_.full_capacity / kValueBlock
               : layout_.full_capacity;
  }
  [[nodiscard]] const std::vector<Part>& parts() const { return parts_; }

private:
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  [[nodiscard]] const Part& PartOf(cache::ComponentId id) const {
    if (!id.value || id.value > parts_.size())
      throw std::invalid_argument("inkling2: invalid component");
    return parts_[id.value - 1];
  }
  [[nodiscard]] std::vector<Piece> PrivatePieces(State& state, const Part& part,
                                                 std::uint32_t position) const;
  [[nodiscard]] void* RowData(State& state, const Part& part) const;
  [[nodiscard]] cache::Completion Transfer(
      State& state, bool load, cache::Stream& stream,
      const std::function<void(gufo::hip::TransferStream&)>& submit);

  std::shared_ptr<GpuEngine> engine_;
  GpuEngine::StateLayout layout_;
  std::uint32_t vocab_;
  std::vector<Part> parts_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  std::uint32_t next_session_{0};
};

class Adapter::State final : public cache::Slot {
public:
  State(Adapter& adapter, cache::MutationGuard& guard, std::uint32_t session)
      : adapter(adapter),
        guard(guard),
        session(session),
        logits(adapter.vocab(), 0.0F) {}

  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };

  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    Release();
  }
  bool IsValid() const noexcept override { return !restoring && !failed; }
  void RequireIdle() const {
    if (restoring || failed || pending)
      throw std::logic_error("inkling2: the slot is not ready for execution");
  }
  // Positions [first, end) are about to be written, including rows a
  // verification step may drop again.
  void BeforeWrite(std::uint32_t first, std::uint32_t end) {
    RequireIdle();
    for (const Part& part : adapter.parts()) {
      if (part.kind == PartKind::kKeys)
        guard.BeforeOverwrite(part.descriptor.id, first, end);
      else if (part.kind == PartKind::kValues &&
               first / kValueBlock < end / kValueBlock)
        guard.BeforeOverwrite(part.descriptor.id, first / kValueBlock,
                              end / kValueBlock);
    }
  }
  void GuardAll() {
    for (const Part& part : adapter.parts())
      if (part.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(part.descriptor.id, 0, adapter.Capacity(part));
  }
  void Release() noexcept {
    for (const Part& part : adapter.parts())
      if (part.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(part.descriptor.id, 0, adapter.Capacity(part));
  }

  Adapter& adapter;
  cache::MutationGuard& guard;
  std::uint32_t session;
  bool restoring{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kMetaBytes> meta{};
  // The soft-capped logits of the token after the last one processed.
  std::vector<float> logits;
  bool has_logits{false};
};

Adapter::State& Adapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("inkling2: slot of another adapter");
  return state;
}

std::unique_ptr<cache::Slot> Adapter::CreateSlot(cache::MutationGuard& guard) {
  if (next_session_ >= engine_->sessions())
    throw std::runtime_error("inkling2: more slots than sessions");
  return std::make_unique<State>(*this, guard, next_session_++);
}

std::vector<cache::ComponentPosition> Adapter::Positions(
    const cache::Slot& slot) const {
  const State& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("inkling2: cannot query an invalid slot");
  return PositionsAt(engine_->position(state.session));
}

std::vector<Rows> Adapter::PlanPrefill(const cache::Slot& slot, Rows first,
                                       Rows end) const {
  const State& state = As(slot);
  if (!state.IsValid() || first > end || end > engine_->context())
    throw std::invalid_argument("inkling2: invalid prefill plan");
  std::vector<Rows> result;
  const Rows capacity = engine_->max_batch();
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}

void Adapter::BeginRestore(cache::Slot& slot,
                           std::span<const cache::ComponentPosition> positions) {
  State& state = As(slot);
  state.RequireIdle();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("inkling2: incomplete restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("inkling2: missing or duplicate frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered.front().valid_rows > engine_->context())
    throw std::invalid_argument("inkling2: frontier beyond the context");
  const auto target = static_cast<std::uint32_t>(ordered.front().valid_rows);
  if (ordered != PositionsAt(target))
    throw std::invalid_argument("inkling2: inconsistent component frontiers");
  state.GuardAll();
  engine_->Synchronize();
  state.expected = std::move(ordered);
  state.coverage.assign(descriptors_.size(), {});
  state.meta.fill(std::byte{});
  state.has_logits = false;
  state.restoring = true;
}

cache::Completion Adapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

// The byte layout of one private component at a frontier.
std::vector<Adapter::Piece> Adapter::PrivatePieces(State& state,
                                                   const Part& part,
                                                   std::uint32_t position) const {
  std::vector<Piece> pieces;
  std::size_t offset = 0;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false) {
    if (bytes && pointer)
      pieces.push_back({pointer, offset, bytes, host});
    offset += bytes;
  };
  switch (part.kind) {
    case PartKind::kFrame:
      add(state.meta.data(), kMetaBytes, true);
      add(state.logits.data(), std::size_t{vocab_} * 4, true);
      add(engine_->History(state.session), layout_.history_bytes);
      break;
    case PartKind::kRing:
      add(engine_->Ring(state.session, part.visit, false), layout_.ring_bytes);
      add(engine_->Ring(state.session, part.visit, true), layout_.ring_bytes);
      break;
    case PartKind::kTail: {
      // A frontier on a block boundary has no partial block: zeros.
      const bool partial = position % kValueBlock != 0;
      const std::size_t block = position / kValueBlock;
      for (std::uint32_t head = 0; head < layout_.heads; ++head) {
        auto* base = static_cast<std::byte*>(
            engine_->Full(state.session, part.visit, head, true));
        add(partial ? base + block * layout_.value_chunk_bytes : nullptr,
            layout_.value_chunk_bytes);
      }
      break;
    }
    default:
      throw std::invalid_argument("inkling2: component is not private state");
  }
  return pieces;
}

void* Adapter::RowData(State& state, const Part& part) const {
  if (part.kind != PartKind::kKeys && part.kind != PartKind::kValues)
    throw std::invalid_argument("inkling2: component has no rows");
  return engine_->Full(state.session, part.visit, part.head,
                       part.kind == PartKind::kValues);
}

cache::Completion Adapter::CapturePrivate(const cache::Slot& slot,
                                          cache::ComponentId id,
                                          std::span<std::byte> bytes,
                                          cache::Stream& stream) {
  if (bytes.size() != PartOf(id).descriptor.state_bytes)
    throw std::invalid_argument("inkling2: incorrect private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}

cache::Completion Adapter::CapturePrivatePiece(const cache::Slot& slot,
                                               cache::ComponentId id,
                                               std::size_t offset,
                                               std::span<std::byte> bytes,
                                               cache::Stream& stream) {
  State& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("inkling2: cannot capture an invalid frontier");
  const Part& part = PartOf(id);
  const auto total = part.descriptor.state_bytes;
  if (part.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("inkling2: invalid private piece");
  const std::uint32_t position = engine_->position(state.session);
  std::array<std::byte, kMetaBytes> host_meta{};
  auto pieces = PrivatePieces(state, part, position);
  if (part.kind == PartKind::kFrame) {
    const Meta meta{.position = position,
                    .has_logits = state.has_logits ? 1u : 0u};
    std::memcpy(host_meta.data(), &meta, sizeof(meta));
    pieces.front().pointer = host_meta.data();
  }
  engine_->Synchronize();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges; only bytes no piece covers are
        // cleared.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        for (const Piece& piece : pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const Piece& piece : pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            CheckHip(gufo::hip::CopySnapshotToHost(destination, source,
                                                   end - first,
                                                   transfer.Native()));
        }
      });
}

cache::Completion Adapter::LoadPrivate(cache::Slot& slot, cache::ComponentId id,
                                       std::span<const std::byte> bytes,
                                       cache::Stream& stream) {
  State& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("inkling2: load requires an unfailed restore");
  try {
    if (bytes.size() != PartOf(id).descriptor.state_bytes)
      throw std::invalid_argument("inkling2: incorrect private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion Adapter::LoadPrivatePiece(cache::Slot& slot,
                                            cache::ComponentId id,
                                            std::size_t offset,
                                            std::span<const std::byte> bytes,
                                            cache::Stream& stream) {
  State& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("inkling2: load requires an unfailed restore");
  try {
    const Part& part = PartOf(id);
    const auto total = part.descriptor.state_bytes;
    if (part.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("inkling2: invalid private piece");
    const auto pieces = PrivatePieces(
        state, part,
        static_cast<std::uint32_t>(state.expected.front().valid_rows));
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const Piece& piece : pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              CheckHip(gufo::hip::CopySnapshotToDevice(
                  destination, source, end - first, transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion Adapter::CopyRowsOut(const cache::Slot& slot,
                                       cache::ComponentId id, Rows first,
                                       Rows end, std::span<std::byte> bytes,
                                       cache::Stream& stream) {
  State& state = As(const_cast<cache::Slot&>(slot));
  const Part& part = PartOf(id);
  if (!state.IsValid())
    throw std::logic_error("inkling2: cannot read an invalid slot");
  const auto positions = Positions(slot);
  const auto row = part.descriptor.row_bytes;
  if (!row || first > end || end > positions[id.value - 1].valid_rows ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("inkling2: invalid row capture");
  auto* source = bytes.empty() ? nullptr
                               : static_cast<std::byte*>(RowData(state, part)) +
                                     first * row;
  engine_->Synchronize();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (!bytes.empty())
          CheckHip(gufo::hip::CopySnapshotToHost(bytes.data(), source,
                                                 bytes.size(),
                                                 transfer.Native()));
      });
}

cache::Completion Adapter::CopyRowsIn(cache::Slot& slot, cache::ComponentId id,
                                      Rows first, Rows end,
                                      std::span<const std::byte> bytes,
                                      cache::Stream& stream) {
  State& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("inkling2: load requires an unfailed restore");
  try {
    const Part& part = PartOf(id);
    const auto row = part.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("inkling2: invalid row restore");
    auto* destination =
        bytes.empty() ? nullptr
                      : static_cast<std::byte*>(RowData(state, part)) +
                            first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (!bytes.empty())
            CheckHip(gufo::hip::CopySnapshotToDevice(
                destination, bytes.data(), bytes.size(), transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool Adapter::Validate(cache::Slot& slot,
                       std::span<const cache::ComponentPosition> positions) {
  State& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("inkling2: validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Meta meta{};
  std::memcpy(&meta, state.meta.data(), sizeof(meta));
  const auto target = state.expected.front().valid_rows;
  if (meta.magic != Meta{}.magic || meta.version != 1 ||
      meta.position != target || meta.has_logits > 1)
    return refuse();
  try {
    engine_->SetPosition(state.session, meta.position);
  } catch (const std::exception&) {
    return refuse();
  }
  state.has_logits = meta.has_logits != 0;
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}

bool Adapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    State& state = As(slot);
    if (state.pending)
      return false;
    state.Release();
    engine_->Reset(state.session);
    state.failed = false;
    state.restoring = false;
    state.has_logits = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

// ---- The runner -------------------------------------------------------------

// What the model is writing, read off its control tokens. The server sees
// reasoning as <think>...</think>, a tool call as <tool_call>...</tool_call>
// and the answer as plain text. kHeader is between <|message_model|> and the
// content type: only a name can stand there (the tool's, repeated inside the
// call), and it is not shown.
enum class Output { kNone, kHeader, kThinking, kText, kTool, kOther };
struct Framing {
  Output mode{Output::kNone};
  // An answer message was already written in this response. The model may
  // write several; they are shown as paragraphs.
  bool text_before{false};
};

struct ControlIds {
  std::uint32_t message_model, content_thinking, content_text, end_message,
      invoke_tool;
};

using Lookup = gufo::speculative::PromptLookup<std::uint32_t>;
// A verification step of n drafts costs about as much as 1 + n/2 plain
// steps, so the first copies are short and grow while every draft is kept.
constexpr std::size_t kFirstDrafts = 4;
constexpr std::size_t kMostDrafts = 16;

class State final : public ext::TextRunnerState {
public:
  State(Adapter& adapter, Adapter::State& slot)
      : adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    tokens.clear();
    lookup.Clear();
    output = {};
  }
  [[nodiscard]] Adapter::State& slot() const { return *slot_; }
  [[nodiscard]] std::uint32_t session() const { return slot_->session; }

  // The tokens of the context, as far as the runner has seen them.
  std::vector<std::uint32_t> tokens;
  Framing output;
  // Drafts copied from the context: GUFO's index of the session's tokens,
  // and how many drafts the next verification step may carry.
  Lookup lookup;
  std::size_t draft_budget{kFirstDrafts};
  // The context declares tools and switches thinking off: the token that
  // opens a thinking block is not sampled (see Runner::Render).
  bool no_thinking{false};

private:
  Adapter* adapter_;
  Adapter::State* slot_;
};

State& Require(ext::TextRunnerState& state) {
  auto* mine = dynamic_cast<State*>(&state);
  if (!mine)
    throw std::logic_error("inkling2: foreign runner state");
  return *mine;
}
const State& Require(const ext::TextRunnerState& state) {
  return Require(const_cast<ext::TextRunnerState&>(state));
}

int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoi(v) : fallback;
}

class Runner final : public ext::HipTextModelRunner {
public:
  Runner(std::shared_ptr<const gufo::core::GgufReader> reader, Config config,
         std::uint32_t max_context, std::size_t sessions,
         std::string artifact_fingerprint)
      : reader_(std::move(reader)),
        config_(std::move(config)),
        model_(LoadModel(*reader_, config_)),
        tokenizer_(*reader_),
        max_context_(max_context),
        sessions_(std::max<std::size_t>(sessions, 1)),
        max_batch_(static_cast<std::uint32_t>(
            std::clamp(EnvInt("INKLING2_SERVE_BATCH", 2048), 16, 4096))),
        lookup_(EnvInt("INKLING2_SERVE_LOOKUP", 1) != 0),
        lookup_min_(static_cast<std::size_t>(std::clamp(
            EnvInt("INKLING2_SERVE_LOOKUP_MIN",
                   static_cast<int>(Lookup::kMinMatch)),
            static_cast<int>(Lookup::kKeyTokens),
            static_cast<int>(Lookup::kMaxMatch)))),
        artifact_fingerprint_(std::move(artifact_fingerprint)),
        ids_{tokenizer_.RequireControl("<|message_model|>"),
             tokenizer_.RequireControl("<|content_thinking|>"),
             tokenizer_.RequireControl("<|content_text|>"),
             tokenizer_.RequireControl("<|end_message|>"),
             tokenizer_.RequireControl("<|content_invoke_tool_json|>")} {
    effort_zero_ = {tokenizer_.RequireControl("<|message_system|>"),
                    ids_.content_text};
    tokenizer_.Encode("Thinking effort level: 0", &effort_zero_);
    effort_zero_.push_back(ids_.end_message);
    tool_declare_ = {tokenizer_.RequireControl("<|message_system|>")};
    tokenizer_.Encode("tool_declare", &tool_declare_);
    tool_declare_.push_back(tokenizer_.RequireControl("<|content_xml|>"));
    if (tokenizer_.size() != config_.n_vocab)
      throw std::runtime_error(
          "inkling2: the vocabulary and the output head differ in size");
  }

  // ---- description

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    sampling::TextSamplingValues values;
    values.temperature =
        reader_->GetMetadataFloat32("general.sampling.temp").value_or(1.0F);
    return {
        .model_id = "inkling-2-mini",
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            ext::TextRunnerCapabilities{
                .incremental_prefill = true,
                .final_token_advance_required = true,
                .incremental_text_is_exact = true,
                .multi_token_decode = lookup_,
                .prefix_reuse = true,
            },
        .sampling_defaults =
            sampling::TextSamplingDefaults{values, values, values},
        .scheduling =
            ext::TextSchedulingProfile{
                .pass_tokens = max_batch_,
                .max_layer_groups = 1,
                .max_yield_decode_width = 1,
            },
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = std::nullopt,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = ext::HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = ext::HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans()
      const override {
    return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        tokenizer_.size(), [this](std::uint32_t id) {
          // What a grammar reads for each token. Control tokens are empty,
          // so a grammar never forbids the model's own framing, except the
          // one that opens a tool call.
          return sampling::ConstraintVocabulary::Piece{
              id == ids_.invoke_tool    ? std::string(kToolBlockOpen)
              : tokenizer_.IsControl(id) ? std::string{}
                                         : std::string(tokenizer_.Bytes(id)),
              id == tokenizer_.eos()};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return tokenizer_.size();
  }
  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(kToolDialectId);
  }

  // The reasoning budget works on tokens; the server's parsers on the text
  // Piece() writes for them.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {"<|content_thinking|>", "<|end_message|>", {"<|end_message|>"}};
  }
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    return {"<think>", "</think>"};
  }

  // ---- text

  [[nodiscard]] std::vector<ext::TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return tokenizer_.EncodeWithControl(text);
  }

  [[nodiscard]] std::optional<std::vector<ext::TextRunnerToken>>
  RenderAndTokenize(const ext::ChatRequest& request) const override {
    auto chat = Render(request);
    if (chat.tokens.empty())
      return std::nullopt;
    return std::move(chat.tokens);
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ext::ChatRequest& request) const override {
    return Render(request).text;
  }

  // An answer the prompt opened (thinking off, no tools) is this request's
  // own framing, not part of what a later turn continues from.
  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PreparePrompt(
      const ext::ChatRequest& request) const override {
    auto prepared = ext::TextModelRunner::PreparePrompt(request);
    if (prepared && prepared->tokens.size() > 2 &&
        prepared->tokens.back() == ids_.content_text &&
        prepared->tokens[prepared->tokens.size() - 2] == ids_.message_model)
      prepared->cache_prefix_tokens = prepared->tokens.size() - 2;
    return prepared;
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ext::ChatRequest& request) const override {
    return ThinkingOff(request) ? ext::InitialOutputState::kText
                                : ext::InitialOutputState::kAuto;
  }

  [[nodiscard]] std::string Decode(
      std::span<const ext::TextRunnerToken> tokens) const override {
    std::string text;
    Framing output;
    for (const ext::TextRunnerToken token : tokens)
      text += Piece(token, &output);
    return text;
  }

  // ---- continuation

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    return CreateComponentCacheResourcesForSessions(ledger, ram, sessions_);
  }
  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResourcesForSessions(cache::ResourceLedger& ledger,
                                           std::size_t ram,
                                           std::size_t sessions) const override {
    (void)ram;
    if (!engine_) {
      GpuEngine::Options options;
      options.context = max_context_;
      options.sessions =
          static_cast<std::uint32_t>(std::max(sessions, sessions_));
      options.max_batch = max_batch_;
      engine_ = std::make_shared<GpuEngine>(model_, options);
    }
    return gufo::hip::CreateHipComponentCacheResources(
        std::make_unique<Adapter>(engine_, tokenizer_.size(), Identity()),
        ledger,
        {[] { return ext::HostSnapshotCeilingBytes(); },
         [](std::string_view) {}, [](std::string_view) {}});
  }
  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& mine = dynamic_cast<Adapter&>(adapter);
    return std::make_unique<State>(mine, mine.As(slot));
  }
  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    State& s = Require(state);
    if (engine_->position(s.session()) != position)
      throw std::logic_error("inkling2: restored position does not match");
    if (s.tokens.size() > position)
      s.tokens.resize(position);
    s.lookup.Clear();
    s.no_thinking = false;
    s.output = {};
  }
  [[nodiscard]] bool HasDecodeFrontier(
      const ext::TextRunnerState& state) const override {
    return Require(state).slot().has_logits;
  }
  void PreparePrefixReuse(
      ext::TextRunnerState& state,
      std::span<const ext::TextRunnerToken> prefix) const override {
    State& s = Require(state);
    if (engine_->position(s.session()) != prefix.size())
      throw std::logic_error("inkling2: reused prefix does not match");
    s.tokens.assign(prefix.begin(), prefix.end());
    s.output = OutputAfter(prefix);
    s.no_thinking = NoThinking(prefix);
    if (s.slot().has_logits)
      Mask(s, s.slot().logits.data());
    s.lookup.BeginRequest();
    s.draft_budget = kFirstDrafts;
  }

  // ---- execution

  [[nodiscard]] ext::TextPrefillStep Prefill(
      ext::TextRunnerState& state, std::span<const ext::TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    State& s = Require(state);
    if (offset != engine_->position(s.session()))
      throw std::logic_error("inkling2: prefill offset does not match");
    if (offset >= prompt.size())
      throw std::logic_error("inkling2: prefill has no remaining input");
    if (prompt.size() > max_context_)
      throw std::invalid_argument("inkling2: the prompt exceeds the context");
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, max_batch_});
    const std::size_t end = offset + consumed;
    const bool final = end == prompt.size();
    s.slot().BeforeWrite(static_cast<std::uint32_t>(offset),
                         static_cast<std::uint32_t>(end));
    std::vector<GpuEngine::Row> rows;
    rows.reserve(consumed);
    for (std::size_t i = offset; i < end; ++i) {
      if (prompt[i] >= tokenizer_.size())
        throw std::invalid_argument("inkling2: token outside the vocabulary");
      rows.push_back({s.session(), prompt[i]});
    }
    s.slot().has_logits = false;
    engine_->Step(rows,
                  final ? GpuEngine::Head::kLast : GpuEngine::Head::kNone);
    s.no_thinking = NoThinking(prompt.first(end));
    if (final)
      KeepLogits(&s);
    s.tokens.assign(prompt.begin(), prompt.begin() + end);
    s.output = OutputAfter(prompt.first(end));
    s.lookup.BeginRequest();
    s.draft_budget = kFirstDrafts;
    return {.consumed_tokens = consumed, .decode_ready = final};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    State& s = Require(state);
    if (engine_->position(s.session()) >= max_context_)
      return {.stop = true, .piece = {}};
    if (!s.slot().has_logits)
      throw std::logic_error("inkling2: no logits at this position");
    const std::span<const float> logits(s.slot().logits);
    const std::uint32_t token = sampler.Sample(logits);
    const bool stop = s.stop_at_eos() && token == tokenizer_.eos();
    Framing output = s.output;
    return {.stop = stop,
            .token = token,
            .piece = stop ? std::string{} : Piece(token, &output),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(ext::TextRunnerState& state,
               ext::TextRunnerToken token) const override {
    State& s = Require(state);
    const std::uint32_t position = engine_->position(s.session());
    if (position >= max_context_)
      throw std::runtime_error("inkling2: the context is full");
    if (token >= tokenizer_.size())
      throw std::invalid_argument("inkling2: token outside the vocabulary");
    s.slot().BeforeWrite(position, position + 1);
    s.slot().has_logits = false;
    engine_->Step({{s.session(), token}}, GpuEngine::Head::kLast);
    KeepLogits(&s);
    if (s.tokens.size() == position)
      s.tokens.push_back(token);
    (void)Piece(token, &s.output);
  }

  // One step, with drafts copied from the context when the last tokens
  // repeat an earlier passage (GUFO's prompt lookup decides when). The model
  // verifies the drafts in the same step: at each drafted position the token
  // is drawn from the model's own distribution as always, and the draft is
  // kept only if it is the token drawn. The output is therefore distributed
  // exactly as without drafts; a wrong draft only costs the larger step.
  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    State& s = Require(state);
    const auto& config = sampler.config();
    const std::uint32_t position = engine_->position(s.session());
    if (!lookup_ || max_tokens < 2 || config.logprobs || config.constraint ||
        config.reasoning_budget || config.penalties_enabled() ||
        sampler.ForcedToken() || s.tokens.size() != position ||
        position + 2 > max_context_ || !s.slot().has_logits)
      return ext::TextModelRunner::DecodeStep(state, max_tokens, sampler);
    const bool sampled = config.uses_random_sampling();
    s.lookup.Extend(s.tokens, sampled);
    if (!s.lookup.Step(position))
      return ext::TextModelRunner::DecodeStep(state, max_tokens, sampler);

    ext::TextDecodeStep step;
    auto first = SelectNext(state, sampler);
    if (first.stop) {
      step.stop = true;
      step.terminal_logprobs = std::move(first.logprobs);
      return step;
    }
    const std::uint32_t anchor = first.token;
    const std::uint32_t tail[1] = {anchor};
    const auto match = s.lookup.Find(
        s.tokens, tail,
        sampled ? std::min(2 * lookup_min_, Lookup::kMaxMatch) : lookup_min_);
    std::size_t drafts = 0;
    if (match.length != 0 && match.start < s.lookup.IndexedTokens())
      drafts = std::min<std::size_t>(
          {s.draft_budget, max_tokens - 1,
           std::size_t{max_context_} - position - 1, std::size_t{max_batch_} - 1,
           s.lookup.IndexedTokens() - match.start});
    step.selections.push_back(std::move(first));
    if (drafts == 0) {
      Advance(state, anchor);
      return step;
    }

    std::vector<GpuEngine::Row> rows;
    rows.push_back({s.session(), anchor});
    for (std::size_t i = 0; i < drafts; ++i)
      rows.push_back({s.session(), s.tokens[match.start + i]});
    s.slot().BeforeWrite(position,
                         position + static_cast<std::uint32_t>(rows.size()));
    s.slot().has_logits = false;
    engine_->Step(rows, GpuEngine::Head::kAll, true);
    s.tokens.push_back(anchor);
    (void)Piece(anchor, &s.output);
    std::size_t kept = 0;
    for (; kept < drafts; ++kept) {
      std::vector<float> logits = engine_->capped_logits(kept);
      Mask(s, logits.data());
      const std::uint32_t token =
          sampler.Sample(std::span<const float>(logits.data(), tokenizer_.size()));
      if (token != rows[kept + 1].token) {
        // The token drawn stands; the next step starts from it.
        sampler.DeferSample(token);
        break;
      }
      if (s.stop_at_eos() && token == tokenizer_.eos()) {
        step.stop = true;
        break;
      }
      step.selections.push_back(
          {.stop = false, .token = token, .piece = Piece(token, &s.output)});
      s.tokens.push_back(token);
    }
    engine_->Accept(s.session(), static_cast<std::uint32_t>(1 + kept));
    {
      const std::vector<float> logits = engine_->capped_logits(kept);
      std::copy(logits.begin(), logits.begin() + tokenizer_.size(),
                s.slot().logits.begin());
      Mask(s, s.slot().logits.data());
      s.slot().has_logits = true;
    }
    s.lookup.Observe(match.start, drafts, kept);
    s.draft_budget = kept == drafts ? std::min(s.draft_budget * 2, kMostDrafts)
                                    : kFirstDrafts;
    step.draft_rounds = 1;
    step.lookup_tokens = drafts;
    step.lookup_accepted_tokens = kept;
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    return engine_->position(Require(state).session());
  }

private:
  void KeepLogits(State* s) const {
    const std::vector<float> capped = engine_->capped_logits(0);
    std::copy(capped.begin(), capped.begin() + tokenizer_.size(),
              s->slot().logits.begin());
    Mask(*s, s->slot().logits.data());
    s->slot().has_logits = true;
  }

  // The text the server sees for a token, and what the model is writing
  // after it.
  [[nodiscard]] std::string Piece(std::uint32_t token, Framing* output) const {
    if (!tokenizer_.IsControl(token))
      return output->mode == Output::kHeader
                 ? std::string{}
                 : std::string(tokenizer_.Bytes(token));
    if (token == ids_.message_model) {
      output->mode = Output::kHeader;
      return {};
    }
    if (token == ids_.invoke_tool) {
      output->mode = Output::kTool;
      return std::string(kToolBlockOpen);
    }
    if (token == ids_.content_thinking) {
      output->mode = Output::kThinking;
      return "<think>";
    }
    if (token == ids_.content_text) {
      output->mode = Output::kText;
      const bool again = output->text_before;
      output->text_before = true;
      return again ? "\n\n" : std::string{};
    }
    if (token == ids_.end_message) {
      const Output was = output->mode;
      output->mode = Output::kNone;
      return was == Output::kThinking ? "</think>"
             : was == Output::kTool   ? std::string(kToolBlockClose)
                                      : std::string{};
    }
    if (token == tokenizer_.eos())
      return {};
    output->mode = Output::kOther;
    return std::string(tokenizer_.Bytes(token));
  }

  [[nodiscard]] RenderedChat Render(const ext::ChatRequest& request) const {
    std::vector<ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& in : request.messages) {
      for (const auto& image : in.images) {
        (void)image;
        throw std::invalid_argument("model does not support image input");
      }
      ChatMessage message;
      switch (in.role) {
        case ext::ChatRole::kSystem:
        case ext::ChatRole::kDeveloper:
          message.role = ChatMessage::Role::kSystem;
          break;
        case ext::ChatRole::kAssistant:
          message.role = ChatMessage::Role::kAssistant;
          break;
        case ext::ChatRole::kTool:
          message.role = ChatMessage::Role::kTool;
          break;
        case ext::ChatRole::kUser:
          message.role = ChatMessage::Role::kUser;
          break;
      }
      message.content = in.content + in.framing_suffix;
      message.reasoning = in.thought;
      message.name = in.name;
      message.tool_call_id = in.tool_call_id;
      for (const auto& call : in.tool_calls) {
        ChatToolCall out;
        out.id = call.id;
        out.name = call.name;
        for (const auto& argument : call.arguments)
          out.arguments.push_back(
              {argument.name, argument.value, argument.is_string});
        message.tool_calls.push_back(std::move(out));
      }
      messages.push_back(std::move(message));
    }
    std::vector<ChatTool> tools;
    if (request.tool_choice != ext::ChatRequest::ToolChoice::kNone)
      for (const auto& tool : request.tools)
        tools.push_back(
            {tool.name, tool.description, tool.parameters_json, "function"});
    ChatOptions options;
    const auto& reasoning = request.reasoning;
    if (reasoning.enabled.has_value() && !*reasoning.enabled) {
      options.effort = 0.0;
    } else if (reasoning.effort.has_value()) {
      switch (*reasoning.effort) {
        case gufo::ReasoningEffort::kMinimal:
          options.effort = 0.1;
          break;
        case gufo::ReasoningEffort::kLow:
          options.effort = 0.2;
          break;
        case gufo::ReasoningEffort::kMedium:
          options.effort = 0.7;
          break;
        case gufo::ReasoningEffort::kHigh:
          options.effort = 0.9;
          break;
        case gufo::ReasoningEffort::kXHigh:
        case gufo::ReasoningEffort::kMax:
          options.effort = 0.99;
          break;
      }
    }
    RenderedChat chat = RenderChat(tokenizer_, messages, tools, options);
    // Thinking off. The template writes "Thinking effort level: 0" for it,
    // and the model may still open a thinking block. Two measures, each for
    // the case where it was seen to work:
    //  - Without tools, the prompt opens the answer itself, so the model
    //    continues with the answer.
    //  - With tools, the model must stay free to start with a call, so the
    //    prompt is left alone and only the token that opens a thinking block
    //    is taken out before sampling (NoThinking). Used without tools, that
    //    made the model write its deliberation into the answer.
    if (ThinkingOff(request) && tools.empty()) {
      chat.tokens.push_back(ids_.message_model);
      chat.tokens.push_back(ids_.content_text);
      chat.text += "<|message_model|><|content_text|>";
    }
    return chat;
  }

  [[nodiscard]] static bool Declares(
      std::span<const ext::TextRunnerToken> tokens,
      const std::vector<std::uint32_t>& line) {
    return std::search(tokens.begin(), tokens.end(), line.begin(),
                       line.end()) != tokens.end();
  }
  // The context declares tools and switches thinking off.
  [[nodiscard]] bool NoThinking(
      std::span<const ext::TextRunnerToken> tokens) const {
    return Declares(tokens, effort_zero_) && Declares(tokens, tool_declare_);
  }
  void Mask(const State& s, float* logits) const {
    if (s.no_thinking)
      logits[ids_.content_thinking] = -1e30F;
  }

  [[nodiscard]] static bool ThinkingOff(const ext::ChatRequest& request) {
    return request.reasoning.enabled.has_value() && !*request.reasoning.enabled;
  }

  // What the model is writing when the context ends in `tokens`.
  [[nodiscard]] Framing OutputAfter(
      std::span<const ext::TextRunnerToken> tokens) const {
    if (tokens.empty())
      return {};
    if (tokens.back() == ids_.content_text)
      return {Output::kText, true};
    if (tokens.back() == ids_.content_thinking)
      return {Output::kThinking, false};
    if (tokens.back() == ids_.message_model)
      return {Output::kHeader, false};
    return {};
  }

  [[nodiscard]] cache::Identity Identity() const {
    std::ostringstream identity;
    if (artifact_fingerprint_.empty()) {
      identity << "schema=gufo-process-text-continuation-v1\n"
               << "model_instance=" << static_cast<const void*>(this) << '\n';
    } else {
      identity << "schema=gufo-text-continuation-v2\n"
               << "model_kind=inkling2_loop\n"
               << "artifact_id=" << ext::kGgufIdentityScheme << ':'
               << artifact_fingerprint_ << '\n'
               << "tokenizer=inkling2-bpe-v1\n"
               << "chat_template=inkling2-cpp-v1\n";
    }
    identity << "state_abi=" << kStateAbi << '\n'
             << "payload_layout=inkling2-rings-whole-v1\n"
             << "context_tokens=" << max_context_ << '\n'
             << "prefill_rows=" << max_batch_ << '\n';
    const std::string canonical = identity.str();
    return {canonical.begin(), canonical.end()};
  }

  std::shared_ptr<const gufo::core::GgufReader> reader_;
  Config config_;
  Model model_;
  Tokenizer tokenizer_;
  std::uint32_t max_context_;
  std::size_t sessions_;
  std::uint32_t max_batch_;
  bool lookup_;
  // Tokens of the output that must repeat an earlier passage before a copy
  // is drafted from it (greedy requests; sampled ones need twice as many).
  std::size_t lookup_min_;
  std::string artifact_fingerprint_;
  ControlIds ids_;
  std::vector<std::uint32_t> effort_zero_, tool_declare_;
  mutable std::shared_ptr<GpuEngine> engine_;
};

}  // namespace

std::shared_ptr<gufo::ext::TextModelRunner> CreateTextRunner(
    const gufo::ext::TextRunnerOptions& options, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error)
      *error = std::move(message);
    return nullptr;
  };
  try {
    std::shared_ptr<const gufo::core::GgufReader> reader = options.reader;
    if (!reader) {
      std::string load_error;
      auto owner = gufo::core::GgufReader::OpenFile(
          options.model_path, &load_error, {.allow_unsized_tensors = true});
      if (!owner)
        return fail("Failed to open GGUF: " + load_error);
      reader = std::shared_ptr<const gufo::core::GgufReader>(std::move(owner));
    }
    if (!options.vision_model_path.empty())
      return fail("Inkling 2 Mini: image input is not supported yet");
    if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled)
      return fail("Inkling 2 Mini: no speculative backend is supported");
    Config config;
    std::string config_error;
    if (!LoadConfig(*reader, &config, &config_error))
      return fail(config_error);
    // The file allows 1,048,576 tokens; a session's caches take 33 KB per
    // token of context, so the default is a quarter of that.
    std::uint32_t max_context = options.max_context;
    if (max_context == 0)
      max_context = std::min<std::uint32_t>(config.n_ctx, 262144);
    if (max_context < 2 || max_context > config.n_ctx)
      return fail("Inkling 2 Mini: unsupported context size");
    std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
    if (!options.disk_cache.directory.empty() && fingerprint.empty())
      fingerprint = gufo::core::GgufIdentityHex(*reader);
    return std::make_shared<Runner>(std::move(reader), std::move(config),
                                    max_context, options.session_count,
                                    std::move(fingerprint));
  } catch (const std::exception& e) {
    return fail(std::string("Inkling 2 Mini: ") + e.what());
  }
}

std::optional<gufo::sampling::TextSamplingDefaults> SamplingDefaults(
    const gufo::core::GgufReader& reader) {
  gufo::sampling::TextSamplingValues values;
  values.temperature =
      reader.GetMetadataFloat32("general.sampling.temp").value_or(1.0F);
  return gufo::sampling::TextSamplingDefaults{values, values, values};
}

}  // namespace inkling2
