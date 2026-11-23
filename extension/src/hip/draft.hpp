#pragma once

// Kernels of the DSpark drafter (docs/DSPARK.md) on the device.
//
// Two groups. The first runs inside every step of the main model: it turns
// the four hidden states of each row into the drafter's context keys and
// values and stores them in the session's rings (DraftMix, two products by
// Gemm, DraftStore). The second is a draft: a block of 8 rows through the
// drafter's three layers, then the recurrent head, one slot after the other,
// without a host round trip in between.
//
// The drafter only proposes; the main model verifies every proposal. So this
// arithmetic need not match any reference bit for bit.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "decode.hpp"

namespace inkling2::gpu {

inline constexpr int kDraftLayers = 3;
inline constexpr int kDraftBlock = 8;   // rows of a draft: the anchor + 7
inline constexpr int kDraftFF = 9216;
inline constexpr int kDraftRank = 256;  // Markov rank
inline constexpr int kDraftJoint = 3 * kDraftRank;
inline constexpr int kDraftJointIn = 2 * kDraftRank + 2 * kE;
inline constexpr int kDraftWindow = 1024;  // layers 0 and 2

/// Where one batch row's context keys and values go. A ring holds `cap`
/// positions: position p of key/value head g is at
/// ring[((p % cap) * kNKV + g) * kHD], FP16.
struct DraftRow {
  std::uint16_t* k[kDraftLayers];
  std::uint16_t* v[kDraftLayers];
  std::int32_t cap[kDraftLayers];
  std::int32_t store[kDraftLayers];  // 0: this row is not stored in the layer
  std::int32_t pos;
  std::int32_t pad;
};

/// Per row: the three layers' context inputs and the head's feature,
/// out[b * 4 * kE + r * kE], each the fused hidden states, normalized.
struct DraftMixArgs {
  const float* feat;        // three blocks of rows (f5, f23, f41), stride kE
  std::size_t feat_block;   // floats from one block to the next
  const float* feat_final;  // f48, stride kE
  float fuse[kDraftLayers + 1][4];
  const float* norm[kDraftLayers + 1];
  float eps;
  float* out;
  int batch;
};
void DraftMix(const DraftMixArgs& args, hipStream_t stream);

/// Normalizes and rotates the context keys and stores keys and values.
struct DraftStoreArgs {
  const float* k_raw;  // stride kDraftLayers * kHK
  const float* v_raw;
  const float* k_norm[kDraftLayers];  // kHD
  const DraftRow* rows;
  float eps;
  int batch;
};
void DraftStore(const DraftStoreArgs& args, hipStream_t stream);

/// The block's first row is the anchor's embedding, the others the mask
/// vector. Also clears the recurrent state and the flags of the last draft.
struct DraftInitArgs {
  const std::uint16_t* tok_embd;  // BF16
  const float* tok_norm;
  float tok_eps;
  const float* mask;
  std::uint32_t anchor;
  float* x;                // stride kE
  float* state;            // kDraftRank
  std::uint32_t* tokens;   // kDraftBlock; [0] = anchor
  std::int32_t* alive;
};
void DraftInit(const DraftInitArgs& args, hipStream_t stream);

/// x += add (when set), then dst = rms_norm(x) * w, for kDraftBlock rows.
struct DraftResNormArgs {
  float* x;          // stride kE
  const float* add;  // stride kE or null
  const float* w;
  float eps;
  float* dst;        // stride kE
};
void DraftResNorm(const DraftResNormArgs& args, hipStream_t stream);

/// Per-head norm and rotation of the block's queries and keys, in place.
struct DraftQkArgs {
  float* q;  // stride kHQ
  float* k;  // stride kHK
  const float* q_norm;
  const float* k_norm;
  float eps;
  int pos0;
};
void DraftQk(const DraftQkArgs& args, hipStream_t stream);

/// Every block row attends over ring positions [first, pos0) and over the
/// block's own eight rows, one softmax over both.
struct DraftAttnArgs {
  const float* q;      // stride kHQ
  const float* k_blk;  // stride kHK
  const float* v_blk;
  const std::uint16_t* k_ring;
  const std::uint16_t* v_ring;
  int cap, first, pos0;
  float* scratch;  // kDraftBlock * kNH * (pos0 - first + kDraftBlock)
  float* out;      // stride kHQ
};
void DraftAttn(const DraftAttnArgs& args, hipStream_t stream);

/// gate = silu(gate) * up over kDraftBlock * kDraftFF values.
void DraftSilu(float* gate, const float* up, hipStream_t stream);

/// One slot of the recurrent head, slot = 1 .. 7. Reads the previous token
/// from tokens[slot - 1]; writes tokens[slot] and confidence[slot], and
/// clears *alive when the confidence is below min_confidence, which turns
/// the later slots into no-ops. `logits` keeps the slot's logits (before the
/// soft cap), for whoever needs the whole distribution afterwards.
struct DraftSlotArgs {
  int slot;
  const float* joint;       // kDraftJoint rows of kDraftJointIn, F32
  const float* joint_bias;
  const std::uint16_t* markov;  // BF16, rows of kDraftRank
  const float* h;           // stride kE: the block's normalized outputs
  const float* f_rnn;       // kE
  float* state;             // kDraftRank
  float* j_raw;             // kDraftJoint
  float* latent;            // kDraftRank
  const float* base;        // stride vocab_stride: output . h per slot (slot - 1)
  // output . hidden_out as FP16, one column of the product after the other:
  // m[c * vocab_stride + v]; and output . bias.
  const std::uint16_t* m;
  const float* m_bias;
  float* logits;            // vocab
  float* block_max;         // one per 256 vocabulary entries
  std::int32_t* block_arg;
  float* block_sum;
  float* top;               // 1
  std::uint32_t* tokens;
  float* confidence;
  std::int32_t* alive;
  float softcap;
  float min_confidence;
  int vocab, vocab_stride;
  // When set, entry v of the head's matrices is token ids[v] (a shortlist of
  // the vocabulary); otherwise entry v is token v.
  const std::uint32_t* ids;
  // temperature > 0: the slot's token is drawn from softmax(capped logits /
  // temperature) over the entries (Gumbel noise from `seed`, the slot and
  // the entry) instead of being the most likely one, and probability[slot]
  // is the chance it had. confidence[slot] is the largest chance either way.
  float temperature;
  std::uint32_t seed;
  float* probability;
  float* block_top;  // one per 256 entries: the largest logit
  float* picked;     // 1: the logit of the slot's token
};
void DraftSlot(const DraftSlotArgs& args, hipStream_t stream);

/// One entry of the shortlist to (re)fill: the head's matrices for `token`
/// go to entry `slot`.
struct DraftListEntry {
  std::int32_t slot;
  std::uint32_t token;
};
struct DraftListArgs {
  const DraftListEntry* entries;
  int count;
  const std::uint16_t* output;  // FP16 output matrix, rows of kE
  const std::uint16_t* m;       // whole vocabulary, columns `vocab_stride` apart
  const float* m_bias;
  int vocab_stride;
  std::uint16_t* list_output;   // `slots` rows of kE
  std::uint16_t* list_m;        // columns `slots` apart
  float* list_bias;
  std::uint32_t* list_ids;
  int slots;
};
void DraftListFill(const DraftListArgs& args, hipStream_t stream);

}  // namespace inkling2::gpu
