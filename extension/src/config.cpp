#include "config.hpp"

#include <cmath>
#include <initializer_list>
#include <string_view>
#include <variant>

namespace inkling2 {
namespace {

using gufo::core::GgufReader;
using gufo::core::GgufTensorInfo;

constexpr std::string_view kArch = "inkling2_loop";

std::string Key(std::string_view suffix) {
  return std::string(kArch) + "." + std::string(suffix);
}

std::vector<std::uint64_t> UintArray(const GgufReader& reader,
                                     const std::string& key) {
  const auto* value = reader.FindMetadata(key);
  if (!value)
    return {};
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&value->value))
    return *u;
  if (const auto* i = std::get_if<std::vector<std::int64_t>>(&value->value))
    return {i->begin(), i->end()};
  return {};
}

// A value stored either as one scalar or as one entry per layer; every entry
// must agree.
bool UniformUint(const GgufReader& reader, const std::string& key,
                 std::uint32_t* out) {
  if (const auto scalar = reader.GetMetadataUint32(key)) {
    *out = *scalar;
    return true;
  }
  const auto values = UintArray(reader, key);
  if (values.empty())
    return false;
  for (const auto v : values)
    if (v != values.front())
      return false;
  *out = static_cast<std::uint32_t>(values.front());
  return true;
}

bool Need(bool ok, std::string_view what, std::string* error) {
  if (!ok && error && error->empty())
    *error = "inkling2: missing or invalid " + std::string(what);
  return ok;
}

std::string TypeName(const GgufTensorInfo& tensor) {
  if (static_cast<std::uint16_t>(tensor.type) == kGgmlTypeNvfp4)
    return "NVFP4";
  return std::string(gufo::core::ToString(tensor.type));
}

}  // namespace

bool Probe(const GgufReader& reader) {
  if (reader.GetMetadataString("general.architecture").value_or("") != kArch)
    return false;
  // Inkling 2 Mini only: 24 layers of width 2304 with 128 experts, and the
  // loop core [6, 18) run three times.
  return reader.GetMetadataUint32(Key("block_count")) == 24u &&
         reader.GetMetadataUint32(Key("embedding_length")) == 2304u &&
         reader.GetMetadataUint32(Key("expert_count")) == 128u &&
         reader.GetMetadataUint32(Key("loop.start")) == 6u &&
         reader.GetMetadataUint32(Key("loop.end")) == 18u &&
         reader.GetMetadataUint32(Key("loop.count")) == 3u;
}

bool LoadConfig(const GgufReader& reader, Config* c, std::string* error) {
  if (error)
    error->clear();
  auto u32 = [&](std::string_view suffix, std::uint32_t* out) {
    const auto v = reader.GetMetadataUint32(Key(suffix));
    if (v)
      *out = *v;
    return Need(v.has_value() && *v != 0, suffix, error);
  };
  auto f32 = [&](std::string_view suffix, float* out, bool positive = true) {
    const auto v = reader.GetMetadataFloat32(Key(suffix));
    if (v)
      *out = *v;
    return Need(v.has_value() && std::isfinite(*v) && (!positive || *v > 0),
                suffix, error);
  };

  bool ok = u32("block_count", &c->n_layer) &&
            u32("embedding_length", &c->n_embd) &&
            u32("context_length", &c->n_ctx) &&
            u32("expert_count", &c->n_expert) &&
            u32("vocab_size", &c->n_vocab_padded) &&
            u32("vocab_size_unpadded", &c->n_vocab) &&
            u32("attention.key_length", &c->head_dim) &&
            u32("attention.sliding_window", &c->n_swa) &&
            u32("attention.relative_length", &c->n_rel) &&
            u32("attention.relative_extent", &c->n_rel_extent) &&
            u32("attention.log_scale_floor", &c->log_scale_floor) &&
            u32("shortconv.l_cache", &c->n_conv) &&
            u32("attn_res.block_size", &c->res_block) &&
            u32("loop.start", &c->loop_start) &&
            u32("loop.end", &c->loop_end) &&
            u32("loop.count", &c->loop_count) &&
            u32("loop.gate_rank", &c->loop_gate_rank) &&
            f32("attention.layer_norm_rms_epsilon", &c->norm_eps) &&
            f32("attention.qk_clip", &c->qk_clip) &&
            f32("attention.relative_clip", &c->rel_clip) &&
            f32("attention.log_scale_alpha", &c->log_scale_alpha, false) &&
            f32("activation.situ_beta", &c->situ_beta) &&
            f32("activation.situ_linear_beta", &c->situ_linear_beta);
  if (!ok)
    return false;

  ok = Need(UniformUint(reader, Key("attention.head_count"), &c->n_head),
            "attention.head_count", error) &&
       Need(UniformUint(reader, Key("attention.head_count_kv"), &c->n_head_kv),
            "attention.head_count_kv", error) &&
       Need(UniformUint(reader, Key("expert_used_count"), &c->n_expert_used),
            "expert_used_count", error) &&
       Need(UniformUint(reader, Key("expert_feed_forward_length"),
                        &c->n_ff_exp),
            "expert_feed_forward_length", error);
  if (!ok)
    return false;

  if (const auto v = reader.GetMetadataFloat32(Key("expert_weights_scale")))
    c->expert_weights_scale = *v;
  if (const auto v = reader.GetMetadataFloat32(Key("final_logit_softcapping")))
    c->logit_softcap = *v;

  // A layer is sliding-window when it has the single shared input norm; the
  // full-attention layers carry separate Q, K and V input norms instead.
  c->is_swa.assign(c->n_layer, false);
  for (std::uint32_t il = 0; il < c->n_layer; ++il) {
    const std::string blk = "blk." + std::to_string(il) + ".";
    const bool swa = reader.HasTensor(blk + "attn_norm.weight");
    const bool full = reader.HasTensor(blk + "attn_q_inp_norm.weight");
    if (!Need(swa != full, blk + "attention input norm", error))
      return false;
    c->is_swa[il] = swa;
  }

  const bool loop_ok =
      c->loop_start > 0 && c->loop_start < c->loop_end &&
      c->loop_end <= c->n_layer && c->loop_count >= 2 &&
      c->loop_start % c->res_block == 0 && c->loop_end % c->res_block == 0;
  return Need(loop_ok, "loop core", error) &&
         Need(c->n_head_kv != 0 && c->n_head % c->n_head_kv == 0,
              "attention head counts", error) &&
         Need(c->n_expert_used <= c->n_expert, "expert counts", error) &&
         Need(c->n_vocab <= c->n_vocab_padded, "vocabulary sizes", error) &&
         Need(c->n_conv >= 2 && c->n_conv <= 64, "shortconv.l_cache", error);
}

bool CheckTensors(const GgufReader& reader, const Config& c,
                  Inventory* inventory, std::string* error) {
  if (error)
    error->clear();
  using Dims = std::initializer_list<std::uint64_t>;
  auto want = [&](const std::string& name, Dims dims) {
    const GgufTensorInfo* t = reader.FindTensor(name);
    if (!t)
      return Need(false, "tensor " + name, error);
    // GGUF drops trailing dimensions of 1, so compare without them.
    std::vector<std::uint64_t> expected(dims), actual(t->dimensions);
    while (expected.size() > 1 && expected.back() == 1)
      expected.pop_back();
    while (actual.size() > 1 && actual.back() == 1)
      actual.pop_back();
    return Need(expected == actual, "shape of " + name, error);
  };

  const std::uint64_t e = c.n_embd, hq = std::uint64_t{c.head_dim} * c.n_head,
                      hk = std::uint64_t{c.head_dim} * c.n_head_kv;
  if (!want("token_embd.weight", {e, c.n_vocab_padded}) ||
      !want("token_embd_norm.weight", {e}) ||
      !want("output.weight", {e, c.n_vocab_padded}) ||
      !want("output_norm.weight", {e}) || !want("output_res_query.weight", {e}))
    return false;

  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    const bool swa = c.is_swa[il];
    const std::uint64_t extent = swa ? c.n_swa : c.n_rel_extent;
    bool ok = want(b + "attn_q.weight", {e, hq}) &&
              want(b + "attn_k.weight", {e, hk}) &&
              want(b + "attn_v.weight", {e, hk}) &&
              want(b + "attn_output.weight", {hq, e}) &&
              want(b + "attn_gate.weight", {e, hq}) &&
              want(b + "attn_q_clip.weight", {c.head_dim, c.n_head}) &&
              want(b + "attn_sink_proj.weight", {e, c.n_head}) &&
              want(b + "attn_rel.weight",
                   {e, std::uint64_t{c.n_rel} * c.n_head}) &&
              want(b + "attn_rel_bias.weight", {c.n_rel, extent}) &&
              want(b + "attn_rel_clip.weight", {c.n_rel, c.n_head}) &&
              want(b + "attn_k_conv.weight", {c.n_conv, hk}) &&
              want(b + "attn_v_conv.weight", {c.n_conv, hk}) &&
              want(b + "attn_post_conv.weight", {c.n_conv, e}) &&
              want(b + "ffn_post_conv.weight", {c.n_conv, e}) &&
              want(b + "ffn_res_query.weight", {e}) &&
              want(b + "ffn_norm.weight", {e}) &&
              want(b + "ffn_gate_inp.weight", {e, c.n_expert}) &&
              want(b + "ffn_gate_up_exps.weight",
                   {e, 2ull * c.n_ff_exp, c.n_expert}) &&
              want(b + "ffn_down_exps.weight", {c.n_ff_exp, e, c.n_expert});
    if (ok && swa) {
      ok = want(b + "attn_norm.weight", {e});
    } else if (ok) {
      const std::uint64_t taps = std::uint64_t{c.n_conv} * c.n_head_kv;
      ok = want(b + "attn_q_inp_norm.weight", {e}) &&
           want(b + "attn_k_inp_norm.weight", {e}) &&
           want(b + "attn_v_inp_norm.weight", {e}) &&
           want(b + "attn_k_conv_proj.weight", {e, taps}) &&
           want(b + "attn_v_conv_proj.weight", {e, taps});
    }
    if (ok && il > 0)
      ok = want(b + "attn_res_query.weight", {e, swa ? 1ull : 3ull});
    if (ok && il >= c.loop_start && il < c.loop_end) {
      ok = want(b + "loop_msg_query.weight", {e, 4});
      for (const char* ctrl : {"q", "k", "v", "ffn"}) {
        const std::string p = b + "loop_" + ctrl + "_ctrl_";
        ok = ok && want(p + "down.weight", {2 * e, c.loop_gate_rank}) &&
             want(p + "gain.weight", {c.loop_gate_rank, e}) &&
             want(p + "forget.weight", {c.loop_gate_rank, e});
      }
    }
    if (!ok)
      return false;
  }

  if (inventory) {
    *inventory = {};
    for (const GgufTensorInfo& t : reader.GetTensors()) {
      ++inventory->tensors;
      inventory->bytes += t.size_bytes;
      auto& slot = inventory->by_type[TypeName(t)];
      ++slot.first;
      slot.second += t.size_bytes;
      if (static_cast<std::uint16_t>(t.type) == kGgmlTypeNvfp4 &&
          t.name.find("_exps.weight") != std::string_view::npos)
        ++inventory->nvfp4_expert_tensors;
    }
  }
  return true;
}

}  // namespace inkling2
