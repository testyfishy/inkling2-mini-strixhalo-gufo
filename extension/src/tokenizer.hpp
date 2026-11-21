#pragma once

#include <gufo/ext/gguf_reader.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace inkling2 {

/// The model's tokenizer, read from the GGUF file: byte-level BPE over pieces
/// cut by the splitting rule of the model's tokenizer.json (the GPT-4o rule),
/// with `ignore_merges` (a piece that is itself a vocabulary entry is one
/// token). No normalization, no token added at either end.
class Tokenizer {
public:
  /// Throws std::runtime_error when the file has no usable vocabulary.
  explicit Tokenizer(const gufo::core::GgufReader& reader);

  /// Appends the tokens of plain text. Never produces a control token: the
  /// spelling of one, such as "<|end_message|>", is encoded as ordinary text.
  void Encode(std::string_view text, std::vector<std::uint32_t>* out) const;

  /// Text in which the spellings of control tokens stand for those tokens.
  [[nodiscard]] std::vector<std::uint32_t> EncodeWithControl(
      std::string_view text) const;

  /// The bytes a token stands for; for a control token, its spelling. Tokens
  /// may end inside a UTF-8 sequence.
  [[nodiscard]] std::string_view Bytes(std::uint32_t id) const {
    return id < bytes_.size() ? std::string_view(bytes_[id])
                              : std::string_view();
  }
  [[nodiscard]] bool IsControl(std::uint32_t id) const {
    return id < control_.size() && control_[id];
  }
  [[nodiscard]] std::optional<std::uint32_t> Control(
      std::string_view spelling) const;
  /// Like Control, but throws std::runtime_error when the file lacks it.
  [[nodiscard]] std::uint32_t RequireControl(std::string_view spelling) const;

  [[nodiscard]] std::uint32_t size() const {
    return static_cast<std::uint32_t>(bytes_.size());
  }
  [[nodiscard]] std::uint32_t eos() const { return eos_; }

  /// The pieces the splitting rule cuts `text` into, as byte offsets of their
  /// ends. For tests.
  static void Split(std::string_view text, std::vector<std::size_t>* ends);

private:
  struct Merge {
    std::uint32_t rank;
    std::uint32_t id;
  };
  void EncodePiece(std::string_view piece,
                   std::vector<std::uint32_t>* out) const;

  std::vector<std::string> bytes_;
  std::vector<bool> control_;
  std::unordered_map<std::string_view, std::uint32_t> vocab_;
  std::unordered_map<std::string_view, std::uint32_t> controls_;
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::uint32_t byte_token_[256];
  std::uint32_t eos_{0};
};

}  // namespace inkling2
