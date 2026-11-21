#include "tokenizer.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <variant>

namespace inkling2 {
namespace {

// ---- Unicode classes -------------------------------------------------------

struct Range {
  std::uint32_t first, last;
  std::uint8_t cls;
};
constexpr Range kRanges[] = {
#include "unicode_tables.inc"
};

// Flags of one character, as the splitting rule reads them.
enum : std::uint8_t {
  kUpperish = 1,   // Lu, Lt, Lm, Lo, M
  kLowerish = 2,   // Ll, Lm, Lo, M
  kLetter = 4,     // L
  kNumber = 8,     // N
  kSpace = 16,     // White_Space
  kNewline = 32,   // \r or \n
};

std::uint8_t ClassFlags(std::uint8_t cls) {
  switch (cls) {
    case 1:  // Lu
    case 3:  // Lt
      return kUpperish | kLetter;
    case 2:  // Ll
      return kLowerish | kLetter;
    case 4:  // Lm
    case 5:  // Lo
      return kUpperish | kLowerish | kLetter;
    case 6:  // M
      return kUpperish | kLowerish;
    case 7:
      return kNumber;
    default:
      return 0;
  }
}

bool IsWhiteSpace(std::uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 ||
         cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
         cp == 0x3000;
}

std::uint8_t Flags(std::uint32_t cp) {
  std::uint8_t flags = 0;
  if (cp < 0x80) {
    if (cp >= 'a' && cp <= 'z')
      return kLowerish | kLetter;
    if (cp >= 'A' && cp <= 'Z')
      return kUpperish | kLetter;
    if (cp >= '0' && cp <= '9')
      return kNumber;
    if (cp == '\r' || cp == '\n')
      return kSpace | kNewline;
    return IsWhiteSpace(cp) ? kSpace : 0;
  }
  const Range* end = std::end(kRanges);
  const Range* it = std::lower_bound(
      std::begin(kRanges), end, cp,
      [](const Range& r, std::uint32_t value) { return r.last < value; });
  if (it != end && it->first <= cp)
    flags = ClassFlags(it->cls);
  if (IsWhiteSpace(cp))
    flags |= kSpace;
  return flags;
}

// Decodes UTF-8 leniently: a byte that starts no valid sequence is one
// character of no class, so every byte of the input lands in some piece.
struct Chars {
  std::vector<std::uint32_t> cp;
  std::vector<std::uint8_t> flags;
  std::vector<std::size_t> offset;  // size + 1 entries
};

void Decode(std::string_view text, Chars* chars) {
  const auto* s = reinterpret_cast<const unsigned char*>(text.data());
  const std::size_t n = text.size();
  chars->cp.reserve(n);
  chars->flags.reserve(n);
  chars->offset.reserve(n + 1);
  std::size_t i = 0;
  while (i < n) {
    std::uint32_t cp = s[i];
    std::size_t len = 1;
    if (cp >= 0x80) {
      std::size_t want = 0;
      std::uint32_t value = 0, low = 0;
      if (cp >= 0xC2 && cp <= 0xDF) {
        want = 1, value = cp & 0x1F, low = 0x80;
      } else if (cp >= 0xE0 && cp <= 0xEF) {
        want = 2, value = cp & 0x0F, low = 0x800;
      } else if (cp >= 0xF0 && cp <= 0xF4) {
        want = 3, value = cp & 0x07, low = 0x10000;
      }
      bool ok = want != 0 && i + want < n;
      if (ok) {
        for (std::size_t k = 1; k <= want; ++k) {
          if ((s[i + k] & 0xC0) != 0x80) {
            ok = false;
            break;
          }
          value = (value << 6) | (s[i + k] & 0x3F);
        }
      }
      if (ok && value >= low && value <= 0x10FFFF &&
          !(value >= 0xD800 && value <= 0xDFFF)) {
        cp = value;
        len = want + 1;
      } else {
        cp = 0xFFFFFFFFu;  // no class
      }
    }
    chars->cp.push_back(cp);
    chars->flags.push_back(cp == 0xFFFFFFFFu ? 0 : Flags(cp));
    chars->offset.push_back(i);
    i += len;
  }
  chars->offset.push_back(n);
}

// ---- The splitting rule ----------------------------------------------------
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
// | \p{N}{1,3}
// |  ?[^\s\p{L}\p{N}]+[\r\n/]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
//
// The first alternative that matches at a position wins, each with the
// backtracking a regular-expression engine does. The case-insensitive
// contractions are matched for ASCII letters only.
class Splitter {
public:
  explicit Splitter(const Chars& chars)
      : cp_(chars.cp), f_(chars.flags), n_(chars.cp.size()) {}

  // End of the piece that starts at character i.
  std::size_t Next(std::size_t i) const {
    if (std::size_t e = Word(i, true))
      return e;
    if (std::size_t e = Word(i, false))
      return e;
    if (f_[i] & kNumber) {
      std::size_t k = i + 1;
      while (k < n_ && k < i + 3 && (f_[k] & kNumber))
        ++k;
      return k;
    }
    if (std::size_t e = Punctuation(i))
      return e;
    if (f_[i] & kSpace) {
      std::size_t w = i;
      std::size_t last_newline = n_;
      while (w < n_ && (f_[w] & kSpace)) {
        if (f_[w] & kNewline)
          last_newline = w;
        ++w;
      }
      if (last_newline != n_)
        return last_newline + 1;
      if (w == n_)
        return w;
      if (w - i >= 2)
        return w - 1;
      return w;
    }
    return i + 1;  // unreachable: every character matches an alternative
  }

private:
  bool Prefix(std::size_t i) const {
    return !(f_[i] & (kNewline | kLetter | kNumber));
  }
  std::size_t Contraction(std::size_t k) const {
    if (k + 1 >= n_ || cp_[k] != '\'')
      return k;
    const auto lower = [&](std::size_t at) -> std::uint32_t {
      const std::uint32_t c = cp_[at];
      return c >= 'A' && c <= 'Z' ? c + 32 : c;
    };
    const std::uint32_t a = lower(k + 1);
    // U+017F (long s) folds to s.
    if (a == 's' || a == 0x17F || a == 't' || a == 'm' || a == 'd')
      return k + 2;
    if (k + 2 < n_) {
      const std::uint32_t b = lower(k + 2);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
          (a == 'l' && b == 'l'))
        return k + 3;
    }
    return k;
  }
  // Alternatives 1 (lower_required) and 2. Zero when there is no match.
  std::size_t Word(std::size_t i, bool lower_required) const {
    for (int prefix = 1; prefix >= 0; --prefix) {
      if (prefix && !Prefix(i))
        continue;
      const std::size_t s = i + static_cast<std::size_t>(prefix);
      std::size_t j = s;
      while (j < n_ && (f_[j] & kUpperish))
        ++j;
      if (!lower_required) {
        if (j == s)
          continue;
        std::size_t k = j;
        while (k < n_ && (f_[k] & kLowerish))
          ++k;
        return Contraction(k);
      }
      // The upper run gives characters back until a lower run follows it.
      for (std::size_t jj = j;; --jj) {
        std::size_t k = jj;
        while (k < n_ && (f_[k] & kLowerish))
          ++k;
        if (k > jj)
          return Contraction(k);
        if (jj == s)
          break;
      }
    }
    return 0;
  }
  std::size_t Punctuation(std::size_t i) const {
    const std::size_t s = cp_[i] == ' ' ? i + 1 : i;
    std::size_t k = s;
    while (k < n_ && !(f_[k] & (kSpace | kLetter | kNumber)))
      ++k;
    if (k == s)
      return 0;
    while (k < n_ && (cp_[k] == '\r' || cp_[k] == '\n' || cp_[k] == '/'))
      ++k;
    return k;
  }

  const std::vector<std::uint32_t>& cp_;
  const std::vector<std::uint8_t>& f_;
  std::size_t n_;
};

// ---- GGUF vocabulary -------------------------------------------------------

// The file spells each token with one character per byte (GPT-2's table):
// printable bytes as themselves, the others as U+0100 and up.
struct ByteTable {
  std::uint8_t from_high[256];
  std::uint32_t high_count{0};
  bool direct[256];
  ByteTable() {
    for (int b = 0; b < 256; ++b) {
      direct[b] = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                  (b >= 174 && b <= 255);
      if (!direct[b])
        from_high[high_count++] = static_cast<std::uint8_t>(b);
    }
  }
};

// False when the spelling holds a character outside the table.
bool ToBytes(const ByteTable& table, std::string_view spelling,
             std::string* out) {
  out->clear();
  const auto* s = reinterpret_cast<const unsigned char*>(spelling.data());
  for (std::size_t i = 0; i < spelling.size();) {
    std::uint32_t cp = s[i];
    if (cp < 0x80) {
      ++i;
    } else if ((cp & 0xE0) == 0xC0 && i + 1 < spelling.size()) {
      cp = ((cp & 0x1F) << 6) | (s[i + 1] & 0x3F);
      i += 2;
    } else {
      return false;
    }
    if (cp < 256 && table.direct[cp])
      out->push_back(static_cast<char>(cp));
    else if (cp >= 256 && cp - 256 < table.high_count)
      out->push_back(static_cast<char>(table.from_high[cp - 256]));
    else
      return false;
  }
  return true;
}

std::vector<std::int64_t> IntArray(const gufo::core::GgufReader& reader,
                                   std::string_view key) {
  const auto* value = reader.FindMetadata(key);
  if (!value)
    return {};
  if (const auto* i = std::get_if<std::vector<std::int64_t>>(&value->value))
    return *i;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&value->value))
    return {u->begin(), u->end()};
  return {};
}

constexpr std::int64_t kTypeNormal = 1;

}  // namespace

Tokenizer::Tokenizer(const gufo::core::GgufReader& reader) {
  if (reader.GetMetadataString("tokenizer.ggml.model").value_or("") != "gpt2")
    throw std::runtime_error("inkling2: the file's tokenizer is not BPE");
  const auto spellings = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto types = IntArray(reader, "tokenizer.ggml.token_type");
  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (spellings.empty() || types.size() != spellings.size() || merges.empty())
    throw std::runtime_error("inkling2: the file has no usable vocabulary");

  const ByteTable table;
  bytes_.resize(spellings.size());
  control_.assign(spellings.size(), false);
  for (std::size_t id = 0; id < spellings.size(); ++id) {
    if (types[id] == kTypeNormal && ToBytes(table, spellings[id], &bytes_[id]))
      continue;
    // Control and other added tokens are spelled as they are written.
    bytes_[id].assign(spellings[id]);
    control_[id] = true;
  }
  vocab_.reserve(bytes_.size() * 2);
  std::fill(std::begin(byte_token_), std::end(byte_token_), 0xFFFFFFFFu);
  for (std::uint32_t id = 0; id < bytes_.size(); ++id) {
    if (control_[id]) {
      controls_.emplace(bytes_[id], id);
      continue;
    }
    vocab_.emplace(bytes_[id], id);
    if (bytes_[id].size() == 1)
      byte_token_[static_cast<unsigned char>(bytes_[id][0])] = id;
  }
  for (const std::uint32_t id : byte_token_)
    if (id == 0xFFFFFFFFu)
      throw std::runtime_error(
          "inkling2: the vocabulary lacks a single-byte token");

  merges_.reserve(merges.size() * 2);
  std::string left, right, joined;
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    const std::string_view line = merges[rank];
    const std::size_t space = line.find(' ');
    if (space == std::string_view::npos ||
        !ToBytes(table, line.substr(0, space), &left) ||
        !ToBytes(table, line.substr(space + 1), &right))
      continue;
    joined = left + right;
    const auto a = vocab_.find(left), b = vocab_.find(right),
               ab = vocab_.find(joined);
    if (a == vocab_.end() || b == vocab_.end() || ab == vocab_.end())
      continue;
    merges_.emplace((std::uint64_t{a->second} << 32) | b->second,
                    Merge{static_cast<std::uint32_t>(rank), ab->second});
  }
  eos_ = reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")
             .value_or(0xFFFFFFFFu);
  if (eos_ >= bytes_.size())
    throw std::runtime_error("inkling2: the file names no end token");
}

std::optional<std::uint32_t> Tokenizer::Control(
    std::string_view spelling) const {
  const auto it = controls_.find(spelling);
  if (it == controls_.end())
    return std::nullopt;
  return it->second;
}

std::uint32_t Tokenizer::RequireControl(std::string_view spelling) const {
  const auto id = Control(spelling);
  if (!id)
    throw std::runtime_error("inkling2: the vocabulary lacks " +
                             std::string(spelling));
  return *id;
}

void Tokenizer::Split(std::string_view text, std::vector<std::size_t>* ends) {
  Chars chars;
  Decode(text, &chars);
  const Splitter splitter(chars);
  for (std::size_t i = 0; i < chars.cp.size();) {
    i = splitter.Next(i);
    ends->push_back(chars.offset[i]);
  }
}

void Tokenizer::EncodePiece(std::string_view piece,
                            std::vector<std::uint32_t>* out) const {
  if (const auto whole = vocab_.find(piece); whole != vocab_.end()) {
    out->push_back(whole->second);
    return;
  }
  // Symbols start as single bytes; the adjacent pair with the lowest merge
  // rank (leftmost on ties) is merged until no pair has a merge.
  const std::size_t n = piece.size();
  const auto find = [&](std::uint32_t a, std::uint32_t b) -> const Merge* {
    const auto it = merges_.find((std::uint64_t{a} << 32) | b);
    return it == merges_.end() ? nullptr : &it->second;
  };
  if (n <= 48) {
    std::uint32_t symbols[48];
    std::size_t count = n;
    for (std::size_t i = 0; i < n; ++i)
      symbols[i] = byte_token_[static_cast<unsigned char>(piece[i])];
    while (count > 1) {
      const Merge* best = nullptr;
      std::size_t at = 0;
      for (std::size_t i = 0; i + 1 < count; ++i) {
        const Merge* m = find(symbols[i], symbols[i + 1]);
        if (m && (!best || m->rank < best->rank))
          best = m, at = i;
      }
      if (!best)
        break;
      symbols[at] = best->id;
      for (std::size_t i = at + 1; i + 1 < count; ++i)
        symbols[i] = symbols[i + 1];
      --count;
    }
    out->insert(out->end(), symbols, symbols + count);
    return;
  }
  // Long pieces (runs of spaces or punctuation): the same order of merges
  // from a queue of candidates, stale ones skipped.
  struct Candidate {
    std::uint32_t rank, at, id;
    bool operator>(const Candidate& o) const {
      return rank != o.rank ? rank > o.rank : at > o.at;
    }
  };
  std::vector<std::uint32_t> symbol(n);
  std::vector<std::int64_t> prev(n), next(n);
  std::vector<bool> gone(n, false);
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
  for (std::size_t i = 0; i < n; ++i) {
    symbol[i] = byte_token_[static_cast<unsigned char>(piece[i])];
    prev[i] = static_cast<std::int64_t>(i) - 1;
    next[i] = i + 1 < n ? static_cast<std::int64_t>(i + 1) : -1;
  }
  const auto offer = [&](std::int64_t at) {
    if (at < 0 || next[at] < 0)
      return;
    if (const Merge* m = find(symbol[at], symbol[next[at]]))
      queue.push({m->rank, static_cast<std::uint32_t>(at), m->id});
  };
  for (std::size_t i = 0; i + 1 < n; ++i)
    offer(static_cast<std::int64_t>(i));
  while (!queue.empty()) {
    const Candidate c = queue.top();
    queue.pop();
    if (gone[c.at] || next[c.at] < 0)
      continue;
    const Merge* m = find(symbol[c.at], symbol[next[c.at]]);
    if (!m || m->rank != c.rank || m->id != c.id)
      continue;
    const std::int64_t removed = next[c.at];
    symbol[c.at] = c.id;
    gone[removed] = true;
    next[c.at] = next[removed];
    if (next[removed] >= 0)
      prev[next[removed]] = c.at;
    offer(prev[c.at]);
    offer(c.at);
  }
  for (std::int64_t i = 0; i >= 0; i = next[i])
    out->push_back(symbol[i]);
}

void Tokenizer::Encode(std::string_view text,
                       std::vector<std::uint32_t>* out) const {
  if (text.empty())
    return;
  Chars chars;
  Decode(text, &chars);
  const Splitter splitter(chars);
  for (std::size_t i = 0; i < chars.cp.size();) {
    const std::size_t e = splitter.Next(i);
    EncodePiece(text.substr(chars.offset[i], chars.offset[e] - chars.offset[i]),
                out);
    i = e;
  }
}

std::vector<std::uint32_t> Tokenizer::EncodeWithControl(
    std::string_view text) const {
  std::vector<std::uint32_t> out;
  std::size_t plain = 0, at = 0;
  while ((at = text.find("<|", at)) != std::string_view::npos) {
    const std::size_t close = text.find("|>", at + 2);
    if (close == std::string_view::npos)
      break;
    const auto id = Control(text.substr(at, close + 2 - at));
    if (!id) {
      at += 2;
      continue;
    }
    Encode(text.substr(plain, at - plain), &out);
    out.push_back(*id);
    at = plain = close + 2;
  }
  Encode(text.substr(plain), &out);
  return out;
}

}  // namespace inkling2
