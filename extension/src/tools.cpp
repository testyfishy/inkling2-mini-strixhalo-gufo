#include "tools.hpp"

#include <gufo/ext/json.hpp>

#include <algorithm>

namespace inkling2 {
namespace {

namespace json = gufo::json;

std::size_t SkipSpace(std::string_view text, std::size_t at) {
  while (at < text.size() && (text[at] == ' ' || text[at] == '\n' ||
                              text[at] == '\r' || text[at] == '\t'))
    ++at;
  return at;
}

// The position after the JSON object or array that starts at `at`, or npos
// when the text ends before it closes.
std::size_t EndOfValue(std::string_view text, std::size_t at) {
  int depth = 0;
  bool in_string = false;
  for (std::size_t i = at; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (c == '\\')
        ++i;
      else if (c == '"')
        in_string = false;
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      if (--depth == 0)
        return i + 1;
      if (depth < 0)
        return std::string_view::npos;
    }
  }
  return std::string_view::npos;
}

}  // namespace

void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> /*markers*/,
                   std::vector<gufo::ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial) {
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    const std::size_t begin = text.find(kToolOpen, cursor);
    if (begin == std::string_view::npos)
      break;
    const std::size_t name_start = begin + kToolOpen.size();
    const std::size_t name_end = text.find('"', name_start);
    if (name_end == std::string_view::npos)
      break;  // the name is still being written
    gufo::ext::ScannedToolCall call;
    call.name = std::string(text.substr(name_start, name_end - name_start));
    call.valid = !call.name.empty();
    const auto unfinished = [&] {
      if (partial && call.valid) {
        call.complete = false;
        calls->push_back(std::move(call));
      }
    };
    const std::string_view after_name = text.substr(name_end);
    if (!after_name.starts_with(kToolNameSuffix)) {
      if (kToolNameSuffix.starts_with(after_name)) {
        unfinished();
        break;
      }
      cursor = name_start;  // not a call: ordinary text
      continue;
    }
    const std::size_t args_start =
        SkipSpace(text, name_end + kToolNameSuffix.size());
    if (args_start >= text.size()) {
      unfinished();
      break;
    }
    if (text[args_start] != '{') {
      cursor = name_start;
      continue;
    }
    const std::size_t args_end = EndOfValue(text, args_start);
    if (args_end == std::string_view::npos) {
      unfinished();
      break;
    }
    try {
      const json::Value args =
          json::parse(text.substr(args_start, args_end - args_start));
      for (const auto& [key, value] : args.members()) {
        gufo::ext::ScannedToolArgument argument{
            .name = key,
            .value = value.is_string()
                         ? value.str()
                         : value.dump(json::Value::Numbers::kAsParsed),
            .is_string = value.is_string()};
        // An identical repeat is dropped; a conflicting one makes the call
        // invalid instead of letting the last value win.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& a) { return a.name; });
        if (previous == call.arguments.end())
          call.arguments.push_back(std::move(argument));
        else if (previous->value != argument.value ||
                 previous->is_string != argument.is_string)
          call.valid = false;
      }
    } catch (const std::exception&) {
      call.valid = false;
    }
    std::size_t end = SkipSpace(text, args_end);
    if (end >= text.size()) {
      unfinished();  // the object that holds the call is not closed yet
      break;
    }
    if (text[end] != '}') {
      cursor = args_end;
      continue;
    }
    ++end;
    if (text.substr(end).starts_with(kToolBlockClose))
      end += kToolBlockClose.size();
    call.complete = true;
    const bool accepted = call.valid;
    if (accepted || partial)
      calls->push_back(std::move(call));
    if (spans && accepted)
      spans->emplace_back(begin, end);
    cursor = end;
  }
}

void RegisterToolDialect() {
  namespace sampling = gufo::sampling;
  sampling::JsonConstraint::ToolSyntax syntax;
  syntax.open = std::string(kToolOpen);
  syntax.name_suffix = std::string(kToolNameSuffix);
  // What a grammar must see at the end of a call. The model ends a call with
  // <|end_message|>, and a grammar reads that token as "</think>" (it ends a
  // thinking block the same way; see BuildConstraintVocabulary in serve.cpp),
  // so that is the spelling here. The text the server parses still shows
  // </tool_call>, which ScanToolCalls reads.
  syntax.close = "}</think>";
  syntax.name_delimiters = "\"";
  syntax.values.whitespace = false;
  syntax.values.sorted_keys = true;
  gufo::ext::RegisterToolDialect({
      .markers = {kToolOpen},
      .reasoning_markers = {},
      .closers = {kToolBlockClose},
      .scan_tool_calls = &ScanToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = std::move(syntax),
  });
}

}  // namespace inkling2
