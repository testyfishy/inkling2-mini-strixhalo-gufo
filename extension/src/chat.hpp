#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer.hpp"

namespace inkling2 {

/// A conversation as the model's chat template (tokenizer.chat_template in the
/// file) lays it out. This is that template written out in C++; the test
/// compares its output with the template run by a Jinja interpreter.
struct ChatToolCall {
  struct Argument {
    std::string name;
    std::string value;     // the text of a string, or JSON
    bool is_string{true};
  };
  std::string id;
  std::string name;
  std::vector<Argument> arguments;
};

struct ChatMessage {
  enum class Role { kSystem, kUser, kAssistant, kTool };
  Role role{Role::kUser};
  std::string content;
  std::string reasoning;      // assistant only
  std::string name;           // tool: the tool's name; assistant: optional
  std::string tool_call_id;   // tool: the call it answers
  std::vector<ChatToolCall> tool_calls;  // assistant only
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json;  // a JSON object; empty means {}
  std::string type{"function"};
};

struct ChatOptions {
  /// The "Thinking effort level" the template writes: 0 (no thinking) to 0.99.
  /// The template's own default is 0.9.
  double effort{0.9};
};

struct RenderedChat {
  std::vector<std::uint32_t> tokens;
  /// The same prompt as text, control tokens spelled out.
  std::string text;
};

/// Throws std::invalid_argument for what the template refuses (an effort out
/// of range, tool parameters that are not JSON).
[[nodiscard]] RenderedChat RenderChat(const Tokenizer& tokenizer,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ChatTool>& tools,
                                      const ChatOptions& options);

/// The template's effort levels by name: none 0, minimal 0.1, low 0.2,
/// medium 0.7, high 0.9, xhigh and max 0.99. Negative for an unknown name.
[[nodiscard]] double EffortByName(std::string_view name);

}  // namespace inkling2
