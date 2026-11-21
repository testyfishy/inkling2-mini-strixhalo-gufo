#include "chat.hpp"

#include <gufo/ext/json.hpp>

#include <algorithm>
#include <charconv>
#include <stdexcept>

namespace inkling2 {
namespace {

namespace json = gufo::json;

// A copy with the members of every object sorted by key, as Python's
// json.dumps(sort_keys=True) orders them.
json::Value Sorted(const json::Value& value) {
  if (value.is_array()) {
    json::Value out = json::Value::array();
    for (const auto& item : value.items())
      out.push_back(Sorted(item));
    return out;
  }
  if (value.is_object()) {
    std::vector<const json::Value::Member*> members;
    for (const auto& member : value.members())
      members.push_back(&member);
    std::stable_sort(members.begin(), members.end(),
                     [](const auto* a, const auto* b) {
                       return a->first < b->first;
                     });
    json::Value out = json::Value::object();
    for (const auto* member : members)
      out.append_member(member->first, Sorted(member->second));
    return out;
  }
  return value;
}

std::string Compact(const json::Value& value) {
  return value.dump(json::Value::Numbers::kAsParsed);
}

std::string Quoted(const std::string& text) {
  return Compact(json::Value(text));
}

class Builder {
public:
  explicit Builder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}
  void Control(std::string_view spelling) {
    out_.tokens.push_back(tokenizer_.RequireControl(spelling));
    out_.text += spelling;
  }
  void Text(std::string_view text) {
    tokenizer_.Encode(text, &out_.tokens);
    out_.text += text;
  }
  RenderedChat Take() { return std::move(out_); }

private:
  const Tokenizer& tokenizer_;
  RenderedChat out_;
};

std::string_view RoleToken(ChatMessage::Role role) {
  switch (role) {
    case ChatMessage::Role::kSystem:
      return "<|message_system|>";
    case ChatMessage::Role::kAssistant:
      return "<|message_model|>";
    case ChatMessage::Role::kTool:
      return "<|message_tool|>";
    case ChatMessage::Role::kUser:
      break;
  }
  return "<|message_user|>";
}

void Effort(Builder* b, double effort) {
  if (!(effort >= 0.0 && effort <= 0.99))
    throw std::invalid_argument("reasoning effort must be in [0.0, 0.99]");
  b->Control("<|message_system|>");
  b->Control("<|content_text|>");
  std::string text = "Thinking effort level: ";
  if (effort == 0.0) {
    text += "0";
  } else {
    char buffer[32];
    const auto [end, error] =
        std::to_chars(buffer, buffer + sizeof(buffer), effort);
    (void)error;
    text.append(buffer, end);
  }
  b->Text(text);
  b->Control("<|end_message|>");
}

}  // namespace

double EffortByName(std::string_view name) {
  if (name == "none")
    return 0.0;
  if (name == "minimal")
    return 0.1;
  if (name == "low")
    return 0.2;
  if (name == "medium")
    return 0.7;
  if (name == "high")
    return 0.9;
  if (name == "xhigh" || name == "max")
    return 0.99;
  return -1.0;
}

RenderedChat RenderChat(const Tokenizer& tokenizer,
                        const std::vector<ChatMessage>& messages,
                        const std::vector<ChatTool>& tools,
                        const ChatOptions& options) {
  Builder b(tokenizer);
  if (!tools.empty()) {
    json::Value specs = json::Value::array();
    for (const ChatTool& tool : tools) {
      json::Value parameters = json::Value::object();
      if (!tool.parameters_json.empty()) {
        try {
          parameters = Sorted(json::parse(tool.parameters_json));
        } catch (const std::exception&) {
          throw std::invalid_argument("tool parameters are not JSON: " +
                                      tool.name);
        }
        // The template writes {} for parameters that are empty or null.
        if (parameters.is_null() || (parameters.is_object() && parameters.empty()))
          parameters = json::Value::object();
      }
      json::Value spec = json::Value::object();
      spec.append_member("description", json::Value(tool.description));
      spec.append_member("name", json::Value(tool.name));
      spec.append_member("parameters", std::move(parameters));
      spec.append_member("type",
                         json::Value(tool.type.empty() ? "function" : tool.type));
      specs.push_back(std::move(spec));
    }
    b.Control("<|message_system|>");
    b.Text("tool_declare");
    b.Control("<|content_xml|>");
    b.Text(Compact(specs));
    b.Control("<|end_message|>");
  }

  bool effort_emitted = false;
  for (const ChatMessage& message : messages) {
    if (!effort_emitted && message.role != ChatMessage::Role::kSystem) {
      Effort(&b, options.effort);
      effort_emitted = true;
    }
    const bool assistant = message.role == ChatMessage::Role::kAssistant;
    // The header: the role's token, then for a tool message the tool's name
    // (its own, or the name of the call it answers) and for a named assistant
    // message that name.
    std::string name;
    if (message.role == ChatMessage::Role::kTool) {
      if (!message.name.empty()) {
        name = message.name;
      } else if (!message.tool_call_id.empty()) {
        for (const ChatMessage& previous : messages) {
          if (previous.role != ChatMessage::Role::kAssistant)
            continue;
          for (const ChatToolCall& call : previous.tool_calls)
            if (!call.id.empty() && call.id == message.tool_call_id)
              name += call.name;
        }
      }
    } else if (assistant) {
      name = message.name;
    }
    const auto header = [&] {
      b.Control(RoleToken(message.role));
      if (!name.empty())
        b.Text(name);
    };

    if (assistant && !message.reasoning.empty()) {
      header();
      b.Control("<|content_thinking|>");
      b.Text(message.reasoning);
      b.Control("<|end_message|>");
    }
    if (!message.content.empty() ||
        !(assistant &&
          (!message.tool_calls.empty() || !message.reasoning.empty()))) {
      header();
      b.Control("<|content_text|>");
      b.Text(message.content);
      b.Control("<|end_message|>");
    }
    if (assistant) {
      for (const ChatToolCall& call : message.tool_calls) {
        if (!message.name.empty() && message.name != call.name)
          throw std::invalid_argument("assistant name " + message.name +
                                      " does not match its tool call " +
                                      call.name);
        std::vector<const ChatToolCall::Argument*> arguments;
        for (const auto& argument : call.arguments)
          arguments.push_back(&argument);
        std::stable_sort(arguments.begin(), arguments.end(),
                         [](const auto* x, const auto* y) {
                           return x->name < y->name;
                         });
        std::string body = "{\"name\":" + Quoted(call.name) + ",\"args\":{";
        for (std::size_t i = 0; i < arguments.size(); ++i) {
          if (i)
            body += ",";
          body += Quoted(arguments[i]->name) + ":";
          if (arguments[i]->is_string) {
            body += Quoted(arguments[i]->value);
          } else {
            try {
              body += Compact(json::parse(arguments[i]->value));
            } catch (const std::exception&) {
              body += arguments[i]->value;
            }
          }
        }
        body += "}}";
        b.Control("<|message_model|>");
        b.Text(call.name);
        b.Control("<|content_invoke_tool_json|>");
        b.Text(body);
        b.Control("<|end_message|>");
      }
      b.Control("<|content_model_end_sampling|>");
    }
  }
  if (!effort_emitted)
    Effort(&b, options.effort);
  return b.Take();
}

}  // namespace inkling2
