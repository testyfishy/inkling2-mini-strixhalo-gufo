#pragma once

#include <gufo/ext/text.hpp>

#include <string_view>

namespace inkling2 {

/// Tool calls as the server sees them. The model writes a call as
///   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
/// and the runner shows it to the server as
///   <tool_call>{"name":"NAME","args":{...}}</tool_call>
/// which is the syntax this dialect describes and scans.
inline constexpr std::string_view kToolDialectId = "inkling2";
inline constexpr std::string_view kToolBlockOpen = "<tool_call>";
inline constexpr std::string_view kToolOpen = "<tool_call>{\"name\":\"";
inline constexpr std::string_view kToolNameSuffix = "\",\"args\":";
inline constexpr std::string_view kToolClose = "}</tool_call>";
inline constexpr std::string_view kToolBlockClose = "</tool_call>";

/// gufo::ext::ToolDialect::scan_tool_calls for this dialect.
void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<gufo::ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial);

/// Registers the dialect; call during extension registration.
void RegisterToolDialect();

}  // namespace inkling2
