#pragma once

#include <gufo/ext/text.hpp>

#include <memory>
#include <optional>
#include <string>

namespace inkling2 {

/// The runner GUFO's server drives. Returns null and sets `error` when the
/// file or the options cannot be served.
[[nodiscard]] std::shared_ptr<gufo::ext::TextModelRunner> CreateTextRunner(
    const gufo::ext::TextRunnerOptions& options, std::string* error);

/// The sampling the file recommends (general.sampling.*).
[[nodiscard]] std::optional<gufo::sampling::TextSamplingDefaults>
SamplingDefaults(const gufo::core::GgufReader& reader);

}  // namespace inkling2
