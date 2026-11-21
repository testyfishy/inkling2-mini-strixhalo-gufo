#pragma once

#include <memory>

#include "backend.hpp"

namespace inkling2 {

/// Runs the heavy matrix products on the GPU through HIP. Weights are copied
/// to the device the first time they are used. Throws std::runtime_error when
/// the device cannot be used.
[[nodiscard]] std::unique_ptr<Backend> MakeGpuBackend();

}  // namespace inkling2
