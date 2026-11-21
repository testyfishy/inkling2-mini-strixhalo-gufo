#pragma once

namespace inkling2::gpu {

/// INKLING2_WMMA_BENCH: prints how many multiply-adds per second the FP16
/// matrix instruction reaches on this GPU under several conditions. Returns
/// the process exit code.
int RunWmmaBench();

}  // namespace inkling2::gpu
