#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace inkling2::gpu {

/// y = W x for a BF16 matrix of `rows` rows of `cols` values.
void GemvBf16(const std::uint16_t* w, const float* x, float* y, int rows,
              int cols, hipStream_t stream);

/// Tuning aid: selects the thread layout of GemvBf16 (32, 64, 128 or 256
/// threads per row, or 1004 / 1008 for four / eight rows per block).
void SetGemvLayout(int layout);

/// The selected experts of one MoE layer, fused: for each selected expert,
/// act = situ(gate x * up_scale, up x * up_scale), then
/// out = sum over experts of coef * (down act).
/// `gate_up` holds 2 * ff rows of `n_embd` values per expert (gate rows then up
/// rows) and `down` holds n_embd rows of ff values per expert. `act` is scratch
/// of count * ff floats. `nvfp4` selects NVFP4 storage, otherwise BF16.
struct ExpertArgs {
  const std::uint8_t* gate_up;
  const std::uint8_t* down;
  const float* x;
  const std::uint32_t* ids;  // count entries, on the device
  const float* up_scale;     // count entries, on the device
  const float* coef;         // count entries, on the device
  float* act;
  float* out;
  int count, ff, n_embd;
  float situ_beta, situ_linear_beta;
  bool nvfp4;
};
void Experts(const ExpertArgs& args, hipStream_t stream);

}  // namespace inkling2::gpu
