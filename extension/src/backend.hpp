#pragma once

#include <cstddef>
#include <cstdint>

#include "config.hpp"

namespace inkling2 {

inline constexpr std::uint16_t kGgmlTypeF32 = 0;
inline constexpr std::uint16_t kGgmlTypeBf16 = 30;
inline constexpr int kNvfp4Values = 64;  // values per NVFP4 block
inline constexpr int kNvfp4Bytes = 36;   // bytes per NVFP4 block

/// A weight used as `y = W x`, mapped from the GGUF file: dims
/// {n_in, n_out[, n_expert]}, one row of n_in values per output.
struct Mat {
  const std::uint8_t* data{nullptr};
  std::size_t cols{0}, rows{0};
  std::uint16_t type{0};

  [[nodiscard]] std::size_t RowBytes() const {
    if (type == kGgmlTypeF32)
      return cols * 4;
    if (type == kGgmlTypeBf16)
      return cols * 2;
    return cols / kNvfp4Values * kNvfp4Bytes;
  }
};

/// Where the heavy matrix products run. The forward pass itself (norms,
/// mixing, convolutions, attention, routing) is shared; a backend only has to
/// reproduce these two operations.
class Backend {
public:
  virtual ~Backend() = default;

  struct Job {
    const Mat* mat;
    const float* x;  // mat->cols values
    float* y;        // mat->rows values
  };
  /// y = W x for every job. Jobs of one call may share inputs.
  virtual void Mul(const Job* jobs, std::size_t count) = 0;

  static constexpr std::size_t kMaxExperts = 16;
  struct ExpertCall {
    const Mat* gate_up;  // rows = 2 * ff: gate rows then up rows
    const Mat* down;     // rows = n_embd, cols = ff
    const float* x;
    std::size_t count;
    std::uint32_t ids[kMaxExperts];
    float up_scale[kMaxExperts];  // per-expert scale of the gate/up product
    float coef[kMaxExperts];      // router weight times the down scale
    float situ_beta, situ_linear_beta;
    float* out;  // sum over experts of coef * down(situ(gate, up))
  };
  virtual void Experts(const ExpertCall& call) = 0;

  [[nodiscard]] virtual const char* name() const = 0;
};

/// y = W x on the CPU, for F32, BF16 and NVFP4 matrices.
void CpuMul(const Mat& m, const float* x, float* y);

}  // namespace inkling2
