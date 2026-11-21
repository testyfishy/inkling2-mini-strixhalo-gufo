#include "gpu.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hip/ops.hpp"

namespace inkling2 {
namespace {

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("inkling2 gpu: ") + what + ": " +
                             hipGetErrorString(status));
}

class GpuBackend final : public Backend {
public:
  GpuBackend() {
    Check(hipStreamCreate(&stream_), "create stream");
    if (const char* layout = std::getenv("INKLING2_GEMV_LAYOUT"))
      gpu::SetGemvLayout(std::atoi(layout));
    // Staging: one host-side copy in and one out per call.
    Check(hipMalloc(&in_, kInBytes), "allocate input staging");
    Check(hipMalloc(&out_, kOutBytes), "allocate output staging");
    Check(hipMalloc(&act_, kActBytes), "allocate expert scratch");
    host_in_.resize(kInBytes / 4);
    host_out_.resize(kOutBytes / 4);
  }

  ~GpuBackend() override {
    (void)hipStreamSynchronize(stream_);
    if (std::getenv("INKLING2_GPU_STATS"))
      std::fprintf(stderr,
                   "gpu stats: projections %.3f s in %zu calls, experts %.3f s "
                   "in %zu calls, uploads %.3f s (%.2f GiB)\n",
                   mul_seconds_, mul_calls_, expert_seconds_, expert_calls_,
                   upload_seconds_,
                   static_cast<double>(uploaded_bytes_) / (1ull << 30));
    for (auto& [host, device] : weights_)
      (void)hipFree(device);
    (void)hipFree(in_);
    (void)hipFree(out_);
    (void)hipFree(act_);
    (void)hipStreamDestroy(stream_);
  }

  void Mul(const Job* jobs, std::size_t count) override {
    const Timer timer(&mul_seconds_, &upload_seconds_);
    ++mul_calls_;
    // Pack the distinct inputs into the staging buffer.
    std::size_t in_used = 0, out_used = 0;
    const float* seen[8];
    std::size_t seen_at[8], n_seen = 0;
    std::size_t x_at[16], y_at[16];
    if (count > 16)
      throw std::runtime_error("inkling2 gpu: too many jobs in one call");
    for (std::size_t i = 0; i < count; ++i) {
      const Mat& m = *jobs[i].mat;
      if (m.type != kGgmlTypeBf16)
        throw std::runtime_error("inkling2 gpu: projection is not BF16");
      std::size_t found = n_seen;
      for (std::size_t s = 0; s < n_seen; ++s)
        if (seen[s] == jobs[i].x)
          found = s;
      if (found == n_seen) {
        seen[n_seen] = jobs[i].x;
        seen_at[n_seen++] = in_used;
        std::memcpy(host_in_.data() + in_used, jobs[i].x, m.cols * 4);
        in_used += m.cols;
      }
      x_at[i] = seen_at[found];
      y_at[i] = out_used;
      out_used += m.rows;
    }
    if (in_used * 4 > kInBytes || out_used * 4 > kOutBytes)
      throw std::runtime_error("inkling2 gpu: staging buffer too small");
    Check(hipMemcpyAsync(in_, host_in_.data(), in_used * 4,
                         hipMemcpyHostToDevice, stream_),
          "copy inputs");
    for (std::size_t i = 0; i < count; ++i) {
      const Mat& m = *jobs[i].mat;
      gpu::GemvBf16(static_cast<const std::uint16_t*>(Device(m, 1)),
                    Floats(in_) + x_at[i], Floats(out_) + y_at[i],
                    static_cast<int>(m.rows), static_cast<int>(m.cols),
                    stream_);
    }
    Check(hipMemcpyAsync(host_out_.data(), out_, out_used * 4,
                         hipMemcpyDeviceToHost, stream_),
          "copy outputs");
    Check(hipStreamSynchronize(stream_), "synchronize");
    for (std::size_t i = 0; i < count; ++i)
      std::memcpy(jobs[i].y, host_out_.data() + y_at[i],
                  jobs[i].mat->rows * 4);
  }

  void Experts(const ExpertCall& call) override {
    const Timer timer(&expert_seconds_, &upload_seconds_);
    ++expert_calls_;
    const Mat& up = *call.gate_up;
    const Mat& down = *call.down;
    const bool nvfp4 = up.type == kGgmlTypeNvfp4;
    if (up.type != down.type || (!nvfp4 && up.type != kGgmlTypeBf16))
      throw std::runtime_error("inkling2 gpu: unsupported expert storage");
    const std::size_t e = down.rows, ff = down.cols, n = call.count;
    if (n * ff * 4 > kActBytes)
      throw std::runtime_error("inkling2 gpu: expert scratch too small");
    // Staging layout: x, ids, up_scale, coef.
    float* host = host_in_.data();
    std::memcpy(host, call.x, e * 4);
    std::memcpy(host + e, call.ids, n * 4);
    std::memcpy(host + e + n, call.up_scale, n * 4);
    std::memcpy(host + e + 2 * n, call.coef, n * 4);
    Check(hipMemcpyAsync(in_, host, (e + 3 * n) * 4, hipMemcpyHostToDevice,
                         stream_),
          "copy expert inputs");
    const std::size_t n_expert = 128;
    gpu::ExpertArgs args{};
    args.gate_up = static_cast<const std::uint8_t*>(Device(up, n_expert));
    args.down = static_cast<const std::uint8_t*>(Device(down, n_expert));
    args.x = Floats(in_);
    args.ids = reinterpret_cast<const std::uint32_t*>(Floats(in_) + e);
    args.up_scale = Floats(in_) + e + n;
    args.coef = Floats(in_) + e + 2 * n;
    args.act = Floats(act_);
    args.out = Floats(out_);
    args.count = static_cast<int>(n);
    args.ff = static_cast<int>(ff);
    args.n_embd = static_cast<int>(e);
    args.situ_beta = call.situ_beta;
    args.situ_linear_beta = call.situ_linear_beta;
    args.nvfp4 = nvfp4;
    gpu::Experts(args, stream_);
    Check(hipMemcpyAsync(call.out, out_, e * 4, hipMemcpyDeviceToHost, stream_),
          "copy expert output");
    Check(hipStreamSynchronize(stream_), "synchronize");
  }

  [[nodiscard]] const char* name() const override { return "hip"; }

private:
  static constexpr std::size_t kInBytes = 1u << 20;
  static constexpr std::size_t kOutBytes = 4u << 20;  // holds the logits
  static constexpr std::size_t kActBytes = 1u << 20;

  static float* Floats(void* p) { return static_cast<float*>(p); }

  // The device copy of a weight, uploaded on first use. `copies` is the number
  // of stacked matrices in the tensor (the expert count for expert tensors).
  void* Device(const Mat& m, std::size_t copies) {
    const auto it = weights_.find(m.data);
    if (it != weights_.end())
      return it->second;
    const std::size_t bytes = m.rows * m.RowBytes() * copies;
    const auto upload_start = std::chrono::steady_clock::now();
    void* device = nullptr;
    Check(hipMalloc(&device, bytes), "allocate weight");
    Check(hipMemcpy(device, m.data, bytes, hipMemcpyHostToDevice),
          "upload weight");
    weights_.emplace(m.data, device);
    uploaded_bytes_ += bytes;
    upload_seconds_ += std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - upload_start)
                           .count();
    return device;
  }

  // Adds the elapsed time to `total`, less any upload time spent meanwhile.
  struct Timer {
    Timer(double* total, const double* uploads)
        : total_(total), uploads_(uploads), uploads_before_(*uploads) {}
    ~Timer() {
      *total_ += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start_)
                     .count() -
                 (*uploads_ - uploads_before_);
    }
    double* total_;
    const double* uploads_;
    double uploads_before_;
    std::chrono::steady_clock::time_point start_{
        std::chrono::steady_clock::now()};
  };

  double mul_seconds_{0}, expert_seconds_{0}, upload_seconds_{0};
  std::size_t mul_calls_{0}, expert_calls_{0};
  hipStream_t stream_{};
  void* in_{};
  void* out_{};
  void* act_{};
  std::vector<float> host_in_, host_out_;
  std::unordered_map<const std::uint8_t*, void*> weights_;
  std::size_t uploaded_bytes_{0};
};

}  // namespace

std::unique_ptr<Backend> MakeGpuBackend() {
  return std::make_unique<GpuBackend>();
}

}  // namespace inkling2
