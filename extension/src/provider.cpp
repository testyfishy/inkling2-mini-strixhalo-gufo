#include <chrono>
#include <cstdio>
#include <string>

#include <gufo/ext/generated/inkling2_native_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "config.hpp"

namespace {

// Milestone 1: load and validate the file, report what was found, and say
// plainly that nothing can be generated yet.
int ReportOnly(const gufo::core::GgufReader& reader) {
  inkling2::Config config;
  inkling2::Inventory inventory;
  std::string error;
  if (!inkling2::LoadConfig(reader, &config, &error) ||
      !inkling2::CheckTensors(reader, config, &inventory, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::size_t swa = 0;
  for (const bool s : config.is_swa)
    swa += s ? 1 : 0;
  std::printf("inkling2-mini-native: file accepted\n");
  std::printf("  layers %u (%zu sliding-window, %zu full attention), "
              "loop [%u, %u) x%u = %u layer visits per token\n",
              config.n_layer, swa, config.is_swa.size() - swa,
              config.loop_start, config.loop_end, config.loop_count,
              config.visits());
  std::printf("  width %u, heads %u/%u x %u, experts %u (top %u, ff %u), "
              "vocab %u, context %u\n",
              config.n_embd, config.n_head, config.n_head_kv, config.head_dim,
              config.n_expert, config.n_expert_used, config.n_ff_exp,
              config.n_vocab, config.n_ctx);
  std::printf("  tensors %zu, %.2f GiB\n", inventory.tensors,
              static_cast<double>(inventory.bytes) / (1ull << 30));
  for (const auto& [type, slot] : inventory.by_type)
    std::printf("    %-6s %4zu tensors %8.2f GiB\n", type.c_str(), slot.first,
                static_cast<double>(slot.second) / (1ull << 30));
  std::printf("  experts stored as NVFP4: %zu of %u tensors\n",
              inventory.nvfp4_expert_tensors, 2 * config.n_layer);
  std::fprintf(stderr, "inkling2-mini-native: the GPU runner is not "
                       "implemented yet; nothing was generated\n");
  return 2;
}

int Prompt(const gufo::ext::TextPromptOptions&,
           const gufo::core::GgufReader& reader,
           std::chrono::steady_clock::time_point) {
  return ReportOnly(reader);
}

}  // namespace

void inkling2_native_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_inkling2_native::
                           kProvider_inkling2_mini_native);
  registry.AddTextProvider({
      .id = "inkling2-mini-native",
      .probe = nullptr,
      .probe_gguf = &inkling2::Probe,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = &Prompt,
      .chat_driver = nullptr,
      // The NVFP4 expert tensors are a storage type GUFO's core reader does
      // not size; this provider reads them from their reserved bytes.
      .accepts_unsized_tensors = true,
  });
}

void inkling2_native_register(gufo::ext::Registry& registry) {
  (void)registry;
}
