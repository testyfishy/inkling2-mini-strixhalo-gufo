#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <gufo/ext/generated/inkling2_native_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "backend.hpp"
#include "config.hpp"
#include "decoder.hpp"
#include "gpu.hpp"
#include "model.hpp"
#include "reference.hpp"

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

// Development aid until the tokenizer exists: with INKLING2_REF_TOKENS set to
// comma-separated token ids, run the CPU reference on them and print the
// per-visit sums and the top next-token logits, to compare with llama.cpp.
int RunReference(const gufo::core::GgufReader& reader, const char* ids) {
  inkling2::Config config;
  std::string error;
  if (!inkling2::LoadConfig(reader, &config, &error) ||
      !inkling2::CheckTensors(reader, config, nullptr, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint32_t> tokens;
  for (const char* p = ids; *p;) {
    char* end = nullptr;
    tokens.push_back(static_cast<std::uint32_t>(std::strtoul(p, &end, 10)));
    if (end == p)
      break;
    p = *end == ',' ? end + 1 : end;
  }
  if (tokens.empty())
    return 1;
  try {
    // INKLING2_GPU=1 runs the heavy matrix products on the GPU.
    std::unique_ptr<inkling2::Backend> gpu;
    if (const char* v = std::getenv("INKLING2_GPU"); v && *v == '1')
      gpu = inkling2::MakeGpuBackend();
    inkling2::Reference reference(reader, config, gpu.get());
    std::vector<float> logits;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < tokens.size(); ++i)
      logits = reference.Feed(tokens[i], i + 1 == tokens.size());
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    const auto& trace = reference.trace();
    for (std::size_t s = 0; s < trace.l_out.size(); ++s)
      std::printf("slot %2zu  attn_out %14.6f  ffn_out %14.6f  l_out %14.6f\n",
                  s, trace.attn_out[s], trace.ffn_out[s], trace.l_out[s]);
    std::printf("result_norm %14.6f\n", trace.result_norm);
    for (const auto& [name, sum] : trace.debug)
      std::printf("dbg %-14s %16.6f\n", name.c_str(), sum);
    std::vector<std::uint32_t> order(logits.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
      order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
    double norm = 0;
    for (const float x : logits)
      norm += std::exp(static_cast<double>(x) - logits[order[0]]);
    const double log_norm = std::log(norm) + logits[order[0]];
    for (int i = 0; i < 10; ++i)
      std::printf("top %2d  token %6u  logit %10.5f  logprob %10.5f\n", i,
                  order[i], logits[order[i]], logits[order[i]] - log_norm);
    // Greedy continuation, to compare token for token with llama.cpp.
    const char* where = gpu ? gpu->name() : "cpu";
    if (const char* n = std::getenv("INKLING2_REF_GEN")) {
      const int count = std::atoi(n);
      std::printf("greedy:");
      const auto gen_start = std::chrono::steady_clock::now();
      for (int i = 0; i < count; ++i) {
        const auto best = static_cast<std::uint32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::printf(" %u", best);
        logits = reference.Feed(best, true);
      }
      const double gen_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        gen_start)
              .count();
      std::printf("\ngeneration: %d tokens in %.2f s (%.2f tokens/s, %s)\n",
                  count, gen_seconds, count / gen_seconds, where);
    }
    std::printf("prompt: %zu tokens in %.2f s (%.2f tokens/s, %s; on the GPU "
                "this includes uploading weights on first use)\n",
                tokens.size(), seconds, tokens.size() / seconds, where);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}

// Development aid: INKLING2_GPU=2 decodes the same token ids entirely on the
// GPU. INKLING2_COMPARE=1 also runs the CPU reference and prints both layer
// traces side by side.
int RunDecoder(const gufo::core::GgufReader& reader, const char* ids) {
  inkling2::Config config;
  std::string error;
  if (!inkling2::LoadConfig(reader, &config, &error) ||
      !inkling2::CheckTensors(reader, config, nullptr, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint32_t> tokens;
  for (const char* p = ids; *p;) {
    char* end = nullptr;
    tokens.push_back(static_cast<std::uint32_t>(std::strtoul(p, &end, 10)));
    if (end == p)
      break;
    p = *end == ',' ? end + 1 : end;
  }
  if (tokens.empty())
    return 1;
  try {
    const bool compare = std::getenv("INKLING2_COMPARE") != nullptr;
    const inkling2::Model model = inkling2::LoadModel(reader, config);
    const auto load_start = std::chrono::steady_clock::now();
    inkling2::GpuDecoder decoder(model, 8192);
    std::printf("decoder: weights on the device in %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              load_start)
                    .count());
    decoder.SetDebug(compare);
    std::vector<float> logits;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (i + 1 == tokens.size())
        logits = decoder.Feed(tokens[i]);
      else
        decoder.FeedSilent(tokens[i]);
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (compare) {
      inkling2::Reference reference(reader, config);
      for (std::size_t i = 0; i < tokens.size(); ++i)
        (void)reference.Feed(tokens[i], false);
      const auto& expected = reference.trace().l_out;
      const auto& actual = decoder.layer_sums();
      for (std::size_t s = 0; s < expected.size(); ++s)
        std::printf("slot %2zu  l_out cpu %14.5f  gpu %14.5f  diff %10.5f\n", s,
                    expected[s], actual[s], actual[s] - expected[s]);
    }
    std::vector<std::uint32_t> order(logits.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
      order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
    double norm = 0;
    for (const float x : logits)
      norm += std::exp(static_cast<double>(x) - logits[order[0]]);
    const double log_norm = std::log(norm) + logits[order[0]];
    for (int i = 0; i < 10; ++i)
      std::printf("top %2d  token %6u  logit %10.5f  logprob %10.5f\n", i,
                  order[i], logits[order[i]], logits[order[i]] - log_norm);
    std::printf("prompt: %zu tokens in %.3f s (%.2f tokens/s, gpu decoder)\n",
                tokens.size(), seconds, tokens.size() / seconds);
    if (const char* n = std::getenv("INKLING2_REF_GEN")) {
      const int count = std::atoi(n);
      std::uint32_t next = order[0];
      std::printf("greedy:");
      const std::size_t launches_before = decoder.launches();
      const auto gen_start = std::chrono::steady_clock::now();
      for (int i = 0; i < count; ++i) {
        std::printf(" %u", next);
        next = decoder.FeedGreedy(next);
      }
      const double gen_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        gen_start)
              .count();
      std::printf("\ngeneration: %d tokens in %.2f s (%.2f tokens/s, gpu "
                  "decoder, %.0f launches per token)\n",
                  count, gen_seconds, count / gen_seconds,
                  static_cast<double>(decoder.launches() - launches_before) /
                      count);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}

int Prompt(const gufo::ext::TextPromptOptions&,
           const gufo::core::GgufReader& reader,
           std::chrono::steady_clock::time_point) {
  if (const char* ids = std::getenv("INKLING2_REF_TOKENS")) {
    if (const char* v = std::getenv("INKLING2_GPU"); v && *v == '2')
      return RunDecoder(reader, ids);
    return RunReference(reader, ids);
  }
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
