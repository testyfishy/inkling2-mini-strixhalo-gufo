#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "drafter.hpp"
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

// Development aid: INKLING2_GPU=2 runs the same token ids through the GPU
// engine. Settings, all optional:
//   INKLING2_SESSIONS   sessions decoded in parallel (session s gets the
//                       prompt rotated by s tokens, so the sessions differ)
//   INKLING2_BATCH      prompt rows per step (1 feeds the prompt token by token)
//   INKLING2_REPEAT     repeat the token list to make a longer prompt
//   INKLING2_ROTATE     rotate the prompt, to reproduce one session of a
//                       multi-session run on its own
//   INKLING2_REF_GEN    greedy tokens to generate per session
//   INKLING2_PROFILE    per-kernel timing of the generation phase
//   INKLING2_DSPARK     path of the DSpark drafter file: after the plain
//                       generation, generate again with drafts
//                       (INKLING2_DRAFT_N per step, default 7) and report how
//                       many the model accepted
int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoi(v) : fallback;
}

int RunDecoder(const gufo::core::GgufReader& reader, const char* ids) {
  inkling2::Config config;
  std::string error;
  if (!inkling2::LoadConfig(reader, &config, &error) ||
      !inkling2::CheckTensors(reader, config, nullptr, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<std::uint32_t> base;
  for (const char* p = ids; *p;) {
    char* end = nullptr;
    base.push_back(static_cast<std::uint32_t>(std::strtoul(p, &end, 10)));
    if (end == p)
      break;
    p = *end == ',' ? end + 1 : end;
  }
  if (base.empty())
    return 1;
  const int sessions = std::max(1, EnvInt("INKLING2_SESSIONS", 1));
  const int prompt_batch = std::max(1, EnvInt("INKLING2_BATCH", 1));
  const int repeat = std::max(1, EnvInt("INKLING2_REPEAT", 1));
  const int rotate = EnvInt("INKLING2_ROTATE", 0);
  const int generate = EnvInt("INKLING2_REF_GEN", 0);
  std::vector<std::uint32_t> tokens;
  for (int r = 0; r < repeat; ++r)
    tokens.insert(tokens.end(), base.begin(), base.end());
  // INKLING2_TEXT names a llama-perplexity logits file; its token chunks then
  // serve as prompts, a different chunk per session, INKLING2_PROMPT_LEN
  // tokens each. That gives sessions real, different text.
  std::vector<std::int32_t> text;
  std::int32_t text_ctx = 0, text_chunks = 0;
  if (const char* path = std::getenv("INKLING2_TEXT")) {
    std::FILE* file = std::fopen(path, "rb");
    char magic[8];
    std::int32_t vocab = 0;
    const bool ok = file && std::fread(magic, 1, 8, file) == 8 &&
                    std::fread(&text_ctx, 4, 1, file) == 1 &&
                    std::fread(&vocab, 4, 1, file) == 1 &&
                    std::fread(&text_chunks, 4, 1, file) == 1;
    if (ok) {
      text.resize(std::size_t(text_ctx) * text_chunks);
      if (std::fread(text.data(), 4, text.size(), file) != text.size())
        text.clear();
    }
    if (file)
      std::fclose(file);
    if (text.empty()) {
      std::fprintf(stderr, "inkling2: cannot read prompts from %s\n", path);
      return 1;
    }
  }
  const std::size_t n =
      text.empty() ? tokens.size()
                   : static_cast<std::size_t>(std::min<int>(
                         text_ctx, EnvInt("INKLING2_PROMPT_LEN", 64)));
  auto prompt_token = [&](int session, std::size_t i) -> std::uint32_t {
    if (!text.empty())
      return static_cast<std::uint32_t>(
          text[std::size_t((session + rotate) % text_chunks) * text_ctx + i]);
    return tokens[(i + static_cast<std::size_t>(session + rotate)) % n];
  };

  try {
    const inkling2::Model model = inkling2::LoadModel(reader, config);
    const auto load_start = std::chrono::steady_clock::now();
    inkling2::GpuEngine::Options options;
    options.sessions = static_cast<std::uint32_t>(sessions);
    options.max_batch =
        static_cast<std::uint32_t>(std::max(sessions, prompt_batch));
    options.context = static_cast<std::uint32_t>(EnvInt("INKLING2_CONTEXT", 8192));
    options.int8_experts = EnvInt("INKLING2_INT8_EXPERTS", 1) != 0;
    options.int8_single = EnvInt("INKLING2_INT8_SINGLE", 1) != 0;
    const char* dspark = std::getenv("INKLING2_DSPARK");
    if (dspark && sessions == 1) {
      options.capture_features = true;
      options.max_batch = std::max<std::uint32_t>(options.max_batch, 8);
    } else {
      dspark = nullptr;
    }
    inkling2::GpuEngine engine(model, options);
    std::printf("engine: %d session(s), weights on the device in %.1f s\n",
                sessions,
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              load_start)
                    .count());

    // Prompt: each session in turn, `prompt_batch` rows per step.
    std::vector<std::uint32_t> next(sessions);
    std::vector<float> first_logits;
    const bool profile_prompt = std::getenv("INKLING2_PROFILE_PROMPT") != nullptr;
    engine.SetProfile(profile_prompt);
    const auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < sessions; ++s) {
      for (std::size_t i = 0; i < n; i += prompt_batch) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j)
          rows.push_back({static_cast<std::uint32_t>(s), prompt_token(s, j)});
        const bool final_chunk = i + prompt_batch >= n;
        engine.Step(rows, final_chunk);
        if (final_chunk) {
          next[s] = engine.argmax(0);
          if (s == 0)
            first_logits = engine.capped_logits(0);
        }
      }
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    std::printf("prompt: %zu tokens x %d session(s) in %.3f s (%.1f tokens/s, "
                "%d row(s) per step)\n",
                n, sessions, seconds, n * sessions / seconds, prompt_batch);

    if (profile_prompt) {
      const double steps = std::ceil(double(n) / prompt_batch) * sessions;
      for (const auto& [name, phase_seconds] : engine.phases())
        std::printf("  %-34s %8.3f ms per prompt step", name.c_str(),
                    phase_seconds / steps * 1e3),
            std::printf("%c", 10);
      engine.SetProfile(false);
    }
    std::vector<std::uint32_t> order(first_logits.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
      order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return first_logits[a] > first_logits[b];
                      });
    double norm = 0;
    for (const float x : first_logits)
      norm += std::exp(static_cast<double>(x) - first_logits[order[0]]);
    const double log_norm = std::log(norm) + first_logits[order[0]];
    for (int i = 0; i < 5; ++i)
      std::printf("top %2d  token %6u  logit %10.5f  logprob %10.5f\n", i,
                  order[i], first_logits[order[i]],
                  first_logits[order[i]] - log_norm);

    // INKLING2_FAKE_CONTEXT: time generation as if each session already held
    // that many tokens (see GpuEngine::FakeContext). The tokens generated
    // afterwards are meaningless.
    if (const int fake = EnvInt("INKLING2_FAKE_CONTEXT", 0); fake > 0) {
      for (int s = 0; s < sessions; ++s)
        engine.FakeContext(static_cast<std::uint32_t>(s),
                           static_cast<std::uint32_t>(fake));
      std::printf("context faked to about %d tokens per session (position %u)",
                  fake, engine.position(0));
      std::printf("%c", 10);
    }
    // INKLING2_DEEP_PROMPT: after faking the context, time that many more
    // prompt tokens for session 0, to see prompt speed at depth.
    if (const int deep = EnvInt("INKLING2_DEEP_PROMPT", 0); deep > 0) {
      const bool profile_deep = std::getenv("INKLING2_PROFILE_DEEP") != nullptr;
      const auto phases_before = engine.phases();
      engine.SetProfile(profile_deep);
      const auto deep_start = std::chrono::steady_clock::now();
      for (int i = 0; i < deep; i += prompt_batch) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (int j = i; j < std::min(deep, i + prompt_batch); ++j)
          rows.push_back({0, prompt_token(0, static_cast<std::size_t>(j) % n)});
        engine.Step(rows, false);
      }
      const double deep_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        deep_start)
              .count();
      std::printf("deep prompt: %d tokens at position %u in %.2f s (%.1f "
                  "tokens/s, %d row(s) per step)",
                  deep, engine.position(0) - deep, deep_seconds,
                  deep / deep_seconds, prompt_batch);
      std::printf("%c", 10);
      if (profile_deep) {
        const auto phases_after = engine.phases();
        const double steps = std::ceil(double(deep) / prompt_batch);
        for (std::size_t p = 0; p < phases_after.size(); ++p)
          std::printf("  %-34s %8.3f ms per step%c", phases_after[p].first.c_str(),
                      (phases_after[p].second - phases_before[p].second) /
                          steps * 1e3,
                      10);
        engine.SetProfile(false);
      }
    }
    if (generate > 0) {
      std::vector<std::vector<std::uint32_t>> out(sessions);
      const std::size_t launches_before = engine.launches();
      engine.SetProfile(std::getenv("INKLING2_PROFILE") != nullptr);
      const auto gen_start = std::chrono::steady_clock::now();
      for (int i = 0; i < generate; ++i) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (int s = 0; s < sessions; ++s) {
          out[s].push_back(next[s]);
          rows.push_back({static_cast<std::uint32_t>(s), next[s]});
        }
        engine.Step(rows, true);
        for (int s = 0; s < sessions; ++s)
          next[s] = engine.argmax(static_cast<std::size_t>(s));
      }
      const double gen_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        gen_start)
              .count();
      for (int s = 0; s < sessions; ++s) {
        std::printf("greedy session %d:", s);
        for (const std::uint32_t token : out[s])
          std::printf(" %u", token);
        std::printf("\n");
      }
      std::printf("generation: %d tokens x %d session(s) in %.2f s: %.2f "
                  "tokens/s total, %.2f per session, %.0f launches per step\n",
                  generate, sessions, gen_seconds,
                  generate * sessions / gen_seconds, generate / gen_seconds,
                  static_cast<double>(engine.launches() - launches_before) /
                      generate);
      if (const int k = EnvInt("INKLING2_VERIFY_TEST", 0); k > 0 && sessions == 1) {
        // Replay session 0 from scratch, verifying k "drafted" tokens per
        // step. Every third draft block is corrupted from its second token.
        engine.Reset(0);
        for (std::size_t i = 0; i < n; i += prompt_batch) {
          std::vector<inkling2::GpuEngine::Row> rows;
          for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j)
            rows.push_back({0, prompt_token(0, j)});
          engine.Step(rows, i + prompt_batch >= n);
        }
        const std::vector<std::uint32_t>& truth = out[0];
        std::vector<std::uint32_t> replay;
        std::uint32_t current = engine.argmax(0);
        int steps = 0, kept = 0;
        // INKLING2_VERIFY_CORRUPT=0 keeps every draft correct (the best case).
        const bool corrupt = EnvInt("INKLING2_VERIFY_CORRUPT", 1) != 0;
        if (const int fake = EnvInt("INKLING2_FAKE_CONTEXT", 0); fake > 0)
          engine.FakeContext(0, static_cast<std::uint32_t>(fake));
        const auto verify_start = std::chrono::steady_clock::now();
        while (replay.size() < truth.size()) {
          replay.push_back(current);
          const std::size_t at = replay.size();  // drafts for truth[at...]
          std::vector<inkling2::GpuEngine::Row> rows{{0, current}};
          for (int d = 0; d < k && at + d < truth.size(); ++d) {
            std::uint32_t draft = truth[at + d];
            if (corrupt && steps % 3 == 2 && d >= 1)
              draft = (draft + 7) % engine.vocab();  // a wrong draft
            rows.push_back({0, draft});
          }
          engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
          // Row i's prediction is checked against the draft in row i + 1.
          std::size_t good = 0;
          while (good + 1 < rows.size() &&
                 engine.argmax(good) == rows[good + 1].token)
            ++good;
          current = engine.argmax(good);
          for (std::size_t i = 1; i <= good && replay.size() < truth.size(); ++i)
            replay.push_back(rows[i].token);
          engine.Accept(0, static_cast<std::uint32_t>(good + 1));
          kept += static_cast<int>(good);
          ++steps;
        }
        replay.resize(truth.size());
        const double verify_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          verify_start)
                .count();
        std::printf("verify test: %d steps for %zu tokens in %.2f s (%.1f "
                    "tokens/s with given drafts, no drafter cost), %d drafts "
                    "kept, output %s plain generation",
                    steps, truth.size(), verify_seconds,
                    truth.size() / verify_seconds, kept,
                    replay == truth ? "EQUALS" : "DIFFERS FROM");
        std::printf("%c", 10);
      }
      if (dspark) {
        // Generate the same tokens again, this time with the drafter
        // proposing tokens and the model verifying them in one step.
        const int draft_n = std::clamp(EnvInt("INKLING2_DRAFT_N", 7), 1, 7);
        inkling2::Drafter drafter(dspark, model);
        const inkling2::Drafter::Head lm_head = [&](const float* hidden) {
          return engine.HeadArgmax(hidden);
        };
        engine.Reset(0);
        double sums[4] = {};
        std::uint32_t current = 0;
        bool head_ok = true;
        const auto inject_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < n; i += prompt_batch) {
          std::vector<inkling2::GpuEngine::Row> rows;
          for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j)
            rows.push_back({0, prompt_token(0, j)});
          const bool final_chunk = i + prompt_batch >= n;
          engine.Step(rows, final_chunk);
          if (final_chunk)
            current = engine.argmax(0);
          std::vector<float> last;
          for (std::size_t r = 0; r < rows.size(); ++r) {
            last = engine.features(r);
            for (int f = 0; f < 4; ++f)
              for (std::size_t k = 0; k < 2304; ++k)
                sums[f] += last[std::size_t(f) * 2304 + k];
            drafter.Inject(last.data());
          }
          // The captured final state through the head must give the step's
          // own prediction.
          if (final_chunk)
            head_ok = engine.HeadArgmax(last.data() + 3 * 2304) == current;
        }
        std::printf("drafter: %zu positions injected in %.2f s; feature sums "
                    "f5 %.6f f23 %.6f f41 %.6f f48 %.6f; head check %s",
                    n,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - inject_start)
                        .count(),
                    sums[0], sums[1], sums[2], sums[3],
                    head_ok ? "ok" : "FAILED");
        std::printf("%c", 10);
        const std::vector<std::uint32_t>& truth = out[0];
        std::vector<std::uint32_t> replay;
        int steps = 0, drafted = 0, kept = 0;
        int kept_at[8] = {};  // steps that kept at least i drafts
        double draft_seconds = 0;
        const auto spec_start = std::chrono::steady_clock::now();
        while (replay.size() < truth.size()) {
          replay.push_back(current);
          const auto draft_start = std::chrono::steady_clock::now();
          const std::vector<std::uint32_t> drafts =
              drafter.Draft(current, draft_n, lm_head);
          draft_seconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - draft_start)
                               .count();
          std::vector<inkling2::GpuEngine::Row> rows{{0, current}};
          for (const std::uint32_t draft : drafts)
            rows.push_back({0, draft});
          engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
          std::size_t good = 0;
          while (good + 1 < rows.size() &&
                 engine.argmax(good) == rows[good + 1].token)
            ++good;
          current = engine.argmax(good);
          if (steps < 4) {
            std::printf("  step %d drafts:", steps);
            for (const std::uint32_t draft : drafts)
              std::printf(" %u", draft);
            std::printf("  model:");
            for (std::size_t r = 0; r + 1 < rows.size(); ++r)
              std::printf(" %u", engine.argmax(r));
            std::printf("  kept %zu%c", good, 10);
          }
          for (std::size_t r = 1; r <= good; ++r)
            replay.push_back(rows[r].token);
          // Features have to be read before the next head call or step.
          std::vector<std::vector<float>> features;
          for (std::size_t r = 0; r <= good; ++r)
            features.push_back(engine.features(r));
          engine.Accept(0, static_cast<std::uint32_t>(good + 1));
          for (const auto& f : features)
            drafter.Inject(f.data());
          drafted += static_cast<int>(drafts.size());
          kept += static_cast<int>(good);
          for (std::size_t r = 1; r <= good; ++r)
            ++kept_at[r];
          ++steps;
        }
        const double spec_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          spec_start)
                .count();
        const std::size_t produced = replay.size();
        replay.resize(truth.size());
        for (std::size_t i = 0; i < truth.size(); ++i)
          if (replay[i] != truth[i]) {
            std::printf("  first difference at token %zu: drafted run %u, "
                        "plain run %u%c",
                        i, replay[i], truth[i], 10);
            break;
          }
        std::printf("drafted generation: %zu tokens in %d steps (%.2f tokens "
                    "per step), %d of %d drafts kept (%.1f%%); output %s plain "
                    "generation",
                    produced, steps, double(produced) / steps, kept, drafted,
                    100.0 * kept / std::max(1, drafted),
                    replay == truth ? "EQUALS" : "DIFFERS FROM");
        std::printf("%c  steps keeping at least 1..7 drafts:", 10);
        for (int r = 1; r <= draft_n; ++r)
          std::printf(" %d", kept_at[r]);
        std::printf("%c  time: %.2f s in all, %.2f s of it in the CPU drafter "
                    "prototype; %.1f tokens/s if drafting were free%c",
                    10, spec_seconds, draft_seconds,
                    produced / std::max(1e-9, spec_seconds - draft_seconds), 10);
      }
      if (std::getenv("INKLING2_PROFILE"))
        for (const auto& [name, phase_seconds] : engine.phases())
          std::printf("  %-34s %7.3f ms per step\n", name.c_str(),
                      phase_seconds / generate * 1e3);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}

// Quality check against llama.cpp: INKLING2_KLD names a file written by
// `llama-perplexity --kl-divergence-base` on the same GGUF. The same token
// chunks are run through the GPU engine and, as llama.cpp does, the second
// half of each chunk is scored: perplexity of both, mean KL divergence from
// llama.cpp's distribution to ours, and how often the top token agrees.
// INKLING2_KLD_CHUNKS limits the chunks; INKLING2_BATCH sets rows per step.
int RunQuality(const gufo::core::GgufReader& reader, const char* path) {
  inkling2::Config config;
  std::string error;
  if (!inkling2::LoadConfig(reader, &config, &error) ||
      !inkling2::CheckTensors(reader, config, nullptr, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::FILE* file = std::fopen(path, "rb");
  if (!file) {
    std::fprintf(stderr, "inkling2: cannot open %s\n", path);
    return 1;
  }
  char magic[8];
  std::int32_t n_ctx = 0, n_vocab = 0, n_chunk = 0;
  bool ok = std::fread(magic, 1, 8, file) == 8 &&
            std::memcmp(magic, "_logits_", 8) == 0 &&
            std::fread(&n_ctx, 4, 1, file) == 1 &&
            std::fread(&n_vocab, 4, 1, file) == 1 &&
            std::fread(&n_chunk, 4, 1, file) == 1;
  if (!ok || n_vocab != static_cast<std::int32_t>(config.n_vocab) ||
      n_ctx < 4 || n_chunk < 1) {
    std::fprintf(stderr, "inkling2: %s is not a matching logits file\n", path);
    std::fclose(file);
    return 1;
  }
  std::vector<std::int32_t> tokens(std::size_t(n_chunk) * n_ctx);
  ok = std::fread(tokens.data(), 4, tokens.size(), file) == tokens.size();
  const int chunks = std::min<int>(n_chunk, EnvInt("INKLING2_KLD_CHUNKS", n_chunk));
  const int batch = std::max(1, EnvInt("INKLING2_BATCH", 16));
  const int first = n_ctx / 2;
  const int scored = n_ctx - 1 - first;
  // One record per scored position: a scale, an offset, then 16-bit log-probs.
  const std::size_t record = 2 * ((std::size_t(n_vocab) + 1) / 2) + 4;
  std::vector<std::uint16_t> base(std::size_t(scored) * record);

  try {
    const inkling2::Model model = inkling2::LoadModel(reader, config);
    inkling2::GpuEngine::Options options;
    options.sessions = 1;
    options.max_batch = static_cast<std::uint32_t>(batch);
    options.context = static_cast<std::uint32_t>(std::max(n_ctx, 1024));
    options.int8_experts = EnvInt("INKLING2_INT8_EXPERTS", 1) != 0;
    options.int8_single = EnvInt("INKLING2_INT8_SINGLE", 1) != 0;
    inkling2::GpuEngine engine(model, options);

    double nll = 0, nll_base = 0, kld_sum = 0, kld_max = 0;
    std::size_t count = 0, same_top = 0;
    std::vector<double> klds;
    std::vector<float> log_p(n_vocab);
    const auto start = std::chrono::steady_clock::now();
    for (int chunk = 0; chunk < chunks && ok; ++chunk) {
      ok = std::fread(base.data(), 2, base.size(), file) == base.size();
      if (!ok)
        break;
      const std::int32_t* tok = tokens.data() + std::size_t(chunk) * n_ctx;
      engine.Reset(0);
      // Positions 0 .. n_ctx - 2 are fed; logits are needed from `first` on.
      for (int i = 0; i < n_ctx - 1; i += batch) {
        const int end = std::min(n_ctx - 1, i + batch);
        std::vector<inkling2::GpuEngine::Row> rows;
        for (int j = i; j < end; ++j)
          rows.push_back({0, static_cast<std::uint32_t>(tok[j])});
        const bool need = end > first;
        engine.Step(rows, need ? inkling2::GpuEngine::Head::kAll
                               : inkling2::GpuEngine::Head::kNone);
        for (int j = std::max(i, first); j < end; ++j) {
          const std::vector<float> logits = engine.capped_logits(j - i);
          int top = 0;
          for (int v = 1; v < n_vocab; ++v)
            if (logits[v] > logits[top])
              top = v;
          double sum = 0;
          for (int v = 0; v < n_vocab; ++v)
            sum += std::exp(static_cast<double>(logits[v]) - logits[top]);
          const double log_sum = std::log(sum) + logits[top];
          const std::uint16_t* rec = base.data() + std::size_t(j - first) * record;
          float scale, min_log_prob;
          std::memcpy(&scale, rec, 4);
          std::memcpy(&min_log_prob, rec + 2, 4);
          const std::uint16_t* q = rec + 4;
          // As llama.cpp computes it: its stored log-probabilities are used as
          // they are, and entries at or below its floor of -16 are skipped.
          int base_top = 0;
          double kld = 0;
          for (int v = 0; v < n_vocab; ++v) {
            log_p[v] = scale * q[v] + min_log_prob;
            if (q[v] > q[base_top])
              base_top = v;
            if (log_p[v] > -16.0f)
              kld += std::exp(static_cast<double>(log_p[v])) *
                     (log_p[v] - (logits[v] - log_sum));
          }
          const int target = tok[j + 1];
          nll += log_sum - logits[target];
          nll_base += -log_p[target];
          kld_sum += kld;
          kld_max = std::max(kld_max, kld);
          klds.push_back(kld);
          same_top += top == base_top ? 1 : 0;
          ++count;
        }
      }
      std::printf("chunk %d/%d: ppl %.4f (llama.cpp %.4f), mean KLD %.6f, "
                  "same top token %.2f%%\n",
                  chunk + 1, chunks, std::exp(nll / count),
                  std::exp(nll_base / count), kld_sum / count,
                  100.0 * same_top / count);
      std::fflush(stdout);
    }
    std::fclose(file);
    if (count == 0) {
      std::fprintf(stderr, "inkling2: no positions scored\n");
      return 1;
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    std::sort(klds.begin(), klds.end());
    std::printf("\nquality over %zu positions (%d chunks of %d tokens, second "
                "half scored):\n",
                count, chunks, n_ctx);
    std::printf("  perplexity, this engine   %.4f\n", std::exp(nll / count));
    std::printf("  perplexity, llama.cpp     %.4f\n", std::exp(nll_base / count));
    std::printf("  mean KLD vs llama.cpp     %.6f\n", kld_sum / count);
    std::printf("  median / 99%% / max KLD    %.6f / %.6f / %.6f\n",
                klds[klds.size() / 2], klds[klds.size() * 99 / 100], kld_max);
    std::printf("  same top token            %.2f%%\n",
                100.0 * same_top / count);
    std::printf("  evaluated in %.1f s (%.1f tokens/s, %d rows per step)\n",
                seconds, std::size_t(chunks) * (n_ctx - 1) / seconds, batch);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}

int Prompt(const gufo::ext::TextPromptOptions&,
           const gufo::core::GgufReader& reader,
           std::chrono::steady_clock::time_point) {
  if (const char* kld = std::getenv("INKLING2_KLD"))
    return RunQuality(reader, kld);
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
