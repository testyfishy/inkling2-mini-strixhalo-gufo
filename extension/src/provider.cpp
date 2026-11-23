#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gufo/ext/generated/inkling2_native_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include <gufo/ext/json.hpp>

#include "backend.hpp"
#include "chat.hpp"
#include "config.hpp"
#include "decoder.hpp"
#include "drafter.hpp"
#include "gpu.hpp"
#include "hip/bench.hpp"
#include "model.hpp"
#include "reference.hpp"
#include "serve.hpp"
#include "tokenizer.hpp"

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

// INKLING2_TOKENIZE=<text file>: tokenizes the file and writes the token ids
// as 32-bit integers to INKLING2_TOKENIZE_OUT. With INKLING2_TOKENIZE_CONTROL
// the spellings of control tokens in the text become those tokens.
int RunTokenize(const gufo::core::GgufReader& reader, const char* path) {
  try {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot read the text file");
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    auto t0 = std::chrono::steady_clock::now();
    const inkling2::Tokenizer tokenizer(reader);
    const double load_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    const bool control = std::getenv("INKLING2_TOKENIZE_CONTROL") != nullptr;
    t0 = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> ids;
    if (control)
      ids = tokenizer.EncodeWithControl(text);
    else
      tokenizer.Encode(text, &ids);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    std::string back;
    for (const std::uint32_t id : ids)
      back += tokenizer.Bytes(id);
    std::printf("tokenizer: %u entries, loaded in %.2f s; %zu bytes -> %zu "
                "tokens in %.3f s (%.0f tokens/s); decodes back to the text: %s",
                tokenizer.size(), load_s, text.size(), ids.size(), seconds,
                ids.size() / std::max(seconds, 1e-9),
                back == text ? "yes" : "NO");
    std::printf("%c", 10);
    if (const char* out_path = std::getenv("INKLING2_TOKENIZE_OUT")) {
      std::ofstream out(out_path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(ids.data()),
                static_cast<std::streamsize>(ids.size() * 4));
      if (!out)
        throw std::runtime_error("cannot write the token file");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s%c", e.what(), 10);
    return 1;
  }
  return 0;
}

// INKLING2_CHAT=<JSON file>: an array of conversations in the OpenAI format
// ({"messages": [...], "tools": [...], "reasoning_effort": "low"}). Renders
// each with the chat template and writes case-N.txt (the prompt as text) and
// case-N.i32 (its tokens) into INKLING2_CHAT_OUT.
int RunChatCases(const gufo::core::GgufReader& reader, const char* path) {
  namespace json = gufo::json;
  try {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot read the conversations");
    const std::string source((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    const json::Value cases = json::parse(source);
    const inkling2::Tokenizer tokenizer(reader);
    const char* out_dir = std::getenv("INKLING2_CHAT_OUT");
    std::size_t n = 0;
    for (const json::Value& item : cases.items()) {
      std::vector<inkling2::ChatMessage> messages;
      if (const json::Value* list = item.find("messages"))
        for (const json::Value& m : list->items()) {
          inkling2::ChatMessage message;
          const std::string role = m.member_str("role");
          message.role = role == "system" || role == "developer"
                             ? inkling2::ChatMessage::Role::kSystem
                         : role == "assistant"
                             ? inkling2::ChatMessage::Role::kAssistant
                         : role == "tool" ? inkling2::ChatMessage::Role::kTool
                                          : inkling2::ChatMessage::Role::kUser;
          message.content = m.member_str("content");
          message.reasoning = m.member_str("reasoning_content");
          message.name = m.member_str("name");
          message.tool_call_id = m.member_str("tool_call_id");
          if (const json::Value* calls = m.find("tool_calls"))
            for (const json::Value& c : calls->items()) {
              inkling2::ChatToolCall call;
              call.id = c.member_str("id");
              if (const json::Value* fn = c.find("function")) {
                call.name = fn->member_str("name");
                if (const json::Value* args = fn->find("arguments"))
                  for (const auto& [key, value] : args->members())
                    call.arguments.push_back(
                        {key,
                         value.is_string()
                             ? value.str()
                             : value.dump(json::Value::Numbers::kAsParsed),
                         value.is_string()});
              }
              message.tool_calls.push_back(std::move(call));
            }
          messages.push_back(std::move(message));
        }
      std::vector<inkling2::ChatTool> tools;
      if (const json::Value* list = item.find("tools"))
        for (const json::Value& t : list->items()) {
          const json::Value* fn = t.find("function");
          if (!fn)
            fn = &t;
          inkling2::ChatTool tool;
          tool.name = fn->member_str("name");
          tool.description = fn->member_str("description");
          if (const json::Value* p = fn->find("parameters"))
            tool.parameters_json = p->dump(json::Value::Numbers::kAsParsed);
          tool.type = t.member_str("type", "function");
          tools.push_back(std::move(tool));
        }
      inkling2::ChatOptions options;
      if (const json::Value* effort = item.find("reasoning_effort"))
        options.effort = effort->is_string()
                             ? inkling2::EffortByName(effort->str())
                             : effort->as_double();
      const inkling2::RenderedChat chat =
          inkling2::RenderChat(tokenizer, messages, tools, options);
      const bool same = chat.tokens == tokenizer.EncodeWithControl(chat.text);
      std::printf("case %zu: %zu tokens; equal to tokenizing the text with "
                  "control tokens: %s",
                  n, chat.tokens.size(), same ? "yes" : "no");
      std::printf("%c", 10);
      if (out_dir) {
        const std::string base =
            std::string(out_dir) + "/case-" + std::to_string(n);
        std::ofstream text(base + ".txt", std::ios::binary);
        text.write(chat.text.data(),
                   static_cast<std::streamsize>(chat.text.size()));
        std::ofstream ids(base + ".i32", std::ios::binary);
        ids.write(reinterpret_cast<const char*>(chat.tokens.data()),
                  static_cast<std::streamsize>(chat.tokens.size() * 4));
      }
      ++n;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s%c", e.what(), 10);
    return 1;
  }
  return 0;
}

// INKLING2_SWEEP=<file of raw 32-bit token ids>: processes the text as one
// prompt and, at each depth of INKLING2_SWEEP_POINTS (comma separated),
// reports how fast the prompt was processed since the previous point, how
// fast the last full step before the point ran, and how long a step that
// generates one token takes there (the median of 12 tentative steps, which
// compute everything and are then dropped), likewise a step that verifies 8
// drafted tokens. INKLING2_BATCH sets the rows per prompt step.
int RunDepthSweep(const gufo::core::GgufReader& reader, const char* path) {
  using Clock = std::chrono::steady_clock;
  const auto seconds = [](Clock::time_point from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
  };
  try {
    std::vector<std::uint32_t> text;
    if (std::FILE* file = std::fopen(path, "rb")) {
      std::uint32_t id;
      while (std::fread(&id, 4, 1, file) == 1)
        text.push_back(id);
      std::fclose(file);
    }
    std::vector<std::size_t> points;
    {
      const char* list = std::getenv("INKLING2_SWEEP_POINTS");
      std::string spec = list ? list
                              : "1024,4096,16384,32768,65536,100000,131072,"
                                "196608,262144,393216,524288,655360,786432,"
                                "917504,1040000";
      for (std::size_t at = 0; at < spec.size();) {
        const std::size_t comma = std::min(spec.find(',', at), spec.size());
        points.push_back(static_cast<std::size_t>(
            std::atoll(spec.substr(at, comma - at).c_str())));
        at = comma + 1;
      }
    }
    while (!points.empty() && points.back() + 16 > text.size())
      points.pop_back();
    if (points.empty())
      throw std::runtime_error("the token file is shorter than every point");
    inkling2::Config config;
    std::string error;
    if (!inkling2::LoadConfig(reader, &config, &error))
      throw std::runtime_error(error);
    const inkling2::Model model = inkling2::LoadModel(reader, config);
    const inkling2::Tokenizer tokenizer(reader);
    const std::size_t batch =
        static_cast<std::size_t>(std::max(EnvInt("INKLING2_BATCH", 2048), 16));
    inkling2::GpuEngine::Options options;
    options.sessions = 1;
    options.max_batch = static_cast<std::uint32_t>(batch);
    options.context =
        static_cast<std::uint32_t>((points.back() + 16 + 1023) / 1024 * 1024);
    inkling2::GpuEngine engine(model, options);
    std::printf("depth sweep: %zu tokens of text, %zu rows per prompt step, "
                "context %u%c",
                text.size(), batch, options.context, 10);
    std::printf("%9s %12s %12s %10s %9s %9s %10s  %s%c", "depth", "prompt t/s",
                "prompt t/s", "seconds", "1 token", "gen t/s", "8 drafts",
                "next token", 10);
    std::printf("%9s %12s %12s %10s %9s %9s %10s%c", "", "(interval)",
                "(at depth)", "(total)", "(ms)", "", "(ms)", 10);
    std::fflush(stdout);
    std::size_t pos = 0;
    double total = 0;
    for (const std::size_t point : points) {
      const std::size_t from = pos;
      const auto t0 = Clock::now();
      double last_rate = 0;
      while (pos < point) {
        const std::size_t rows_now = std::min(batch, point - pos);
        std::vector<inkling2::GpuEngine::Row> rows;
        rows.reserve(rows_now);
        for (std::size_t i = 0; i < rows_now; ++i)
          rows.push_back({0, text[pos + i]});
        const auto step0 = Clock::now();
        engine.Step(rows, inkling2::GpuEngine::Head::kNone);
        engine.Synchronize();
        if (rows_now == batch)
          last_rate = static_cast<double>(rows_now) / seconds(step0);
        pos += rows_now;
      }
      const double interval = seconds(t0);
      total += interval;
      // One generated token at this depth.
      std::vector<double> ms;
      std::uint32_t next = 0;
      for (int rep = 0; rep < 12; ++rep) {
        const auto g0 = Clock::now();
        engine.Step({{0, text[pos]}}, inkling2::GpuEngine::Head::kLast, true);
        next = engine.argmax(0);
        ms.push_back(seconds(g0) * 1e3);
        engine.Accept(0, 0);
      }
      std::sort(ms.begin(), ms.end());
      const double one = ms[ms.size() / 2];
      std::vector<double> ms8;
      for (int rep = 0; rep < 3; ++rep) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (std::size_t i = 0; i < 9; ++i)
          rows.push_back({0, text[pos + i]});
        const auto g0 = Clock::now();
        engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
        (void)engine.argmax(8);
        ms8.push_back(seconds(g0) * 1e3);
        engine.Accept(0, 0);
      }
      std::sort(ms8.begin(), ms8.end());
      std::string piece(tokenizer.Bytes(next));
      for (char& c : piece)
        if (c == 10 || c == 13)
          c = ' ';
      const bool right = next == text[pos + 1];
      std::printf("%9zu %12.1f %12.1f %10.1f %9.2f %9.2f %10.1f  \"%s\"%s%c",
                  point, static_cast<double>(point - from) / interval,
                  last_rate, total, one, 1e3 / one, ms8[1], piece.c_str(),
                  right ? " (as in the text)" : "", 10);
      std::fflush(stdout);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s%c", e.what(), 10);
    return 1;
  }
  return 0;
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
  // INKLING2_TOKENS names a file of raw 32-bit token ids: one long prompt.
  if (const char* path = std::getenv("INKLING2_TOKENS")) {
    std::FILE* file = std::fopen(path, "rb");
    if (file) {
      std::int32_t id;
      while (std::fread(&id, 4, 1, file) == 1)
        text.push_back(id);
      std::fclose(file);
    }
    if (text.empty()) {
      std::fprintf(stderr, "inkling2: cannot read tokens from %s\n", path);
      return 1;
    }
    text_ctx = static_cast<std::int32_t>(text.size());
    text_chunks = 1;
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
    // INKLING2_DSPARK_GPU=1: the drafter on the device (2: also the host
    // prototype beside it, to compare their proposals).
    const int dspark_gpu = dspark ? EnvInt("INKLING2_DSPARK_GPU", 0) : 0;
    if (dspark && sessions == 1) {
      options.capture_features = dspark_gpu != 1;
      options.max_batch = std::max<std::uint32_t>(options.max_batch, 8);
      if (dspark_gpu) {
        options.drafter = inkling2::LoadDrafter(dspark, model);
        options.draft_window = static_cast<std::uint32_t>(
            EnvInt("INKLING2_DRAFT_WINDOW", 8192));
        options.draft_vocab = static_cast<std::uint32_t>(
            EnvInt("INKLING2_DRAFT_VOCAB", 32768));
      }
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
    // INKLING2_SNAPSHOT_TEST: save session 0 to host memory, overwrite it by
    // processing other tokens from position 0, and restore it; what follows
    // must then be what a run without this gives.
    if (EnvInt("INKLING2_SNAPSHOT_TEST", 0) != 0 && sessions == 1) {
      auto t0 = std::chrono::steady_clock::now();
      const std::vector<std::uint8_t> saved = engine.Save(0);
      const double save_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      engine.Reset(0);
      const std::size_t scramble = std::min<std::size_t>(n, 4096);
      for (std::size_t i = 0; i < scramble; i += prompt_batch) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (std::size_t j = i; j < std::min(scramble, i + prompt_batch); ++j)
          rows.push_back({0, prompt_token(0, n - 1 - j)});  // the prompt backwards
        engine.Step(rows, false);
      }
      t0 = std::chrono::steady_clock::now();
      engine.Restore(0, saved);
      const double restore_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
      std::printf("snapshot: %u positions, %.2f GB, saved in %.2f s, restored "
                  "in %.2f s after the session was overwritten",
                  engine.position(0), saved.size() / 1e9, save_s, restore_s);
      std::printf("%c", 10);
    }
    if (generate > 0) {
      std::vector<std::vector<std::uint32_t>> out(sessions);
      const std::size_t launches_before = engine.launches();
      engine.SetProfile(std::getenv("INKLING2_PROFILE") != nullptr);
      const auto gen_start = std::chrono::steady_clock::now();
      // INKLING2_SHORTLIST_SIM: how often the next token is among the prompt's
      // tokens, the tokens generated so far and the K most likely tokens of
      // every earlier step (K = 16, 64, 256, 1024).
      const bool shortlist_sim = EnvInt("INKLING2_SHORTLIST_SIM", 0) != 0;
      constexpr int kSimK[4] = {16, 64, 256, 1024};
      std::vector<std::vector<char>> listed(
          4, std::vector<char>(shortlist_sim ? engine.vocab() : 0, 0));
      std::size_t sim_hits[4][2] = {}, sim_size[4] = {}, sim_late = 0;
      if (shortlist_sim)
        for (auto& set : listed)
          for (std::size_t i = 0; i < n; ++i)
            set[prompt_token(0, i)] = 1;
      for (int i = 0; i < generate; ++i) {
        std::vector<inkling2::GpuEngine::Row> rows;
        for (int s = 0; s < sessions; ++s) {
          out[s].push_back(next[s]);
          rows.push_back({static_cast<std::uint32_t>(s), next[s]});
        }
        if (shortlist_sim && i > 0) {
          const float* raw = engine.logits(0);
          std::vector<std::uint32_t> order(engine.vocab());
          for (std::uint32_t t = 0; t < order.size(); ++t)
            order[t] = t;
          std::partial_sort(order.begin(), order.begin() + 1024, order.end(),
                            [&](std::uint32_t a, std::uint32_t b) {
                              return raw[a] > raw[b];
                            });
          sim_late += i >= 32 ? 1 : 0;
          for (int k = 0; k < 4; ++k) {
            // Drafts cover up to seven positions ahead; what matters is
            // whether the token was listed before its own step.
            sim_hits[k][0] += listed[k][next[0]];
            sim_hits[k][1] += i >= 32 ? listed[k][next[0]] : 0;
            for (int j = 0; j < kSimK[k]; ++j)
              listed[k][order[j]] = 1;
            listed[k][next[0]] = 1;
          }
        }
        engine.Step(rows, true);
        for (int s = 0; s < sessions; ++s)
          next[s] = engine.argmax(static_cast<std::size_t>(s));
      }
      const double gen_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        gen_start)
              .count();
      if (shortlist_sim) {
        for (int k = 0; k < 4; ++k) {
          for (const char c : listed[k])
            sim_size[k] += c;
          std::printf("shortlist sim, %4d most likely per step: next token "
                      "listed in %.1f%% of steps (%.1f%% after the first 32); "
                      "%zu tokens listed%c",
                      kSimK[k], 100.0 * sim_hits[k][0] / std::max(1, generate - 1),
                      100.0 * sim_hits[k][1] / std::max<std::size_t>(1, sim_late),
                      sim_size[k], 10);
        }
      }
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
      if (const int k = EnvInt("INKLING2_STALE_TEST", 0); k > 0 && sessions == 1) {
        // A tentative step of k drafts after the prompt, once with whatever
        // the generation above left in the caches beyond the prompt and once
        // with the caches zeroed first: the logits of every row must be the
        // same, since no row may read past its own position.
        const std::vector<std::uint32_t>& truth = out[0];
        std::vector<std::vector<std::uint64_t>> hashes;
        for (int pass = 0; pass < 2; ++pass) {
          if (pass == 0)
            engine.Reset(0);
          else
            engine.Scrub(0);
          for (std::size_t i = 0; i < n; i += prompt_batch) {
            std::vector<inkling2::GpuEngine::Row> rows;
            for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j)
              rows.push_back({0, prompt_token(0, j)});
            engine.Step(rows, i + prompt_batch >= n);
          }
          const int warm = EnvInt("INKLING2_STALE_WARM", 0);
          for (int i = 0; i < warm; ++i)
            engine.Step({{0, truth[static_cast<std::size_t>(i)]}}, true);
          std::vector<inkling2::GpuEngine::Row> rows;
          for (int d = 0; d <= k; ++d)
            rows.push_back({0, truth[static_cast<std::size_t>(warm + d)]});
          engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
          std::vector<std::uint64_t> h;
          for (std::size_t r = 0; r < rows.size(); ++r) {
            const float* raw = engine.logits(r);
            std::uint64_t hash = 1469598103934665603ull;
            for (std::uint32_t j = 0; j < engine.vocab(); ++j) {
              std::uint32_t bits;
              std::memcpy(&bits, raw + j, 4);
              hash = (hash ^ bits) * 1099511628211ull;
            }
            h.push_back(hash);
          }
          engine.Accept(0, 0);
          hashes.push_back(h);
        }
        std::printf("stale test, %d rows at position %u: rows whose logits "
                    "differ:",
                    k + 1, engine.position(0));
        int bad = 0;
        for (std::size_t r = 0; r < hashes[0].size(); ++r)
          if (hashes[0][r] != hashes[1][r]) {
            std::printf(" %zu", r);
            ++bad;
          }
        std::printf("%s%c", bad ? "" : " none", 10);
      }
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
      if (const int k = std::min(EnvInt("INKLING2_LOOKUP", 0), 15);
          k > 0 && sessions == 1) {
        // Generate again from scratch with drafts copied from the context:
        // when the last tokens also occur earlier, what followed them there
        // is proposed, and the model verifies the proposals in one step.
        const int min_match = std::max(1, EnvInt("INKLING2_LOOKUP_MIN", 3));
        engine.Reset(0);
        std::vector<std::uint32_t> ctx;
        for (std::size_t i = 0; i < n; i += prompt_batch) {
          std::vector<inkling2::GpuEngine::Row> rows;
          for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j) {
            rows.push_back({0, prompt_token(0, j)});
            ctx.push_back(rows.back().token);
          }
          engine.Step(rows, i + prompt_batch >= n);
        }
        const std::vector<std::uint32_t>& truth = out[0];
        std::vector<std::uint32_t> replay;
        std::uint32_t current = engine.argmax(0);
        int steps = 0, kept = 0, proposed = 0, plain_steps = 0;
        // How many drafts to risk: it grows while every draft is kept and
        // falls back when one is refused; after a step that kept none, a few
        // steps go without drafts unless the match is long.
        int budget = std::min(k, 3), cooldown = 0;
        const auto lookup_start = std::chrono::steady_clock::now();
        while (replay.size() < truth.size()) {
          replay.push_back(current);
          ctx.push_back(current);
          // The most recent earlier place where the longest suffix (up to 16
          // tokens, at least min_match) occurs.
          std::size_t follow = 0;
          const std::size_t size = ctx.size();
          int best = 0;
          for (std::size_t end = size - 1; end-- > 0 && best < 16;) {
            int m = 0;
            while (m < 16 && static_cast<std::size_t>(m) <= end &&
                   ctx[end - m] == ctx[size - 1 - m])
              ++m;
            if (m > best) {
              best = m;
              follow = end + 1;
            }
          }
          std::vector<inkling2::GpuEngine::Row> rows{{0, current}};
          if (best >= min_match && (cooldown == 0 || best >= 8))
            for (int d = 0; d < budget && follow + d < size; ++d)
              rows.push_back({0, ctx[follow + d]});
          proposed += static_cast<int>(rows.size()) - 1;
          plain_steps += rows.size() == 1 ? 1 : 0;
          engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
          std::size_t good = 0;
          while (good + 1 < rows.size() &&
                 engine.argmax(good) == rows[good + 1].token)
            ++good;
          current = engine.argmax(good);
          for (std::size_t i = 1; i <= good && replay.size() < truth.size(); ++i) {
            replay.push_back(rows[i].token);
            ctx.push_back(rows[i].token);
          }
          engine.Accept(0, static_cast<std::uint32_t>(good + 1));
          kept += static_cast<int>(good);
          ++steps;
          if (rows.size() == 1) {
            cooldown = std::max(0, cooldown - 1);
          } else if (good + 1 == rows.size()) {
            budget = std::min(k, 2 * budget + 1);
            cooldown = 0;
          } else {
            budget = std::max<int>(1, static_cast<int>(good));
            cooldown = good == 0 ? 4 : 0;
          }
        }
        replay.resize(truth.size());
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          lookup_start)
                .count();
        std::size_t same = 0;
        while (same < truth.size() && replay[same] == truth[same])
          ++same;
        std::printf("lookup drafting: %zu tokens in %d steps (%d without a "
                    "draft) in %.2f s: %.1f tokens/s; %d of %d drafts kept; "
                    "output %s plain generation",
                    truth.size(), steps, plain_steps, seconds,
                    truth.size() / seconds, kept, proposed,
                    same == truth.size() ? "EQUALS" : "DIFFERS FROM");
        if (same < truth.size())
          std::printf(" (from token %zu)", same);
        std::printf("%c", 10);
      }
      if (dspark && dspark_gpu) {
        // Generate again with the drafter on the device proposing tokens:
        // the engine keeps the drafter's context itself.
        const int draft_n = std::clamp(EnvInt("INKLING2_DRAFT_N", 7), 1, 7);
        const double gate = [] {
          const char* v = std::getenv("INKLING2_DRAFT_GATE");
          return v && *v ? std::atof(v) : 0.0;
        }();
        const double temp = [] {
          const char* v = std::getenv("INKLING2_DRAFT_TEMP");
          return v && *v ? std::atof(v) : 0.0;
        }();
        std::mt19937 rng(static_cast<unsigned>(EnvInt("INKLING2_DRAFT_SEED", 1)));
        const auto pick = [&](std::size_t row) -> std::uint32_t {
          if (temp <= 0)
            return engine.argmax(row);
          const std::vector<float> capped = engine.capped_logits(row);
          const float top = *std::max_element(capped.begin(), capped.end());
          std::vector<double> cum(capped.size());
          double sum = 0;
          for (std::size_t i = 0; i < capped.size(); ++i)
            cum[i] = sum += std::exp((capped[i] - top) / temp);
          const double u =
              std::uniform_real_distribution<double>(0.0, sum)(rng);
          return static_cast<std::uint32_t>(
              std::lower_bound(cum.begin(), cum.end(), u) - cum.begin());
        };
        std::unique_ptr<inkling2::Drafter> host;
        if (dspark_gpu == 2)
          host = std::make_unique<inkling2::Drafter>(dspark, model);
        const inkling2::Drafter::Head lm_head = [&](const float* hidden) {
          return engine.HeadArgmax(hidden);
        };
        engine.Reset(0);
        std::uint32_t current = 0;
        const auto prompt_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < n; i += prompt_batch) {
          std::vector<inkling2::GpuEngine::Row> rows;
          for (std::size_t j = i; j < std::min(n, i + prompt_batch); ++j)
            rows.push_back({0, prompt_token(0, j)});
          const bool final_chunk = i + prompt_batch >= n;
          engine.Step(rows, final_chunk);
          if (final_chunk)
            current = engine.argmax(0);
          if (host)
            for (std::size_t r = 0; r < rows.size(); ++r)
              host->Inject(engine.features(r).data());
        }
        std::printf("prompt with the drafter's context kept: %zu tokens in "
                    "%.3f s%c",
                    n,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - prompt_start)
                        .count(),
                    10);
        if (const int fake = EnvInt("INKLING2_FAKE_CONTEXT", 0);
            fake > 0 && !host)
          engine.FakeContext(0, static_cast<std::uint32_t>(fake));
        const std::vector<std::uint32_t>& truth = out[0];
        std::vector<std::uint32_t> replay;
        int steps = 0, drafted = 0, kept = 0, plain_steps = 0;
        int same_first = 0, same_all = 0, compared = 0;
        double draft_seconds = 0;
        const auto spec_start = std::chrono::steady_clock::now();
        while (replay.size() < truth.size()) {
          replay.push_back(current);
          const auto draft_start = std::chrono::steady_clock::now();
          const std::vector<inkling2::GpuEngine::DraftToken> drafts =
              engine.Draft(0, current, draft_n, static_cast<float>(gate));
          draft_seconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - draft_start)
                               .count();
          if (host) {
            const std::vector<std::uint32_t> ref =
                host->Draft(current, draft_n, lm_head);
            std::size_t same = 0;
            while (same < drafts.size() && same < ref.size() &&
                   drafts[same].token == ref[same])
              ++same;
            ++compared;
            same_first += same >= 1 ? 1 : 0;
            same_all += same == ref.size() ? 1 : 0;
            if (steps < 3) {
              std::printf("  step %d device:", steps);
              for (const auto& d : drafts)
                std::printf(" %u(%.2f)", d.token, d.confidence);
              std::printf("  host:");
              for (const std::uint32_t t : ref)
                std::printf(" %u", t);
              std::printf("%c", 10);
            }
          }
          std::vector<inkling2::GpuEngine::Row> rows{{0, current}};
          for (const auto& d : drafts)
            rows.push_back({0, d.token});
          plain_steps += drafts.empty() ? 1 : 0;
          engine.Step(rows, inkling2::GpuEngine::Head::kAll, true);
          std::size_t good = 0;
          std::uint32_t drawn = pick(0);
          while (good + 1 < rows.size() && drawn == rows[good + 1].token)
            drawn = pick(++good);
          current = drawn;
          std::printf("S %zu", good);
          for (const auto& d : drafts)
            std::printf(" %.3f", d.confidence);
          std::printf("%c", 10);
          for (std::size_t r = 1; r <= good; ++r)
            replay.push_back(rows[r].token);
          std::vector<std::vector<float>> features;
          if (host)
            for (std::size_t r = 0; r <= good; ++r)
              features.push_back(engine.features(r));
          engine.Accept(0, static_cast<std::uint32_t>(good + 1));
          for (const auto& f : features)
            host->Inject(f.data());
          drafted += static_cast<int>(drafts.size());
          kept += static_cast<int>(good);
          ++steps;
        }
        const double spec_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          spec_start)
                .count();
        const std::size_t produced = replay.size();
        replay.resize(truth.size());
        std::printf("device drafter: %zu tokens in %d steps (%d without a "
                    "draft) in %.2f s: %.1f tokens/s; %d of %d drafts kept "
                    "(%.1f%%); %.2f ms per draft call; output %s plain "
                    "generation%c",
                    produced, steps, plain_steps, spec_seconds,
                    produced / spec_seconds, kept, drafted,
                    100.0 * kept / std::max(1, drafted),
                    1e3 * draft_seconds / std::max(1, steps),
                    temp > 0 ? "is sampled, not compared with"
                             : replay == truth ? "EQUALS" : "DIFFERS FROM",
                    10);
        if (host)
          std::printf("  device against host prototype over %d steps: first "
                      "draft the same in %d, all seven in %d%c",
                      compared, same_first, same_all, 10);
      } else if (dspark) {
        // Generate the same tokens again, this time with the drafter
        // proposing tokens and the model verifying them in one step.
        const int draft_n = std::clamp(EnvInt("INKLING2_DRAFT_N", 7), 1, 7);
        inkling2::Drafter drafter(dspark, model);
        // The drafter's own probability of each token it proposes, to see
        // how well it predicts which drafts the model keeps.
        std::vector<float> confidence;
        const inkling2::Drafter::Head lm_head = [&](const float* hidden) {
          const std::uint32_t best = engine.HeadArgmax(hidden);
          const std::vector<float> capped = engine.capped_logits(0);
          double sum = 0;
          for (const float x : capped)
            sum += std::exp(static_cast<double>(x) - capped[best]);
          confidence.push_back(static_cast<float>(1.0 / sum));
          return best;
        };
        // INKLING2_DRAFT_TEMP: draw the model's tokens at this temperature
        // instead of taking the most likely one (the run then has no plain
        // run to equal).
        const double temp = [] {
          const char* v = std::getenv("INKLING2_DRAFT_TEMP");
          return v && *v ? std::atof(v) : 0.0;
        }();
        std::mt19937 rng(static_cast<unsigned>(EnvInt("INKLING2_DRAFT_SEED", 1)));
        const auto pick = [&](std::size_t row) -> std::uint32_t {
          if (temp <= 0)
            return engine.argmax(row);
          const std::vector<float> capped = engine.capped_logits(row);
          const float top = *std::max_element(capped.begin(), capped.end());
          std::vector<double> cum(capped.size());
          double sum = 0;
          for (std::size_t i = 0; i < capped.size(); ++i)
            cum[i] = sum += std::exp((capped[i] - top) / temp);
          const double u =
              std::uniform_real_distribution<double>(0.0, sum)(rng);
          return static_cast<std::uint32_t>(
              std::lower_bound(cum.begin(), cum.end(), u) - cum.begin());
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
          std::uint32_t drawn = pick(0);
          while (good + 1 < rows.size() && drawn == rows[good + 1].token)
            drawn = pick(++good);
          current = drawn;
          std::printf("S %zu", good);
          for (const float c : confidence)
            std::printf(" %.3f", c);
          std::printf("%c", 10);
          confidence.clear();
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
        for (std::size_t i = 0; i < truth.size() && temp <= 0; ++i)
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
    // By position: the scored positions of a chunk in eight equal parts.
    if (!klds.empty() && chunks > 0) {
      const std::size_t per_chunk =
          klds.size() / static_cast<std::size_t>(chunks);
      std::printf("  mean KLD by eighth of the scored positions:");
      for (std::size_t part = 0; part < 8; ++part) {
        double sum = 0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < klds.size(); ++i)
          if ((i % per_chunk) * 8 / per_chunk == part) {
            sum += klds[i];
            ++n;
          }
        std::printf(" %.5f", n ? sum / n : 0.0);
      }
      std::printf("%c", 10);
    }
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
  if (const char* text = std::getenv("INKLING2_TOKENIZE"))
    return RunTokenize(reader, text);
  if (const char* cases = std::getenv("INKLING2_CHAT"))
    return RunChatCases(reader, cases);
  if (std::getenv("INKLING2_LAUNCH_BENCH"))
    return inkling2::gpu::RunLaunchBench();
  if (std::getenv("INKLING2_WMMA_BENCH"))
    return inkling2::gpu::RunWmmaBench();
  if (const char* tokens = std::getenv("INKLING2_SWEEP"))
    return RunDepthSweep(reader, tokens);
  if (const char* kld = std::getenv("INKLING2_KLD"))
    return RunQuality(reader, kld);
  if (const char* ids = std::getenv("INKLING2_REF_TOKENS")) {
    if (const char* v = std::getenv("INKLING2_GPU"); v && *v == '2')
      return RunDecoder(reader, ids);
    return RunReference(reader, ids);
  }
  return ReportOnly(reader);
}

bool DevelopmentMode() {
  for (const char* name :
       {"INKLING2_REF_TOKENS", "INKLING2_KLD", "INKLING2_TOKENIZE",
        "INKLING2_CHAT", "INKLING2_REPORT", "INKLING2_SWEEP",
        "INKLING2_WMMA_BENCH", "INKLING2_LAUNCH_BENCH"})
    if (std::getenv(name))
      return true;
  return false;
}

}  // namespace

void inkling2_native_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_inkling2_native::
                           kProvider_inkling2_mini_native);
  registry.AddTextProvider({
      .id = "inkling2-mini-native",
      .probe = nullptr,
      .probe_gguf = &inkling2::Probe,
      .create = &inkling2::CreateTextRunner,
      .bench_driver = nullptr,
      // The development modes (INKLING2_* settings) keep their own driver;
      // without one, `gufo prompt` runs through the runner like `gufo serve`.
      .prompt_driver = DevelopmentMode() ? &Prompt : nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &inkling2::SamplingDefaults,
      // The NVFP4 expert tensors are a storage type GUFO's core reader does
      // not size; this provider reads them from their reserved bytes.
      .accepts_unsized_tensors = true,
      // The runner decides how many drafts a step carries (serve.cpp).
      .speculative_defaults = {{.backend = "dspark",
                                .max_draft_tokens = 7,
                                .min_draft_tokens = 1,
                                .draft_policy = {},
                                .draft_policies = {}}},
  });
}
