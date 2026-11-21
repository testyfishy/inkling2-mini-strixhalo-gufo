#!/bin/bash
# Run llama.cpp's own DSpark drafter on a chat prompt, to compare draft acceptance with this
# extension. Needs a HIP build of the llama.cpp branch that carries the drafter
# (ggml-org/llama.cpp#32218) in $WORK/llama-dspark, built like llama-hip-build.sh does.
#
#   MODELS=/path/to/models ./dspark-oracle.sh <model.gguf> <drafter.gguf> "<user message>"
#
# Prints the prompt's token ids (feed them to the extension with INKLING2_REF_TOKENS), the
# generated token ids, llama.cpp's timings and its draft acceptance line.
# Nothing else may be using the GPU while this runs.
set -e
WORK=${WORK:-$HOME/inkling-bench/llama-hip}
MODELS=${MODELS:-$HOME/models}
N_MAX=${N_MAX:-7}
N_PREDICT=${N_PREDICT:-160}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python3 - "$3" > "$TMP/chat.json" <<'EOF'
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}]}))
EOF
cat > "$TMP/inner.sh" <<'EOF'
#!/bin/bash
./llama-server -m "/models/$1" -md "/models/$2" --spec-type draft-dspark --spec-draft-n-max "$3" \
  -ngl 99 -ngld 99 -c 4096 --jinja --host 127.0.0.1 --port 8099 --no-webui > /tmp/server.log 2>&1 &
for i in $(seq 1 240); do
  curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok && break
  sleep 1
done
post() { curl -s -m 600 "http://127.0.0.1:8099/$1" -H 'Content-Type: application/json' -d "@$2"; }
post apply-template /req/chat.json > /tmp/template.json
python3 - <<'PY'
import json
prompt = json.load(open("/tmp/template.json"))["prompt"]
json.dump({"content": prompt, "add_special": False, "parse_special": True}, open("/tmp/tok.json", "w"))
PY
post tokenize /tmp/tok.json > /tmp/tokens.json
python3 - "$4" <<'PY'
import json, sys
tokens = json.load(open("/tmp/tokens.json"))["tokens"]
print("prompt tokens:", ",".join(str(t) for t in tokens))
json.dump({"prompt": tokens, "n_predict": int(sys.argv[1]), "temperature": 0, "top_k": 1,
           "return_tokens": True, "cache_prompt": False, "stream": False}, open("/tmp/req.json", "w"))
PY
post completion /tmp/req.json > /tmp/response.json
python3 - <<'PY'
import json
r = json.load(open("/tmp/response.json"))
print("generated tokens:", " ".join(str(t) for t in r.get("tokens", [])))
print("timings:", json.dumps(r.get("timings", {})))
PY
grep -a "draft acceptance" /tmp/server.log
kill %1
EOF
podman run --rm --network=none --userns=keep-id:uid=1000,gid=1000 --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --group-add keep-groups --ulimit memlock=-1 \
  -v "$MODELS":/models:ro -v "$WORK":/workspace -v "$TMP":/req:ro \
  -w /workspace/llama-dspark/build/bin -e LD_LIBRARY_PATH=/workspace/llama-dspark/build/bin \
  ghcr.io/gufo-org/toolboxes/gufo-dev:edge bash /req/inner.sh "$1" "$2" "$N_MAX" "$N_PREDICT"
