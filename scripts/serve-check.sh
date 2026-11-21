#!/bin/bash
# Functional and speed checks of the runner through `gufo serve llm`. Runs where `gufo` and
# `curl` are (inside the GUFO runtime container), with the extension installed, the GPU free
# and the request bodies of serve-requests.py in the current directory:
#
#   python3 scripts/serve-requests.py make prose.txt code.cpp work/
#   cd work && MODEL=/models/<file>.gguf bash ../scripts/serve-check.sh
#   python3 ../scripts/serve-requests.py compare . copy edit copys
#
# CONTEXT (default 32768) must be 131072 for the p100k requests. Every request has a time
# limit, so a stuck server ends the script instead of blocking it.
MODEL=${MODEL:?set MODEL to the GGUF path}
PORT=${PORT:-8011}
URL=http://127.0.0.1:$PORT
launch() {  # environment settings as NAME=VALUE words; sessions
  rm -f serve.log
  env $1 gufo serve llm --model "$MODEL" --context ${CONTEXT:-32768} --sessions ${2:-1} \
    --cache-ram-bytes ${CACHE:-8589934592} --host 127.0.0.1 --port $PORT --log-level info > serve.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 200); do
    [ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" $URL/v1/models)" = 200 ] && break
    kill -0 $SERVER 2>/dev/null || break
    sleep 1
  done
  echo "##### server [$1] sessions=${2:-1} ready after ${i}s"
}
stop() {  # prints the server's own figures for every request
  grep -E "rror|WARN" serve.log | cut -c1-300 | head -8
  grep -E "event=completed method=POST" serve.log | sed -E 's/.*(path=[^ ]+).*(prompt_tokens=)/\1 \2/; s/queue_depth.*shared_prefix_wait_ms=[0-9.]+ //; s/cache_schema.*(lookup_)/\1/; s/rss_mib.*//' | cut -c1-330
  kill $SERVER; wait $SERVER 2>/dev/null
}
post() {  # request name, output suffix, bytes of the answer to show
  [ -f $1.json ] || return
  curl -s -m ${LIMIT:-600} -o o-$1$2.json -w "$1$2: http %{http_code} in %{time_total}s\n" \
    $URL/v1/chat/completions -H 'Content-Type: application/json' -d @$1.json
  if [ -n "$3" ]; then head -c $3 o-$1$2.json | sed -E 's/"usage":\{.*//; s/^.*"message"://'; echo; fi
}

# The same outputs without drafts, to compare text and speed.
launch "INKLING2_SERVE_LOOKUP=0"
post copy -off; post edit -off; post copys -off
stop

launch "" 2
post copy -on; post edit -on; post copys -on
for r in q1 q2 t1 t2 t3 t4 t6 j1 j2; do post $r "" 700; done
post t5; grep -o '"delta":{[^}]*}[^}]*}' o-t5.json | head -5 | cut -c1-200
post prose; post p4k "" 400; post p4k "" 400; post p4kb "" 400
post p32k "" 400; post p32kb "" 400
if [ "${CONTEXT:-32768}" -ge 131072 ]; then post p100k "" 400; post p100k "" 400; post p100kb "" 400; fi
echo "--- two requests at once"
post q2 -a 400 & A=$!
post prose -b 300 & B=$!
wait $A $B
echo "--- other endpoints"
curl -s -m 60 -o o-msg.json -w "messages: http %{http_code} in %{time_total}s\n" $URL/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","max_tokens":200,"messages":[{"role":"user","content":"What is 12 * 12?"}]}'
head -c 500 o-msg.json; echo
curl -s -m 60 -o o-resp.json -w "responses: http %{http_code} in %{time_total}s\n" $URL/v1/responses \
  -H 'Content-Type: application/json' -d '{"model":"inkling-2-mini","max_output_tokens":200,"input":"What is 12 * 12?"}'
head -c 300 o-resp.json; echo
stop
