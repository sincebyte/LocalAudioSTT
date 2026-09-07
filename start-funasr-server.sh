#!/usr/bin/env bash
# One-click start of the FunASR llama.cpp / GGUF OpenAI-compatible server (Fun-ASR-Nano).
#
# Runs the server in the BACKGROUND, prints a short startup log, reports
# success/failure, then exits. The server keeps running after the script exits.
#
# Usage:  ./start-funasr-server.sh
#         FUNASR_PORT=9000 ./start-funasr-server.sh    # override port
#         FUNASR_HOST=0.0.0.0 ./start-funasr-server.sh  # listen on all interfaces
#         FUNASR_PROMPT='...' ./start-funasr-server.sh  # custom recognition prompt (see below)
#         FUNASR_ORGANIZER=llm ./start-funasr-server.sh # full LLM text reorganization
#
# Endpoints:
#   POST http://127.0.0.1:8001/v1/audio/transcriptions   (OpenAI-compatible)
#   POST http://127.0.0.1:8001/v1/text/reformat          (可选: 手动整理一大段累积文本)
#   GET  http://127.0.0.1:8001/health
#
# Transcription text is organized per request according to FUNASR_ORGANIZER:
#   none - raw transcript only
#   rule - deterministic local formatting (default; no extra model call)
#   llm  - full reorganization through a general chat model (needs llama-server)
#
# The Fun-ASR-tuned Qwen3 used for audio stops immediately on pure text, so the
# 'llm' organizer (and the manual /v1/text/reformat endpoint) require a llama.cpp
# llama-server hosting a general chat GGUF. That server is only started when
# FUNASR_ORGANIZER=llm or FUNASR_REFORMAT=1, so the default run is one process.
#
# NOTE: use the FUNASR_* variables to override settings; the generic PORT
# variable is intentionally NOT used (it collides with unrelated shell envs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${FUNASR_HOST:-127.0.0.1}"
PORT="${FUNASR_PORT:-8001}"
BINARY="${FUNASR_BINARY:-$SCRIPT_DIR/llama-funasr-cli}"
# LLM 精度实验: q8_0 为默认(速度/效果平衡)。BF16 在 CPU 解码下太慢(~90s/请求), 已弃用; q5km 也可用
MODEL="${FUNASR_MODEL:-$SCRIPT_DIR/gguf/qwen3-0.6b-q8_0.gguf}"
ENCODER="${FUNASR_ENCODER:-$SCRIPT_DIR/gguf/funasr-encoder-f16.gguf}"
VAD="${FUNASR_VAD:-$SCRIPT_DIR/gguf/fsmn-vad.gguf}"
TIMEOUT="${FUNASR_TIMEOUT:-300}"
PERSISTENT="${FUNASR_PERSISTENT:-1}"   # keep the model resident (1) or spawn per request (0)
# 识别提示词(语音转写阶段)模板: 覆盖 语言范围/英文识别/标点/错别字纠正。
# 作用在音频解码前, 是弱偏置; 可直接用 FUNASR_PROMPT 整体替换。
PROMPT="${FUNASR_PROMPT:你是语音转写校对助手，把口语转写逐句整理成规范的书面文本，保留每一句的原意与全部内容，删除“嗯、啊、呃、那个、就是说”等口头语，说错又纠正的只保留最后正确的说法。说话人只会说中文和英文，以中文为主。请为每句话加上恰当标点。结合上下文纠正同音错别字，人名、地名、数字、单位尽量准确。数字一律写成阿拉伯数字。注意：clear 和 发送 是指令词}"
# 转写结果整理档位: none|rule|llm (rule 为默认, 不调文本模型)
ORGANIZER="${FUNASR_ORGANIZER:-rule}"
LOG_FILE="${FUNASR_LOG_FILE:-$SCRIPT_DIR/funasr-server/server.log}"

case "$ORGANIZER" in
  none|rule|llm) ;;
  *) echo "错误: FUNASR_ORGANIZER 只支持 none|rule|llm, 得到: $ORGANIZER" >&2; exit 1 ;;
esac

# --- 文本整理 llama-server (通用 chat 模型, 仅 llm 档/手动整理时启用) ------
# FUNASR_REFORMAT=1 额外拉起 llama-server, 供 /v1/text/reformat 手动整段整理。
REFORMAT_ENABLE="${FUNASR_REFORMAT:-0}"
[ "$ORGANIZER" = "llm" ] && REFORMAT_ENABLE=1
REFORMAT_PORT="${FUNASR_REFORMAT_PORT:-8082}"
REFORMAT_HOST="${FUNASR_REFORMAT_HOST:-127.0.0.1}"
# 文本整理模型: 默认优先用 Q8 量化后的通用聊天版 (快), 没有则退回 BF16 全精度版。
# fun-asr 调优版 q8/q5km 不能做纯文本整理, 不要指到那些文件。
REFORMAT_GGUF="${FUNASR_REFORMAT_GGUF:-}"
if [ -z "$REFORMAT_GGUF" ]; then
  REFORMAT_GGUF="$SCRIPT_DIR/gguf/qwen3-0.6b-chat-q8_0.gguf"
  [ -f "$REFORMAT_GGUF" ] || REFORMAT_GGUF="$SCRIPT_DIR/gguf/Qwen3-0.6B-BF16.gguf"
fi
LLAMA_SERVER="${FUNASR_LLAMA_SERVER:-$SCRIPT_DIR/llama-server}"
# llama-server 不在仓库根目录时, 回退到本地构建产物
if [ ! -x "$LLAMA_SERVER" ] && [ -x "$SCRIPT_DIR/.build/build-llama-server/bin/llama-server" ]; then
  LLAMA_SERVER="$SCRIPT_DIR/.build/build-llama-server/bin/llama-server"
fi
# llama-server 侧的 model id (任意字符串, 服务端不校验) 与可选自定义排版指令
REFORMAT_MODEL_ID="${FUNASR_REFORMAT_MODEL:-qwen3-0.6b}"
REFORMAT_PROMPT="${FUNASR_REFORMAT_PROMPT:-}"

for f in "$BINARY" "$MODEL" "$ENCODER" "$VAD"; do
  [ -f "$f" ] || { echo "错误: 缺少文件 $f（先运行 download-funasr-model.sh nano 下载模型）" >&2; exit 1; }
done
if [ "$REFORMAT_ENABLE" = "1" ]; then
  [ -x "$LLAMA_SERVER" ] || { echo "错误: 找不到 llama-server ($LLAMA_SERVER)。$([ "$ORGANIZER" = llm ] && echo 'FUNASR_ORGANIZER=llm 需要它' || echo 'FUNASR_REFORMAT=1 需要它')；请先编译或设 FUNASR_ORGANIZER=rule/none" >&2; exit 1; }
  [ -f "$REFORMAT_GGUF" ] || { echo "错误: 缺少整理模型 $REFORMAT_GGUF（需通用 chat 模型，非 fun-asr 调优版）" >&2; exit 1; }
fi

# Kill anything already listening on the target ports, so the server starts fresh.
for p in "$PORT" "$( [ "$REFORMAT_ENABLE" = "1" ] && echo "$REFORMAT_PORT" )"; do
  [ -z "$p" ] && continue
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "端口 $p 被占用，正在停止旧的进程 ..."
    lsof -nP -tiTCP:"$p" -sTCP:LISTEN | while read -r pid; do
      echo "  正在终止 pid $pid"
      kill "$pid" 2>/dev/null || true
    done
    sleep 1
    if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "错误: 端口 $p 在 kill 之后仍被占用" >&2
      exit 1
    fi
  fi
done

echo "启动 FunASR GGUF 服务于 http://$HOST:$PORT ..."
echo "  binary : $BINARY"
echo "  model  : $MODEL"
echo "  encoder: $ENCODER"
echo "  vad    : $VAD"
echo "  mode   : $([ "$PERSISTENT" = "1" ] && echo "常驻内存(persistent)" || echo "每请求拉起子进程(oneshot)")"
echo "  prompt : $PROMPT"
echo "  organizer: $ORGANIZER"
if [ "$REFORMAT_ENABLE" = "1" ]; then
  echo "  reformat : http://$REFORMAT_HOST:$REFORMAT_PORT  (model $REFORMAT_MODEL_ID @ $REFORMAT_GGUF)"
else
  echo "  reformat : 未启用 (llm 档或 FUNASR_REFORMAT=1 才拉起 llama-server)"
fi
echo "  log    : $LOG_FILE"

: > "$LOG_FILE"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

REFORMAT_URL_ARGS=()
if [ "$REFORMAT_ENABLE" = "1" ]; then
  echo "启动文本整理 llama-server ($REFORMAT_GGUF) ..."
  "$LLAMA_SERVER" \
    -m "$REFORMAT_GGUF" \
    --host "$REFORMAT_HOST" --port "$REFORMAT_PORT" \
    -c 4096 -t 8 \
    >> "$LOG_FILE" 2>&1 < /dev/null &
  REFORMAT_PID=$!
  PIDS+=("$REFORMAT_PID")
  disown "$REFORMAT_PID" 2>/dev/null || true
  REFORMAT_URL_ARGS=(--reformat-url "http://$REFORMAT_HOST:$REFORMAT_PORT/v1")
  if [ -n "$REFORMAT_MODEL_ID" ]; then
    REFORMAT_URL_ARGS+=(--reformat-model "$REFORMAT_MODEL_ID")
  fi
  if [ -n "$REFORMAT_PROMPT" ]; then
    REFORMAT_URL_ARGS+=(--reformat-prompt "$REFORMAT_PROMPT")
  fi
fi

PERSISTENT_FLAG=""
if [ "$PERSISTENT" = "1" ]; then
  PERSISTENT_FLAG="--persistent"
fi
# nohup 在无控制终端的上下文(SSH/agent/面板)会报 "can't detach from console" 并导致后台进程被杀,
# 改用 < /dev/null + disown: 真实终端与无 tty 环境都能稳定后台运行。
python3 "$SCRIPT_DIR/funasr-server/funasr_gguf_server.py" \
  --host "$HOST" \
  --port "$PORT" \
  --binary "$BINARY" \
  --model "$MODEL" \
  --vad "$VAD" \
  --extra-arg "--enc $ENCODER" \
  --timeout "$TIMEOUT" \
  --prompt "$PROMPT" \
  --organizer "$ORGANIZER" \
  "${REFORMAT_URL_ARGS[@]+"${REFORMAT_URL_ARGS[@]}"}" \
  $PERSISTENT_FLAG \
  >> "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
PIDS+=("$SERVER_PID")
disown "$SERVER_PID" 2>/dev/null || true

# Wait for all /health endpoints to come up (up to ~15s).
UP=0
for _ in $(seq 1 30); do
  HEALTH_OK=1
  if ! curl -sf "http://$HOST:$PORT/health" -o /dev/null; then
    HEALTH_OK=0
  fi
  if [ "$REFORMAT_ENABLE" = "1" ] && [ "$REFORMAT_PID" != "" ] \
     && ! curl -sf "http://$REFORMAT_HOST:$REFORMAT_PORT/health" -o /dev/null; then
    HEALTH_OK=0
  fi
  if [ "$HEALTH_OK" = "1" ]; then UP=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  sleep 0.5
done

trap - EXIT
if [ "$UP" = "1" ]; then
  echo
  echo "启动成功 ✓  http://$HOST:$PORT   (PID $SERVER_PID)  organizer=$ORGANIZER"
  if [ "$REFORMAT_ENABLE" = "1" ]; then
    echo "文本整理 llama-server ✓  (PID $REFORMAT_PID)"
  fi
  if grep -q '^==== reformat-prompt-begin ====$' "$LOG_FILE" 2>/dev/null; then
    echo "整理提示词(第二段)内容:"
    awk '/^==== reformat-prompt-begin ====$/{f=1;next} /^==== reformat-prompt-end ====$/{f=0} f' "$LOG_FILE"
  fi
  echo "停止服务:  kill $SERVER_PID${REFORMAT_PID:+ $REFORMAT_PID} （或再次运行本脚本自动重启）"
  echo "日志文件:  $LOG_FILE"
  exit 0
else
  echo
  echo "启动失败 ✗  请查看日志 $LOG_FILE" >&2
  tail -n 20 "$LOG_FILE" >&2 || true
  cleanup
  exit 1
fi
