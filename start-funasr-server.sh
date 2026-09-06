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
#
# Endpoint: POST http://127.0.0.1:8001/v1/audio/transcriptions  (OpenAI-compatible)
# Health:   GET  http://127.0.0.1:8001/health
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
# 可选: 自定义提示词/词表, 替换默认的 "语音转写:", 用于向解码器注入指令词偏置(方案A)。
# 例: FUNASR_PROMPT='语音转写, 可能出现的指令词: CLEAR、发送、上一条' ./start-funasr-server.sh
PROMPT="${FUNASR_PROMPT:-}"
LOG_FILE="${FUNASR_LOG_FILE:-$SCRIPT_DIR/funasr-server/server.log}"

for f in "$BINARY" "$MODEL" "$ENCODER" "$VAD"; do
  [ -f "$f" ] || { echo "错误: 缺少文件 $f（先运行 download-funasr-model.sh nano 下载模型）" >&2; exit 1; }
done

# Kill anything already listening on the target port, so the server starts fresh.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "端口 $PORT 被占用，正在停止旧的进程 ..."
  lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN | while read -r pid; do
    echo "  正在终止 pid $pid"
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "错误: 端口 $PORT 在 kill 之后仍被占用" >&2
    exit 1
  fi
fi

echo "启动 FunASR GGUF 服务于 http://$HOST:$PORT ..."
echo "  binary : $BINARY"
echo "  model  : $MODEL"
echo "  encoder: $ENCODER"
echo "  vad    : $VAD"
echo "  mode   : $([ "$PERSISTENT" = "1" ] && echo "常驻内存(persistent)" || echo "每请求拉起子进程(oneshot)")"
if [ -n "$PROMPT" ]; then
  echo "  prompt : $PROMPT"
else
  echo "  prompt : (默认) 语音转写："
fi
echo "  log    : $LOG_FILE"

: > "$LOG_FILE"
PERSISTENT_FLAG=""
if [ "$PERSISTENT" = "1" ]; then
  PERSISTENT_FLAG="--persistent"
fi
PROMPT_ARGS=()
if [ -n "$PROMPT" ]; then
  PROMPT_ARGS=(--prompt "$PROMPT")
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
  "${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}" \
  $PERSISTENT_FLAG \
  >> "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true

# Wait for /health up to ~10s to report success or failure.
UP=0
for _ in $(seq 1 20); do
  if curl -sf "http://$HOST:$PORT/health" -o /dev/null; then
    UP=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if [ "$UP" = "1" ]; then
  echo
  echo "启动成功 ✓  http://$HOST:$PORT   (PID $SERVER_PID)"
  echo "停止服务:  kill $SERVER_PID  （或再次运行本脚本自动重启）"
  echo "日志文件:  $LOG_FILE"
  exit 0
else
  echo
  echo "启动失败 ✗  请查看日志 $LOG_FILE" >&2
  tail -n 20 "$LOG_FILE" >&2 || true
  exit 1
fi
