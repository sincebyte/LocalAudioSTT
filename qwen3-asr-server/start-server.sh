#!/usr/bin/env bash
# One-click start of the Qwen3-ASR-1.7B transcription service.
#
# Runs in the BACKGROUND:
#   1. llama-server (Metal) hosting Qwen3-ASR-1.7B Q8_0 + mmproj  -> :ENGINE_PORT
#   2. the Python OpenAI-compatible wrapper with cleanup pipeline  -> :SERVICE_PORT
#
#   ./start-server.sh
#   QASR_PORT=8021 ./start-server.sh          # override service port
#   QASR_ENGINE_PORT=8093 ./start-server.sh   # override engine port
#   QASR_ORGANIZER=none ./start-server.sh     # raw transcript only
#   QASR_NGL=0 ./start-server.sh              # CPU-only (if not built with Metal)
#
# Endpoints:
#   POST http://127.0.0.1:8011/v1/audio/transcriptions   (OpenAI-compatible)
#   GET  http://127.0.0.1:8011/health
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${QASR_HOST:-127.0.0.1}"
SERVICE_PORT="${QASR_PORT:-8011}"
ENGINE_PORT="${QASR_ENGINE_PORT:-8083}"
ENGINE="${QASR_ENGINE_BIN:-$SCRIPT_DIR/bin/llama-server}"
MODEL="${QASR_MODEL:-$SCRIPT_DIR/gguf/Qwen3-ASR-1.7B-Q8_0.gguf}"
MMPROJ="${QASR_MMPROJ:-$SCRIPT_DIR/gguf/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf}"
ORGANIZER="${QASR_ORGANIZER:-rule}"
NGL="${QASR_NGL:-99}"
CTX="${QASR_CTX:-4096}"
PARALLEL="${QASR_PARALLEL:-1}"   # single-user: 1 slot is enough; KV memory scales with slots*ctx
TIMEOUT="${QASR_TIMEOUT:-120}"
LOG_FILE="${QASR_LOG_FILE:-$SCRIPT_DIR/server.log}"

case "$ORGANIZER" in
  none|rule) ;;
  *) echo "错误: QASR_ORGANIZER 只支持 none|rule, 得到: $ORGANIZER" >&2; exit 1 ;;
esac

for f in "$ENGINE" "$MODEL" "$MMPROJ"; do
  [ -f "$f" ] || { echo "错误: 缺少文件 $f
   - 先运行 ./build-llama-server.sh --install (编译 Metal 版 llama-server)
   - 再运行 ./download-models.sh (下载 Qwen3-ASR-1.7B GGUF)" >&2; exit 1; }
done

# Kill anything already listening on the target ports, so the service starts fresh.
for p in "$SERVICE_PORT" "$ENGINE_PORT"; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "端口 $p 被占用, 正在停止旧进程 ..."
    lsof -nP -tiTCP:"$p" -sTCP:LISTEN | while read -r pid; do
      kill "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
done

: > "$LOG_FILE"
PIDS=()
cleanup() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "启动 Qwen3-ASR-1.7B 服务 ..."
echo "  engine : llama-server(Metal) :$ENGINE_PORT  ngl=$NGL  ctx=$CTX  parallel=$PARALLEL"
echo "  model  : $MODEL"
echo "  mmproj : $MMPROJ"
echo "  service: http://$HOST:$SERVICE_PORT  organizer=$ORGANIZER"
echo "  log    : $LOG_FILE"

# 1) Qwen3-ASR audio engine (resident llama-server).
"$ENGINE" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -c "$CTX" \
  -np "$PARALLEL" \
  -ngl "$NGL" \
  --host "$HOST" --port "$ENGINE_PORT" \
  >> "$LOG_FILE" 2>&1 < /dev/null &
ENGINE_PID=$!
PIDS+=("$ENGINE_PID")
disown "$ENGINE_PID" 2>/dev/null || true

# 2) Python wrapper.
python3 "$SCRIPT_DIR/server/qwen3_asr_server.py" \
  --host "$HOST" \
  --port "$SERVICE_PORT" \
  --engine-url "http://$HOST:$ENGINE_PORT/v1" \
  --organizer "$ORGANIZER" \
  --timeout "$TIMEOUT" \
  >> "$LOG_FILE" 2>&1 < /dev/null &
SERVICE_PID=$!
PIDS+=("$SERVICE_PID")
disown "$SERVICE_PID" 2>/dev/null || true

# Wait for both /health endpoints (up to ~60s: engine model load takes a while).
UP=0
for _ in $(seq 1 120); do
  OK=1
  curl -sf "http://$HOST:$ENGINE_PORT/health" -o /dev/null || OK=0
  curl -sf "http://$HOST:$SERVICE_PORT/health" -o /dev/null || OK=0
  if [ "$OK" = "1" ]; then UP=1; break; fi
  kill -0 "$ENGINE_PID" 2>/dev/null || break
  sleep 0.5
done

trap - EXIT
if [ "$UP" = "1" ]; then
  echo
  echo "启动成功 ✓  http://$HOST:$SERVICE_PORT/v1/audio/transcriptions"
  echo "  engine PID $ENGINE_PID   service PID $SERVICE_PID"
  echo "  （或直接调 llama-server 原生端点 http://$HOST:$ENGINE_PORT/v1/audio/transcriptions）"
  echo "停止:  kill $ENGINE_PID $SERVICE_PID   (或再次运行本脚本自动重启)"
  echo "日志:   $LOG_FILE"
  exit 0
else
  echo
  echo "启动失败 ✗ 请查看日志 $LOG_FILE" >&2
  tail -n 25 "$LOG_FILE" >&2 || true
  cleanup
  exit 1
fi
