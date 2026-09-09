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
# 注意: -c 是"总上下文", 会被并行槽平分(parallel=2 时每槽 = CTX/2)。
# 短句/本机 4096 总量、每槽 2048 已足够(60s 音频约 ~1k token);
# 若要每槽都 4096, 用 CTX=8192。
CTX="${QASR_CTX:-4096}"
# 并发槽: 默认 2 —— 单个麦克风但允许"上一段未转完就开始第二段"时,
# 新请求不必排队等上一段完成(两槽并行; 总上下文不变)。
# 纯串行且想省内存可 QASR_PARALLEL=1。
PARALLEL="${QASR_PARALLEL:-2}"
# 本场景(短句/单用户/Metal)可调项: 留空=用引擎默认
THREADS="${QASR_THREADS:-}"      # e.g. 8: 显式限制 CPU 生成/批处理线程
FLASH="${QASR_FLASH:-auto}"      # auto/on/off: Flash Attention (Metal 长 ctx 更省更稳)
MLOCK="${QASR_MLOCK:-0}"         # 1 = --load-mode mlock: 模型页常驻防换出(略增常驻内存)
TEMP="${QASR_TEMP:-0}"           # 采样温度: 0=贪心确定性 (llama.cpp 默认0.8会导致每次结果抖动)
TIMEOUT="${QASR_TIMEOUT:-120}"
# 转写提示词(精简优化版): 已去掉"删口头语/加标点/数字转阿拉伯"话术(标点由模型
# 自然输出、口头语与数字规范化已在 rule 档/ITN 确定性完成), 保留语言约束/纠错/
# 专名准确, 并含指令词逻辑: 听到 发送 / clear 指令词时单独成句并触发断句。
# 置空 QASR_PROMPT='' 则用引擎内置默认 ASR 提示。
PROMPT="${QASR_PROMPT:-语音转写：说话人只说中文和英文，以中文为主。结合上下文纠正同音错别字，人名、地名尽量准确。注意：若听到指令词“发送”或“clear”，把它作为独立的指令单独成句，并在指令词处触发断句。}"
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
echo "  tune   : threads=${THREADS:-(auto)}  flash=$FLASH  mlock=$MLOCK  temp=$TEMP"
echo "  model  : $MODEL"
echo "  mmproj : $MMPROJ"
echo "  service: http://$HOST:$SERVICE_PORT  organizer=$ORGANIZER"
if [ -n "$PROMPT" ]; then
  echo "  prompt : $PROMPT"
else
  echo "  prompt : (空, 用引擎内置 ASR 提示)"
fi
echo "  log    : $LOG_FILE"

# 1) Qwen3-ASR audio engine (resident llama-server).
ENGINE_ARGS=(
  -m "$MODEL"
  --mmproj "$MMPROJ"
  -c "$CTX"
  -np "$PARALLEL"
  -ngl "$NGL"
  --host "$HOST" --port "$ENGINE_PORT"
)
[ -n "$THREADS" ] && ENGINE_ARGS+=(-t "$THREADS")
case "$FLASH" in
  on|off|auto) ENGINE_ARGS+=(-fa "$FLASH") ;;
  *) echo "错误: QASR_FLASH 只支持 on|off|auto" >&2; exit 1 ;;
esac
if [ "$MLOCK" = "1" ]; then
  ENGINE_ARGS+=(--load-mode mlock)
fi
"$ENGINE" "${ENGINE_ARGS[@]}" \
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
  --prompt "$PROMPT" \
  --temperature "$TEMP" \
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
