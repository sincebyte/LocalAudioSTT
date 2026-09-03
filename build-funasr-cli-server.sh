#!/usr/bin/env bash
# Build the `llama-funasr-cli` binary WITH `--server` (persistent/resident) mode.
#
# The upstream binary is one-shot: it loads the ~1GB of models, transcribes one
# file, and exits. This repo's server keeps the models resident across requests
# via `llama-funasr-cli --server`, which requires this patch on top of the
# upstream source:
#     patches/llama-funasr-cli.server-mode.patch
#
# Usage:
#   ./build-funasr-cli-server.sh                 # build only (binary lands in .build/)
#   ./build-funasr-cli-server.sh --install       # build + replace ./llama-funasr-cli (backs up first)
#
# Faster build (recommended): pre-fetch a shallow llama.cpp and point at it so
# CMake skips the slow full FetchContent clone:
#   git init llama-src && cd llama-src \
#     && git remote add origin https://github.com/ggml-org/llama.cpp.git \
#     && git fetch --depth 1 origin 803b7fcae893e9caaee3921779628fef83ac0965 \
#     && git checkout -q FETCH_HEAD
#   LLAMA_SRC_DIR=/abs/path/to/llama-src ./build-funasr-cli-server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$SCRIPT_DIR/patches/llama-funasr-cli.server-mode.patch"
PINNED_LLAMA="803b7fcae893e9caaee3921779628fef83ac0965"

INSTALL=0
if [ "${1:-}" = "--install" ]; then
  INSTALL=1
fi

BUILD_DIR="${FUNASR_BUILD_DIR:-$SCRIPT_DIR/.build}"
LLAMA_LOCAL="${LLAMA_SRC_DIR:-}"
SRC="$BUILD_DIR/FunASR"
CLI_SRC="$SRC/runtime/llama.cpp/fun-asr-nano/funasr-cli/funasr-cli.cpp"

[ -f "$PATCH" ] || { echo "错误: 找不到补丁 $PATCH" >&2; exit 1; }

echo "==> 1/4 获取 FunASR 源码 (sparse: runtime/llama.cpp)"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/modelscope/FunASR.git "$SRC"
  git -C "$SRC" sparse-checkout set runtime/llama.cpp
fi

echo "==> 2/4 应用 --server 补丁 (可重复执行, 幂等)"
if git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SRC" apply "$PATCH"
  echo "      已应用 patches/llama-funasr-cli.server-mode.patch"
elif grep -q -- '--server' "$CLI_SRC"; then
  echo "      补丁已应用过, 跳过"
else
  echo "错误: 补丁无法应用到 $CLI_SRC" >&2
  exit 1
fi

echo "==> 3/4 配置并编译 llama-funasr-cli (Release)"
CMAKE_ARGS=(-S "$SRC/runtime/llama.cpp" -B "$BUILD_DIR/build-runtime" -DCMAKE_BUILD_TYPE=Release)
if [ -n "$LLAMA_LOCAL" ]; then
  echo "      使用本地 llama.cpp: $LLAMA_LOCAL (跳过 FetchContent 完整 clone)"
  CMAKE_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_LLAMA="$LLAMA_LOCAL")
else
  echo "      未设置 LLAMA_SRC_DIR, CMake 将 FetchContent 拉取固定版本 llama.cpp ($PINNED_LLAMA), 可能较慢"
fi
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR/build-runtime" -j --target llama-funasr-cli

BIN="$BUILD_DIR/build-runtime/bin/llama-funasr-cli"
echo
echo "==> 4/4 产物: $BIN"
# Note: the CLI exits 1 on --help, so capture output and grep the string instead
# of relying on the pipeline exit code (breaks under `set -o pipefail`).
HELP_OUT="$(set +e; "$BIN" --help 2>&1 || true)"
if printf '%s' "$HELP_OUT" | grep -q -- '--server'; then
  echo "      ✓ 二进制已支持 --server 常驻模式"
else
  echo "      ✗ 二进制不支持 --server, 请检查补丁" >&2
  exit 1
fi

if [ "$INSTALL" = "1" ]; then
  DEST="$SCRIPT_DIR/llama-funasr-cli"
  if [ -f "$DEST" ]; then
    CUR_HELP="$(set +e; "$DEST" --help 2>&1 || true)"
    if printf '%s' "$CUR_HELP" | grep -q -- '--server'; then
      echo "      当前二进制已是常驻版, 直接覆盖"
    else
      cp "$DEST" "$DEST.oneshot.bak"
      echo "      已备份旧(一次性)二进制 -> llama-funasr-cli.oneshot.bak"
    fi
  fi
  # Replace via a temp file + rename (fresh inode). On Apple Silicon, cp-ing over
  # a binary that is currently being executed leaves a stale code-signature cache
  # entry for that path, and every new exec of it gets SIGKILL ("Killed: 9").
  cp "$BIN" "$DEST.new"
  mv -f "$DEST.new" "$DEST"
  echo "      已安装到 $DEST"
fi

echo
echo "完成。用法:"
echo "  ./llama-funasr-cli --enc gguf/funasr-encoder-f16.gguf -m gguf/qwen3-0.6b-q5km.gguf \\"
echo "    --vad gguf/fsmn-vad.gguf --server   # 常驻模式 (模型加载一次)"
echo "  或直接 ./start-funasr-server.sh (默认即常驻模式)"
