#!/usr/bin/env bash
# Build llama-server with Metal (Apple GPU) enabled, self-contained in .deps/.
#
# llama.cpp commit is pinned to the same tree that already supports Qwen3-ASR
# audio (mtmd) and the /v1/audio/transcriptions route.
#
#   ./build-llama-server.sh              # build only
#   ./build-llama-server.sh --install    # build + copy to ./bin/
#
# Resulting binaries land in .deps/llama.cpp/build/bin and (with --install)
# ./bin/llama-server + ./bin/llama-quantize.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINNED_LLAMA="803b7fcae893e9caaee3921779628fef83ac0965"
SRC="$SCRIPT_DIR/.deps/llama.cpp"
BUILD="$SRC/build"

mkdir -p "$SRC"

echo "==> 1/3 拉取 llama.cpp (pin $PINNED_LLAMA)"
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$SRC" fetch --depth 1 origin "$PINNED_LLAMA"
  git -C "$SRC" checkout -q FETCH_HEAD
else
  echo "      已存在 $SRC, 跳过拉取 (如需重置可删除 .deps/llama.cpp)"
fi

echo "==> 2/3 配置编译 (GGML_METAL=ON)"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_HTML=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_CURL=OFF
cmake --build "$BUILD" -j --target llama-server llama-quantize

BIN_SERVER="$BUILD/bin/llama-server"
BIN_QUANT="$BUILD/bin/llama-quantize"

echo "==> 3/3 产物"
if "$BIN_SERVER" --version 2>&1 | grep -q "built with"; then
  echo "      ✓ llama-server: $BIN_SERVER"
fi

if [ "${1:-}" = "--install" ]; then
  mkdir -p "$SCRIPT_DIR/bin"
  cp "$BIN_SERVER" "$SCRIPT_DIR/bin/llama-server"
  cp "$BIN_QUANT" "$SCRIPT_DIR/bin/llama-quantize"
  echo "      已安装到 $SCRIPT_DIR/bin/"
fi

echo
echo "完成。用法:"
echo "  ./start-server.sh   # 默认拉起 ./bin/llama-server (Metal) + python 包装服务"
