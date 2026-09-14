#!/usr/bin/env bash
# Download the Qwen3-ASR-1.7B GGUF files used by this project into ./gguf/.
#
#   ./download-models.sh                 # bf16 LLM + bf16 mmproj (default, ~4.7 GB total)
#
# Downloads resume on rerun (-C -) and fall back to the hf-mirror.com mirror if
# huggingface.co is unreachable. (The Q8_0 pair is half the size and ~2x faster
# with negligible ASR quality loss: point QASR_MODEL/QASR_MMPROJ at the *-Q8_0
# files, or swap the names below.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/gguf"

HUGGINGFACE="https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF/resolve/main"
MIRROR="https://hf-mirror.com/ggml-org/Qwen3-ASR-1.7B-GGUF/resolve/main"

dl() { # rel_path
  local out="$SCRIPT_DIR/gguf/$1"
  for base in "$HUGGINGFACE" "$MIRROR"; do
    echo "==> $1"
    echo "    源: $base/$1"
    for _ in 1 2 3 4 5; do
      if curl -L --fail --retry 5 --retry-all-errors -C - -o "$out" "$base/$1"; then
        echo "    ✓ $1 -> $out"
        return 0
      fi
      echo "    重试中 ..."
      sleep 3
    done
  done
  echo "错误: 无法下载 $1 (两个源都失败)" >&2
  return 1
}

dl "Qwen3-ASR-1.7B-bf16.gguf"
dl "mmproj-Qwen3-ASR-1.7B-bf16.gguf"

echo
echo "完成。文件:"
ls -lh "$SCRIPT_DIR"/gguf/*.gguf
