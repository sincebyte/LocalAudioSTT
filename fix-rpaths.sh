#!/usr/bin/env bash
# 修复 macOS 二进制的 LC_RPATH: 把构建时写死的绝对路径改成相对路径。
#
# 背景: llama-server / llama-quantize 及其 dylib 由 cmake 生成, LC_RPATH 里记录
# 的是"构建时的绝对路径"(如 /Users/van/Desktop/.../.deps/llama.cpp/build/bin)。
# 整个项目目录一旦被移动/重命名, dyld 就按旧路径找 dylib, 启动报:
#   dyld[...]: Library not loaded: @rpath/libllama-server-impl.dylib
# 本脚本把绝对 rpath 全部删掉, 换成相对 loader 的路径, 项目可以任意搬迁。
#
#   ./fix-rpaths.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
LIB_DIR="$SCRIPT_DIR/.deps/llama.cpp/build/bin"
# 二进制(dylib 均在此目录)相对路径: bin/llama-server -> ../.deps/llama.cpp/build/bin
BIN_RPATH="@loader_path/../.deps/llama.cpp/build/bin"

# 列出某个 Mach-O 文件里的所有 LC_RPATH
list_rpaths() {
  otool -l "$1" | awk '/LC_RPATH/ { getline; getline; sub(/^[[:space:]]*path /, ""); sub(/ \(offset.*/, ""); print }'
}

# 删除所有绝对(rpath 以 / 开头)的 LC_RPATH, 再加上 $2 指定的相对 rpath
rewrite_rpath() {
  local file="$1" new_rpath="$2" old args=()
  for old in $(list_rpaths "$file"); do
    case "$old" in
      /*) args+=(-delete_rpath "$old") ;;
    esac
  done
  if [ "${#args[@]}" -gt 0 ]; then
    install_name_tool "${args[@]}" "$file"
  fi
  if ! list_rpaths "$file" | grep -Fxq "$new_rpath"; then
    install_name_tool -add_rpath "$new_rpath" "$file"
  fi
  # install_name_tool 会破坏原有 ad-hoc 签名, arm64 上签名无效会被内核直接杀掉
  codesign --force --sign - "$file" >/dev/null 2>&1 || true
  echo "  ✓ $(basename "$file") -> $new_rpath"
}

[ -d "$BIN_DIR" ] || { echo "错误: 不存在 $BIN_DIR" >&2; exit 1; }
[ -d "$LIB_DIR" ] || { echo "错误: 不存在 $LIB_DIR (先运行 ./build-llama-server.sh)" >&2; exit 1; }

echo "修复可执行文件的 rpath:"
for f in "$BIN_DIR/llama-server" "$BIN_DIR/llama-quantize"; do
  [ -f "$f" ] && rewrite_rpath "$f" "$BIN_RPATH"
done

echo "修复 dylib 的 rpath:"
# 只处理真实文件, 符号链接指向同一个 inode, 重复处理会报"文件格式错误"
for f in "$LIB_DIR"/*.dylib; do
  [ -L "$f" ] && continue
  rewrite_rpath "$f" "@loader_path"
done

echo
echo "完成。验证:"
"$BIN_DIR/llama-server" --version 2>&1 | head -5 || true
