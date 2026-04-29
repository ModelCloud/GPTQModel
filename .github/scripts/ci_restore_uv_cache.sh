#!/usr/bin/env bash
set -euo pipefail

tar_file="${1:-/opt/dist/uv.tar.xz}"
cache_dir="${2:-/opt/uv/cache}"
tmp_dir="${cache_dir}/tmp"
last_file="${cache_dir}/lastmodified"

if [[ ! -f "$tar_file" ]]; then
  echo "uv cache archive not found: $tar_file"
  exit 0
fi

tar_mtime="$(stat -c %Y "$tar_file")"
last_mtime="0"
if [[ -f "$last_file" ]]; then
  last_mtime="$(<"$last_file")"
fi

if [[ "$tar_mtime" == "$last_mtime" ]]; then
  echo "uv cache archive unchanged, skip decompress"
  exit 0
fi

echo "decompressing $tar_file into $cache_dir..."
mkdir -p "$tmp_dir"
rm -rf "${tmp_dir:?}/"*
tar -xJf "$tar_file" -C "$tmp_dir"
rm -rf "$cache_dir/uv"
mv "$tmp_dir/uv" "$cache_dir/uv"
printf '%s\n' "$tar_mtime" > "$last_file"

ls -ahl "$cache_dir"
echo "=========="
ls -ahl "$cache_dir/uv"
