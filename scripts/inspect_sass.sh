#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "cuobjdump not found; install the CUDA toolkit or add it to PATH to inspect SASS."
  exit 0
fi

so_path="$(find "$repo_root" -type f -path "*/focal_w4a16/_C*.so" | head -n 1)"
if [[ -z "$so_path" ]]; then
  echo "No built focal_w4a16 extension found. Run: pip install -v -e ."
  exit 0
fi

full_out="$repo_root/scripts/sass_full.txt"
grep_out="$repo_root/scripts/sass_grep.txt"

if ! cuobjdump --dump-sass "$so_path" > "$full_out"; then
  echo "cuobjdump failed for: $so_path"
  exit 0
fi

grep -E "LDG|STG|BF16|FMA|IMAD" "$full_out" > "$grep_out" || true

echo "Extension: $so_path"
echo "Full SASS: $full_out"
echo "Grep SASS: $grep_out"
grep -E "LDG|STG|BF16|FMA|IMAD" "$full_out" | head -n 80 || true
