#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

map_file="VERSION_MAP.md"
[[ -f "$map_file" ]] || fail "Missing VERSION_MAP.md"
[[ -d manuscript/current ]] || fail "Missing manuscript/current"
[[ -d manuscript/archive ]] || fail "Missing manuscript/archive"

rows=$(awk -F'|' '
  /^\|/ {
    if ($0 ~ /^\|[[:space:]]*track[[:space:]]*\|/) next
    if ($0 ~ /^\|[[:space:]]*[-:]+[[:space:]]*\|/) next
    for (i=2; i<=NF-1; i++) {
      gsub(/^[ \t]+|[ \t]+$/, "", $i)
    }
    if ($2 != "") {
      print $2 "|" $3 "|" $4 "|" $5 "|" $6 "|" $7 "|" $8 "|" $9
    }
  }
' "$map_file")

[[ -n "$rows" ]] || fail "No data rows in VERSION_MAP.md"

while IFS='|' read -r track status arxiv_id arxiv_version local_version manuscript_dir main_tex checksum_file; do
  [[ -d "$manuscript_dir" ]] || fail "Missing directory: $manuscript_dir"
  [[ -f "$manuscript_dir/$main_tex" ]] || fail "Missing manuscript: $manuscript_dir/$main_tex"
  [[ -f "$manuscript_dir/$checksum_file" ]] || fail "Missing checksum file: $manuscript_dir/$checksum_file"

  awk '{print $2}' "$manuscript_dir/$checksum_file" | grep -qx "$main_tex" \
    || fail "Main TeX not listed in checksum file: $manuscript_dir/$checksum_file"

  while IFS= read -r fig; do
    [[ -n "$fig" ]] || continue
    [[ -f "$manuscript_dir/$fig" ]] || fail "Missing figure from includegraphics: $manuscript_dir/$fig"
  done < <(rg -o '\\includegraphics\[[^]]*\]\{[^}]+\}' "$manuscript_dir/$main_tex" \
      | sed -E 's/.*\{([^}]+)\}.*/\1/')

  if [[ "$arxiv_id" != "N/A" ]]; then
    rg -q "$arxiv_id" README.md || fail "README.md missing arXiv id: $arxiv_id"
  fi

done <<< "$rows"

if find manuscript -maxdepth 1 -type d -name 'v*' | grep -q .; then
  fail "Legacy manuscript/v* directory still present"
fi

echo "[OK] VERSION_MAP and manuscript artifacts are consistent."
