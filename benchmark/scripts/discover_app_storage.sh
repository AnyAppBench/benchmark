#!/usr/bin/env bash
# discover_app_storage.sh -- run the upstream tasks_guide.md §1-2 procedure
# (find the SQLite DBs and dump their schemas) for one Android package, and
# write the result to benchmark/docs/storage_profiles/<package>.yaml.
#
# This file is the ground truth a cross-app SQLite validator pins to: every
# new (task, app) pair under §5.3 of tasks_guide.md must be backed by a
# committed YAML profile that names the db_path, table_name and column list
# the validator queries.
#
# Usage:
#   bash scripts/discover_app_storage.sh <package_name>
#   bash scripts/discover_app_storage.sh --all       # iterate over CSV
#
# Requirements:
#   - A booted emulator or device with the package installed.
#   - adb on PATH (override via $ADB).
#   - The device should be debuggable so /data/data/<pkg>/ is readable.
#     (Plain Pixel AVD images satisfy this; release-keys hardware does not.)

set -euo pipefail

ADB="${ADB:-adb}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CSV="${REPO_ROOT}/app_generalization_apps.csv"
OUT_DIR="${REPO_ROOT}/docs/storage_profiles"

mkdir -p "$OUT_DIR"

probe_one() {
  local pkg="$1"
  local out="${OUT_DIR}/${pkg}.yaml"
  local app_id_match
  app_id_match=$(awk -F, -v p="$pkg" '$3==p { print $1; exit }' "$CSV" || true)

  echo "==> probing ${pkg}  ->  ${out}"

  if ! "$ADB" shell pm list packages -e 2>/dev/null | tr -d '\r' | grep -Fxq "package:${pkg}"; then
    cat > "$out" <<EOF
package: ${pkg}
app_id: ${app_id_match}
status: not_installed
EOF
    echo "    [SKIP] not installed on device"
    return 0
  fi

  # 1) databases dir listing
  local db_listing
  db_listing=$("$ADB" shell "run-as ${pkg} ls -1 /data/data/${pkg}/databases/ 2>/dev/null || true" \
                | tr -d '\r' | sed '/^$/d')

  # 2) files dir listing (filesystem-backed apps)
  local files_listing
  files_listing=$("$ADB" shell "run-as ${pkg} ls -1 /data/data/${pkg}/files/ 2>/dev/null || true" \
                   | tr -d '\r' | sed '/^$/d')

  # 3) public-storage paths the app may use
  local public_paths
  public_paths=$("$ADB" shell "ls -1 /sdcard/Android/data/${pkg}/files/ 2>/dev/null || true" \
                  | tr -d '\r' | sed '/^$/d')

  {
    echo "package: ${pkg}"
    echo "app_id: ${app_id_match}"
    echo "status: installed"
    echo "private_databases:"
    if [[ -n "$db_listing" ]]; then
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        echo "  - file: ${line}"
        # Try to dump schema if .db extension and accessible.
        if [[ "$line" == *.db ]]; then
          local schema
          schema=$("$ADB" shell "run-as ${pkg} sqlite3 /data/data/${pkg}/databases/${line} '.schema' 2>/dev/null || true" \
                    | tr -d '\r')
          if [[ -n "$schema" ]]; then
            echo "    schema: |"
            printf '      %s\n' "$schema" | sed -e 's/^      $//'
          else
            echo "    schema: <unreadable>  # may need root or debuggable build"
          fi
        fi
      done <<< "$db_listing"
    else
      echo "  []"
    fi
    echo "private_files:"
    if [[ -n "$files_listing" ]]; then
      while IFS= read -r line; do
        echo "  - ${line}"
      done <<< "$files_listing"
    else
      echo "  []"
    fi
    echo "public_files:"
    if [[ -n "$public_paths" ]]; then
      while IFS= read -r line; do
        echo "  - ${line}"
      done <<< "$public_paths"
    else
      echo "  []"
    fi
  } > "$out"

  echo "    [OK] wrote $(wc -l < "$out") lines"
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <package_name> | --all" >&2
  exit 64
fi

if [[ "$1" == "--all" ]]; then
  awk -F, 'NR>1 && $3 != "" { print $3 }' "$CSV" | while read -r pkg; do
    [[ -z "$pkg" ]] && continue
    probe_one "$pkg" || true
  done
else
  probe_one "$1"
fi

echo
echo "Storage profiles written to ${OUT_DIR}/"
echo "Commit the new YAMLs alongside any cross-app validator that references them."
