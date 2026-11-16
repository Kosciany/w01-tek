#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [OUTPUT_DIR]\n' "$0" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-/tmp/wojtek-point-frames}"

printf 'Saving frames to %s\n' "$output_dir"
printf 'Requires an already-running simulation with its camera.\n'
while true; do
  if ! IFS= read -r -p 'Settle the robot, then press Enter to capture (q to finish): ' answer; then
    printf '\n'
    break
  fi
  case "$answer" in
    q|Q) break ;;
    '') python3 "$script_dir/capture_frames.py" --out "$output_dir" ;;
    *) printf 'Press Enter to capture or q to finish.\n' ;;
  esac
done

printf 'Capture session finished. Frames are in %s\n' "$output_dir"