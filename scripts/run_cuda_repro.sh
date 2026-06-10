#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-cuda:0}"
REPORT_DEVICE="${DEVICE//:/_}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-.cache/xdg}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" reports

python run_repro.py env

python run_repro.py run \
  --category channel \
  --device "${DEVICE}" \
  --json-report "reports/${REPORT_DEVICE}.json"
