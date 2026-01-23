#!/usr/bin/env bash
set -euo pipefail

retry_delay_seconds="${RETRY_DELAY_SECONDS:-0}"

if [[ "${retry_delay_seconds}" != "0" ]]; then
  echo "Retry delay configured; sleeping for ${retry_delay_seconds} seconds before import."
  sleep "${retry_delay_seconds}"
fi

exec python import_from_geniesim.py
