#!/usr/bin/env bash
set -euo pipefail

matches=$(rg -n "image: .*:latest" k8s/**/*.yaml || true)
if [[ -n "$matches" ]]; then
  echo "Found disallowed :latest image tags in k8s manifests:"
  echo "$matches"
  exit 1
fi
