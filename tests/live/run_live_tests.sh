#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

RUN_ORACLE_LIVE_TESTS=1 uv run \
  --env-file "$repo_root/.env" \
  pytest "$repo_root/tests/live/resolver_live_scenarios.py" -v -s
