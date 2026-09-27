#!/usr/bin/env bash
# One foreground command: build if missing, inspect device, prepare, train, evaluate.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if ! docker image inspect "${SCHNITZELJAGD_IMAGE:-schnitz-spark:0.4}" >/dev/null 2>&1; then
  "$root/scripts/spark.sh" build
fi
"$root/scripts/spark.sh" run schnitz doctor --require-spark
exec "$root/scripts/spark.sh" run schnitz launch "$@"
