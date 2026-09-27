#!/usr/bin/env bash
# First real-model validation, intentionally only two optimizer steps.
set -euo pipefail
config="${1:-configs/lfm25_230m_spark.yaml}"
output="${2:-runs/lfm-smoke-$(date +%Y%m%d-%H%M%S)}"
schnitz doctor --require-spark
python -m pytest -q -m integration
schnitz model-probe --config "$config" --output "${output}-model-probe.json"
schnitz train --config "$config" --output "$output" --steps 2
schnitz evaluate --run "$output" --count 2
