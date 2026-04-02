#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RULE_OUT="${1:-/tmp/nrr_phi_rule_based_output.txt}"
VAL_OUT="${2:-/tmp/nrr_phi_operator_validation.json}"
LLM_AUDIT_OUT="${3:-/tmp/nrr_phi_llm_transcript_audit.json}"

cd "$ROOT"
python3 experiments/rule_based_experiment.py > "$RULE_OUT"
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output "$VAL_OUT"
python3 scripts/audit_llm_transcripts.py --output "$LLM_AUDIT_OUT"
