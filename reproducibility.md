# Reproducibility (NRR-Phi)

## Scope

This repository snapshot bundles the current manuscript source package together with
the deterministic offline checks used for the current Phi line, including a
bundled transcript-audit path for the main-text LLM extraction summary.
As of 2026-03-27 JST, the public arXiv line is `2601.19933v5`
(repo snapshot `manuscript/current/paper2_nrr-phi_v39.tex`), while the
current local replacement candidate is `manuscript/current/paper2_nrr-phi_v42.tex`.
`VERSION_MAP.md` remains a full-repo provenance map and may list historical rows
that are not bundled into a narrow review drop.

## Stable review-package commands

- Build the current manuscript to temp output:
  - `bash scripts/build_current_manuscript.sh`
  - output: `/tmp/nrr-phi_current_build/paper2_nrr-phi_v42.pdf`
- Verify the current review-package checksum manifest:
  - `bash scripts/verify_current_package.sh`
- Reproduce the primary checks to temp outputs:
  - `bash scripts/run_primary_checks.sh`
  - outputs:
    - `/tmp/nrr_phi_rule_based_output.txt`
    - `/tmp/nrr_phi_operator_validation.json`
    - `/tmp/nrr_phi_llm_transcript_audit.json`

## Current review package

- Main TeX: `manuscript/current/paper2_nrr-phi_v42.tex`
- Current manuscript figures: `manuscript/current/figure1.png` to `figure5.png`
- Checksum manifest: `manuscript/current/checksums_sha256.txt`
- Bundled rule-based artifact: `results/rule_based_output.json`
- Bundled operator-validation artifact: `results/operator_validation_results.json`
- Bundled transcript-audit manifest: `prompts/llm_audit_manifest.json`
- Bundled transcript-audit script: `scripts/audit_llm_transcripts.py`
- Bundled transcript-audit summary: `results/llm_transcript_audit_summary.json`
- Bundled LLM prompt+response transcripts:
  - `prompts/GPTprompts_for_kei.txt`
  - `prompts/Geminiprompts_for_kei_2.txt`
  - `prompts/claudeprompts_for_kei_2_2.txt`
- Public arXiv note: the current public arXiv source snapshot is
  `manuscript/current/paper2_nrr-phi_v39.tex`; the `v40` and `v41` packages
  named here are prior derived lines, the `v38` package remains an older
  historical row in the provenance map, and the `v42` package named here is the current prepared
  replacement candidate.

## Checksum policy

- `manuscript/current/checksums_sha256.txt` covers the tracked files that define the
  current review package for the latest manuscript line in `manuscript/current/`.
- Coverage includes the current main `.tex` file and each figure asset consumed by
  that current manuscript from `manuscript/current/`.
- Coverage excludes `checksums_sha256.txt` itself, older manuscript versions kept
  outside the current package, and repo-specific artifacts outside
  `manuscript/current/` unless a separate manifest is provided.
- For reviewer audit of the main-text LLM extraction table and the combined
  `H = 1.087` line, this repo now bundles:
  - the original prompt+response transcript files in `prompts/`
  - an explicit sentence-set and aggregation manifest in
    `prompts/llm_audit_manifest.json`
  - a local audit script in `scripts/audit_llm_transcripts.py`
  - a generated audit summary in `results/llm_transcript_audit_summary.json`

## Environment

- Python: 3.13.7 (`python3`)
- Main libraries: NumPy >= 1.20
- OS: Darwin 25.2.0 arm64

## Fixed protocol settings

- Primary checks: rule-based extractor for Table 2 and operator-validation rerun for Appendix D
- Seed: N/A (deterministic scripts)
- Temperature: N/A (offline scripts; no live LLM sampling)
- Trials: 1 pass per command (deterministic input set)

## Artifact map

| Artifact | Command | Output |
|---|---|---|
| Paper Table 2 rule-based rerun text capture | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_rule_based_output.txt` |
| Bundled rule-based JSON artifact | N/A (tracked artifact) | `results/rule_based_output.json` |
| Appendix D rerun summary | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_operator_validation.json` |
| Appendix D bundled artifact | N/A (tracked artifact) | `results/operator_validation_results.json` |
| LLM transcript audit rerun | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_llm_transcript_audit.json` |
| Bundled LLM transcript audit summary | N/A (tracked artifact) | `results/llm_transcript_audit_summary.json` |
| Bundled LLM transcript audit manifest | N/A (tracked artifact) | `prompts/llm_audit_manifest.json` |
| Bundled LLM prompt+response transcripts for the main-text extraction table | N/A (tracked artifacts) | `prompts/GPTprompts_for_kei.txt`, `prompts/Geminiprompts_for_kei_2.txt`, `prompts/claudeprompts_for_kei_2_2.txt` |
| Current manuscript build | `bash scripts/build_current_manuscript.sh` | `/tmp/nrr-phi_current_build/paper2_nrr-phi_v42.pdf` |
| Current package checksum verification | `bash scripts/verify_current_package.sh` | stdout verification for `manuscript/current/checksums_sha256.txt` |
| Current manuscript source snapshot | N/A (tracked artifact) | `manuscript/current/paper2_nrr-phi_v42.tex` |
| Full-repo provenance map | N/A (tracked artifact) | `VERSION_MAP.md` |

## Known limitations

- `results/operator_validation_results.json` is the bundled tracked artifact; reruns should write to a separate path to avoid overwriting the snapshot.
- The transcript-audit script reconstructs the manuscript summaries from bundled
  transcripts and confidence weights, but it still audits fixed saved outputs
  from free-tier web interfaces rather than replaying live web runs.
- Rule-based coverage is limited to implemented marker patterns (EN/JP).
