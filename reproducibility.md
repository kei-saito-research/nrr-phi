# Reproducibility (NRR-Phi)

## Scope

This repository snapshot bundles the current manuscript source package together with
the deterministic offline checks used for the current Phi line.
As of 2026-03-27 JST, the public arXiv line remains `2601.19933v4`
(repo snapshot `manuscript/current/paper2_nrr-phi_v38.tex`), while the
current local replacement candidate is `manuscript/current/paper2_nrr-phi_v39.tex`.

## Stable review-package commands

- Build the current manuscript to temp output:
  - `bash scripts/build_current_manuscript.sh`
  - output: `/tmp/nrr-phi_current_build/paper2_nrr-phi_v39.pdf`
- Verify the current review-package checksum manifest:
  - `bash scripts/verify_current_package.sh`
- Reproduce the primary checks to temp outputs:
  - `bash scripts/run_primary_checks.sh`
  - outputs:
    - `/tmp/nrr_phi_rule_based_output.json`
    - `/tmp/nrr_phi_operator_validation.json`

## Current review package

- Main TeX: `manuscript/current/paper2_nrr-phi_v39.tex`
- Current manuscript figures: `manuscript/current/figure1.png` to `figure5.png`
- Checksum manifest: `manuscript/current/checksums_sha256.txt`
- Public arXiv note: the current public arXiv source snapshot is
  `manuscript/current/paper2_nrr-phi_v38.tex`; the `v39` package named here is
  the prepared replacement candidate.

## Checksum policy

- `manuscript/current/checksums_sha256.txt` covers the tracked files that define the
  current review package for the latest manuscript line in `manuscript/current/`.
- Coverage includes the current main `.tex` file and each figure asset consumed by
  that current manuscript from `manuscript/current/`.
- Coverage excludes `checksums_sha256.txt` itself, older manuscript versions kept
  outside the current package, and repo-specific artifacts outside
  `manuscript/current/` unless a separate manifest is provided.

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
| Paper Table 2 rule-based extraction summary | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_rule_based_output.json` |
| Appendix D rerun summary | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_operator_validation.json` |
| Appendix D bundled artifact | N/A (tracked artifact) | `results/operator_validation_results.json` |
| Current manuscript build | `bash scripts/build_current_manuscript.sh` | `/tmp/nrr-phi_current_build/paper2_nrr-phi_v39.pdf` |
| Current package checksum verification | `bash scripts/verify_current_package.sh` | stdout verification for `manuscript/current/checksums_sha256.txt` |
| Current manuscript source snapshot | N/A (tracked artifact) | `manuscript/current/paper2_nrr-phi_v39.tex` |
| Version map | N/A (tracked artifact) | `VERSION_MAP.md` |

## Known limitations

- `results/operator_validation_results.json` is the bundled tracked artifact; reruns should write to a separate path to avoid overwriting the snapshot.
- LLM prompt/response files are archival artifacts; free-tier model build IDs are not fully fixed.
- Rule-based coverage is limited to implemented marker patterns (EN/JP).
