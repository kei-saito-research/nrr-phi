# Reproducibility

## Environment
- Python: 3.13.7 (`python3`)
- Main libraries: NumPy >= 1.20
- OS: Darwin 25.2.0 arm64

## Fixed settings
- model: Rule-based extractor for Table 2; operator-validation implementation for Appendix-D checks
- seed: N/A (deterministic scripts)
- temperature: N/A (offline scripts; no live LLM sampling)
- trials: 1 pass per command (deterministic input set)

## Stable review-package commands
- Build current manuscript to temp output:
  - `bash scripts/build_current_manuscript.sh`
  - output: `/tmp/nrr-phi_current_build/paper2_nrr-phi_v38.pdf`
- Verify current package checksums:
  - `bash scripts/verify_current_package.sh`
- Reproduce the primary checks to temp outputs:
  - `bash scripts/run_primary_checks.sh`
  - outputs:
    - `/tmp/nrr_phi_rule_based_output.json`
    - `/tmp/nrr_phi_operator_validation.json`

## Run commands
```bash
pip install -r requirements.txt
bash scripts/run_primary_checks.sh
```

## Artifact map
| Table/Figure | Command | Output file |
|---|---|---|
| Paper Table 2 (rule-based extraction summary) | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_rule_based_output.json` |
| Appendix D rerun summary (local output) | `bash scripts/run_primary_checks.sh` | `/tmp/nrr_phi_operator_validation.json` |
| Appendix D bundled artifact (tracked snapshot) | N/A (tracked artifact) | `results/operator_validation_results.json` |
| Public manuscript source (current snapshot) | N/A (tracked artifact) | `manuscript/current/paper2_nrr-phi_v38.tex` |
| Public manuscript figures (current snapshot) | N/A (tracked artifact) | `manuscript/current/figure1.png` ... `manuscript/current/figure5.png` |
| Current package checksum verification | `bash scripts/verify_current_package.sh` | stdout verification for `manuscript/current/checksums_sha256.txt` |
| Current manuscript build | `bash scripts/build_current_manuscript.sh` | `/tmp/nrr-phi_current_build/paper2_nrr-phi_v38.pdf` |
| Version map | N/A (tracked artifact) | `VERSION_MAP.md` |

## Known limitations
- `results/operator_validation_results.json` is the bundled tracked artifact; reruns should write to a separate path to avoid overwriting the snapshot.
- LLM prompt/response files are archival artifacts; free-tier model build IDs are not fully fixed.
- Rule-based coverage is limited to implemented marker patterns (EN/JP).
