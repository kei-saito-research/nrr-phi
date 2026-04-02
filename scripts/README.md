# Scripts

This directory contains the stable manuscript and audit wrappers for the current
NRR-Phi repository surface.

## Stable entrypoints

- `build_current_manuscript.sh`
  - builds the latest manuscript in `manuscript/current/` to a temp output directory
- `run_primary_checks.sh`
  - reruns the deterministic rule-based, operator-validation, and transcript-audit checks to temp outputs
- `verify_active_review_surface.sh`
  - verifies that `manuscript/current/` contains only the current `.tex` / `.pdf` pair and checks `manuscript/checksums_active_review_surface_sha256.txt`
- `verify_current_package.sh`
  - verifies the active review surface first and then checks `manuscript/checksums_current_package_sha256.txt`

`audit_llm_transcripts.py` remains a bundled audit helper, while the four
entrypoints above define the stable current-package interface.
