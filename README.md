# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

NRR-Phi defines text-to-state mapping for **ambiguity-preserving inference** in LLM systems. The focus is not a universal incapacity claim about standard LLMs; it is to provide an explicit layer that maps competing interpretations into one retained state so later context can update that state without unnecessary re-enumeration. This repository contains the manuscript snapshot, reproducibility code, and operator-validation assets for the Phi line. It targets measurable handling of lexical, structural, and epistemic ambiguity through explicit extraction stages, auditable artifacts, and clear limits on what the mapping does and does not claim to solve.

**Quick links**
- [arXiv: 2601.19933](https://arxiv.org/abs/2601.19933)
- [Positioning (NRR vs related approaches)](./docs/positioning.md)
- [Search Keywords and Reader Guide](./docs/keywords.md)

**EN/JA query terms**
- `early commitment` = `早期確定`
- `ambiguity-preserving inference` = `曖昧性保持推論`

Part of the Non-Resolution Reasoning (NRR) research program. This repository presents the text-to-state mapping line and bundles the manuscript, transcript-audit artifacts, and verification scripts for that line.

## NRR Series Hub (Start here)

For the cross-paper map and current series links, start here:
- [NRR Series Hub](https://github.com/kei-saito-research/nrr-series-hub)

Version mapping source of truth: [`VERSION_MAP.md`](./VERSION_MAP.md)
For the current package snapshot and transcript-audit artifacts, start with the sections below; `VERSION_MAP.md` remains the broader version history reference.

NRR is not an anti-LLM framework.
NRR does not replace standard LLM use.
NRR optimizes when to commit and when to defer, under explicit conditions.
Series numbering policy: `paper3` is permanently skipped and never reused.

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18793563.svg)](https://doi.org/10.5281/zenodo.18793563)

## Package Structure

```
nrr-phi/
├── data/
│   ├── rule_based_data.json
│   └── operator_validation_states.json
├── results/
│   ├── rule_based_output.json
│   └── operator_validation_results.json
├── experiments/
│   ├── rule_based_experiment.py
│   └── run_operator_validation.py
├── prompts/                      # Prompt+response transcripts for the main-text LLM table
│   └── llm_audit_manifest.json   # Included sentence IDs and aggregation contract
├── notebooks/
├── manuscript/
│   ├── current/
│   │   ├── paper2_nrr-phi_v44.tex
│   │   └── paper2_nrr-phi_v44.pdf
│   ├── figures/
│   │   ├── figure1.png ... figure5.png
│   ├── checksums_active_review_surface_sha256.txt
│   ├── checksums_current_package_sha256.txt
│   └── archive/
│       └── ...                   # Full-repo history
├── scripts/
│   ├── audit_llm_transcripts.py
│   ├── README.md
│   ├── build_current_manuscript.sh
│   ├── run_primary_checks.sh
│   ├── verify_active_review_surface.sh
│   └── verify_current_package.sh
├── VERSION_MAP.md
├── reproducibility.md
├── requirements.txt
├── README.md
├── LICENSE
└── results/
    ├── llm_transcript_audit_summary.json
    ├── operator_validation_results.json
    └── rule_based_output.json
```

## Manuscript Artifacts

Published and local version mapping is maintained in [`VERSION_MAP.md`](./VERSION_MAP.md).

- Current public arXiv line: `2601.19933v5`
- Current manuscript snapshot: `manuscript/current/paper2_nrr-phi_v44.tex`
- Earlier package versions remain recorded in `VERSION_MAP.md`.

## Package Entry Points

Primary package entrypoints:
- `bash scripts/build_current_manuscript.sh`
- `bash scripts/verify_active_review_surface.sh`
- `bash scripts/verify_current_package.sh`
- `bash scripts/run_primary_checks.sh`

## Quick Start

```bash
pip install -r requirements.txt
python3 experiments/rule_based_experiment.py > results/rule_based_output.txt
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output results/operator_validation_rerun_summary.json
```

## Reproducibility

See [`reproducibility.md`](./reproducibility.md) for environment, fixed settings, commands, and artifact mapping.
The bundled `prompts/` files plus `prompts/llm_audit_manifest.json`,
`scripts/audit_llm_transcripts.py`, and
`results/llm_transcript_audit_summary.json` form the bundled audit path
for the main-text LLM extraction table and combined `H = 1.087` line.

## Related Repositories

- [NRR-Core](https://github.com/kei-saito-research/nrr-core)
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime)
- [NRR-Transfer](https://github.com/kei-saito-research/nrr-transfer)
- [NRR-Coupled](https://github.com/kei-saito-research/nrr-coupled)
- [NRR-Projection](https://github.com/kei-saito-research/nrr-projection)
- [NRR-Patterns](https://github.com/kei-saito-research/nrr-patterns)

## Collaboration Style

I support written technical Q&A, concept clarification, and small evaluation design.

Typical flow:
1. you send questions and context,
2. I return a structured technical response,
3. if needed, I provide an English-ready version for external sharing.

Scope: research interpretation and evaluation planning.  
Out of scope: production integration, implementation outsourcing, ongoing operations, and SLA/deadline commitments.  
Contact: kei.saito.research@gmail.com

## License

CC BY 4.0. See [LICENSE](LICENSE).
