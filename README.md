# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

NRR-Phi defines text-to-state mapping for **ambiguity-preserving inference** in LLM systems. The focus is not a universal incapacity claim about standard LLMs; it is to provide an explicit layer that maps competing interpretations into one retained state so later context can update that state without unnecessary re-enumeration. This repository contains reproducibility code, operator-validation assets, and public-track manuscript mapping for the NRR series. It targets measurable handling of lexical, structural, and epistemic ambiguity through explicit extraction stages, auditable artifacts, and clear limits on what the mapping does and does not claim to solve.

**Quick links**
- [arXiv: 2601.19933](https://arxiv.org/abs/2601.19933)
- [Positioning (NRR vs related approaches)](./docs/positioning.md)
- [Search Keywords and Weekly Rank Log](./docs/keywords.md)

**EN/JA query terms**
- `early commitment` = `早期確定`
- `ambiguity-preserving inference` = `曖昧性保持推論`

Part of the Non-Resolution Reasoning (NRR) research program. In the current spine, this repository is the text-to-state layer that feeds later interface, transfer, coupled-propagation, projection, and integrated `paper7` comparison work, with downstream carry-forward into Energy and Guarantee.

## NRR Series Hub (Start here)

For the cross-paper map and current series links, start here:
- [NRR Series Hub](https://github.com/kei-saito-research/nrr-series-hub)

Version mapping source of truth: [`VERSION_MAP.md`](./VERSION_MAP.md)
For narrow review surfaces, use the current-package and transcript-audit sections
below first; the version map remains a full-repo provenance record and may list
historical rows that are not bundled into a given drop.

NRR is not an anti-LLM framework.
NRR does not replace standard LLM use.
NRR optimizes when to commit and when to defer, under explicit conditions.
Series numbering policy: `paper3` is permanently skipped and never reused.

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18793563.svg)](https://doi.org/10.5281/zenodo.18793563)

## Bundled Review-Surface Structure

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
│   │   ├── paper2_nrr-phi_v42.tex
│   │   ├── figure1.png ... figure5.png
│   │   └── checksums_sha256.txt
│   └── archive/
│       └── ...                   # Full-repo history; not required in narrow review surfaces
├── scripts/
│   ├── audit_llm_transcripts.py
│   ├── build_current_manuscript.sh
│   ├── run_primary_checks.sh
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

Published baseline and local archive mapping is maintained in [`VERSION_MAP.md`](./VERSION_MAP.md).

- Current public arXiv line: `2601.19933v5`
- Current narrow review-surface candidate: `manuscript/current/paper2_nrr-phi_v42.tex`
- Full-repo provenance rows, including the public-current `v39` line, the prior
  derived `v40`/`v41` lines, and older historical rows including the former
  public `v38` line, remain recorded in `VERSION_MAP.md` and may be omitted
  from narrow review drops.

## Review Surface Contract

This README describes the current-candidate audit surface. Full-repo provenance
history remains recorded in `VERSION_MAP.md`, but narrow review drops only need
the assets explicitly listed in `reproducibility.md`.

Stable review-package entrypoints:
- `bash scripts/build_current_manuscript.sh`
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
`results/llm_transcript_audit_summary.json` form the reviewer-facing audit path
for the main-text LLM extraction table and combined `H = 1.087` line.

## Related Repositories

- [NRR-Core](https://github.com/kei-saito-research/nrr-core)
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime)
- [NRR-Transfer](https://github.com/kei-saito-research/nrr-transfer)
- [NRR-Coupled](https://github.com/kei-saito-research/nrr-coupled)
- [NRR-Projection](https://github.com/kei-saito-research/nrr-projection)
- [NRR-Principles](https://github.com/kei-saito-research/nrr-principles)

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
