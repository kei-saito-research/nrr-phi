# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

NRR-Phi defines text-to-state mapping for **ambiguity-preserving inference** in LLM systems. The focus is preventing **premature commitment in LLM decoding** when inputs carry competing interpretations, then deciding **defer vs commit** with explicit conditions. Instead of forcing a single parse too early, the pipeline keeps multiple candidates in state form so later context can resolve them without unnecessary backtracking. This repository contains reproducibility code, operator-validation assets, and public-track manuscript mapping for the NRR series. It targets measurable handling of lexical, structural, and epistemic ambiguity and the reduction of **semantic collapse** in practical inference workflows. The intent is implementation clarity: explicit extraction stages, auditable artifacts, and clear limits on what the mapping does not claim to solve.

**Quick links**
- [arXiv: 2601.19933](https://arxiv.org/abs/2601.19933)
- [Positioning (NRR vs related approaches)](./docs/positioning.md)
- [Search Keywords and Weekly Rank Log](./docs/keywords.md)

**EN/JA query terms**
- `early commitment` = `早期確定`
- `ambiguity-preserving inference` = `曖昧性保持推論`

Part of the Non-Resolution Reasoning (NRR) research program.

## NRR Series Hub (Start here)

For the cross-paper map and current series links, start here:
- [NRR Series Hub](https://github.com/kei-saito-research/nrr-series-hub)

Version mapping source of truth: [`VERSION_MAP.md`](./VERSION_MAP.md)

NRR is not an anti-LLM framework.
NRR does not replace standard LLM use.
NRR optimizes when to commit and when to defer, under explicit conditions.
Series numbering policy: `paper3` is permanently skipped and never reused.

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18793563.svg)](https://doi.org/10.5281/zenodo.18793563)

## Repository Structure

```
nrr-phi/
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── phi_mapping.py
│   └── conflict_detection.py
├── data/
│   ├── rule_based_data.json
│   └── operator_validation_states.json
├── results/
│   ├── rule_based_output.json
│   └── operator_validation_results.json
├── experiments/
│   ├── rule_based_experiment.py
│   └── run_operator_validation.py
├── prompts/
├── notebooks/
├── manuscript/
│   ├── current/
│   │   ├── paper2_nrr-phi_v39.tex
│   │   ├── figure1.png ... figure5.png
│   │   └── checksums_sha256.txt
│   └── archive/
│       ├── public-v25/
│       ├── local-v37/
│       └── local-v36/
├── scripts/
│   └── verify_versions.sh
├── VERSION_MAP.md
├── reproducibility.md
├── requirements.txt
├── README.md
└── LICENSE
```

## Manuscript Artifacts

Published baseline and local archive mapping is maintained in [`VERSION_MAP.md`](./VERSION_MAP.md).

- Current public arXiv line: `2601.19933v4`
- Current public source snapshot in repo: `manuscript/current/paper2_nrr-phi_v38.tex`
- Current local replacement candidate: `manuscript/current/paper2_nrr-phi_v39.tex`
- Archived public baseline snapshot: `manuscript/archive/public-v25/paper2_nrr-phi_v25.tex`
- Archived local draft snapshot: `manuscript/archive/local-v37/paper2_nrr-phi_v37.tex`

## Version Verification

```bash
./scripts/verify_versions.sh
```

Stable review-package entrypoints:
- `bash scripts/build_current_manuscript.sh`
- `bash scripts/verify_current_package.sh`
- `bash scripts/run_primary_checks.sh`

## Quick Start

```bash
pip install -r requirements.txt
python3 experiments/rule_based_experiment.py > results/rule_based_output.json
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output results/operator_validation_rerun_summary.json
```

## Reproducibility

See [`reproducibility.md`](./reproducibility.md) for environment, fixed settings, commands, and artifact mapping.

## Related Repositories

- [NRR-Core](https://github.com/kei-saito-research/nrr-core)
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime)
- [NRR-Transfer](https://github.com/kei-saito-research/nrr-transfer)

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
