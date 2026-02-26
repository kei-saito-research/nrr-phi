# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

Reference implementation and reproducibility package for:

> Saito, K. (2026). *NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference.* arXiv:2601.19933

Part of the Non-Resolution Reasoning (NRR) research program.
Program Map (series hub): [NRR Program Map](https://github.com/kei-saito-research/nrr-core/blob/main/PROGRAM_MAP.md)
Version mapping source of truth: [`VERSION_MAP.md`](./VERSION_MAP.md)

NRR is not an anti-LLM framework.
NRR does not replace standard LLM use.
NRR optimizes when to commit and when to defer, under explicit conditions.

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
│   │   ├── paper2_nrr-phi_v25.tex
│   │   ├── figure1.png ... figure5.png
│   │   └── checksums_sha256.txt
│   └── archive/
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

- Public arXiv line: `2601.19933v3`
- Current public snapshot in repo: `manuscript/current/paper2_nrr-phi_v25.tex`
- Archived local draft snapshot: `manuscript/archive/local-v36/paper2_nrr-phi_v36.tex`

## Version Verification

```bash
./scripts/verify_versions.sh
```

## Quick Start

```bash
pip install -r requirements.txt
python3 experiments/rule_based_experiment.py > results/rule_based_output.txt
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output results/operator_validation_results.json
```

## Reproducibility

See [`reproducibility.md`](./reproducibility.md) for environment, fixed settings, commands, and artifact mapping.

## Related Repositories

- [NRR-Core](https://github.com/kei-saito-research/nrr-core)
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime)
- [NRR-Transfer](https://github.com/kei-saito-research/nrr-transfer)

## License

CC BY 4.0. See [LICENSE](LICENSE).
