# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

Reference implementation and reproducibility package for:

Saito, K. (2026). NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference.
arXiv:2601.19933

Part of the Non-Resolution Reasoning (NRR) research program.
Program Map (series hub): https://github.com/kei-saito-research/nrr-core/blob/main/PROGRAM_MAP.md

NRR is not an anti-LLM framework.
NRR does not replace standard LLM use.
NRR optimizes when to commit and when to defer, under explicit conditions.

## Publication handling

This repository is maintained as a code/data reproducibility package.
Manuscript text artifacts (`.tex`, `.pdf`) are not included in this snapshot.

## Repository structure

```
nrr-phi/
|-- src/
|   |-- state.py
|   |-- phi_mapping.py
|   `-- conflict_detection.py
|-- data/
|   |-- rule_based_data.json
|   `-- operator_validation_states.json
|-- experiments/
|   |-- rule_based_experiment.py
|   `-- run_operator_validation.py
|-- results/
|   |-- rule_based_output.json
|   `-- operator_validation_results.json
|-- prompts/
|-- notebooks/
|-- reproducibility.md
|-- requirements.txt
|-- README.md
`-- LICENSE
```

## Quick start

```bash
pip install -r requirements.txt
python3 experiments/rule_based_experiment.py > results/rule_based_output.txt
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output results/operator_validation_results.json
```

## Reproducibility

See `reproducibility.md` for environment, fixed settings, commands, and artifact mapping.

## Related repositories

- https://github.com/kei-saito-research/nrr-core
- https://github.com/kei-saito-research/nrr-ime
- https://github.com/kei-saito-research/nrr-transfer

## License

CC BY 4.0. See `LICENSE`.
