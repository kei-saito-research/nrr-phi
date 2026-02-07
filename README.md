# NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference

Reference implementation and reproducibility package for:

> Saito, K. (2026). *NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference.* arXiv:2601.19933

Part of the Non-Resolution Reasoning (NRR) research program.

NRR-Core: [arXiv:2512.13478](https://arxiv.org/abs/2512.13478)

## Repository Structure

```
nrr-phi/
├── src/                          # Core library
│   ├── __init__.py
│   ├── state.py                  # NRRState, Interpretation classes
│   ├── phi_mapping.py            # φ mapping: text → NRR state
│   └── conflict_detection.py     # Linguistic marker detection (EN/JP)
│
├── data/                         # Input data
│   ├── rule_based_data.json              # 40 sentences (adv 20 + hdg 20)
│   └── operator_validation_states.json   # 580 test cases (Appendix D)
│
├── results/                      # Experiment outputs
│   ├── rule_based_output.json            # Rule-based extraction results
│   └── operator_validation_results.json  # Operator validation results
│
├── experiments/                  # Reproducibility scripts
│   ├── rule_based_experiment.py          # Main text: Table 2
│   └── run_operator_validation.py        # Appendix D: Table 7, Figures 4-5
│
├── prompts/                      # LLM experiment prompts + responses
│   ├── GPTprompts_for_kei.txt            # ChatGPT prompts + outputs
│   ├── Geminiprompts_for_kei_2.txt       # Gemini prompts + outputs
│   └── claudeprompts_for_kei_2_2.txt     # Claude prompts + outputs
│
├── notebooks/                    # Experiment notebooks
│   └── operator_validation.ipynb         # Operator validation notebook
│
├── LICENSE                       # CC BY 4.0
├── README.md
└── requirements.txt              # Python dependencies
```

## Quick Start

### Rule-based extraction (Table 2)

```bash
cd experiments
python rule_based_experiment.py
```

### Operator validation (Appendix D, Table 7)

```bash
cd experiments
python run_operator_validation.py --data ../data/operator_validation_states.json
```

### Using the φ mapping library

```python
from src.phi_mapping import phi

state = phi("I want to quit, but I don't want to quit.")
print(state)           # NRRState(|S|=2, H=1.000, lang=EN, cat=adversative)
print(state.entropy)   # 1.0

state_jp = phi("辞めたいけど、辞めたくない。", lang="JP")
print(state_jp)        # NRRState(|S|=2, H=1.000, lang=JP, cat=adversative)
```

## Experiment Summary

| Experiment | Data | Script | Results |
|---|---|---|---|
| Rule-based (Table 2) | data/rule_based_data.json | experiments/rule_based_experiment.py | results/rule_based_output.json |
| LLM-based (Table 3) | prompts/*.txt | Manual (free-tier web UI) | Embedded in prompt files |
| Operator validation (Table 7) | data/operator_validation_states.json | experiments/run_operator_validation.py | results/operator_validation_results.json |

## Key Results

- **68 sentences** across 5 ambiguity categories (EN + JP)
- Mean state entropy H = **1.087 bits** (vs H = 0 for collapse-based models)
- **0% collapse** for all principle-satisfying operators (580 test cases)
- **2,740 total measurements** in operator validation

## Requirements

- Python 3.8+
- NumPy

## Related Repositories

- [NRR-Core](https://github.com/kei-saito-research/nrr-core) - Foundational framework *(arXiv:2512.13478)*
- [NRR-IME](https://github.com/kei-saito-research/nrr-ime) - Structure-aware optimization
- [NRR-Universal](https://github.com/kei-saito-research/nrr-universal) - Universal generality validation

## Citation

```bibtex
@article{saito2026nrrphi,
  title={NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference},
  author={Saito, Kei},
  journal={arXiv preprint arXiv:2601.19933},
  year={2026}
}
```

## License

CC BY 4.0 — See [LICENSE](LICENSE) for details.
