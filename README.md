# nrr-phi-mapping

Reference implementation for the φ-mapping function described in:

> **Saito, K. (2026). NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference.**  
> arXiv:2601.19933

This repository provides the **text-to-state mapping** (φ function) that converts natural language text into NRR states—non-collapsed representations that preserve semantic ambiguity as multiple co-existing interpretations.

## Overview

Standard NLP pipelines resolve ambiguity during encoding, collapsing multiple interpretations into a single representation. The φ-mapping takes the opposite approach:
```
φ: Text → S = { (v₁, c₁, w₁), (v₂, c₂, w₂), ... }
```

Each interpretation `(vᵢ, cᵢ, wᵢ)` consists of:
- **vᵢ**: Semantic content (meaning)
- **cᵢ**: Contextual identifier (interpretation role)
- **wᵢ**: Activation weight (non-normalized)

**Key property**: Weights are *not* constrained to sum to 1. Multiple interpretations can maintain high activation simultaneously, unlike softmax-normalized probability distributions.

## Repository Structure
```
nrr-phi-mapping/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── rule_based_states.json     # 40-state dataset (EN + JP)
├── src/
│   ├── __init__.py
│   ├── state.py                   # NRR State representation
│   ├── conflict_detection.py      # Linguistic marker detection
│   └── phi_mapping.py             # Core φ function
├── experiments/
│   └── run_rule_based.py          # Experiment runner + Paper 2 verification
└── results/
    └── rule_based_output.json     # Reproduction of Paper 2 Table 2
```

## Quick Start
```bash
# Clone the repository
git clone https://github.com/kei-saito-research/nrr-phi-mapping.git
cd nrr-phi-mapping

# Run the experiment (no dependencies required beyond Python 3.7+)
python experiments/run_rule_based.py

# Save results to JSON
python experiments/run_rule_based.py --output results/my_output.json
```

### As a library
```python
from src import phi, NRRState

# English adversative
state = phi("I want to quit, but I also don't want to quit.")
print(state.state_size)  # 2
print(state.entropy)     # 1.0 (balanced binary)

# Japanese hedging
state = phi("その仕事に応募すべきかもしれない。", lang="JP")
print(state.state_size)  # 2
print(state.entropy)     # 1.0

# Inspect interpretations
for interp in state.interpretations:
    print(f"  {interp.context}: {interp.meaning} (w={interp.weight})")
```

## Dataset

The `data/rule_based_states.json` file contains 40 semantically ambiguous sentences:

| Category | English | Japanese | Total | Method |
|----------|---------|----------|-------|--------|
| Adversative | 10 | 10 | 20 | Rule-based |
| Hedging | 10 | 10 | 20 | Rule-based |

**Adversative** sentences contain contrastive markers that create opposing interpretations:
- EN: *but*, *however*, *yet*, *although*
- JP: けど (*kedo*), しかし (*shikashi*), だが (*daga*), でも (*demo*)

**Hedging** sentences contain epistemic markers that create uncertain/certain interpretation pairs:
- EN: *maybe*, *perhaps*, *might*, *probably*
- JP: かもしれない (*kamoshirenai*), たぶん (*tabun*), だろう (*darou*)

## Reproducing Paper 2 Results

Running the experiment reproduces Table 2 from Paper 2:
```
VERIFICATION AGAINST PAPER 2 TABLE 2

Category         Paper 2 |S|   Actual |S|   Match     Paper 2 H     Actual H   Match
-------------------------------------------------------------------------------------
  adversative           2.10         2.10       ✓         1.037        1.037       ✓
  hedging               2.05         2.05       ✓         1.002        1.002       ✓
-------------------------------------------------------------------------------------
  ✓ All values match Paper 2 Table 2.
```

Where:
- **|S|** = mean number of co-existing interpretations per state
- **H(S)** = mean Shannon entropy over interpretation weights

## NRR Paper Series

This repository accompanies Paper 2 of the NRR research series:

1. **NRR-Core**: [NRR-Core: Non-Resolution Reasoning as a Computational Framework for Contextual Identity and Ambiguity Preservation](https://arxiv.org/abs/2512.13478) — Theoretical foundations and A≠A≈A principle
2. **NRR-Phi**: [NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference](https://arxiv.org/abs/2601.19933) — φ-mapping and ambiguity-preserving states *(this repository)*

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation
```bibtex
@article{saito2026nrrphi,
  title={NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference},
  author={Saito, Kei},
  journal={arXiv preprint arXiv:2601.19933},
  year={2026}
}
```
