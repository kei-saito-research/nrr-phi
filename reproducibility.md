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

## Run commands
```bash
pip install -r requirements.txt
python3 experiments/rule_based_experiment.py > results/rule_based_output.json
python3 experiments/run_operator_validation.py \
  --data data/operator_validation_states.json \
  --output results/operator_validation_rerun_summary.json
```

## Artifact map
| Table/Figure | Command | Output file |
|---|---|---|
| Paper Table 2 (rule-based extraction summary) | `python3 experiments/rule_based_experiment.py > results/rule_based_output.json` | `results/rule_based_output.json` |
| Appendix D rerun summary (local output) | `python3 experiments/run_operator_validation.py --data data/operator_validation_states.json --output results/operator_validation_rerun_summary.json` | `results/operator_validation_rerun_summary.json` |
| Appendix D bundled artifact (tracked snapshot) | N/A (tracked artifact) | `results/operator_validation_results.json` |
| Public manuscript source (current snapshot) | N/A (tracked artifact) | `manuscript/current/paper2_nrr-phi_v38.tex` |
| Public manuscript figures (current snapshot) | N/A (tracked artifact) | `manuscript/current/figure1.png` ... `manuscript/current/figure5.png` |
| Version map | N/A (tracked artifact) | `VERSION_MAP.md` |

## Known limitations
- `results/operator_validation_results.json` is the bundled tracked artifact; reruns should write to a separate path to avoid overwriting the snapshot.
- LLM prompt/response files are archival artifacts; free-tier model build IDs are not fully fixed.
- Rule-based coverage is limited to implemented marker patterns (EN/JP).
