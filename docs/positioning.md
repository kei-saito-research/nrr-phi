# NRR-Phi Positioning

This note gives a short public-facing view of where the NRR-Phi line sits among nearby approaches. The focus is ambiguity-preserving inference: keeping multiple plausible interpretations available when context is still incomplete, then allowing later context to support commitment when appropriate.

NRR-Phi is not a replacement for standard LLM use. In this line, the contribution is a text-to-state mapping that keeps competing interpretations explicit enough to be updated, inspected, and reused across later turns or later evidence.

## How It Relates to Nearby Approaches

| Approach | Typical focus | How NRR-Phi differs |
| --- | --- | --- |
| Fuzzy reasoning | Represents graded truth or soft category boundaries. | NRR-Phi focuses on carrying multiple discrete interpretations forward as explicit state candidates. |
| Calibrated abstention | Decides whether to answer or abstain under low confidence. | NRR-Phi focuses on preserving alternatives internally before final output-time commitment. |
| Word sense disambiguation | Chooses one sense from local context. | NRR-Phi focuses on delaying collapse when later context may still change the appropriate interpretation. |

## What This Repository Covers

- Text-to-state mapping for ambiguity-preserving inference.
- Handling of lexical, structural, and epistemic ambiguity under explicit conditions.
- Manuscript, reproducibility, and audit materials for the Phi line.

Formal definitions, experiments, and limitations remain in the manuscript and reproducibility materials.

## Navigation

- [README](../README.md)
- [Reproducibility](../reproducibility.md)
- [Search Keywords and Reader Guide](./keywords.md)
- [arXiv: 2601.19933](https://arxiv.org/abs/2601.19933)
