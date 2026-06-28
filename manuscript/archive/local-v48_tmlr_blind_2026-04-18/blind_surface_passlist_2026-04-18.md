# Blind Surface Passlist

Artifact directory:
- `manuscript/archive/local-v48_tmlr_blind_2026-04-18`

Artifact files:
- `paper2_nrr-phi_v48_blind.tex`
- `paper2_nrr-phi_v48_blind.pdf`
- `checksums_sha256.txt`

Blind-surface checks completed on 2026-04-18:
- Author name, ORCID, affiliation, and contact email removed from the manuscript header.
- Copyright and license block removed from the blinded manuscript.
- Explicit repository and series-hub links absent from the blinded manuscript surface.
- Acknowledgments section removed from the blinded manuscript.
- Prior self-citation converted to anonymized placeholder citation key `anon2025framework`.
- Explicit vendor-name acknowledgment text absent from the blinded manuscript.
- Title generalized to avoid direct reuse of the public `NRR-Phi` naming surface.
- Series-identifying terminology (`NRR-*`, `Non-Resolution Reasoning`, arXiv IDs, repo-name strings) removed from the blinded manuscript text surface.
- Blind manuscript compiled successfully to `paper2_nrr-phi_v48_blind.pdf`.

Leak-search command used:

```bash
rg -n -i 'Kei Saito|ORCID|Independent Researcher|kei.saito.research@gmail.com|github.com/kei-saito-research|nrr-series-hub|Acknowledgments|Anthropic|OpenAI|Google|arXiv:2512\\.13478|arXiv:2601\\.19933|NRR-(Phi|Core|IME|Transfer|Coupled|Guarantee|Principles|Boundary|Energy)|Non-Resolution Reasoning|nrr-(phi|series-hub|core)' \
  manuscript/archive/local-v48_tmlr_blind_2026-04-18/paper2_nrr-phi_v48_blind.tex
```

Observed result:
- No matches.
