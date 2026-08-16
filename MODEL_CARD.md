# PxDDI Model Card

## Intended use
Research prototype demonstrating patient-context-conditioned DDI prediction.
NOT validated for clinical decision-making.

## Training data
- TWOSIDES (200,000 sampled real interaction pairs)
- Negative sampling: random unreported pairs, labeled "no interaction
  reported" — NOT confirmed safe. See Limitations.
- FAERS-derived toxicity signal: 339 real drugs with structural matches
  (via PubChem bridge), single quarter (2023Q4), not deduplicated at
  case level.

## Architecture
Dual-view GAT encoder, 128 hidden dims, multi-task (DDI risk + toxicity).

## Evaluation
| Split | AUROC | F1 |
|---|---|---|
| Transductive | 0.9735 | 0.9170 |
| S1 (both unseen) | 0.5021 | 0.3530 |
| S2 (one unseen) | 0.7474 | 0.6387 |

## Known Limitations / Not Yet Implemented
1. **S1 Generalization**: Performance is near-random — model cannot reliably predict interactions for entirely novel drug pairs.
2. **Negative Labels**: Represent "not reported" in TWOSIDES, not "confirmed safe."
3. **Patient-Context Module**: Exists architecturally but is NOT trained on linked patient-outcome data — disabled at inference.
4. **Toxicity Coverage**: Limited to 339 drugs; missing = unknown, not zero risk.
5. **Clinical Validation**: No external/temporal/clinical validation performed.
6. **Symmetry**: Evaluated for symmetry and found to be highly order-sensitive (A+B != B+A). This is a known limitation.
7. **Security**: Missing auth, CORS tightening, rate-limiting, and audit logs.
8. **Clinical Rules Engine**: Not implemented.
9. **Deployment Hardening**: Model server and ChemBERTa infrastructure not hardened for production scale.

## Ablation: ChemBERTa-2 encoder
Tested as alternative encoder (50k pairs, 50 epochs). Did not improve
S1 generalization (AUROC 0.4769 vs GNN's 0.5021). Not deployed.
