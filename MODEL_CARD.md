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

## Known limitations
1. S1 generalization is near-random — model cannot reliably predict
   interactions for entirely novel drug pairs.
2. Negative labels represent "not reported" not "confirmed safe."
3. Patient-context module exists architecturally but is NOT trained
   on linked patient-outcome data — disabled at inference.
4. Toxicity coverage limited to 339 drugs; missing = unknown, not zero risk.
5. No external/temporal/clinical validation performed.
6. Not evaluated for symmetry (A+B vs B+A may differ).

## Ablation: ChemBERTa-2 encoder
Tested as alternative encoder (50k pairs, 50 epochs). Did not improve
S1 generalization (AUROC 0.4769 vs GNN's 0.5021). Not deployed.
