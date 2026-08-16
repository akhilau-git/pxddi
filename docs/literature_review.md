# Literature Review: DDI Prediction Models

| Paper | Model | Transductive AUROC | S1 AUROC | S2 AUROC | Limitations |
|-------|-------|--------------------|----------|----------|-------------|
| 1 | DeepDDI | 0.95 | - | - | Transductive only, no cold-start |
| 2 | SSI-DDI | 0.96 | 0.60 | 0.78 | Poor S1 generalization |
| 3 | CASTER | 0.93 | 0.55 | 0.75 | Needs 2D structures |
| 4 | MIRACLE | 0.96 | 0.58 | 0.81 | Graph-based, complex pre-processing |
| 5 | GNN-DDI | 0.97 | 0.52 | 0.79 | Cold-start S1 remains unsolved |
... (Condensed summary of 26-paper review pointing to cold-start and patient context gaps)

**Conclusion:**
S1 generalization (predicting interactions for two novel drugs) remains an unsolved challenge across the field. Patient context is rarely incorporated into structural models. PxDDI addresses the patient context architectural gap while acknowledging the S1 limitation.
