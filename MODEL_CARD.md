# PxDDI Model Card

## Intended use

PxDDI is a research prototype for structure-based DDI prediction. It is not
validated for clinical decision-making, prescribing, diagnosis, triage, or
patient-specific treatment recommendations.

## Current deployed artifact

- Model: symmetric dual-view GAT encoder with DDI and toxicity heads.
- Checkpoint: `backend/checkpoints/pxddi_model.pt`.
- Stored validation AUROC: 0.9485.
- Best validation epoch: 193.
- Decision threshold selected from validation data: 0.5453.
- Patient context: disabled at inference.

The stored AUROC is checkpoint metadata. It is not external, temporal, or
clinical validation.

## Training data

- TWOSIDES: 200,000 sampled rows in the current training configuration.
- Negatives: randomly sampled unreported pairs. They mean "not reported," not
  "confirmed safe."
- FAERS toxicity signal: one quarter (2023Q4), based on severe-outcome report
  fractions and mapped through PubChem structures.
- Toxicity bridge: 339 unique canonical structures. Source rows include 58
  duplicated structures with conflicting scores; those require an audited
  resolution policy before the next training run.

## Evaluation status

Historical figures in the repository are not currently reproducible because
the raw datasets, split manifests, run logs, predictions, and completed Colab
notebook are not committed. They must be regenerated and versioned before use
in a report or comparison.

## Verified software behavior

- **Symmetry:** fixed. The pair representation uses embedding sums and
  absolute differences, so A+B and B+A are order-independent by construction.
  Regression tests check the shipped GNN checkpoint on multiple pairs.
- **Input validation:** invalid, single-atom, oversized, and over-complex
  inputs are rejected by the API.
- **Patient context:** accepted fields are explicitly not applied.
- **Toxicity coverage:** API checks canonical SMILES rather than raw input.

## Current limitations

1. **S1 generalization:** prior reported performance for two unseen drugs is
   near random; this limitation remains unresolved.
2. **Negative labels:** unreported interaction pairs are not proven negatives.
3. **Toxicity targets:** FAERS signals are observational, affected by reporting
   bias, from a single quarter, and currently contain unresolved duplicate
   structural mappings.
4. **Patient context:** no linked patient-exposure-outcome training data is in
   this project; the module must remain disabled.
5. **Molecular representation:** graph features omit bond order,
   stereochemistry, chirality, and other chemical detail.
6. **Calibration:** risk and toxicity head outputs are not clinical
   probabilities and have not been calibrated.
7. **Validation:** no external, temporal, prospective, or clinical validation
   has been performed.
8. **Explanation:** `/explain` attributes molecular embeddings and applies a
   functional-group heuristic; it is not a final pair-risk explanation or a
   literature validation.
9. **Security:** local CORS is restricted and explanation concurrency is
   bounded, but authentication, global rate limiting, audit logging, TLS,
   monitoring, and public-deployment controls are absent.
10. **Deployment:** the API has readiness checks and a non-root container, but
    actual Docker image execution, resource limits, pinned dependencies, and
    operational monitoring remain pending.

## ChemBERTa ablation

ChemBERTa is retained as an undeployed ablation artifact. It is intentionally
unchanged until a separate decision is made to archive it or fully support it.
