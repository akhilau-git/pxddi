# PxDDI Model Card

## Intended use

PxDDI is a research prototype for structure-based DDI prediction. It is not
validated for clinical decision-making, prescribing, diagnosis, triage, or
patient-specific treatment recommendations.

## Committed reference artifact

- Model: symmetric dual-view GAT encoder with DDI and toxicity heads.
- Checkpoint: `backend/checkpoints/pxddi_model.pt`.
- Stored validation AUROC: 0.8972.
- Best validation epoch: 195.
- Decision threshold selected from validation data: 0.4404.
- Patient context: disabled at inference.

The stored AUROC is checkpoint metadata. It is not external, temporal, or
clinical validation. A later local or Colab checkpoint must be accompanied by
its own run manifest, split files, predictions, and checkpoint hash before it
replaces this reference in project documentation.

## Training data

- TWOSIDES: 200,000 sampled rows in the current training configuration.
- Negatives: randomly sampled unreported pairs. They mean "not reported," not
  "confirmed safe."
- FAERS toxicity signal: one quarter (2023Q4), based on severe-outcome report
  fractions and mapped through PubChem structures.
- Toxicity bridge: 339 unique canonical structures. Source rows include 58
  duplicated structures with conflicting scores. The reproducible training
  pipeline excludes those structures from toxicity supervision rather than
  guessing a score, leaving 281 clean structures and saving the conflict table
  as a run artifact.

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
   bias, and from a single quarter. Conflicting structural mappings are now
   excluded conservatively, but still require scientific audit before broader
   claims are made.
4. **Patient context:** no linked patient-exposure-outcome training data is in
   this project; the module must remain disabled.
5. **Molecular representation:** the deployed legacy checkpoint still omits
   bond order, stereochemistry, chirality, and other chemical detail. The
   separate edge-aware candidate adds these atom and bond features, but has not
   solved S1 generalization and is not promoted.
6. **Calibration:** the deployed legacy checkpoint is uncalibrated. Candidate
   checkpoints can store a calibration mapping fitted only on the internal
   validation split, which must not be presented as calibrated cold-start,
   external, or clinical performance.
7. **Validation:** no external, temporal, prospective, or clinical validation
   has been performed.
8. **Explanation:** `/explain` attributes molecular embeddings and applies a
   functional-group heuristic; it is not a final pair-risk explanation or a
   literature validation. It is intentionally unavailable if an edge-aware
   candidate is ever promoted, until a compatible explanation method is
   implemented and evaluated.
9. **Security:** local CORS is restricted and explanation concurrency is
   bounded, but authentication, global rate limiting, audit logging, TLS,
   monitoring, and public-deployment controls are absent.
10. **Deployment:** the API has readiness checks and a non-root container, but
    actual Docker image execution, resource limits, pinned dependencies, and
    operational monitoring remain pending.

## Candidate model and evaluation workflow

The current deployed artifact uses the legacy 13-feature GAT schema. New
candidate training uses an edge-aware GATv2 schema with atom and bond features
for bond type, bond stereo, atom chirality, hybridization, charge, ring status,
and aromaticity. Candidate checkpoints are stored separately and must not
replace the deployed artifact until a controlled comparison reports all of:

- Transductive, S1, and S2 metrics.
- Raw and validation-calibrated Brier/ECE values.
- Ablation/baseline comparison.
- Repeated-seed uncertainty intervals when making comparative claims.

The experiment suite additionally refuses a cross-model comparison when the
matched runs do not have identical TWOSIDES input and split-manifest hashes.
For repeated runs it reports paired candidate-minus-reference bootstrap
intervals; a one-seed screening run intentionally has no confidence interval.

The training audit produces a counterion-candidate review table. It does not
invent parent mappings for isolated ions or salts: an authoritative source and
manual review are required before any mapping is approved.

## ChemBERTa ablation

ChemBERTa is retained as an undeployed ablation artifact. It is intentionally
unchanged until a separate decision is made to archive it or fully support it.
