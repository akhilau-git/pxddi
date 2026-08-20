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

The Phase 7 evaluator is implemented but has not yet generated accepted
candidate results. It will report AUROC, average precision, MCC, Brier/ECE,
validation-thresholded decision metrics, stratified test-set bootstrap
intervals, structural-novelty slices, confidence-ranked errors, conformal
abstention coverage, and hardware-specific efficiency records. These are
research measurements on reported-versus-unreported labels, not clinical
performance measures.

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
   A separate motif-edge-aware candidate additionally uses fixed SMARTS motif
   counts. It is untrained in the committed project and its motifs are not
   validated causal explanations.
   A separate cross-attention edge-aware candidate lets atom embeddings from
   the two drugs exchange pair-isolated attention messages. It is untrained,
   not deployed, and its attention weights are not validated explanations.
6. **Calibration:** the deployed legacy checkpoint is uncalibrated. Candidate
   checkpoints can store a calibration mapping fitted on a disjoint internal
   validation-calibration partition, which must not be presented as calibrated
   cold-start, external, or clinical performance.
   New candidate run artifacts additionally include validation-only
   split-conformal sets and nearest-training-drug ECFP similarity flags. These
   are abstention/review signals under stated assumptions, not clinical
   confidence, a safety guarantee, or a cure for S1/S2 distribution shift.
7. **Validation:** no external, temporal, prospective, or clinical validation
   has been performed.
8. **Explanation:** `/explain` attributes molecular embeddings and applies a
   functional-group heuristic; it is not a final pair-risk explanation or a
   literature validation. An optional, offline candidate audit now performs
   single-component atom/bond/motif occlusion on a bounded evaluation subset,
   with local fidelity/sufficiency, score-symmetry, and canonical-reencoding
   checks. Cross-attention atom and configured-SMARTS motif associations are
   also exported only as internal associations. None of these outputs are causal mechanisms, chemical proof,
   or clinically validated explanations. Candidate models remain unavailable
   through `/explain` unless a separately reviewed API-compatible method is
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

The normal Transductive/S1/S2 protocol and the separate Murcko
scaffold-disjoint protocol must be trained and reported as distinct studies.
The latter uses scaffold-disjoint training, validation, and test partitions,
so its score cannot be merged into a normal-split table. Both protocols still
need executed Colab artifacts before their metrics can appear in the paper.

The experiment suite additionally refuses a cross-model comparison when the
matched runs do not have identical TWOSIDES input and split-manifest hashes.
For repeated runs it reports paired candidate-minus-reference bootstrap
intervals. With five or more matched seeds it also reports a two-sided
Wilcoxon signed-rank test with Holm correction across the study comparisons; a
one-seed screening run intentionally has neither interval nor hypothesis test.

The screening suite includes a non-deployment ECFP/Morgan-fingerprint +
linear-logistic baseline. It is a necessary comparison point, not a claim of
novelty or clinical utility. Its pair vector uses fingerprint sums and absolute
differences so, like the GNN, it is order-independent.

The suite also includes two explicit motif ablations: motif-edge-aware DDI-only
and motif-edge-aware multi-task. They compare the fixed 17-feature SMARTS motif
view against the same edge-aware graph encoder. The component must not be
described as helpful until repeated matched-seed S1/S2 evidence supports it.

Two additional cross-attention ablations—cross-attention edge-aware DDI-only
and multi-task—test direct atom-level interaction reasoning against the same
edge-aware reference. The attention layer is pair-isolated and the final head
is still symmetric, but it must not be presented as a mechanism explanation or
accuracy improvement until repeated matched-seed S1/S2, calibration, and
efficiency evidence supports it.

For a trained nonlegacy candidate, setting
`PXDDI_RUN_CANDIDATE_EXPLANATIONS=1` writes a bounded offline occlusion audit
under the immutable run artifact directory. It uses raw (not calibrated) model
probabilities, stores the selected evaluation examples and all stated warnings,
and renders matching indexed SVG molecular figures. It does not alter the
candidate checkpoint, legacy checkpoint, or API. It is
evidence about local model behaviour only; a separate stability study and
expert chemical review are still needed before it could support scientific
interpretation.

`analyze_explanation_stability.py` can compare shared explained pairs across
two or more candidate seeds. It reports top-atom, motif, and cross-motif
association overlap plus raw-score variation; this is an agreement audit, not
validation of an explanation or chemical mechanism.

The training audit produces a counterion-candidate review table. It does not
invent parent mappings for isolated ions or salts: an authoritative source and
manual review are required before any mapping is approved.

## Fixed-split ensemble and abstention workflow

The Phase 6 ensemble launcher trains three to five independently initialized
members on the same audited data sample and exact split, then averages only
their verified raw prediction rows. It refuses to combine members with a
different source-data hash, split-manifest evidence, architecture, loss
configuration, or row provenance. A fresh ensemble calibrator, threshold, and
conformal rule use disjoint post-training validation partitions.

The resulting prediction artifact supplies member-score standard deviation,
conformal ambiguity, structural-domain distance, and an explicit abstention
status. It is deliberately offline and cannot replace, alter, or be served as
the deployed legacy checkpoint. The disagreement threshold is a transparent
research setting—not a calibrated clinical threshold—and must be assessed on
the actual S1/S2 results before any research claim.

## ChemBERTa ablation

ChemBERTa is retained as an undeployed ablation artifact. It is intentionally
unchanged until a separate decision is made to archive it or fully support it.
