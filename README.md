# PxDDI: Research Drug-Drug Interaction Prototype

PxDDI is a research prototype for structure-based drug-drug interaction (DDI)
prediction. It combines a symmetric molecular graph model with a toxicity head
and a patient-context architecture. It is **not for clinical use**.

Patient context is intentionally disabled at inference: the repository does
not contain linked patient, drug-pair, and outcome data needed to train that
module safely.

## Committed reference GNN artifact

The checkpoint at `backend/checkpoints/pxddi_model.pt` currently reports:

- Best validation epoch: 195
- Stored validation AUROC: 0.8972
- Validation-selected decision threshold: 0.4404
- Training cap: 200,000 rows

These are metadata for the committed reference artifact, not external or
clinical validation. A locally replaced or newly trained checkpoint must be
described by its own Colab `run_manifest.json`, not by copying these figures.
The API also marks its risk estimate as uncalibrated.

## Historical evaluation figures

The following previously reported figures are retained for traceability, but
cannot yet be independently reproduced from this repository because raw data,
split manifests, prediction files, and completed Colab outputs are not stored
here.

| Split | AUROC | F1 |
|---|---:|---:|
| Transductive | 0.9735 | 0.9170 |
| S1 (both unseen drugs) | 0.5021 | 0.3530 |
| S2 (one unseen drug) | 0.7474 | 0.6387 |

Do not treat these figures as current, reproducible evidence until the next
audited Colab run produces versioned data, split manifests, metrics, and
prediction artifacts.

## Local setup

Use CPython 3.10.11. Install development dependencies and run tests:

```powershell
C:\Python310\python.exe -m pip install -r requirements-dev.txt
C:\Python310\python.exe -m pytest -q
```

Start the research API:

```powershell
C:\Python310\python.exe -m uvicorn backend.main:app --reload
```

Useful endpoints:

- `GET /health` — process liveness and model metadata.
- `GET /ready` — deployment readiness; fails if required toxicity coverage
  data is unavailable.
- `POST /predict` — research-only pair prediction.
- `POST /explain` — slower embedding-level explanation.

The default frontend origin is `http://localhost:3000`. Set
`PXDDI_ALLOWED_ORIGINS` to a comma-separated allowlist for another deployment.

## Colab training

Training is intentionally run in Google Colab with a GPU. Install the
Colab-compatible non-PyTorch packages from `requirements_colab.txt`, set
`PXDDI_DATA_BASE` to the Drive data directory when needed, and run
`src/training/train_full_pipeline_v2.py`. ChemBERTa remains disabled.

`requirements_colab.txt` fixes `torch-geometric` at version 2.7.0, the version
used by the reviewed candidate run. If a Colab runtime previously imported a
different PyG version or reports a circular-import error, restart the runtime
after installing the requirements before starting training. The run manifest
records the resolved package versions, so the installed environment remains
auditable.

Google Drive shortcuts may be read-only even when the source data inside them
is accessible. In that case set `PXDDI_DATA_BASE` to the shared shortcut and
`PXDDI_RESULTS_BASE` to a writable folder in your own `MyDrive`. Normal GNN
and ECFP baseline runs will then read the TWOSIDES CSV and toxicity bridge from
the shortcut while storing artifacts, latest results, and new candidate
checkpoints under the writable results folder. The run manifest records both
locations. This never alters the deployed backend checkpoint.

Each run creates `artifacts/run_<timestamp>/` in Drive containing an initial
and final manifest, resolved package/GPU environment, source revision, model
summary, numeric training-history CSV, clean toxicity-label and input-quality
audits, exact split CSVs and hashes, checkpoint hash, prediction CSVs, metrics
JSON, and PNG/PDF figures. The pipeline excludes structures with conflicting
toxicity scores rather than choosing a score by CSV row order or averaging
them, and removes graph-incompatible SMILES before splitting.

After a successful run, `latest_results/` in the Drive data folder is refreshed
with the newest figures, manifests, summaries, training history, and audits.
This gives one stable location for the current results; the timestamped run
folders remain as reproducibility evidence and are not deleted automatically.

### Candidate-model evaluation

The deployed `backend/checkpoints/pxddi_model.pt` remains the audited legacy
GAT checkpoint. New training defaults to an **edge-aware candidate** checkpoint
at `checkpoints/candidates/pxddi_edge_aware_candidate.pt`; it does not replace
the deployed model automatically. The candidate uses richer atom features and
bond order, stereo, chirality, conjugation, aromaticity, and ring features.

An additional untrained research candidate,
`motif_edge_aware_gat_v1`, can be selected with
`PXDDI_MODEL_ARCHITECTURE=motif_edge_aware_gat_v1`. It fuses the edge-aware
graph embedding with 17 fixed SMARTS motif-count features, including carbonyl,
amide, ester, aromatic-ring, amine, halogen, and other common chemical motifs.
The motif vocabulary is explicit in `src/data_prep/molecular_motifs.py` and is
recorded in each run manifest. These are experimental chemical-prior features,
not proven DDI mechanisms or validated explanations.

Another separate untrained candidate,
`cross_attention_edge_aware_gat_v1`, can be selected with
`PXDDI_MODEL_ARCHITECTURE=cross_attention_edge_aware_gat_v1`. After the
edge-aware encoder produces atom embeddings, each atom in Drug A attends only
to atoms in its paired Drug B, and vice versa. Attention is isolated within
each DDI row—no molecule can attend to a different pair in the batch. Its
shared two-way projections and final sum/absolute-difference pair head preserve
the A–B/B–A symmetry guarantee. Attention values are not evidence of a chemical
mechanism or validated explanations.

### Candidate explanation audit (optional)

For a **trained candidate only**, set
`PXDDI_RUN_CANDIDATE_EXPLANATIONS=1` when launching
`src/training/train_full_pipeline_v2.py`. The run then writes a small,
deterministically selected evaluation subset to
`artifacts/run_<timestamp>/explanations/candidate_occlusion_explanations.json`.
For each example it records raw-score changes after masking one atom, bond
feature, or (where applicable) SMARTS motif; top-atom fidelity/sufficiency
checks; A–B versus B–A score symmetry; and canonical-SMILES re-encoding score
stability. The cross-attention candidate additionally records its strongest
pair-isolated atom-to-atom attention associations and configured SMARTS-motif
A↔B association summaries. Matching SVG molecule figures are saved beside the
JSON: orange means masking locally reduced the raw model score and blue means
it increased it. These colours show model sensitivity only, not a toxicophore
or causal mechanism.

This is intentionally disabled by default, because masking each component is
slower than ordinary inference. It is an offline research audit, not a backend
endpoint, and its output must not be described as causal chemical evidence or
clinical explanation. The latest-results mirror also copies the explanation
and prediction files from a completed run for easy inspection.

To evaluate repeated-seed explanation stability rather than selecting a
visually preferred run, create matching candidate explanation artifacts for at
least two seeds, then run
`src/training/analyze_explanation_stability.py` with
`PXDDI_EXPLANATION_ARTIFACTS` set to their comma-separated JSON paths. The
report measures overlap of top atoms, motifs, and cross-drug motif associations
only for pairs explained by every supplied run, together with raw-score
variation. It reports agreement; it does not prove a causal explanation.

### Uncertainty and structural-domain audit

Every new run partitions the post-training validation predictions by class into
three disjoint roles: Platt-calibration fitting, decision-threshold selection,
and binary split-conformal fitting (default `PXDDI_CONFORMAL_ALPHA=0.1`). The
exact roles, scores, and hash are stored in the run artifact. Each saved test
prediction states whether its conformal set is a single label, both labels, or
an empty set; both-label and empty sets are marked `conformal_abstain=true`.
The prediction CSV also records Bernoulli score entropy and a
nearest-training-drug ECFP/Tanimoto structural-domain flag (default minimum
similarity `0.4`).

These are research guardrails, not clinical confidence. The conformal coverage
assumption may fail under the S1/S2 distribution shifts, and molecular
similarity does not measure novelty of a DDI pair or prove reliability. An OOD
flag tells the reviewer to inspect or abstain; an unflagged result is still not
validated or safe.

### Fixed-split ensemble and safe abstention

`src/training/run_fixed_split_ensemble.py` implements the Phase 6 research
ensemble. It trains **three to five** same-architecture members with different
model seeds but one fixed data/split seed, verifies their data and row-level
prediction provenance, averages their raw scores, and then fits a fresh
ensemble calibration, threshold, and conformal rule on disjoint validation
roles. It never overwrites or serves `backend/checkpoints/pxddi_model.pt`.

The ensemble prediction CSV stores every member score, its standard deviation,
conformal set, structural-domain flag, and a transparent abstention status. A
pair is marked `insufficient_evidence_for_reliable_unseen_drug_prediction`
when the conformal set is ambiguous/empty, members disagree beyond the chosen
research threshold, or either drug is outside the nearest-training-drug
structural domain. This is a review/abstention rule—not a clinical guarantee.

For a later Colab run, point `PXDDI_DATA_BASE` at the shared input-data folder
and `PXDDI_RESULTS_BASE` / `PXDDI_ENSEMBLES_BASE` at writable folders in your
own Drive, then run for example:

```bash
PXDDI_ENSEMBLE_ARCHITECTURE=cross_attention_edge_aware_gat_v1 \
PXDDI_ENSEMBLE_SEEDS=11,23,37 \
PXDDI_ENSEMBLE_SPLIT_SEED=42 \
PXDDI_ENSEMBLE_EPOCHS=200 \
python src/training/run_fixed_split_ensemble.py
```

Member candidates must be trained again through this command. Older runs used
one seed for both data splitting and model initialization, so they are not
valid fixed-split ensemble members.

Candidate training applies validation-AUROC early stopping after at least 40
epochs, with a default patience of 30 non-improving epochs. Set
`PXDDI_EARLY_STOPPING_PATIENCE=0` only when a deliberate fixed-length run is
needed; the final manifest always records whether early stopping occurred.

The backend continues to load the legacy checkpoint unless an operator
explicitly selects a reviewed file through `PXDDI_CHECKPOINT_PATH`. A selected
edge-aware candidate is given the compatible rich graph schema automatically;
this is a loading compatibility feature, not an approval to deploy the
candidate.

Run one candidate first, inspect its Transductive/S1/S2 metrics and calibration
artifacts, then decide whether it should replace the legacy model. For a
controlled baseline/ablation study, run
`src/training/run_experiment_suite.py`. Its screening preset compares the
ECFP/Morgan + linear-logistic baseline with four GNN ablations once. Its paper
preset repeats the directly comparable legacy and edge-aware multi-task GNNs
across five seeds. The suite verifies that each matched seed used the same
TWOSIDES input and exact split hashes, and saves paired bootstrap confidence
intervals. The ECFP baseline can be included in any deliberate run through
`PXDDI_EXPERIMENT_NAMES=ecfp_sgd_logistic,...`. It uses symmetric
`ECFP_a+ECFP_b` and `|ECFP_a-ECFP_b|` features, so reversing the drug order
does not change its score. A screening result is directional evidence only; it
is not statistical proof. Neither mode promotes a model automatically.

The ablation suite now also exposes `motif_edge_aware_ddi_only` and
`motif_edge_aware_multitask`. Do not treat their presence as a result: keep the
motif component only if repeated, matched-seed results improve S1/S2 over the
edge-aware GATv2 reference without degrading calibration.

It also exposes `cross_attention_edge_aware_ddi_only` and
`cross_attention_edge_aware_multitask`. Keep the cross-drug component only if
it improves repeated matched-seed S1/S2 results over the same edge-aware GATv2
reference, without an unacceptable calibration or efficiency regression.

If `PXDDI_DATA_BASE` is a read-only Google Drive shortcut, keep it pointed at
the shared data and set `PXDDI_EXPERIMENTS_BASE` to a writable folder in your
own Drive before running the experiment suite. The suite then reads source data
from the shortcut but writes study artifacts and candidate checkpoints to the
writable location. `PXDDI_EXPERIMENTS_BASE` controls the complete study folder;
its child runs already receive their individual writable artifact and checkpoint
paths automatically.

External validation is supported by `src/training/evaluate_external_dataset.py`
only after you supply an independently sourced, documented dataset and its
provenance metadata. The repository does not download or fabricate such data.

`notebooks/pxddi_training_run.ipynb` is intentionally empty in this local
repository because the live work is done in Colab. After the next audited run,
download and commit the executed notebook together with the configuration,
data/split manifests, logs, metrics, plots, and checkpoint hash.

## Known limitations

- S1 generalization to two unseen drugs is currently near random.
- Random unreported TWOSIDES pairs are not confirmed-safe negatives.
- The raw FAERS toxicity bridge covers 339 unique structures; 58 have
  conflicting source scores. The next audited training run excludes those
  conflicts, leaving 281 clean toxicity-label structures and a CSV audit file.
- The deployed legacy checkpoint is uncalibrated. Candidate checkpoints may
  include validation-fitted calibration, but that is not evidence of calibrated
  cold-start, external, or clinical performance.
- Patient context, external validation, clinical rules, and production
  authentication are not implemented.

See `MODEL_CARD.md` for full limitations and the current research scope.
