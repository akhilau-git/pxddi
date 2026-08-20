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
`src/training/run_experiment_suite.py`. Its screening preset compares four
configurations once; its paper preset repeats them over multiple seeds and
saves seed-level bootstrap confidence intervals. Neither mode promotes a model
automatically.

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
