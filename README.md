# PxDDI: Research Drug-Drug Interaction Prototype

PxDDI is a research prototype for structure-based drug-drug interaction (DDI)
prediction. It combines a symmetric molecular graph model with a toxicity head
and a patient-context architecture. It is **not for clinical use**.

Patient context is intentionally disabled at inference: the repository does
not contain linked patient, drug-pair, and outcome data needed to train that
module safely.

## Committed reference GNN artifact

The checkpoint at `backend/checkpoints/pxddi_model.pt` currently reports:

- Best validation epoch: 193
- Stored validation AUROC: 0.9485
- Validation-selected decision threshold: 0.5453
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

Each run creates `artifacts/run_<timestamp>/` in Drive containing an initial
and final manifest, resolved package/GPU environment, source revision, model
summary, numeric training-history CSV, clean toxicity-label and input-quality
audits, exact split CSVs and hashes, checkpoint hash, prediction CSVs, metrics
JSON, and PNG/PDF figures. The pipeline excludes structures with conflicting
toxicity scores rather than choosing a score by CSV row order or averaging
them, and removes graph-incompatible SMILES before splitting.

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
- The prediction score is uncalibrated and is not a clinical probability.
- Patient context, external validation, clinical rules, and production
  authentication are not implemented.

See `MODEL_CARD.md` for full limitations and the current research scope.
