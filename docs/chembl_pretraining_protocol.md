# ChEMBL encoder-pretraining protocol

This protocol evaluates whether self-supervised molecular pretraining improves
PxDDI's unseen-drug (S1) performance. It is a research candidate, not a
backend deployment procedure.

## Evidence available

The audited ChEMBL 37 `chemreps` source contains 2,897,804 valid canonical
SMILES records. It has exact structural overlap with 414 of 645 valid PxDDI
structures. The supplied ChEMBL files do **not** contain molecule--target
activity records, so this phase uses only unlabelled molecular structures.

Before selection, the pretraining script recreates the PxDDI standard split
from the source hash, data cap, split seed, negative-sampling strategy, and
the `split_aware_standard_v1` sampling protocol. It
excludes every molecule outside `transductive_train`, including transductive
validation/test, S1-dev/test, and S2-dev/test molecules. It saves that split
audit inside the encoder checkpoint. Fine-tuning rejects a checkpoint unless
those values exactly match the DDI run.

## Colab configuration

Use the real paths independently: code can be a normal Drive checkout, while
shared data can be a Drive shortcut. In Colab, run the following after mounting
Drive and pulling the current repository revision:

```python
%cd /content/drive/MyDrive/pxddi-data/pxddi
!git pull origin main

%env PXDDI_DATA_BASE=/content/drive/.shortcut-targets-by-id/1EK5SEg3iwEAEUBzwrCOsj_Y0huxGZklA/pxddi-data
%env PXDDI_RESULTS_BASE=/content/drive/MyDrive/pxddi-results
%env PXDDI_DATA_CAP=200000
%env PXDDI_SPLIT_SEED=42
%env PXDDI_NEGATIVE_SAMPLING_STRATEGY=degree_matched
%env PXDDI_NEGATIVE_SAMPLING_PROTOCOL=split_aware_standard_v1
%env PXDDI_HIDDEN_CHANNELS=128
```

## Step 1 — Pretrain the encoder

Start with the complete configured candidate. It creates a new timestamped
folder and never changes the deployed DDI checkpoint:

```python
%env PXDDI_PRETRAIN_SEED=2026
%env PXDDI_PRETRAIN_MAX_MOLECULES=50000
%env PXDDI_PRETRAIN_EPOCHS=20
%env PXDDI_PRETRAIN_BATCH_SIZE=256

!python src/training/pretrain_chembl_encoder.py
```

The final output prints the encoder checkpoint path. Keep it in your writable
`pxddi-results/pretraining/...` folder. Inspect its
`pretraining_manifest.json` before proceeding. It must show:

- `pretraining_leakage_policy` equal to
  `exclude_all_non_train_twosides_structures_v1`;
- `split_seed` 42;
- `data_cap` 200000;
- `negative_sampling_strategy` `degree_matched`;
- `negative_sampling_protocol` `split_aware_standard_v1`;
- a checkpoint SHA-256.

## Step 2 — One-seed screening comparison

Replace the placeholder with the path printed by step 1. This compares a
scratch edge-aware GAT against the ChEMBL-warm-started edge-aware GAT on the
same split. It does not promote either candidate.

```python
%env PXDDI_CHEMBL_PRETRAINED_ENCODER_PATH=PASTE_THE_PRINTED_ENCODER_CHECKPOINT_PATH
%env PXDDI_EXPERIMENTS_BASE=/content/drive/MyDrive/pxddi-results/experiments
%env PXDDI_EXPERIMENT_PRESET=screening
%env PXDDI_EXPERIMENT_SEEDS=42
%env PXDDI_EXPERIMENT_SPLIT_SEED=42
%env PXDDI_EXPERIMENT_NEGATIVE_SAMPLING_PROTOCOL=split_aware_standard_v1
%env PXDDI_EXPERIMENT_EPOCHS=200
%env PXDDI_EXPERIMENT_NAMES=edge_aware_multitask,edge_aware_chembl_pretrained_multitask
%env PXDDI_EXPERIMENT_REFERENCE=edge_aware_multitask

!python src/training/run_experiment_suite.py
```

Review `experiment_results.csv`, the two run manifests, and `S1` metrics. A
one-seed result is only a screening signal; it cannot establish an improvement.

## Step 3 — Paper-quality repeated comparison

Run this only if the screening candidate is technically complete and does not
regress badly. The pretraining checkpoint may be reused because the split stays
fixed at 42; only DDI model initialization changes across the five seeds.

```python
%env PXDDI_EXPERIMENT_PRESET=paper
%env PXDDI_EXPERIMENT_SEEDS=11,23,37,53,71
%env PXDDI_EXPERIMENT_SPLIT_SEED=42
%env PXDDI_EXPERIMENT_EPOCHS=200
%env PXDDI_EXPERIMENT_NAMES=edge_aware_multitask,edge_aware_chembl_pretrained_multitask
%env PXDDI_EXPERIMENT_REFERENCE=edge_aware_multitask

!python src/training/run_experiment_suite.py
```

The study is credible only when its `experiment_results.csv` records equal
split-manifest signatures for both models at every matched seed. Base any
claim on S1 AUROC, PR-AUC, MCC, calibration, paired confidence intervals, and
the Holm-adjusted paired test—not on accuracy alone.

## Promotion rule

Do not copy the resulting candidate checkpoint into `backend/checkpoints` yet.
Promotion requires an explicit review showing reproducible S1/S2 benefit over
the scratch edge-aware GAT, no unacceptable calibration regression, and no
provenance or leakage-audit failure.
