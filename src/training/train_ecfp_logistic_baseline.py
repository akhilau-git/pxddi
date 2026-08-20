"""Reproducible ECFP + linear-logistic baseline for PxDDI research studies.

This is a deliberately simple, fast molecular-machine-learning baseline.  It
uses the exact same pre-split input audit, sampled unreported pairs, and
Transductive/S1/S2 split logic as the GNN pipeline.  It is not a deployment
model and never changes ``backend/checkpoints/pxddi_model.pt``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.splits import build_binary_pair_dataset, create_splits
from src.models.calibration import apply_calibrator, fit_platt_calibrator
from src.training.train_full_pipeline_v2 import (
    _prepare_positive_edges,
    calculate_metrics,
    get_file_hash,
    plot_benchmark_comparison,
    plot_evaluation,
    plot_training_curves,
    resolve_results_base,
    save_split_manifests,
    save_training_history,
    select_validation_threshold,
)


def _positive_int_from_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _non_negative_int_from_environment(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f'{name} must be zero or a positive integer.')
    return value


SEED = _positive_int_from_environment('PXDDI_SEED', 42)
DATA_CAP = _positive_int_from_environment('PXDDI_DATA_CAP', 200000)
ECFP_RADIUS = _positive_int_from_environment('PXDDI_ECFP_RADIUS', 2)
ECFP_NUM_BITS = _positive_int_from_environment('PXDDI_ECFP_NUM_BITS', 1024)
ECFP_EPOCHS = _positive_int_from_environment('PXDDI_ECFP_EPOCHS', 30)
ECFP_BATCH_SIZE = _positive_int_from_environment('PXDDI_ECFP_BATCH_SIZE', 1024)
EARLY_STOPPING_PATIENCE = _non_negative_int_from_environment(
    'PXDDI_ECFP_EARLY_STOPPING_PATIENCE', 6
)
EARLY_STOPPING_MIN_EPOCHS = _positive_int_from_environment(
    'PXDDI_ECFP_EARLY_STOPPING_MIN_EPOCHS', 8
)
if ECFP_NUM_BITS < 128:
    raise ValueError('PXDDI_ECFP_NUM_BITS must be at least 128.')
if EARLY_STOPPING_MIN_EPOCHS > ECFP_EPOCHS:
    raise ValueError('PXDDI_ECFP_EARLY_STOPPING_MIN_EPOCHS must not exceed PXDDI_ECFP_EPOCHS.')

DRIVE_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
TWOSIDES_EDGES = DRIVE_BASE / 'twosides' / 'drug_drug_edges.csv'
RESULTS_BASE = resolve_results_base()
ARTIFACTS_BASE = Path(os.environ.get('PXDDI_ARTIFACTS_BASE', RESULTS_BASE / 'artifacts'))
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_ARTIFACTS_DIR = ARTIFACTS_BASE / f'run_{RUN_ID}'
DEFAULT_CHECKPOINT_PATH = (
    RESULTS_BASE / 'checkpoints' / 'baselines' / 'pxddi_ecfp_sgd_logistic.npz'
)
CHECKPOINT_PATH = Path(os.environ.get('PXDDI_CHECKPOINT_PATH', DEFAULT_CHECKPOINT_PATH))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


class MorganFingerprintCache:
    """Cache molecular ECFP vectors so repeated drug pairs stay inexpensive."""

    def __init__(self, radius: int = ECFP_RADIUS, num_bits: int = ECFP_NUM_BITS) -> None:
        self.radius = radius
        self.num_bits = num_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=num_bits,
            includeChirality=True,
        )
        self._vectors: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, smiles: str) -> np.ndarray:
        normalized = str(smiles).strip()
        if normalized in self._vectors:
            self.hits += 1
            return self._vectors[normalized]
        molecule = Chem.MolFromSmiles(normalized)
        if molecule is None:
            raise ValueError(f'Cannot create an ECFP fingerprint from invalid SMILES: {smiles!r}.')
        bit_vector = self.generator.GetFingerprint(molecule)
        vector = np.zeros(self.num_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bit_vector, vector)
        self._vectors[normalized] = vector
        self.misses += 1
        return vector

    def summary(self) -> dict[str, Any]:
        requests = self.hits + self.misses
        return {
            'fingerprint_type': 'ECFP (Morgan)',
            'radius': self.radius,
            'num_bits_per_drug': self.num_bits,
            'pair_feature_dimension': self.num_bits * 2,
            'include_chirality': True,
            'unique_smiles_cached': len(self._vectors),
            'requests': requests,
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'cache_hit_rate': self.hits / requests if requests else None,
        }


def pair_feature_matrix(
    dataframe: pd.DataFrame,
    cache: MorganFingerprintCache,
) -> sparse.csr_matrix:
    """Create an order-independent ECFP pair representation for one batch.

    The per-drug ECFPs are combined as ``a+b`` and ``|a-b|``.  Swapping the two
    drugs therefore gives exactly the same vector, just as the GNN's pair head
    uses an embedding sum and absolute difference.
    """
    required = {'source', 'target'}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f'Pair data is missing columns: {sorted(missing)}.')
    if dataframe.empty:
        return sparse.csr_matrix((0, cache.num_bits * 2), dtype=np.float32)
    left = np.vstack([cache.get(smiles) for smiles in dataframe['source']])
    right = np.vstack([cache.get(smiles) for smiles in dataframe['target']])
    combined = np.concatenate((left + right, np.abs(left - right)), axis=1)
    return sparse.csr_matrix(combined, dtype=np.float32)


def iter_pair_batches(
    dataframe: pd.DataFrame,
    batch_size: int,
    indices: np.ndarray | None = None,
) -> Iterator[pd.DataFrame]:
    ordered_indices = np.arange(len(dataframe)) if indices is None else indices
    for start in range(0, len(ordered_indices), batch_size):
        yield dataframe.iloc[ordered_indices[start:start + batch_size]]


def probability_from_linear_weights(
    features: sparse.csr_matrix,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Compute stable positive-class probabilities from a saved linear state."""
    logits = np.asarray(features @ coefficients.reshape(-1, 1)).reshape(-1) + intercept
    clipped_logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped_logits))


def collect_predictions(
    dataframe: pd.DataFrame,
    cache: MorganFingerprintCache,
    coefficients: np.ndarray,
    intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for batch in iter_pair_batches(dataframe, ECFP_BATCH_SIZE):
        labels.append(batch['label'].to_numpy(dtype=int))
        predictions.append(
            probability_from_linear_weights(pair_feature_matrix(batch, cache), coefficients, intercept)
        )
    if not labels:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    return np.concatenate(labels), np.concatenate(predictions)


def safe_save_baseline_checkpoint(
    coefficients: np.ndarray,
    intercept: float,
    metadata: dict[str, Any],
    path: Path,
) -> str:
    """Atomically write numeric baseline parameters and return their SHA-256 hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open('wb') as destination:
            np.savez_compressed(
                destination,
                coefficients=np.asarray(coefficients, dtype=np.float32),
                intercept=np.asarray([intercept], dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        with np.load(temporary_path, allow_pickle=False) as loaded:
            if loaded['coefficients'].shape != np.asarray(coefficients).shape:
                raise ValueError('Baseline checkpoint validation failed: coefficient shape changed.')
            if loaded['intercept'].shape != (1,):
                raise ValueError('Baseline checkpoint validation failed: intercept shape changed.')
        checkpoint_hash = get_file_hash(temporary_path)
        os.replace(temporary_path, path)
        return checkpoint_hash
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_prediction_artifact(
    name: str,
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    threshold: float,
    prediction_dir: Path,
    calibration: dict[str, Any],
) -> Path:
    if len(dataframe) != len(labels):
        raise RuntimeError('Baseline prediction provenance does not match its evaluated split.')
    table = dataframe[['source', 'target', 'label', 'label_evidence']].copy()
    table['raw_prediction_score'] = raw_predictions
    table['calibrated_prediction_score'] = calibrated_predictions
    table['prediction_score'] = calibrated_predictions
    table['calibration_status'] = calibration.get('status', 'not_fitted')
    table['calibration_method'] = calibration.get('method')
    table['threshold'] = threshold
    table['predicted_label'] = (calibrated_predictions >= threshold).astype(int)
    path = prediction_dir / f'{name.lower().replace(" ", "_")}_predictions.csv'
    table.to_csv(path, index=False)
    return path


def baseline_manifest() -> dict[str, Any]:
    if not TWOSIDES_EDGES.is_file():
        raise FileNotFoundError(f'Missing TWOSIDES edges: {TWOSIDES_EDGES}.')
    return {
        'run_id': RUN_ID,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'random_seed': SEED,
        'input_data_base': str(DRIVE_BASE),
        'results_base': str(RESULTS_BASE),
        'input_sha256': {
            'twosides_edges': get_file_hash(TWOSIDES_EDGES),
            'training_source': get_file_hash(Path(__file__)),
        },
        'configuration': {
            'data_cap': DATA_CAP,
            'model_architecture': 'ecfp_sgd_logistic_v1',
            'feature_schema': 'ecfp_morgan_sum_and_absolute_difference',
            'ecfp_radius': ECFP_RADIUS,
            'ecfp_num_bits_per_drug': ECFP_NUM_BITS,
            'pair_feature_dimension': ECFP_NUM_BITS * 2,
            'ecfp_epochs': ECFP_EPOCHS,
            'batch_size': ECFP_BATCH_SIZE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_min_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'use_toxicity_pair_features': False,
            'toxicity_loss_weight': 0.0,
            'negative_label_meaning': 'unreported_twosides_sampled',
            'model_role': 'non-deployment classical baseline',
        },
    }


def main() -> None:
    # This shared audit deliberately gives the baseline and GNN identical clean
    # positives before the deterministic negative sampling and cold-start split.
    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=False)
    figure_dir = RUN_ARTIFACTS_DIR / 'figures'
    prediction_dir = RUN_ARTIFACTS_DIR / 'predictions'
    audit_dir = RUN_ARTIFACTS_DIR / 'audits'
    figure_dir.mkdir()
    prediction_dir.mkdir()

    manifest = baseline_manifest()
    write_json(RUN_ARTIFACTS_DIR / 'run_manifest_initial.json', manifest)
    positives, input_quality_summary = _prepare_positive_edges(audit_dir)
    full_dataset = build_binary_pair_dataset(
        positives, source_col='source', target_col='target', neg_ratio=1.0, seed=SEED
    )
    dataset_summary = {
        'effective_positive_pairs': int((full_dataset['label'] == 1.0).sum()),
        'sampled_unreported_negative_pairs': int((full_dataset['label'] == 0.0).sum()),
        'total_pair_rows_before_split': int(len(full_dataset)),
        'negative_label_meaning': 'unreported_twosides_sampled',
    }
    write_json(audit_dir / 'dataset_summary.json', dataset_summary)
    splits = create_splits(full_dataset, drug_a_col='source', drug_b_col='target', seed=SEED)
    split_manifest = save_split_manifests(splits, RUN_ARTIFACTS_DIR)

    cache = MorganFingerprintCache()
    classifier = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=1e-5, max_iter=1,
        learning_rate='optimal', random_state=SEED, average=True,
    )
    history = {'epoch': [], 'loss': [], 'auroc': [], 'f1': []}
    best_auroc = float('-inf')
    best_coefficients: np.ndarray | None = None
    best_intercept: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, ECFP_EPOCHS + 1):
        order = np.random.default_rng(SEED + epoch).permutation(len(splits['transductive_train']))
        first_batch = True
        for batch in iter_pair_batches(splits['transductive_train'], ECFP_BATCH_SIZE, order):
            features = pair_feature_matrix(batch, cache)
            labels = batch['label'].to_numpy(dtype=int)
            classifier.partial_fit(features, labels, classes=np.asarray([0, 1]) if first_batch else None)
            first_batch = False

        coefficients = classifier.coef_[0].copy()
        intercept = float(classifier.intercept_[0])
        train_labels, train_predictions = collect_predictions(
            splits['transductive_train'], cache, coefficients, intercept
        )
        validation_labels, validation_predictions = collect_predictions(
            splits['validation'], cache, coefficients, intercept
        )
        threshold = select_validation_threshold(validation_labels, validation_predictions)
        validation_metrics = calculate_metrics(validation_labels, validation_predictions, threshold)
        history['epoch'].append(epoch)
        history['loss'].append(float(log_loss(train_labels, train_predictions, labels=[0, 1])))
        history['auroc'].append(float(validation_metrics['auroc']))
        history['f1'].append(float(validation_metrics['f1']))
        print(
            f'Epoch {epoch}/{ECFP_EPOCHS}: loss={history["loss"][-1]:.4f}; '
            f'validation AUROC={validation_metrics["auroc"]:.4f}; F1={validation_metrics["f1"]:.4f}'
        )

        if validation_metrics['auroc'] > best_auroc:
            best_auroc = float(validation_metrics['auroc'])
            best_coefficients = coefficients
            best_intercept = intercept
            best_epoch = epoch
            epochs_without_improvement = 0
            safe_save_baseline_checkpoint(
                best_coefficients,
                best_intercept,
                {'epoch': epoch, 'validation_auroc': best_auroc, 'status': 'uncalibrated'},
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1
        if (
            EARLY_STOPPING_PATIENCE > 0
            and epoch >= EARLY_STOPPING_MIN_EPOCHS
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            print(
                f'Early stopping at epoch {epoch}; keeping the best baseline from epoch {best_epoch}.'
            )
            break

    if best_coefficients is None or best_intercept is None or best_epoch is None:
        raise RuntimeError('Baseline training did not produce a valid checkpoint.')

    validation_labels, validation_raw_predictions = collect_predictions(
        splits['validation'], cache, best_coefficients, best_intercept
    )
    calibration = fit_platt_calibrator(validation_labels, validation_raw_predictions)
    validation_calibrated_predictions = apply_calibrator(validation_raw_predictions, calibration)
    frozen_threshold = select_validation_threshold(
        validation_labels, validation_calibrated_predictions
    )
    checkpoint_hash = safe_save_baseline_checkpoint(
        best_coefficients,
        best_intercept,
        {
            'epoch': best_epoch,
            'validation_auroc': best_auroc,
            'threshold': frozen_threshold,
            'calibration': calibration,
            'model_architecture': 'ecfp_sgd_logistic_v1',
        },
        CHECKPOINT_PATH,
    )

    plot_training_curves(history, figure_dir)
    history_summary = save_training_history(history, RUN_ARTIFACTS_DIR)
    results: dict[str, dict[str, Any]] = {}
    for name, split_key in (
        ('Transductive', 'transductive_test'), ('S1', 's1_test'), ('S2', 's2_test')
    ):
        labels, raw_predictions = collect_predictions(
            splits[split_key], cache, best_coefficients, best_intercept
        )
        calibrated_predictions = apply_calibrator(raw_predictions, calibration)
        metrics = calculate_metrics(
            labels, calibrated_predictions, frozen_threshold, raw_predictions=raw_predictions
        )
        prediction_path = save_prediction_artifact(
            name, splits[split_key], labels, raw_predictions, calibrated_predictions,
            frozen_threshold, prediction_dir, calibration,
        )
        metrics['prediction_path'] = str(prediction_path)
        metrics['prediction_sha256'] = get_file_hash(prediction_path)
        results[name] = metrics
        plot_evaluation(
            name, labels, calibrated_predictions, metrics, figure_dir,
            raw_predictions=raw_predictions,
        )
    plot_benchmark_comparison(results, figure_dir)
    write_json(RUN_ARTIFACTS_DIR / 'results_summary.json', results)

    manifest.update({
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': {
            'path': str(CHECKPOINT_PATH),
            'sha256': checkpoint_hash,
            'epoch': best_epoch,
            'validation_auroc': best_auroc,
            'validation_selected_threshold': frozen_threshold,
        },
        'calibration': calibration,
        'model_summary': {
            'model_class': 'SGDClassifier',
            'model_architecture': 'ecfp_sgd_logistic_v1',
            'trainable_parameters': int(best_coefficients.size + 1),
            'pair_representation': 'ECFP_a_plus_b + absolute_ECFP_a_minus_b',
            'output_heads': ['interaction_risk'],
        },
        'training_history': history_summary,
        'early_stopping': {
            'enabled': EARLY_STOPPING_PATIENCE > 0,
            'patience': EARLY_STOPPING_PATIENCE,
            'minimum_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'stopped_early': stopped_early,
            'completed_epochs': len(history['epoch']),
            'best_epoch': best_epoch,
        },
        'input_quality': input_quality_summary,
        'dataset': dataset_summary,
        'fingerprint_cache': cache.summary(),
        'split_manifest': split_manifest,
        'results': results,
        'warning': (
            'This baseline learns reported-versus-unreported TWOSIDES pairs. '
            'Zero labels are not evidence that a drug pair is safe.'
        ),
    })
    write_json(RUN_ARTIFACTS_DIR / 'run_manifest.json', manifest)
    print(f'Baseline artifacts saved to: {RUN_ARTIFACTS_DIR}')


if __name__ == '__main__':
    main()
