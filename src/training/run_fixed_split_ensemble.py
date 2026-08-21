"""Train and evaluate a fixed-split 3–5 member PxDDI research ensemble.

Each member receives a unique initialization/training-order seed but the same
audited data sample and S1/S2 split.  The script averages raw member scores,
then fits a fresh ensemble calibration, decision threshold, and conformal rule
on separate validation roles.  It never writes to the deployed backend model.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.applicability_domain import MorganApplicabilityDomain
from src.models.calibration import apply_calibrator, fit_platt_calibrator
from src.models.ensemble import (
    ABSTENTION_LABEL,
    apply_safe_abstention,
    combine_member_prediction_tables,
    summarize_safe_abstention,
    validate_ensemble_member_manifests,
)
from src.models.uncertainty import (
    conformal_prediction_sets,
    fit_split_conformal_binary,
    predictive_entropy,
    summarize_conformal_test_labels,
)
from src.training.train_full_pipeline_v2 import (
    calculate_metrics,
    get_file_hash,
    partition_validation_for_posthoc,
    plot_benchmark_comparison,
    plot_evaluation,
    posthoc_validation_partition_summary,
    resolve_results_base,
    save_conformal_calibration_artifact,
    select_validation_threshold,
    write_json,
)


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _open_unit_interval(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0 < value < 1:
        raise ValueError(f'{name} must lie strictly between zero and one.')
    return value


def _closed_unit_interval(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must lie between zero and one.')
    return value


def parse_member_seeds(value: str) -> list[int]:
    """Parse three to five distinct positive ensemble member seeds."""
    seeds = [int(part.strip()) for part in value.split(',') if part.strip()]
    if not 3 <= len(seeds) <= 5:
        raise ValueError('PXDDI_ENSEMBLE_SEEDS must contain three to five seeds.')
    if any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('PXDDI_ENSEMBLE_SEEDS must contain distinct positive seeds.')
    return seeds


DATA_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
RESULTS_BASE = resolve_results_base()
MEMBER_SEEDS = parse_member_seeds(os.environ.get('PXDDI_ENSEMBLE_SEEDS', '11,23,37'))
SPLIT_SEED = _positive_int('PXDDI_ENSEMBLE_SPLIT_SEED', 42)
EPOCHS = _positive_int('PXDDI_ENSEMBLE_EPOCHS', 200)
ARCHITECTURE = os.environ.get('PXDDI_ENSEMBLE_ARCHITECTURE', 'edge_aware_gat_v2')
USE_TOXICITY_PAIR_FEATURES = os.environ.get('PXDDI_ENSEMBLE_USE_TOXICITY_PAIR_FEATURES', 'true')
TOXICITY_LOSS_WEIGHT = float(os.environ.get('PXDDI_ENSEMBLE_TOXICITY_LOSS_WEIGHT', '0.3'))
if TOXICITY_LOSS_WEIGHT < 0:
    raise ValueError('PXDDI_ENSEMBLE_TOXICITY_LOSS_WEIGHT must be non-negative.')
CONFORMAL_ALPHA = _open_unit_interval('PXDDI_ENSEMBLE_CONFORMAL_ALPHA', 0.1)
OOD_MINIMUM_SIMILARITY = _closed_unit_interval(
    'PXDDI_ENSEMBLE_OOD_MINIMUM_SIMILARITY', 0.4
)
DISAGREEMENT_THRESHOLD = _closed_unit_interval(
    'PXDDI_ENSEMBLE_DISAGREEMENT_THRESHOLD', 0.10
)
ENSEMBLES_BASE = Path(
    os.environ.get('PXDDI_ENSEMBLES_BASE', RESULTS_BASE / 'ensembles')
)
TRAINING_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train_full_pipeline_v2.py'


def find_completed_run(artifact_base: Path) -> Path:
    runs = sorted(
        (path for path in artifact_base.glob('run_*') if (path / 'run_manifest.json').is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f'No completed run was found under {artifact_base}.')
    return runs[0]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _prediction_path(manifest: dict[str, Any], split_name: str) -> Path:
    if split_name == 'Validation':
        detail = manifest.get('validation_predictions')
        path_value_key = 'path'
    else:
        detail = manifest.get('results', {}).get(split_name)
        path_value_key = 'prediction_path'
    if not isinstance(detail, dict) or not detail.get(path_value_key):
        raise ValueError(f'Run manifest lacks a {split_name} prediction artifact.')
    if split_name == 'Validation' and detail.get('split_role') != (
        'posthoc_validation_reserved_before_training'
    ):
        raise ValueError(
            'Ensemble members must provide validation predictions reserved before '
            'training; older runs reused early-stopping validation and are not valid '
            'ensemble members.'
        )
    path = Path(detail[path_value_key])
    if not path.is_file():
        raise FileNotFoundError(f'Prediction artifact is missing: {path}')
    expected_hash = detail.get('sha256') if split_name == 'Validation' else detail.get('prediction_sha256')
    if expected_hash and get_file_hash(path) != expected_hash:
        raise ValueError(f'Prediction artifact hash does not match the run manifest: {path}')
    return path


def _posthoc_indices(labels: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    partition = partition_validation_for_posthoc(labels, seed=SPLIT_SEED)
    if not partition['independent_roles']:
        raise ValueError(
            'This ensemble has too few examples per validation class to create '
            'independent calibration, threshold, and conformal roles.'
        )
    return partition, partition['indices']


def _write_ensemble_prediction(
    table: pd.DataFrame,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return get_file_hash(destination)


def train_members(study_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Run isolated members and return their immutable completed artifacts."""
    members = []
    for model_seed in MEMBER_SEEDS:
        member_dir = study_dir / 'members' / f'model_seed_{model_seed}'
        artifact_base = member_dir / 'artifacts'
        checkpoint_path = member_dir / 'checkpoints' / f'ensemble_member_{model_seed}.pt'
        environment = os.environ.copy()
        environment.update({
            'PXDDI_DATA_BASE': str(DATA_BASE),
            'PXDDI_RESULTS_BASE': str(RESULTS_BASE),
            'PXDDI_ARTIFACTS_BASE': str(artifact_base),
            'PXDDI_CHECKPOINT_PATH': str(checkpoint_path),
            'PXDDI_PUBLISH_LATEST_RESULTS': 'false',
            'PXDDI_RUN_CANDIDATE_EXPLANATIONS': 'false',
            'PXDDI_SEED': str(model_seed),
            'PXDDI_MODEL_SEED': str(model_seed),
            'PXDDI_SPLIT_SEED': str(SPLIT_SEED),
            'PXDDI_EPOCHS': str(EPOCHS),
            'PXDDI_MODEL_ARCHITECTURE': ARCHITECTURE,
            'PXDDI_USE_TOXICITY_PAIR_FEATURES': USE_TOXICITY_PAIR_FEATURES,
            'PXDDI_TOXICITY_LOSS_WEIGHT': str(TOXICITY_LOSS_WEIGHT),
        })
        print(f'Training ensemble member model_seed={model_seed}.')
        subprocess.run([sys.executable, str(TRAINING_SCRIPT)], check=True, env=environment)
        run_dir = find_completed_run(artifact_base)
        members.append((run_dir, _load_json(run_dir / 'run_manifest.json')))
    return members


def _append_ensemble_columns(
    table: pd.DataFrame,
    calibrated_scores: np.ndarray,
    threshold: float,
    conformal: dict[str, Any],
    domain: MorganApplicabilityDomain,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Add calibrated uncertainty and abstention columns to one ensemble split."""
    result = table.copy()
    labels = pd.to_numeric(result['label'], errors='raise').to_numpy(dtype=int)
    conformal_sets = conformal_prediction_sets(calibrated_scores, conformal)
    domain_scores = domain.score_pairs(result['source'].tolist(), result['target'].tolist())
    abstention = apply_safe_abstention(
        result['ensemble_member_standard_deviation'].to_numpy(dtype=float),
        conformal_sets['abstain'],
        domain_scores['structural_ood_flag'],
        DISAGREEMENT_THRESHOLD,
    )
    result['calibrated_prediction_score'] = calibrated_scores
    result['prediction_score'] = calibrated_scores
    result['decision_threshold_used'] = threshold
    result['predicted_label'] = (calibrated_scores >= threshold).astype(int)
    result['predictive_entropy_nats'] = predictive_entropy(calibrated_scores)
    result['conformal_no_interaction_p_value'] = conformal_sets['no_interaction_p_value']
    result['conformal_interaction_p_value'] = conformal_sets['interaction_p_value']
    result['conformal_prediction_set'] = conformal_sets['prediction_set']
    result['conformal_prediction_set_size'] = conformal_sets['set_size']
    result['conformal_abstain'] = conformal_sets['abstain']
    for column, values in domain_scores.items():
        result[column] = values
    for column, values in abstention.items():
        result[column] = values
    metrics = calculate_metrics(
        labels, calibrated_scores, threshold, raw_predictions=result['raw_prediction_score'].to_numpy()
    )
    metrics['uncertainty'] = summarize_conformal_test_labels(labels, conformal_sets)
    metrics['uncertainty'].update({
        'method': conformal['method'],
        'alpha': conformal['alpha'],
        'interpretation_warning': conformal['interpretation_warning'],
    })
    metrics['safe_abstention'] = {
        **summarize_safe_abstention(abstention),
        'label': ABSTENTION_LABEL,
        'member_standard_deviation_threshold': DISAGREEMENT_THRESHOLD,
        'interpretation_warning': (
            'Abstention is a research review rule combining member disagreement, '
            'conformal ambiguity, and structural-domain distance. It is not a clinical '
            'safety guarantee or proof that non-abstained pairs are reliable.'
        ),
    }
    return result, metrics, domain_scores


def main() -> None:
    study_id = datetime.now(timezone.utc).strftime('ensemble_%Y%m%dT%H%M%SZ')
    study_dir = ENSEMBLES_BASE / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    write_json(study_dir / 'ensemble_plan.json', {
        'study_id': study_id,
        'member_model_seeds': MEMBER_SEEDS,
        'split_seed': SPLIT_SEED,
        'architecture': ARCHITECTURE,
        'epochs_per_member': EPOCHS,
        'conformal_alpha': CONFORMAL_ALPHA,
        'member_standard_deviation_threshold': DISAGREEMENT_THRESHOLD,
        'structural_ood_minimum_similarity': OOD_MINIMUM_SIMILARITY,
        'promotion_policy': (
            'This is an offline research ensemble. It does not overwrite or deploy '
            'backend/checkpoints/pxddi_model.pt.'
        ),
    })
    member_runs = train_members(study_dir)
    manifests = [manifest for _, manifest in member_runs]
    member_summary = validate_ensemble_member_manifests(manifests)
    member_records = [
        {
            'run_directory': str(run_dir),
            'checkpoint_path': manifest['checkpoint']['path'],
            'checkpoint_sha256': manifest['checkpoint']['sha256'],
            'model_seed': manifest['configuration']['model_seed'],
            'split_seed': manifest['configuration']['split_seed'],
        }
        for run_dir, manifest in member_runs
    ]

    validation_tables = [
        pd.read_csv(_prediction_path(manifest, 'Validation')) for manifest in manifests
    ]
    ensemble_validation = combine_member_prediction_tables(validation_tables)
    labels = ensemble_validation['label'].to_numpy(dtype=int)
    posthoc_partition, posthoc_indices = _posthoc_indices(labels)
    calibration = fit_platt_calibrator(
        labels[posthoc_indices['calibration']],
        ensemble_validation['raw_prediction_score'].to_numpy()[posthoc_indices['calibration']],
        fitted_on='ensemble_validation_calibration_partition',
    )
    validation_calibrated = apply_calibrator(
        ensemble_validation['raw_prediction_score'].to_numpy(), calibration
    )
    threshold = float(select_validation_threshold(
        labels[posthoc_indices['threshold']],
        validation_calibrated[posthoc_indices['threshold']],
    ))
    conformal = fit_split_conformal_binary(
        labels[posthoc_indices['conformal']],
        validation_calibrated[posthoc_indices['conformal']],
        alpha=CONFORMAL_ALPHA,
        fitted_on='ensemble_validation_conformal_partition',
    )

    train_split_path = Path(member_summary['split_manifest']['transductive_train']['path'])
    train_split = pd.read_csv(train_split_path)
    domain = MorganApplicabilityDomain(minimum_similarity=OOD_MINIMUM_SIMILARITY)
    domain_summary = domain.fit(
        set(train_split['source'].tolist()).union(train_split['target'].tolist())
    )
    artifact_dir = study_dir / 'artifacts'
    prediction_dir = artifact_dir / 'predictions'
    figure_dir = artifact_dir / 'figures'
    uncertainty_dir = artifact_dir / 'uncertainty'
    prediction_dir.mkdir(parents=True)
    figure_dir.mkdir()
    posthoc_validation = ensemble_validation.copy()
    posthoc_validation['posthoc_validation_role'] = 'not_assigned'
    for role, indices in posthoc_indices.items():
        posthoc_validation.loc[indices, 'posthoc_validation_role'] = role
    posthoc_validation['calibrated_prediction_score'] = validation_calibrated
    posthoc_validation_path = uncertainty_dir / 'posthoc_validation_partition.csv'
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    posthoc_validation.to_csv(posthoc_validation_path, index=False)
    posthoc_summary = {
        **posthoc_validation_partition_summary(labels, posthoc_partition),
        'path': str(posthoc_validation_path),
        'sha256': get_file_hash(posthoc_validation_path),
    }
    conformal_artifact = save_conformal_calibration_artifact(conformal, uncertainty_dir)

    validation_output, validation_metrics, _ = _append_ensemble_columns(
        ensemble_validation, validation_calibrated, threshold, conformal, domain
    )
    validation_path = prediction_dir / 'validation_predictions.csv'
    validation_hash = _write_ensemble_prediction(validation_output, validation_path)
    results: dict[str, dict[str, Any]] = {}
    for split_name in ('Transductive', 'S1', 'S2'):
        tables = [pd.read_csv(_prediction_path(manifest, split_name)) for manifest in manifests]
        combined = combine_member_prediction_tables(tables)
        calibrated = apply_calibrator(combined['raw_prediction_score'].to_numpy(), calibration)
        output, metrics, _ = _append_ensemble_columns(
            combined, calibrated, threshold, conformal, domain
        )
        path = prediction_dir / f'{split_name.lower()}_predictions.csv'
        metrics['prediction_path'] = str(path)
        metrics['prediction_sha256'] = _write_ensemble_prediction(output, path)
        results[split_name] = metrics
        plot_evaluation(
            split_name,
            output['label'].to_numpy(dtype=int),
            calibrated,
            metrics,
            figure_dir,
            raw_predictions=output['raw_prediction_score'].to_numpy(),
        )
    plot_benchmark_comparison(results, figure_dir)
    write_json(study_dir / 'results_summary.json', results)
    manifest = {
        'study_id': study_id,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'member_summary': member_summary,
        'members': member_records,
        'calibration': calibration,
        'validation_predictions': {
            'path': str(validation_path),
            'sha256': validation_hash,
            'metrics': validation_metrics,
        },
        'uncertainty': {
            'conformal': {
                key: value for key, value in conformal.items()
                if key != 'validation_nonconformity_scores'
            },
            'conformal_validation_artifact': conformal_artifact,
            'posthoc_validation_partition': posthoc_summary,
            'applicability_domain': domain_summary,
            'safe_abstention_label': ABSTENTION_LABEL,
        },
        'results': results,
        'limitations': [
            'The ensemble averages internally evaluated research model scores; it is not a clinical model.',
            'Unreported TWOSIDES negatives are not proven safe negatives.',
            'Conformal coverage relies on an exchangeability assumption that may fail under cold-start shift.',
            'Structural-domain similarity is drug-level only and does not prove DDI-pair reliability.',
            'No ensemble member is promoted or exposed through the backend by this script.',
        ],
    }
    write_json(study_dir / 'ensemble_manifest.json', manifest)
    print(f'Ensemble artifacts saved to: {study_dir}')


if __name__ == '__main__':
    main()
