"""Reproducible Colab training pipeline for the research-only PxDDI GNN.

This script intentionally keeps ChemBERTa disabled. It saves all run-specific
artifacts to Google Drive so a later paper result can be traced to its input
data, split manifests, checkpoint, predictions, and plots.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = PROJECT_ROOT / 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))

DRIVE_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
# Training code must come from ``PROJECT_ROOT`` above (normally /content/pxddi
# in Colab).  DRIVE_BASE intentionally contains data, prior checkpoints, and
# outputs only.  Prepending a stale Drive checkout here would make a run's Git
# revision disagree with the code it actually imported.

from data_prep.prepare_twosides import (
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_RICH,
    LEGACY_NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    graph_compatibility_reason as source_graph_compatibility_reason,
    smiles_to_graph,
)
from data_prep.molecular_motifs import (
    MOTIF_FEATURE_DIM,
    MOTIF_FEATURE_NAMES,
    MOTIF_SCHEMA_SMARTS_COUNTS_V1,
    motif_metadata,
)
from data_prep.pubchem_bridge import canonicalize, resolve_toxicity_bridge
from data_prep.splits import build_binary_pair_dataset, create_splits, deduplicate_unordered_pairs
from data_prep.scaffold_splits import create_scaffold_disjoint_splits
from models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_LEGACY,
    MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
    PxDDIModel,
)
from models.calibration import (
    apply_calibrator,
    expected_calibration_error,
    fit_platt_calibrator,
)
from models.candidate_explainability import (
    EXPLANATION_LIMITATIONS,
    EXPLANATION_METHOD,
    explain_pair_with_occlusion,
    render_occlusion_svg,
    select_representative_indices,
)
from models.applicability_domain import MorganApplicabilityDomain
from models.uncertainty import (
    conformal_prediction_sets,
    fit_split_conformal_binary,
    predictive_entropy,
    summarize_conformal_test_labels,
)
from evaluation.ddi_metrics import (
    bootstrap_confidence_intervals,
    calculate_binary_metrics,
    save_confident_error_analysis,
    selective_prediction_summary,
    structural_similarity_slices,
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


def _boolean_from_environment(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes'}:
        return True
    if normalized in {'0', 'false', 'no'}:
        return False
    raise ValueError(f'{name} must be one of true/false, yes/no, or 1/0.')


def _open_unit_interval_from_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0 < value < 1:
        raise ValueError(f'{name} must lie strictly between zero and one.')
    return value


def _closed_unit_interval_from_environment(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must lie between zero and one.')
    return value


def resolve_results_base(
    configured_path: str | Path | None = None,
    data_base: Path = DRIVE_BASE,
) -> Path:
    """Choose a writable output root independently of a shared data shortcut."""
    if configured_path is None:
        configured_path = os.environ.get('PXDDI_RESULTS_BASE')
        if not configured_path and 'COLAB_RELEASE_TAG' in os.environ:
            configured_path = '/content/drive/MyDrive/pxddi-results'
    return Path(configured_path) if configured_path else data_base


# ``PXDDI_SEED`` remains the backwards-compatible one-knob default.  Ensemble
# training can override the model and split seeds independently so its members
# genuinely differ while evaluating exactly the same rows.
SEED = _positive_int_from_environment('PXDDI_SEED', 42)
MODEL_SEED = _positive_int_from_environment('PXDDI_MODEL_SEED', SEED)
SPLIT_SEED = _positive_int_from_environment('PXDDI_SPLIT_SEED', SEED)
DATA_CAP = _positive_int_from_environment('PXDDI_DATA_CAP', 200000)
EPOCHS = _positive_int_from_environment('PXDDI_EPOCHS', 200)
HIDDEN_CHANNELS = _positive_int_from_environment('PXDDI_HIDDEN_CHANNELS', 128)
BATCH_SIZE = _positive_int_from_environment('PXDDI_BATCH_SIZE', 128)
EARLY_STOPPING_PATIENCE = _non_negative_int_from_environment(
    'PXDDI_EARLY_STOPPING_PATIENCE', 30
)
EARLY_STOPPING_MIN_EPOCHS = _positive_int_from_environment(
    'PXDDI_EARLY_STOPPING_MIN_EPOCHS', 40
)
if EARLY_STOPPING_MIN_EPOCHS > EPOCHS:
    raise ValueError('PXDDI_EARLY_STOPPING_MIN_EPOCHS must not exceed PXDDI_EPOCHS.')
MODEL_SELECTION_VALIDATION_FRACTION = _open_unit_interval_from_environment(
    'PXDDI_MODEL_SELECTION_VALIDATION_FRACTION', 0.5
)
USE_CHEMBERTA = False
MODEL_ARCHITECTURE = os.environ.get(
    'PXDDI_MODEL_ARCHITECTURE', MODEL_ARCHITECTURE_EDGE_AWARE
)
if MODEL_ARCHITECTURE not in {
    MODEL_ARCHITECTURE_LEGACY,
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
}:
    raise ValueError(f'Unsupported PXDDI_MODEL_ARCHITECTURE: {MODEL_ARCHITECTURE}.')
USES_EDGE_FEATURES = MODEL_ARCHITECTURE in {
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
}
USE_MOTIF_FEATURES = MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE
USE_CROSS_DRUG_ATTENTION = (
    MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE
)
MOTIF_HIDDEN_CHANNELS = _positive_int_from_environment(
    'PXDDI_MOTIF_HIDDEN_CHANNELS', 32
)
FEATURE_SCHEMA = (
    FEATURE_SCHEMA_RICH
    if USES_EDGE_FEATURES
    else FEATURE_SCHEMA_LEGACY
)
INPUT_FEATURE_DIM = (
    RICH_NUM_ATOM_FEATURES
    if FEATURE_SCHEMA == FEATURE_SCHEMA_RICH
    else LEGACY_NUM_ATOM_FEATURES
)
USE_TOXICITY_PAIR_FEATURES = _boolean_from_environment(
    'PXDDI_USE_TOXICITY_PAIR_FEATURES', True
)
TOXICITY_LOSS_WEIGHT = float(os.environ.get('PXDDI_TOXICITY_LOSS_WEIGHT', '0.3'))
if TOXICITY_LOSS_WEIGHT < 0:
    raise ValueError('PXDDI_TOXICITY_LOSS_WEIGHT must be non-negative.')
EVALUATION_PROTOCOL = os.environ.get('PXDDI_EVALUATION_PROTOCOL', 'standard').strip().lower()
if EVALUATION_PROTOCOL not in {'standard', 'scaffold_disjoint'}:
    raise ValueError("PXDDI_EVALUATION_PROTOCOL must be 'standard' or 'scaffold_disjoint'.")
BOOTSTRAP_RESAMPLES = _non_negative_int_from_environment('PXDDI_BOOTSTRAP_RESAMPLES', 1000)
ERROR_ANALYSIS_MAX_ROWS = _positive_int_from_environment('PXDDI_ERROR_ANALYSIS_MAX_ROWS', 100)

TWOSIDES_EDGES = DRIVE_BASE / 'twosides' / 'drug_drug_edges.csv'
TOXICITY_BRIDGE = DRIVE_BASE / 'checkpoints' / 'toxicity_smiles_bridge.csv'
RESULTS_BASE = resolve_results_base()
DEFAULT_CHECKPOINT_PATH = (
    RESULTS_BASE / 'checkpoints' / 'pxddi_model.pt'
    if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_LEGACY
    else RESULTS_BASE / 'checkpoints' / 'candidates' / (
        'pxddi_motif_edge_aware_candidate.pt'
        if USE_MOTIF_FEATURES
        else 'pxddi_cross_attention_edge_aware_candidate.pt'
        if USE_CROSS_DRUG_ATTENTION
        else 'pxddi_edge_aware_candidate.pt'
    )
)
CHECKPOINT_PATH = Path(os.environ.get('PXDDI_CHECKPOINT_PATH', DEFAULT_CHECKPOINT_PATH))
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ARTIFACTS_BASE = Path(os.environ.get('PXDDI_ARTIFACTS_BASE', RESULTS_BASE / 'artifacts'))
RUN_ARTIFACTS_DIR = ARTIFACTS_BASE / f'run_{RUN_ID}'
LATEST_RESULTS_DIR = Path(
    os.environ.get('PXDDI_LATEST_RESULTS_DIR', RESULTS_BASE / 'latest_results')
)
PUBLISH_LATEST_RESULTS = _boolean_from_environment('PXDDI_PUBLISH_LATEST_RESULTS', True)
RUN_CANDIDATE_EXPLANATIONS = _boolean_from_environment(
    'PXDDI_RUN_CANDIDATE_EXPLANATIONS', False
)
EXPLANATION_SAMPLES_PER_SPLIT = _positive_int_from_environment(
    'PXDDI_EXPLANATION_SAMPLES_PER_SPLIT', 4
)
EXPLANATION_TOP_K = _positive_int_from_environment('PXDDI_EXPLANATION_TOP_K', 5)
CONFORMAL_ALPHA = _open_unit_interval_from_environment('PXDDI_CONFORMAL_ALPHA', 0.1)
APPLICABILITY_DOMAIN_MINIMUM_SIMILARITY = _closed_unit_interval_from_environment(
    'PXDDI_APPLICABILITY_DOMAIN_MINIMUM_SIMILARITY', 0.4
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_reproducibility(seed: int) -> None:
    """Set every supported random seed without making GPU execution fail hard."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_file_hash(path: str | Path) -> str:
    """Return a complete SHA-256 digest without loading a whole file into RAM."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot JSON-encode {type(value).__name__}.')


def _installed_distribution_version(distribution_name: str) -> str | None:
    """Return an installed package version without making a run fail on metadata."""
    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, Any]:
    """Record the resolved Colab software and GPU environment for reproducibility."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    package_names = {
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'rdkit': 'rdkit',
        'scikit_learn': 'scikit-learn',
        'torch': 'torch',
        'torch_geometric': 'torch-geometric',
    }
    return {
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu_name': gpu_name,
        'package_versions': {
            name: _installed_distribution_version(distribution)
            for name, distribution in package_names.items()
        },
    }


def repository_git_commit() -> str | None:
    """Return the source revision when training from a Git checkout."""
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an auditable JSON artifact with a stable UTF-8 encoding."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open('w', encoding='utf-8') as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, default=_json_default)


def safe_checkpoint_save(state_dict_bundle: dict[str, Any], path: str | Path) -> str:
    """Atomically save, safe-reload, and hash a checkpoint before replacing it."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{final_path.name}.', suffix='.tmp', dir=final_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(state_dict_bundle, temporary_path)
        loaded = torch.load(temporary_path, map_location='cpu', weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError('Checkpoint validation failed: expected a dictionary.')
        if set(loaded) != set(state_dict_bundle):
            raise ValueError('Checkpoint validation failed: keys changed after saving.')
        checkpoint_hash = get_file_hash(temporary_path)
        os.replace(temporary_path, final_path)
        return checkpoint_hash
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# Backward-compatible name used by earlier Colab cells and documentation.
save_checkpoint_safe = safe_checkpoint_save


def build_run_manifest() -> dict[str, Any]:
    """Capture immutable inputs before any model fitting begins."""
    required_paths = {
        'twosides_edges': TWOSIDES_EDGES,
        'toxicity_bridge': TOXICITY_BRIDGE,
        'training_source': Path(__file__),
    }
    if USE_MOTIF_FEATURES:
        required_paths['motif_definitions_source'] = (
            PROJECT_ROOT / 'src' / 'data_prep' / 'molecular_motifs.py'
        )
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing required Colab inputs: {missing}.')
    return {
        'run_id': RUN_ID,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': str(DEVICE),
        'input_data_base': str(DRIVE_BASE),
        'results_base': str(RESULTS_BASE),
        'repository_git_commit': repository_git_commit(),
        'runtime_environment': runtime_environment(),
        'random_seed': MODEL_SEED,
        'model_seed': MODEL_SEED,
        'split_seed': SPLIT_SEED,
        'configuration': {
            'data_cap': DATA_CAP,
            'model_seed': MODEL_SEED,
            'split_seed': SPLIT_SEED,
            'epochs': EPOCHS,
            'hidden_channels': HIDDEN_CHANNELS,
            'batch_size': BATCH_SIZE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_min_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'model_selection_validation_fraction': MODEL_SELECTION_VALIDATION_FRACTION,
            'validation_role_protocol': (
                'stratified_disjoint_model_selection_then_posthoc_three_way_v1'
            ),
            'use_chemberta': USE_CHEMBERTA,
            'model_architecture': MODEL_ARCHITECTURE,
            'feature_schema': FEATURE_SCHEMA,
            'use_motif_features': USE_MOTIF_FEATURES,
            'use_cross_drug_attention': USE_CROSS_DRUG_ATTENTION,
            'cross_drug_attention_type': (
                'pair_isolated_atom_attention_v1'
                if USE_CROSS_DRUG_ATTENTION else None
            ),
            'motif_feature_schema': (
                MOTIF_SCHEMA_SMARTS_COUNTS_V1 if USE_MOTIF_FEATURES else None
            ),
            'motif_feature_dim': MOTIF_FEATURE_DIM if USE_MOTIF_FEATURES else None,
            'motif_hidden_channels': MOTIF_HIDDEN_CHANNELS if USE_MOTIF_FEATURES else None,
            'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
            'toxicity_loss_weight': TOXICITY_LOSS_WEIGHT,
            'edge_feature_dim': (
                NUM_BOND_FEATURES
                if USES_EDGE_FEATURES else None
            ),
            'negative_label_meaning': 'unreported_twosides_sampled',
            'toxicity_conflict_policy': 'exclude_conflicting_structures',
            'evaluation_protocol': EVALUATION_PROTOCOL,
            'test_set_bootstrap': {
                'method': 'stratified_nonparametric_bootstrap',
                'resamples': BOOTSTRAP_RESAMPLES,
                'seed': SPLIT_SEED,
                'metrics': [
                    'auroc', 'average_precision', 'f1', 'mcc',
                    'balanced_accuracy', 'brier_score_calibrated',
                ],
            },
            'error_analysis': {
                'method': 'confidence_ranked_threshold_errors',
                'maximum_rows_per_error_type': ERROR_ANALYSIS_MAX_ROWS,
            },
            'candidate_explanations': {
                'enabled': RUN_CANDIDATE_EXPLANATIONS,
                'method': EXPLANATION_METHOD,
                'samples_per_split': EXPLANATION_SAMPLES_PER_SPLIT,
                'top_k': EXPLANATION_TOP_K,
                'uses_raw_model_probabilities': True,
            },
            'uncertainty': {
                'conformal_method': 'split_conformal_binary_v1',
                'conformal_alpha': CONFORMAL_ALPHA,
                'applicability_domain_method': 'nearest_train_ecfp_tanimoto_v1',
                'applicability_domain_minimum_similarity': (
                    APPLICABILITY_DOMAIN_MINIMUM_SIMILARITY
                ),
            },
        },
        'input_sha256': {name: get_file_hash(path) for name, path in required_paths.items()},
    }


def save_split_manifests(
    splits: dict[str, pd.DataFrame], artifact_dir: Path
) -> dict[str, dict[str, Any]]:
    """Save exact split tables and their hashes before training."""
    split_dir = artifact_dir / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for name, frame in splits.items():
        split_path = split_dir / f'{name}.csv'
        frame.to_csv(split_path, index=False)
        label_counts = frame['label'].value_counts().to_dict() if 'label' in frame else {}
        manifest[name] = {
            'path': str(split_path),
            'sha256': get_file_hash(split_path),
            'rows': len(frame),
            'label_counts': {str(label): int(count) for label, count in label_counts.items()},
        }
    write_json(split_dir / 'split_manifest.json', manifest)
    return manifest


def save_training_history(history: dict[str, list[float]], artifact_dir: Path) -> dict[str, Any]:
    """Save the numeric source data behind the convergence figure."""
    lengths = {len(values) for values in history.values()}
    if len(lengths) != 1:
        raise ValueError('Training history fields must have equal lengths.')
    history_path = artifact_dir / 'training_history.csv'
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        'path': str(history_path),
        'sha256': get_file_hash(history_path),
        'epoch_count': next(iter(lengths), 0),
    }


def publish_latest_results(
    run_artifacts_dir: Path,
    latest_results_dir: Path = LATEST_RESULTS_DIR,
) -> Path:
    """Refresh one easy-to-find latest-results folder after a completed run.

    Timestamped run folders remain immutable evidence. This lightweight mirror
    deliberately overwrites only the current figures and summary files, so a
    Colab user can always open one stable folder for the latest output.
    """
    run_dir = Path(run_artifacts_dir)
    latest_dir = Path(latest_results_dir)
    required_files = (
        Path('run_manifest_initial.json'),
        Path('run_manifest.json'),
        Path('results_summary.json'),
        Path('training_history.csv'),
        Path('audits/toxicity_bridge_conflicts.csv'),
        Path('audits/toxicity_bridge_summary.json'),
        Path('audits/invalid_smiles_exclusions.csv'),
        Path('audits/input_quality_summary.json'),
        Path('audits/dataset_summary.json'),
        Path('audits/counterion_curation_candidates.csv'),
    )
    missing = [str(relative) for relative in required_files if not (run_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Cannot publish incomplete run artifacts; missing: {missing}.'
        )

    latest_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for relative in required_files:
        source = run_dir / relative
        destination = latest_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files.append(str(relative))

    for artifact_subdirectory in (
        'figures', 'predictions', 'explanations', 'uncertainty', 'error_analysis', 'splits'
    ):
        artifact_dir = run_dir / artifact_subdirectory
        if not artifact_dir.exists():
            continue
        for source in sorted(artifact_dir.rglob('*')):
            if source.is_file():
                relative = source.relative_to(run_dir)
                destination = latest_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied_files.append(str(relative))

    write_json(latest_dir / 'latest_run.json', {
        'source_run_directory': str(run_dir),
        'source_run_id': run_dir.name,
        'refreshed_at_utc': datetime.now(timezone.utc).isoformat(),
        'mirrored_files': copied_files,
    })
    return latest_dir


def graph_compatibility_reason(smiles: Any) -> str | None:
    """Use the shared molecular parser to identify unusable training inputs."""
    return source_graph_compatibility_reason(smiles)


def filter_graph_compatible_pairs(
    dataframe: pd.DataFrame,
    source_col: str = 'source',
    target_col: str = 'target',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude graph-incompatible pairs before splitting and retain an audit table."""
    missing = {source_col, target_col}.difference(dataframe.columns)
    if missing:
        raise ValueError(f'Pair data is missing required columns: {sorted(missing)}.')

    keep_rows: list[bool] = []
    exclusions: list[dict[str, Any]] = []
    audit_columns = ['dataset_row_index', *dataframe.columns, 'source_graph_status', 'target_graph_status']
    for row_index, row in dataframe.iterrows():
        source_reason = graph_compatibility_reason(row[source_col])
        target_reason = graph_compatibility_reason(row[target_col])
        source_status = source_reason or 'graph_compatible'
        target_status = target_reason or 'graph_compatible'
        keep_row = source_reason is None and target_reason is None
        keep_rows.append(keep_row)
        if not keep_row:
            exclusions.append({
                'dataset_row_index': str(row_index),
                **row.to_dict(),
                'source_graph_status': source_status,
                'target_graph_status': target_status,
            })

    clean = dataframe.loc[keep_rows].copy().reset_index(drop=True)
    return clean, pd.DataFrame(exclusions, columns=audit_columns)


def save_counterion_curation_candidates(exclusions: pd.DataFrame, audit_dir: Path) -> dict[str, Any]:
    """Create a review queue; never guess a parent drug for an isolated ion."""
    counterion_columns = ['source_graph_status', 'target_graph_status']
    occurrences: dict[str, dict[str, Any]] = {}
    for side in ('source', 'target'):
        status_column = f'{side}_graph_status'
        if status_column not in exclusions:
            continue
        subset = exclusions[
            exclusions[status_column] == 'counterion_or_inorganic_only_structure'
        ]
        for structure, count in subset[side].value_counts().items():
            candidate = occurrences.setdefault(str(structure), {
                'raw_structure': structure,
                'occurrence_count': 0,
                'observed_as': set(),
                'audit_reason': 'counterion_or_inorganic_only_structure',
                'recommended_action': 'exclude_until_authoritative_parent_drug_mapping_is_reviewed',
                'curation_decision': 'pending_manual_review',
                'authoritative_source': '',
                'approved_parent_smiles': '',
                'review_notes': '',
            })
            candidate['occurrence_count'] += count
            candidate['observed_as'].add(side)
    candidates = []
    for candidate in occurrences.values():
        candidate['observed_as'] = '|'.join(sorted(candidate['observed_as']))
        candidates.append(candidate)
    candidate_columns = [
        'raw_structure',
        'occurrence_count',
        'observed_as',
        'audit_reason',
        'recommended_action',
        'curation_decision',
        'authoritative_source',
        'approved_parent_smiles',
        'review_notes',
    ]
    candidates_path = audit_dir / 'counterion_curation_candidates.csv'
    pd.DataFrame(candidates, columns=candidate_columns).to_csv(candidates_path, index=False)
    return {
        'counterion_only_pair_exclusions': (
            (
                exclusions[counterion_columns]
                == 'counterion_or_inorganic_only_structure'
            ).any(axis=1).sum()
        ) if not exclusions.empty else 0,
        'unique_counterion_or_inorganic_structures': len(candidates),
        'curation_candidates_path': str(candidates_path),
        'curation_candidates_sha256': get_file_hash(candidates_path),
        'automatic_parent_mapping_applied': False,
    }


def load_toxicity_lookup(audit_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Save toxicity conflicts to the current run and return clean labels only."""
    bridge = pd.read_csv(TOXICITY_BRIDGE)
    resolved, summary, conflicts = resolve_toxicity_bridge(bridge)
    audit_dir.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(audit_dir / 'toxicity_bridge_conflicts.csv', index=False)
    summary = {
        **summary,
        'conflict_policy': 'exclude_conflicting_structures',
        'source_bridge_sha256': get_file_hash(TOXICITY_BRIDGE),
        'conflict_report_path': str(audit_dir / 'toxicity_bridge_conflicts.csv'),
    }
    write_json(audit_dir / 'toxicity_bridge_summary.json', summary)
    print(
        'Toxicity bridge: '
        f"{summary['source_rows']} source rows; "
        f"{summary['resolved_unique_canonical_structures']} clean structures; "
        f"{summary['excluded_conflicting_structures']} conflicts excluded."
    )
    return dict(zip(resolved['canonical_smiles'], resolved['toxicity_score'])), summary


class GraphCache:
    """Reuse immutable SMILES graphs across train/validation/test loaders."""
    def __init__(self, feature_schema: str, include_motif_features: bool = False) -> None:
        self.feature_schema = feature_schema
        self.include_motif_features = include_motif_features
        self._graphs: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, smiles: str):
        key = smiles.strip() if isinstance(smiles, str) else str(smiles)
        if key in self._graphs:
            self.hits += 1
            graph = self._graphs[key]
        else:
            self.misses += 1
            graph = smiles_to_graph(
                key,
                feature_schema=self.feature_schema,
                include_motif_features=self.include_motif_features,
            )
            self._graphs[key] = graph
        return graph.clone() if graph is not None else None

    def summary(self) -> dict[str, Any]:
        requests = self.hits + self.misses
        return {
            'feature_schema': self.feature_schema,
            'include_motif_features': self.include_motif_features,
            'unique_smiles_cached': len(self._graphs),
            'graph_requests': requests,
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'cache_hit_rate': self.hits / requests if requests else None,
        }


class PxDDIDataset(Dataset):
    """Graph pairs plus provenance retained for later prediction artifacts."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        toxicity_lookup: dict[str, float],
        source_col: str = 'source',
        target_col: str = 'target',
        label_col: str = 'label',
        graph_cache: GraphCache | None = None,
    ) -> None:
        super().__init__()
        self.records = []
        self.metadata: list[dict[str, Any]] = []
        self.skipped_count = 0
        self.graph_cache = graph_cache or GraphCache(
            FEATURE_SCHEMA, include_motif_features=USE_MOTIF_FEATURES
        )
        for row in dataframe.itertuples(index=False):
            source = getattr(row, source_col)
            target = getattr(row, target_col)
            label = float(getattr(row, label_col))
            graph_a = self.graph_cache.get(source)
            graph_b = self.graph_cache.get(target)
            if graph_a is None or graph_b is None:
                self.skipped_count += 1
                continue
            c_src = canonicalize(source)
            toxicity_a = toxicity_lookup.get(c_src) if c_src is not None else None
            c_tgt = canonicalize(target)
            toxicity_b = toxicity_lookup.get(c_tgt) if c_tgt is not None else None
            toxicity_a_known = float(toxicity_a is not None)
            toxicity_b_known = float(toxicity_b is not None)
            self.records.append(
                (
                    graph_a,
                    graph_b,
                    0.0 if toxicity_a is None else toxicity_a,
                    0.0 if toxicity_b is None else toxicity_b,
                    toxicity_a_known,
                    toxicity_b_known,
                    label,
                )
            )
            self.metadata.append(
                {
                    'source': source,
                    'target': target,
                    'label': label,
                    'label_evidence': getattr(row, 'label_evidence', 'unknown'),
                }
            )
        print(f'Built dataset: {len(self.records)} valid pairs; {self.skipped_count} skipped.')

    def len(self) -> int:
        return len(self.records)

    def get(self, idx: int):
        return self.records[idx]


def collate_fn(batch):
    graph_a = [item[0] for item in batch]
    graph_b = [item[1] for item in batch]
    toxicity_a = torch.tensor([item[2] for item in batch], dtype=torch.float)
    toxicity_b = torch.tensor([item[3] for item in batch], dtype=torch.float)
    toxicity_a_known = torch.tensor([item[4] for item in batch], dtype=torch.float)
    toxicity_b_known = torch.tensor([item[5] for item in batch], dtype=torch.float)
    labels = torch.tensor([item[6] for item in batch], dtype=torch.float)
    return (
        Batch.from_data_list(graph_a),
        Batch.from_data_list(graph_b),
        toxicity_a,
        toxicity_b,
        toxicity_a_known,
        toxicity_b_known,
        labels,
    )


def build_loader(
    dataframe: pd.DataFrame,
    toxicity_lookup: dict[str, float],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    graph_cache: GraphCache | None = None,
    generator_seed: int = MODEL_SEED,
) -> DataLoader:
    dataset = PxDDIDataset(dataframe, toxicity_lookup, graph_cache=graph_cache)
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator if shuffle else None,
    )


def multi_task_loss(
    risk_prediction,
    toxicity_a_prediction,
    toxicity_b_prediction,
    risk_label,
    toxicity_a_label,
    toxicity_b_label,
    toxicity_a_known,
    toxicity_b_known,
    toxicity_loss_weight: float = TOXICITY_LOSS_WEIGHT,
):
    bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
    bce = torch.nn.BCELoss(reduction='none')
    ddi_loss = bce_with_logits(risk_prediction, risk_label).mean()
    toxicity_a_loss = (
        bce(toxicity_a_prediction, toxicity_a_label) * toxicity_a_known
    ).sum() / (toxicity_a_known.sum() + 1e-8)
    toxicity_b_loss = (
        bce(toxicity_b_prediction, toxicity_b_label) * toxicity_b_known
    ).sum() / (toxicity_b_known.sum() + 1e-8)
    return ddi_loss + toxicity_loss_weight * (toxicity_a_loss + toxicity_b_loss)


def train_one_epoch(model, loader, optimizer, scaler) -> float:
    if len(loader) == 0:
        raise ValueError('Training loader is empty after SMILES validation.')
    model.train()
    total_loss = 0.0
    for graph_a, graph_b, toxicity_a, toxicity_b, toxicity_a_known, toxicity_b_known, labels in loader:
        graph_a, graph_b = graph_a.to(DEVICE), graph_b.to(DEVICE)
        toxicity_a = toxicity_a.to(DEVICE)
        toxicity_b = toxicity_b.to(DEVICE)
        toxicity_a_known = toxicity_a_known.to(DEVICE)
        toxicity_b_known = toxicity_b_known.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                risk, prediction_a, prediction_b = model(graph_a, graph_b)
                loss = multi_task_loss(
                    risk, prediction_a, prediction_b, labels,
                    toxicity_a, toxicity_b, toxicity_a_known, toxicity_b_known,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            risk, prediction_a, prediction_b = model(graph_a, graph_b)
            loss = multi_task_loss(
                risk, prediction_a, prediction_b, labels,
                toxicity_a, toxicity_b, toxicity_a_known, toxicity_b_known,
            )
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(loader)


def collect_predictions(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for graph_a, graph_b, _, _, _, _, batch_labels in loader:
            graph_a, graph_b = graph_a.to(DEVICE), graph_b.to(DEVICE)
            risk, _, _ = model(graph_a, graph_b)
            predictions.extend(torch.sigmoid(risk).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels, dtype=int), np.asarray(predictions, dtype=float)


def measure_inference_efficiency(model, loader) -> dict[str, Any]:
    """Measure full-loader inference throughput for transparent model comparison.

    The figure includes graph batching and host-to-device transfer because that
    is the practical research inference path.  It is hardware-specific and is
    therefore stored beside GPU/runtime details rather than treated as a
    universal production latency claim.
    """
    if len(loader.dataset) == 0:
        return {
            'status': 'skipped_empty_loader',
            'sample_count': 0,
            'wall_clock_seconds': None,
            'pairs_per_second': None,
            'mean_batch_latency_milliseconds': None,
            'peak_cuda_memory_bytes': None,
        }
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
    batch_count = 0
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for graph_a, graph_b, *_ in loader:
            model(graph_a.to(DEVICE), graph_b.to(DEVICE))
            batch_count += 1
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - start
    sample_count = len(loader.dataset)
    return {
        'status': 'measured',
        'measurement_scope': 'full_loader_inference_including_batching_and_device_transfer',
        'sample_count': sample_count,
        'batch_count': batch_count,
        'wall_clock_seconds': elapsed,
        'pairs_per_second': sample_count / elapsed if elapsed else None,
        'mean_batch_latency_milliseconds': 1000 * elapsed / batch_count
        if batch_count else None,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(DEVICE)
        if DEVICE.type == 'cuda' else None,
        'interpretation_warning': (
            'Inference efficiency depends on the recorded hardware, batch size, '
            'PyTorch/PyG versions, and whether data are already in memory.'
        ),
    }


def select_validation_threshold(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError('Validation split must contain both classes to select a threshold.')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predictions)
    return float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])


def partition_validation_for_model_selection(
    dataframe: pd.DataFrame,
    seed: int = SPLIT_SEED,
    selection_fraction: float = MODEL_SELECTION_VALIDATION_FRACTION,
    label_column: str = 'label',
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Make early stopping independent from post-hoc validation decisions.

    Each binary class is shuffled deterministically, then partitioned into a
    model-selection split and a reserved post-hoc split.  The latter retains at
    least three examples per class so calibration, threshold selection, and
    conformal fitting can each use a distinct role.  A small split is an
    explicit error rather than a silent reuse of model-selection examples.
    """
    if label_column not in dataframe:
        raise ValueError(f'Validation frame is missing {label_column!r}.')
    if not 0 < selection_fraction < 1:
        raise ValueError('selection_fraction must lie strictly between zero and one.')
    targets = dataframe[label_column].to_numpy(dtype=int)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError('Validation labels must be a non-empty one-dimensional array.')
    if not np.isin(targets, (0, 1)).all() or len(np.unique(targets)) < 2:
        raise ValueError('Validation role partitioning requires both binary classes.')

    generator = np.random.default_rng(seed)
    selection_positions: list[int] = []
    posthoc_positions: list[int] = []
    source_counts: dict[str, int] = {}
    for label in (0, 1):
        positions = np.flatnonzero(targets == label)
        source_counts[str(label)] = len(positions)
        if len(positions) < 4:
            raise ValueError(
                'Validation role partitioning requires at least four examples '
                f'of each class; class {label} has {len(positions)}.'
            )
        shuffled = generator.permutation(positions)
        requested_selection = round(len(shuffled) * selection_fraction)
        # Preserve one row per class for each of the three post-hoc roles.
        selection_count = min(max(1, requested_selection), len(shuffled) - 3)
        selection_positions.extend(shuffled[:selection_count].tolist())
        posthoc_positions.extend(shuffled[selection_count:].tolist())

    selection_positions = sorted(selection_positions)
    posthoc_positions = sorted(posthoc_positions)
    model_selection = dataframe.iloc[selection_positions].copy().reset_index(drop=True)
    posthoc = dataframe.iloc[posthoc_positions].copy().reset_index(drop=True)
    summary = {
        'status': 'stratified_disjoint_model_selection_and_posthoc',
        'independent_from_early_stopping': True,
        'seed': seed,
        'requested_model_selection_fraction': selection_fraction,
        'source_rows': len(dataframe),
        'source_label_counts': source_counts,
        'model_selection_rows': len(model_selection),
        'model_selection_label_counts': {
            str(label): (model_selection[label_column] == label).sum()
            for label in (0, 1)
        },
        'posthoc_rows': len(posthoc),
        'posthoc_label_counts': {
            str(label): (posthoc[label_column] == label).sum()
            for label in (0, 1)
        },
        'posthoc_roles': ('calibration', 'threshold', 'conformal'),
    }
    return model_selection, posthoc, summary


def partition_validation_for_posthoc(
    labels: np.ndarray,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Create disjoint calibration, threshold, and conformal validation roles.

    Keeping these roles separate prevents a post-hoc calibrator, decision
    threshold, and conformal nonconformity distribution from each evaluating
    themselves. The input must be the reserved post-hoc validation split, not
    the model-selection split used for early stopping.
    """
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError('Validation labels must be a non-empty one-dimensional array.')
    if not np.isin(targets, (0, 1)).all() or len(np.unique(targets)) < 2:
        raise ValueError('Post-hoc validation partitioning requires both binary classes.')

    class_indices = {label: np.flatnonzero(targets == label) for label in (0, 1)}
    minimum_class_count = min(len(indices) for indices in class_indices.values())
    if minimum_class_count < 3:
        all_indices = np.arange(len(targets), dtype=int)
        return {
            'status': 'reused_validation_insufficient_per_class_count',
            'independent_roles': False,
            'reason': (
                'At least three examples of each class are required for disjoint '
                'calibration, threshold, and conformal roles.'
            ),
            'indices': {
                'calibration': all_indices,
                'threshold': all_indices,
                'conformal': all_indices,
            },
        }

    generator = np.random.default_rng(seed)
    role_indices = {'calibration': [], 'threshold': [], 'conformal': []}
    for label in (0, 1):
        shuffled = generator.permutation(class_indices[label])
        role_sizes = [len(shuffled) // 3] * 3
        for offset in range(len(shuffled) % 3):
            role_sizes[offset] += 1
        start = 0
        for role, role_size in zip(role_indices, role_sizes):
            role_indices[role].extend(shuffled[start:start + role_size].tolist())
            start += role_size
    indices = {
        role: np.asarray(sorted(values), dtype=int)
        for role, values in role_indices.items()
    }
    return {
        'status': 'stratified_disjoint_three_way',
        'independent_roles': True,
        'seed': seed,
        'indices': indices,
    }


def posthoc_validation_partition_summary(
    labels: np.ndarray,
    partition: dict[str, Any],
) -> dict[str, Any]:
    """Produce JSON-safe counts for the three post-hoc validation roles."""
    targets = np.asarray(labels, dtype=int)
    roles = {}
    for role, indices in partition['indices'].items():
        role_labels = targets[indices]
        roles[role] = {
            'rows': len(indices),
            'negative_count': int((role_labels == 0).sum()),
            'positive_count': int((role_labels == 1).sum()),
        }
    return {
        'status': partition['status'],
        'independent_roles': partition['independent_roles'],
        'reason': partition.get('reason'),
        'roles': roles,
    }


def save_posthoc_validation_partition_artifact(
    loader: DataLoader,
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    partition: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    """Save the exact validation roles used by post-hoc analysis."""
    if len(cast(PxDDIDataset, loader.dataset).metadata) != len(labels):
        raise RuntimeError('Validation provenance does not match post-hoc labels.')
    assignments = np.full(len(labels), 'not_assigned', dtype=object)
    for role, indices in partition['indices'].items():
        if np.any(assignments[indices] != 'not_assigned'):
            assignments[indices] = assignments[indices] + '|' + role
        else:
            assignments[indices] = role
    table = pd.DataFrame(cast(PxDDIDataset, loader.dataset).metadata)
    table['label'] = labels
    table['raw_prediction_score'] = raw_predictions
    table['calibrated_prediction_score'] = calibrated_predictions
    table['posthoc_validation_role'] = assignments
    table['posthoc_roles_independent'] = partition['independent_roles']
    path = artifact_dir / 'posthoc_validation_partition.csv'
    table.to_csv(path, index=False)
    return {
        **posthoc_validation_partition_summary(labels, partition),
        'path': str(path),
        'sha256': get_file_hash(path),
    }


def should_stop_early(
    epoch: int,
    epochs_without_improvement: int,
    minimum_epochs: int = EARLY_STOPPING_MIN_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> bool:
    """Return whether a non-improving training run has reached its stop rule."""
    return (
        patience > 0
        and epoch >= minimum_epochs
        and epochs_without_improvement >= patience
    )


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    raw_predictions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the shared rigorous metric set."""
    return calculate_binary_metrics(labels, predictions, threshold, raw_predictions)


def save_prediction_artifact(
    name: str,
    loader: DataLoader,
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    threshold: float,
    prediction_dir: Path,
    calibration: dict[str, Any],
    additional_columns: dict[str, Any] | None = None,
) -> Path:
    metadata = cast(PxDDIDataset, loader.dataset).metadata
    if len(metadata) != len(labels):
        raise RuntimeError('Prediction provenance does not match the evaluated dataset.')
    table = pd.DataFrame(metadata)
    table['label'] = labels
    table['raw_prediction_score'] = raw_predictions
    table['calibrated_prediction_score'] = calibrated_predictions
    table['prediction_score'] = calibrated_predictions
    table['calibration_status'] = calibration.get('status', 'not_fitted')
    table['calibration_method'] = calibration.get('method')
    table['threshold'] = threshold
    table['predicted_label'] = (calibrated_predictions >= threshold).astype(int)
    for column_name, values in (additional_columns or {}).items():
        if len(values) != len(table):
            raise ValueError(
                f'Additional prediction column {column_name!r} does not match '
                'the evaluated dataset length.'
            )
        table[column_name] = values
    prediction_path = prediction_dir / f'{name.lower().replace(" ", "_")}_predictions.csv'
    table.to_csv(prediction_path, index=False)
    return prediction_path


def generate_candidate_explanation_artifact(
    model: PxDDIModel,
    evaluation_data: dict[str, dict[str, Any]],
    explanation_dir: Path,
) -> dict[str, Any]:
    """Write a bounded, offline explanation audit for experimental candidates.

    This deliberately runs only when requested.  Occlusion is much slower than
    normal inference, and the output is a research artifact rather than an API
    feature.  A single failed example is captured in the JSON without losing a
    completed training run or its benchmark results.
    """
    if not RUN_CANDIDATE_EXPLANATIONS:
        return {
            'status': 'disabled',
            'reason': 'Set PXDDI_RUN_CANDIDATE_EXPLANATIONS=1 to generate a bounded offline audit.',
            'method': EXPLANATION_METHOD,
        }
    if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_LEGACY:
        return {
            'status': 'not_applicable',
            'reason': (
                'The deployed legacy model keeps its existing API explanation path. '
                'Candidate occlusion artifacts are reserved for experimental architectures.'
            ),
            'method': EXPLANATION_METHOD,
        }

    explanation_dir.mkdir(parents=True, exist_ok=True)
    graph_builder = lambda smiles: smiles_to_graph(
        smiles,
        feature_schema=FEATURE_SCHEMA,
        include_motif_features=USE_MOTIF_FEATURES,
    )
    artifact: dict[str, Any] = {
        'method': EXPLANATION_METHOD,
        'model_architecture': MODEL_ARCHITECTURE,
        'model_seed': MODEL_SEED,
        'split_seed': SPLIT_SEED,
        'uses_raw_model_probabilities': True,
        'scope_limitations': EXPLANATION_LIMITATIONS,
        'samples_per_split_limit': EXPLANATION_SAMPLES_PER_SPLIT,
        'top_k': EXPLANATION_TOP_K,
        'splits': {},
    }
    for split_name, payload in evaluation_data.items():
        loader = payload['loader']
        labels = payload['labels']
        raw_predictions = payload['raw_predictions']
        calibrated_predictions = payload['calibrated_predictions']
        threshold = payload['threshold']
        selected_indices = select_representative_indices(
            labels, calibrated_predictions, threshold, EXPLANATION_SAMPLES_PER_SPLIT
        )
        examples = []
        for index in selected_indices:
            graph_a, graph_b, *_ = loader.dataset.records[index]
            metadata = cast(PxDDIDataset, loader.dataset).metadata[index]
            example: dict[str, Any] = {
                'dataset_index': index,
                'source': metadata['source'],
                'target': metadata['target'],
                'label': int(labels[index]),
                'label_evidence': metadata['label_evidence'],
                'raw_prediction_score': float(raw_predictions[index]),
                'calibrated_prediction_score': float(calibrated_predictions[index]),
                'validation_selected_threshold': float(threshold),
                'predicted_label': int(calibrated_predictions[index] >= threshold),
            }
            try:
                example['explanation'] = explain_pair_with_occlusion(
                    model,
                    graph_a,
                    graph_b,
                    metadata['source'],
                    metadata['target'],
                    motif_names=MOTIF_FEATURE_NAMES if USE_MOTIF_FEATURES else (),
                    top_k=EXPLANATION_TOP_K,
                    graph_builder=graph_builder,
                )
                visual_dir = explanation_dir / 'figures' / split_name.lower().replace(' ', '_')
                example_prefix = f'example_{index:05d}'
                source_figure = visual_dir / f'{example_prefix}_drug_a.svg'
                target_figure = visual_dir / f'{example_prefix}_drug_b.svg'
                render_occlusion_svg(
                    metadata['source'],
                    example['explanation']['drug_a']['top_atom_occlusions'],
                    example['explanation']['drug_a']['top_bond_occlusions'],
                    source_figure,
                )
                render_occlusion_svg(
                    metadata['target'],
                    example['explanation']['drug_b']['top_atom_occlusions'],
                    example['explanation']['drug_b']['top_bond_occlusions'],
                    target_figure,
                )
                example['explanation_figures'] = {
                    'drug_a_svg': str(source_figure),
                    'drug_b_svg': str(target_figure),
                }
            except Exception as error:  # Keep a completed candidate run auditable.
                example['explanation_error'] = {
                    'type': type(error).__name__,
                    'message': str(error),
                }
            examples.append(example)
        artifact['splits'][split_name] = {
            'evaluated_rows': len(labels),
            'selected_dataset_indices': selected_indices,
            'examples': examples,
        }

    artifact_path = explanation_dir / 'candidate_occlusion_explanations.json'
    write_json(artifact_path, artifact)
    return {
        'status': 'generated',
        'method': EXPLANATION_METHOD,
        'path': str(artifact_path),
        'sha256': get_file_hash(artifact_path),
        'samples_per_split_limit': EXPLANATION_SAMPLES_PER_SPLIT,
        'top_k': EXPLANATION_TOP_K,
        'scope_limitations': EXPLANATION_LIMITATIONS,
    }


def save_conformal_calibration_artifact(
    conformal: dict[str, Any], uncertainty_dir: Path
) -> dict[str, Any]:
    """Persist the validation nonconformity values behind conformal p-values."""
    if conformal.get('status') != 'fitted':
        return {
            'status': conformal.get('status', 'not_fitted'),
            'path': None,
            'sha256': None,
        }
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    path = uncertainty_dir / 'conformal_validation_nonconformity.csv'
    pd.DataFrame({
        'validation_nonconformity_score': conformal['validation_nonconformity_scores']
    }).to_csv(path, index=False)
    return {
        'status': 'saved',
        'path': str(path),
        'sha256': get_file_hash(path),
        'rows': len(conformal['validation_nonconformity_scores']),
    }


def _save_figure(figure, destination: Path) -> None:
    figure.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    figure.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def plot_training_curves(history: dict[str, list[float]], figure_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history['epoch'], history['loss'], color='#0072B2', label='Training loss')
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Loss')
    axes[0].grid(alpha=0.25)
    metric_lines = (
        ('auroc', 'Model-selection AUROC', '#D55E00'),
        ('average_precision', 'Model-selection PR-AUC', '#CC79A7'),
        ('f1', 'Model-selection F1', '#009E73'),
        ('mcc', 'Model-selection MCC', '#56B4E9'),
        ('balanced_accuracy', 'Model-selection balanced accuracy', '#E69F00'),
    )
    for key, label, color in metric_lines:
        if key in history:
            axes[1].plot(history['epoch'], history[key], color=color, label=label)
    axes[1].set(
        title='Model-selection validation performance',
        xlabel='Epoch',
        ylabel='Score',
        ylim=(0, 1),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figure_dir / 'training_curves')


def plot_evaluation(
    name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    metrics: dict[str, Any],
    figure_dir: Path,
    raw_predictions: np.ndarray | None = None,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    if metrics['status'] == 'evaluated':
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, predictions)
        precision, recall, _ = precision_recall_curve(labels, predictions)
        axes[0].plot(false_positive_rate, true_positive_rate, color='#0072B2')
        axes[0].plot([0, 1], [0, 1], linestyle='--', color='#777777')
        axes[0].set(
            title=f'{name}: ROC (AUROC={metrics["auroc"]:.3f})',
            xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1),
        )
        axes[1].plot(recall, precision, color='#D55E00')
        axes[1].set(
            title=f'{name}: PR (AP={metrics["average_precision"]:.3f})',
            xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1),
        )
    else:
        message = 'Skipped: split has fewer than two classes.'
        for axis in axes[:3]:
            axis.text(0.5, 0.5, message, ha='center', va='center', wrap=True)
            axis.set_axis_off()

    matrix = np.asarray(metrics['confusion_matrix'])
    image = axes[2].imshow(matrix, cmap='Blues')
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(matrix[row, column]), ha='center', va='center')
    axes[2].set(
        title=f'{name}: confusion matrix\nthreshold={metrics["threshold"]:.3f}',
        xlabel='Predicted label', ylabel='True label',
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['Unreported', 'Reported'], yticklabels=['Unreported', 'Reported'],
    )
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    if metrics['status'] == 'evaluated':
        fraction_positive, mean_predicted = calibration_curve(labels, predictions, n_bins=10)
        axes[3].plot(mean_predicted, fraction_positive, marker='o', label='Final score')
        if raw_predictions is not None:
            raw_fraction_positive, raw_mean_predicted = calibration_curve(
                labels, raw_predictions, n_bins=10
            )
            axes[3].plot(
                raw_mean_predicted,
                raw_fraction_positive,
                marker='o',
                linestyle='--',
                label='Raw score',
            )
        axes[3].plot([0, 1], [0, 1], linestyle=':', color='#777777', label='Ideal')
        axes[3].set(
            title=f'{name}: calibration',
            xlabel='Mean predicted score',
            ylabel='Observed reported-pair fraction',
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[3].legend()
        axes[3].grid(alpha=0.25)
    else:
        axes[3].set_axis_off()
    figure.tight_layout()
    _save_figure(figure, figure_dir / f'{name.lower().replace(" ", "_")}_evaluation')


def plot_benchmark_comparison(results: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    names = list(results)
    metric_names = [('auroc', 'AUROC'), ('average_precision', 'Average precision'), ('f1', 'F1')]
    x = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, (key, label) in enumerate(metric_names):
        values = [results[name][key] if results[name][key] is not None else np.nan for name in names]
        axis.bar(x + (index - 1) * width, values, width, label=label)
    axis.set(
        title='PxDDI evaluation by split', xlabel='Evaluation split', ylabel='Score',
        xticks=x, xticklabels=names, ylim=(0, 1),
    )
    axis.legend()
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figure_dir / 'benchmark_comparison')


def plot_toxicity_bridge_coverage(summary: dict[str, Any], figure_dir: Path) -> None:
    labels = ['Source rows', 'Mapped rows', 'Unique structures', 'Clean structures']
    values = [
        summary['source_rows'],
        summary['rows_with_canonical_smiles'],
        summary['unique_canonical_structures'],
        summary['resolved_unique_canonical_structures'],
    ]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(labels, values, color=['#0072B2', '#56B4E9', '#E69F00', '#009E73'])
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom')
    axis.set(title='Toxicity bridge coverage after conflict exclusion', ylabel='Structure count')
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figure_dir / 'toxicity_bridge_coverage')


def model_summary(model: PxDDIModel) -> dict[str, Any]:
    """Return a compact architecture and parameter summary for the run manifest."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        'model_class': model.__class__.__name__,
        'encoder_class': model.encoder.__class__.__name__,
        'model_architecture': MODEL_ARCHITECTURE,
        'feature_schema': FEATURE_SCHEMA,
        'input_atom_features': INPUT_FEATURE_DIM,
        'edge_feature_dim': (
            NUM_BOND_FEATURES
            if USES_EDGE_FEATURES else None
        ),
        'use_motif_features': USE_MOTIF_FEATURES,
        'use_cross_drug_attention': USE_CROSS_DRUG_ATTENTION,
        'cross_drug_attention_type': (
            'pair_isolated_atom_attention_v1'
            if USE_CROSS_DRUG_ATTENTION else None
        ),
        'motif_feature_schema': (
            MOTIF_SCHEMA_SMARTS_COUNTS_V1 if USE_MOTIF_FEATURES else None
        ),
        'motif_feature_dim': MOTIF_FEATURE_DIM if USE_MOTIF_FEATURES else None,
        'motif_hidden_channels': MOTIF_HIDDEN_CHANNELS if USE_MOTIF_FEATURES else None,
        'motif_metadata': motif_metadata() if USE_MOTIF_FEATURES else None,
        'hidden_channels': HIDDEN_CHANNELS,
        'use_chemberta': USE_CHEMBERTA,
        'pair_representation': (
            'graph_embedding_plus_pair_isolated_cross_drug_atom_attention_sum + absolute_difference + toxicity_sum + absolute_toxicity_difference'
            if USE_CROSS_DRUG_ATTENTION and USE_TOXICITY_PAIR_FEATURES
            else 'graph_embedding_plus_pair_isolated_cross_drug_atom_attention_sum + absolute_difference'
            if USE_CROSS_DRUG_ATTENTION
            else
            'graph_embedding_plus_motif_embedding_sum + absolute_difference + toxicity_sum + absolute_toxicity_difference'
            if USE_MOTIF_FEATURES and USE_TOXICITY_PAIR_FEATURES
            else 'graph_embedding_plus_motif_embedding_sum + absolute_difference'
            if USE_MOTIF_FEATURES
            else 'embedding_sum + absolute_embedding_difference + toxicity_sum + absolute_toxicity_difference'
            if USE_TOXICITY_PAIR_FEATURES
            else 'embedding_sum + absolute_embedding_difference'
        ),
        'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
        'output_heads': ['interaction_risk', 'drug_a_toxicity', 'drug_b_toxicity'],
        'total_parameters': int(total_parameters),
        'trainable_parameters': int(trainable_parameters),
    }


def _prepare_positive_edges(
    audit_dir: Path, sampling_seed: int = SPLIT_SEED
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate and validate positive pairs before negative sampling or splitting."""
    edges = pd.read_csv(TWOSIDES_EDGES)
    required_columns = {'source', 'target'}
    missing = required_columns.difference(edges.columns)
    if missing:
        raise ValueError(f'TWOSIDES edges are missing required columns: {sorted(missing)}.')
    positives = edges[['source', 'target']].copy()
    positives['label'] = 1.0
    positives = deduplicate_unordered_pairs(positives, 'source', 'target')
    clean_positives, exclusions = filter_graph_compatible_pairs(positives)
    if clean_positives.empty:
        raise ValueError('No graph-compatible positive pairs remain after SMILES validation.')

    audit_dir.mkdir(parents=True, exist_ok=True)
    exclusions_path = audit_dir / 'invalid_smiles_exclusions.csv'
    exclusions.to_csv(exclusions_path, index=False)
    curation_summary = save_counterion_curation_candidates(exclusions, audit_dir)
    sampled = clean_positives.sample(
        n=min(DATA_CAP, len(clean_positives)), random_state=sampling_seed
    ).reset_index(drop=True)
    summary = {
        'raw_unique_positive_pairs': len(positives),
        'excluded_positive_pairs': len(exclusions),
        'clean_positive_pairs': len(clean_positives),
        'sampled_positive_pairs': len(sampled),
        'data_cap': DATA_CAP,
        'exclusion_audit_path': str(exclusions_path),
        'exclusion_audit_sha256': get_file_hash(exclusions_path),
        'counterion_curation': curation_summary,
    }
    write_json(audit_dir / 'input_quality_summary.json', summary)
    print(
        'Positive-pair input audit: '
        f"{summary['raw_unique_positive_pairs']} unique pairs; "
        f"{summary['excluded_positive_pairs']} excluded; "
        f"{summary['sampled_positive_pairs']} used for sampling."
    )
    return sampled, summary


def main() -> None:
    set_reproducibility(MODEL_SEED)
    print(f'Training on: {DEVICE}')
    if DEVICE.type != 'cuda':
        print('Warning: CUDA is unavailable; Colab GPU is recommended for this run.')

    RUN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=False)
    figure_dir = RUN_ARTIFACTS_DIR / 'figures'
    prediction_dir = RUN_ARTIFACTS_DIR / 'predictions'
    audit_dir = RUN_ARTIFACTS_DIR / 'audits'
    figure_dir.mkdir()
    prediction_dir.mkdir()

    manifest = build_run_manifest()
    write_json(RUN_ARTIFACTS_DIR / 'run_manifest_initial.json', manifest)

    toxicity_lookup, toxicity_summary = load_toxicity_lookup(audit_dir)
    positives, input_quality_summary = _prepare_positive_edges(
        audit_dir, sampling_seed=SPLIT_SEED
    )
    full_dataset = build_binary_pair_dataset(
        positives, source_col='source', target_col='target', neg_ratio=1.0, seed=SPLIT_SEED
    )
    dataset_summary = {
        'effective_positive_pairs': (full_dataset['label'] == 1.0).sum(),
        'sampled_unreported_negative_pairs': (full_dataset['label'] == 0.0).sum(),
        'total_pair_rows_before_split': len(full_dataset),
        'negative_label_meaning': 'unreported_twosides_sampled',
    }
    write_json(audit_dir / 'dataset_summary.json', dataset_summary)
    if EVALUATION_PROTOCOL == 'scaffold_disjoint':
        splits, scaffold_audit = create_scaffold_disjoint_splits(
            full_dataset, drug_a_col='source', drug_b_col='target', seed=SPLIT_SEED
        )
        train_split_key = 'scaffold_train'
        validation_split_key = 'scaffold_validation'
        posthoc_validation_split_key = 'scaffold_posthoc_validation'
        test_split_keys = {'Scaffold-disjoint': 'scaffold_test'}
        evaluation_protocol_summary: dict[str, Any] = {
            'name': 'scaffold_disjoint',
            **scaffold_audit,
        }
        write_json(audit_dir / 'scaffold_split_audit.json', evaluation_protocol_summary)
    else:
        splits = create_splits(
            full_dataset, drug_a_col='source', drug_b_col='target', seed=SPLIT_SEED
        )
        train_split_key = 'transductive_train'
        validation_split_key = 'validation'
        posthoc_validation_split_key = 'posthoc_validation'
        test_split_keys = {
            'Transductive': 'transductive_test',
            'S1': 's1_test',
            'S2': 's2_test',
            'S1-dev': 's1_dev',
            'S2-dev': 's2_dev',
        }
        evaluation_protocol_summary: dict[str, Any] = {
            'name': 'standard_transductive_s1_s2',
            'interpretation_warning': (
                'Standard results include a transductive split and drug-identity '
                'cold-start S1/S2 partitions. Run scaffold-disjoint evaluation '
                'separately; it requires a distinct training partition.'
            ),
        }
    model_selection_validation, posthoc_validation, validation_role_partition = (
        partition_validation_for_model_selection(
            splits[validation_split_key],
            seed=SPLIT_SEED,
        )
    )
    # Retain the conventional ``validation`` key for compatibility with the
    # experiment suite. It now means model-selection validation only.
    splits[validation_split_key] = model_selection_validation
    splits[posthoc_validation_split_key] = posthoc_validation
    evaluation_protocol_summary['validation_role_partition'] = validation_role_partition
    split_manifest = save_split_manifests(splits, RUN_ARTIFACTS_DIR)

    graph_cache = GraphCache(
        FEATURE_SCHEMA, include_motif_features=USE_MOTIF_FEATURES
    )
    train_loader = build_loader(
        splits[train_split_key], toxicity_lookup, shuffle=True, graph_cache=graph_cache
    )
    validation_loader = build_loader(
        splits[validation_split_key], toxicity_lookup, shuffle=False, graph_cache=graph_cache
    )
    posthoc_validation_loader = build_loader(
        splits[posthoc_validation_split_key],
        toxicity_lookup,
        shuffle=False,
        graph_cache=graph_cache,
    )
    test_loaders = {
        name: build_loader(
            splits[split_key], toxicity_lookup, shuffle=False, graph_cache=graph_cache
        )
        for name, split_key in test_split_keys.items()
    }
    all_loaders = {
        'train': train_loader,
        'model_selection_validation': validation_loader,
        'posthoc_validation': posthoc_validation_loader,
        **test_loaders,
    }
    unexpected_skips = {
        name: cast(PxDDIDataset, loader.dataset).skipped_count
        for name, loader in all_loaders.items()
        if cast(PxDDIDataset, loader.dataset).skipped_count
    }
    if unexpected_skips:
        raise RuntimeError(
            'Graph validation changed between the pre-split audit and dataset construction: '
            f'{unexpected_skips}. Review the input audit before training.'
        )
    validation_labels = np.asarray([record['label'] for record in cast(PxDDIDataset, validation_loader.dataset).metadata])
    if len(validation_labels) == 0 or len(np.unique(validation_labels)) < 2:
        raise ValueError(
            'Model-selection validation is unusable after SMILES validation; '
            'adjust the data split.'
        )
    posthoc_validation_labels = np.asarray([
        record['label'] for record in cast(PxDDIDataset, posthoc_validation_loader.dataset).metadata
    ])
    if len(posthoc_validation_labels) == 0 or len(np.unique(posthoc_validation_labels)) < 2:
        raise ValueError(
            'Reserved post-hoc validation is unusable after SMILES validation; '
            'adjust the data split.'
        )

    model = PxDDIModel(
        in_channels=INPUT_FEATURE_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        use_chemberta=USE_CHEMBERTA,
        architecture_version=MODEL_ARCHITECTURE,
        edge_feature_dim=(
            NUM_BOND_FEATURES
            if USES_EDGE_FEATURES else None
        ),
        use_toxicity_pair_features=USE_TOXICITY_PAIR_FEATURES,
        motif_feature_dim=MOTIF_FEATURE_DIM if USE_MOTIF_FEATURES else None,
        motif_hidden_channels=MOTIF_HIDDEN_CHANNELS if USE_MOTIF_FEATURES else None,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    history = {
        'epoch': [],
        'loss': [],
        'auroc': [],
        'average_precision': [],
        'f1': [],
        'mcc': [],
        'balanced_accuracy': [],
        'threshold': [],
        'learning_rate': [],
    }
    best_auroc = float('-inf')
    epochs_without_improvement = 0
    stopped_early = False
    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize(DEVICE)
    training_started_at = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler)
        scheduler.step()
        validation_true, validation_predicted = collect_predictions(model, validation_loader)
        threshold = select_validation_threshold(validation_true, validation_predicted)
        validation_metrics = calculate_metrics(validation_true, validation_predicted, threshold)
        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['auroc'].append(validation_metrics['auroc'])
        history['average_precision'].append(validation_metrics['average_precision'])
        history['f1'].append(validation_metrics['f1'])
        history['mcc'].append(validation_metrics['mcc'])
        history['balanced_accuracy'].append(validation_metrics['balanced_accuracy'])
        history['threshold'].append(threshold)
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        print(
            f'Epoch {epoch}/{EPOCHS}: loss={loss:.4f}; '
            f"selection AUROC={validation_metrics['auroc']:.4f}; "
            f"PR-AUC={validation_metrics['average_precision']:.4f}; "
            f"MCC={validation_metrics['mcc']:.4f}; "
            f"balanced accuracy={validation_metrics['balanced_accuracy']:.4f}; "
            f"F1={validation_metrics['f1']:.4f}; "
            f'LR={optimizer.param_groups[0]["lr"]:.6f}'
        )
        if validation_metrics['auroc'] > best_auroc:
            best_auroc = validation_metrics['auroc']
            epochs_without_improvement = 0
            safe_checkpoint_save(
                {
                    'model_state_dict': model.state_dict(),
                    'hidden_channels': HIDDEN_CHANNELS,
                    'in_channels': INPUT_FEATURE_DIM,
                    'architecture_version': MODEL_ARCHITECTURE,
                    'feature_schema': FEATURE_SCHEMA,
                    'edge_feature_dim': (
                        NUM_BOND_FEATURES
                        if USES_EDGE_FEATURES else None
                    ),
                    'motif_feature_schema': (
                        MOTIF_SCHEMA_SMARTS_COUNTS_V1 if USE_MOTIF_FEATURES else None
                    ),
                    'motif_feature_dim': MOTIF_FEATURE_DIM if USE_MOTIF_FEATURES else None,
                    'motif_hidden_channels': (
                        MOTIF_HIDDEN_CHANNELS if USE_MOTIF_FEATURES else None
                    ),
                    'motif_metadata': motif_metadata() if USE_MOTIF_FEATURES else None,
                    'use_cross_drug_attention': USE_CROSS_DRUG_ATTENTION,
                    'cross_drug_attention_type': (
                        'pair_isolated_atom_attention_v1'
                        if USE_CROSS_DRUG_ATTENTION else None
                    ),
                    'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
                    'toxicity_loss_weight': TOXICITY_LOSS_WEIGHT,
                    'toxicity_head_output': 'logits_v1',
                    'use_chemberta': USE_CHEMBERTA,
                    'auroc': float(best_auroc),
                    'model_selection_metric': 'AUROC',
                    'model_selection_split': (
                        f'{validation_split_key}: disjoint model-selection validation '
                        'used for early stopping'
                    ),
                    'epoch': epoch,
                    'data_cap': DATA_CAP,
                    'seed': MODEL_SEED,
                    'model_seed': MODEL_SEED,
                    'split_seed': SPLIT_SEED,
                    'threshold': threshold,
                },
                CHECKPOINT_PATH,
            )
        else:
            epochs_without_improvement += 1
        if should_stop_early(epoch, epochs_without_improvement):
            stopped_early = True
            print(
                'Early stopping: no validation AUROC improvement for '
                f'{epochs_without_improvement} epochs after epoch {EARLY_STOPPING_MIN_EPOCHS}. '
                f'Keeping best checkpoint from epoch {history["epoch"][int(np.argmax(history["auroc"]))]}.'
            )
            break

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)
    training_efficiency = {
        'measurement_scope': 'optimization_and_per_epoch_validation',
        'wall_clock_seconds': time.perf_counter() - training_started_at,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(DEVICE)
        if DEVICE.type == 'cuda' else None,
        'recorded_batch_size': BATCH_SIZE,
        'parameter_count': int(sum(parameter.numel() for parameter in model.parameters())),
        'interpretation_warning': (
            'Training efficiency is hardware- and data-size-specific. Compare it '
            'only across runs with the same runtime environment and split manifest.'
        ),
    }

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    validation_true, validation_raw_predictions = collect_predictions(
        model, posthoc_validation_loader
    )
    posthoc_partition = partition_validation_for_posthoc(
        validation_true, seed=SPLIT_SEED
    )
    calibration_indices = posthoc_partition['indices']['calibration']
    threshold_indices = posthoc_partition['indices']['threshold']
    conformal_indices = posthoc_partition['indices']['conformal']
    calibration = fit_platt_calibrator(
        validation_true[calibration_indices],
        validation_raw_predictions[calibration_indices],
        fitted_on=(
            'posthoc_validation_calibration_partition'
            if posthoc_partition['independent_roles']
            else 'reused_validation_insufficient_per_class_count'
        ),
    )
    validation_calibrated_predictions = apply_calibrator(
        validation_raw_predictions, calibration
    )
    conformal = fit_split_conformal_binary(
        validation_true[conformal_indices],
        validation_calibrated_predictions[conformal_indices],
        alpha=CONFORMAL_ALPHA,
        fitted_on=(
            'posthoc_validation_conformal_partition'
            if posthoc_partition['independent_roles']
            else 'reused_validation_insufficient_per_class_count'
        ),
    )
    conformal_artifact = save_conformal_calibration_artifact(
        conformal, RUN_ARTIFACTS_DIR / 'uncertainty'
    )
    posthoc_partition_artifact = save_posthoc_validation_partition_artifact(
        posthoc_validation_loader,
        validation_true,
        validation_raw_predictions,
        validation_calibrated_predictions,
        posthoc_partition,
        RUN_ARTIFACTS_DIR / 'uncertainty',
    )
    training_smiles = {
        metadata[column]
        for metadata in cast(PxDDIDataset, train_loader.dataset).metadata
        for column in ('source', 'target')
    }
    applicability_domain = MorganApplicabilityDomain(
        minimum_similarity=APPLICABILITY_DOMAIN_MINIMUM_SIMILARITY
    )
    applicability_domain_summary = applicability_domain.fit(training_smiles)
    frozen_threshold = select_validation_threshold(
        validation_true[threshold_indices], validation_calibrated_predictions[threshold_indices]
    )
    checkpoint['threshold'] = frozen_threshold
    checkpoint['calibration'] = calibration
    checkpoint['toxicity_head_output'] = 'logits_v1'
    checkpoint['conformal'] = conformal
    checkpoint['applicability_domain'] = applicability_domain.export_checkpoint_state()
    checkpoint['posthoc_validation_partition'] = {
        key: value for key, value in posthoc_partition_artifact.items()
        if key not in {'path', 'sha256'}
    }
    checkpoint['validation_role_partition'] = validation_role_partition
    checkpoint_hash = safe_checkpoint_save(checkpoint, CHECKPOINT_PATH)
    validation_prediction_path = save_prediction_artifact(
        'Validation',
        posthoc_validation_loader,
        validation_true,
        validation_raw_predictions,
        validation_calibrated_predictions,
        frozen_threshold,
        prediction_dir,
        calibration,
    )
    validation_prediction_summary = {
        'path': str(validation_prediction_path),
        'sha256': get_file_hash(validation_prediction_path),
        'rows': len(validation_true),
        'purpose': (
            'Raw member scores from validation reserved before training. They were '
            'not used for early stopping; a fixed-split ensemble combines them '
            'before its own calibrator, threshold, and conformal rule are fitted.'
        ),
        'split_role': 'posthoc_validation_reserved_before_training',
    }

    plot_training_curves(history, figure_dir)
    history_summary = save_training_history(history, RUN_ARTIFACTS_DIR)
    results: dict[str, dict[str, Any]] = {}
    evaluation_data: dict[str, dict[str, Any]] = {}
    for name, loader in test_loaders.items():
        labels, raw_predictions = collect_predictions(model, loader)
        calibrated_predictions = apply_calibrator(raw_predictions, calibration)
        metrics = calculate_metrics(
            labels,
            calibrated_predictions,
            frozen_threshold,
            raw_predictions=raw_predictions,
        )
        conformal_sets = conformal_prediction_sets(calibrated_predictions, conformal)
        applicability_scores = applicability_domain.score_pairs(
            [metadata['source'] for metadata in cast(PxDDIDataset, loader.dataset).metadata],
            [metadata['target'] for metadata in cast(PxDDIDataset, loader.dataset).metadata],
        )
        additional_prediction_columns = {
            'predictive_entropy_nats': predictive_entropy(calibrated_predictions),
            'conformal_no_interaction_p_value': conformal_sets['no_interaction_p_value'],
            'conformal_interaction_p_value': conformal_sets['interaction_p_value'],
            'conformal_prediction_set': conformal_sets['prediction_set'],
            'conformal_prediction_set_size': conformal_sets['set_size'],
            'conformal_abstain': conformal_sets['abstain'],
            **applicability_scores,
        }
        prediction_path = save_prediction_artifact(
            name,
            loader,
            labels,
            raw_predictions,
            calibrated_predictions,
            frozen_threshold,
            prediction_dir,
            calibration,
            additional_columns=additional_prediction_columns,
        )
        metrics['prediction_path'] = str(prediction_path)
        metrics['prediction_sha256'] = get_file_hash(prediction_path)
        metrics['skipped_invalid_smiles'] = cast(PxDDIDataset, loader.dataset).skipped_count
        metrics['test_set_bootstrap_95ci'] = bootstrap_confidence_intervals(
            labels,
            calibrated_predictions,
            frozen_threshold,
            raw_predictions=raw_predictions,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=SPLIT_SEED,
        )
        metrics['uncertainty'] = summarize_conformal_test_labels(labels, conformal_sets)
        metrics['uncertainty'].update({
            'method': conformal['method'],
            'alpha': conformal['alpha'],
            'interpretation_warning': conformal['interpretation_warning'],
        })
        metrics['selective_prediction'] = selective_prediction_summary(
            labels,
            calibrated_predictions,
            frozen_threshold,
            conformal_sets['abstain'],
        )
        metrics['applicability_domain'] = {
            'structural_ood_count': int(applicability_scores['structural_ood_flag'].sum()),
            'structural_ood_rate': float(applicability_scores['structural_ood_flag'].mean()),
            'minimum_tanimoto_similarity': APPLICABILITY_DOMAIN_MINIMUM_SIMILARITY,
            'interpretation_warning': applicability_domain_summary['interpretation_warning'],
        }
        metrics['structural_similarity_slices'] = structural_similarity_slices(
            labels,
            calibrated_predictions,
            frozen_threshold,
            applicability_scores['pair_minimum_nearest_train_tanimoto'],
            raw_predictions=raw_predictions,
        )
        metrics['error_analysis'] = save_confident_error_analysis(
            cast(PxDDIDataset, loader.dataset).metadata,
            labels,
            raw_predictions,
            calibrated_predictions,
            frozen_threshold,
            RUN_ARTIFACTS_DIR / 'error_analysis',
            name,
            maximum_rows_per_error_type=ERROR_ANALYSIS_MAX_ROWS,
            additional_columns=additional_prediction_columns,
        )
        metrics['inference_efficiency'] = measure_inference_efficiency(model, loader)
        results[name] = metrics
        evaluation_data[name] = {
            'loader': loader,
            'labels': labels,
            'raw_predictions': raw_predictions,
            'calibrated_predictions': calibrated_predictions,
            'threshold': frozen_threshold,
        }
        plot_evaluation(
            name,
            labels,
            calibrated_predictions,
            metrics,
            figure_dir,
            raw_predictions=raw_predictions,
        )
    plot_benchmark_comparison(results, figure_dir)
    plot_toxicity_bridge_coverage(toxicity_summary, figure_dir)
    write_json(RUN_ARTIFACTS_DIR / 'results_summary.json', results)
    explanation_summary = generate_candidate_explanation_artifact(
        model, evaluation_data, RUN_ARTIFACTS_DIR / 'explanations'
    )

    manifest.update({
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': {
            'path': str(CHECKPOINT_PATH),
            'sha256': checkpoint_hash,
            'epoch': checkpoint['epoch'],
            'validation_auroc': checkpoint['auroc'],
            'validation_selected_threshold': frozen_threshold,
        },
        'calibration': calibration,
        'validation_predictions': validation_prediction_summary,
        'uncertainty': {
            'conformal': {
                key: value for key, value in conformal.items()
                if key != 'validation_nonconformity_scores'
            },
            'conformal_validation_artifact': conformal_artifact,
            'posthoc_validation_partition': posthoc_partition_artifact,
            'validation_role_partition': validation_role_partition,
            'applicability_domain': applicability_domain_summary,
        },
        'model_summary': model_summary(model),
        'evaluation_protocol': evaluation_protocol_summary,
        'efficiency': {
            'training': training_efficiency,
            'inference_by_evaluation_split': {
                name: metrics['inference_efficiency'] for name, metrics in results.items()
            },
        },
        'training_history': history_summary,
        'early_stopping': {
            'enabled': EARLY_STOPPING_PATIENCE > 0,
            'patience': EARLY_STOPPING_PATIENCE,
            'minimum_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'stopped_early': stopped_early,
            'completed_epochs': len(history['epoch']),
            'best_epoch': checkpoint['epoch'],
            'model_selection_split': validation_split_key,
            'posthoc_validation_split': posthoc_validation_split_key,
            'posthoc_reserved_before_training': True,
        },
        'toxicity_bridge': toxicity_summary,
        'input_quality': input_quality_summary,
        'dataset': dataset_summary,
        'graph_cache': graph_cache.summary(),
        'split_manifest': split_manifest,
        'results': results,
        'candidate_explanations': explanation_summary,
        'latest_results_directory': str(LATEST_RESULTS_DIR) if PUBLISH_LATEST_RESULTS else None,
    })
    write_json(RUN_ARTIFACTS_DIR / 'run_manifest.json', manifest)
    print(f'Run artifacts saved to: {RUN_ARTIFACTS_DIR}')
    if PUBLISH_LATEST_RESULTS:
        latest_results_dir = publish_latest_results(RUN_ARTIFACTS_DIR)
        print(f'Latest results refreshed at: {latest_results_dir}')


if __name__ == '__main__':
    main()
