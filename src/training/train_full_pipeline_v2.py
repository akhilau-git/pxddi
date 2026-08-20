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
from datetime import datetime, timezone
from typing import Any

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
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))

DRIVE_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
DRIVE_SRC = DRIVE_BASE / 'pxddi' / 'src'
if DRIVE_SRC.exists() and str(DRIVE_SRC) not in sys.path:
    sys.path.insert(0, str(DRIVE_SRC))

from data_prep.prepare_twosides import (
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_RICH,
    LEGACY_NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    graph_compatibility_reason as source_graph_compatibility_reason,
    smiles_to_graph,
)
from data_prep.pubchem_bridge import canonicalize, resolve_toxicity_bridge
from data_prep.splits import build_binary_pair_dataset, create_splits, deduplicate_unordered_pairs
from models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_LEGACY,
    PxDDIModel,
)
from models.calibration import (
    apply_calibrator,
    expected_calibration_error,
    fit_platt_calibrator,
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


SEED = _positive_int_from_environment('PXDDI_SEED', 42)
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
USE_CHEMBERTA = False
MODEL_ARCHITECTURE = os.environ.get(
    'PXDDI_MODEL_ARCHITECTURE', MODEL_ARCHITECTURE_EDGE_AWARE
)
if MODEL_ARCHITECTURE not in {MODEL_ARCHITECTURE_LEGACY, MODEL_ARCHITECTURE_EDGE_AWARE}:
    raise ValueError(f'Unsupported PXDDI_MODEL_ARCHITECTURE: {MODEL_ARCHITECTURE}.')
FEATURE_SCHEMA = (
    FEATURE_SCHEMA_RICH
    if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE
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

TWOSIDES_EDGES = DRIVE_BASE / 'twosides' / 'drug_drug_edges.csv'
TOXICITY_BRIDGE = DRIVE_BASE / 'checkpoints' / 'toxicity_smiles_bridge.csv'
DEFAULT_CHECKPOINT_PATH = (
    DRIVE_BASE / 'checkpoints' / 'candidates' / 'pxddi_edge_aware_candidate.pt'
    if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE
    else DRIVE_BASE / 'checkpoints' / 'pxddi_model.pt'
)
CHECKPOINT_PATH = Path(os.environ.get('PXDDI_CHECKPOINT_PATH', DEFAULT_CHECKPOINT_PATH))
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ARTIFACTS_BASE = Path(os.environ.get('PXDDI_ARTIFACTS_BASE', DRIVE_BASE / 'artifacts'))
RUN_ARTIFACTS_DIR = ARTIFACTS_BASE / f'run_{RUN_ID}'
LATEST_RESULTS_DIR = Path(
    os.environ.get('PXDDI_LATEST_RESULTS_DIR', DRIVE_BASE / 'latest_results')
)
PUBLISH_LATEST_RESULTS = _boolean_from_environment('PXDDI_PUBLISH_LATEST_RESULTS', True)
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
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing required Colab inputs: {missing}.')
    return {
        'run_id': RUN_ID,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': str(DEVICE),
        'repository_git_commit': repository_git_commit(),
        'runtime_environment': runtime_environment(),
        'random_seed': SEED,
        'configuration': {
            'data_cap': DATA_CAP,
            'epochs': EPOCHS,
            'hidden_channels': HIDDEN_CHANNELS,
            'batch_size': BATCH_SIZE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'early_stopping_min_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'use_chemberta': USE_CHEMBERTA,
            'model_architecture': MODEL_ARCHITECTURE,
            'feature_schema': FEATURE_SCHEMA,
            'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
            'toxicity_loss_weight': TOXICITY_LOSS_WEIGHT,
            'edge_feature_dim': (
                NUM_BOND_FEATURES
                if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE else None
            ),
            'negative_label_meaning': 'unreported_twosides_sampled',
            'toxicity_conflict_policy': 'exclude_conflicting_structures',
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
            'rows': int(len(frame)),
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

    figure_dir = run_dir / 'figures'
    for source in sorted(figure_dir.rglob('*')):
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
            candidate['occurrence_count'] += int(count)
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
        'counterion_only_pair_exclusions': int(
            (
                exclusions[counterion_columns]
                == 'counterion_or_inorganic_only_structure'
            ).any(axis=1).sum()
        ) if not exclusions.empty else 0,
        'unique_counterion_or_inorganic_structures': int(len(candidates)),
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
    def __init__(self, feature_schema: str) -> None:
        self.feature_schema = feature_schema
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
            graph = smiles_to_graph(key, feature_schema=self.feature_schema)
            self._graphs[key] = graph
        return graph.clone() if graph is not None else None

    def summary(self) -> dict[str, Any]:
        requests = self.hits + self.misses
        return {
            'feature_schema': self.feature_schema,
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
        self.graph_cache = graph_cache or GraphCache(FEATURE_SCHEMA)
        for row in dataframe.itertuples(index=False):
            source = getattr(row, source_col)
            target = getattr(row, target_col)
            label = float(getattr(row, label_col))
            graph_a = self.graph_cache.get(source)
            graph_b = self.graph_cache.get(target)
            if graph_a is None or graph_b is None:
                self.skipped_count += 1
                continue
            toxicity_a = toxicity_lookup.get(canonicalize(source))
            toxicity_b = toxicity_lookup.get(canonicalize(target))
            toxicity_a_known = float(toxicity_a is not None)
            toxicity_b_known = float(toxicity_b is not None)
            self.records.append(
                (
                    graph_a,
                    graph_b,
                    0.0 if toxicity_a is None else float(toxicity_a),
                    0.0 if toxicity_b is None else float(toxicity_b),
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

    def get(self, index: int):
        return self.records[index]


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
) -> DataLoader:
    dataset = PxDDIDataset(dataframe, toxicity_lookup, graph_cache=graph_cache)
    generator = torch.Generator()
    generator.manual_seed(SEED)
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
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none')
    ddi_loss = binary_cross_entropy(risk_prediction, risk_label).mean()
    toxicity_a_loss = (
        binary_cross_entropy(toxicity_a_prediction, toxicity_a_label) * toxicity_a_known
    ).sum() / (toxicity_a_known.sum() + 1e-8)
    toxicity_b_loss = (
        binary_cross_entropy(toxicity_b_prediction, toxicity_b_label) * toxicity_b_known
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


def select_validation_threshold(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError('Validation split must contain both classes to select a threshold.')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predictions)
    return float(thresholds[np.argmax(true_positive_rate - false_positive_rate)])


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
    """Calculate ranking, decision, and calibration metrics for one split."""
    if raw_predictions is None:
        raw_predictions = predictions
    if len(raw_predictions) != len(predictions):
        raise ValueError('Raw and final predictions must have equal length.')
    predicted_labels = (predictions >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    result: dict[str, Any] = {
        'sample_count': int(len(labels)),
        'positive_count': int((labels == 1).sum()),
        'negative_count': int((labels == 0).sum()),
        'threshold': float(threshold),
        'confusion_matrix': matrix.tolist(),
        'f1': float(f1_score(labels, predicted_labels, zero_division=0)) if len(labels) else None,
        'precision': float(precision_score(labels, predicted_labels, zero_division=0)) if len(labels) else None,
        'recall': float(recall_score(labels, predicted_labels, zero_division=0)) if len(labels) else None,
        'specificity': float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive else None,
        'mcc': float(matthews_corrcoef(labels, predicted_labels))
        if len(np.unique(labels)) > 1 else None,
        'brier_score_raw': float(brier_score_loss(labels, raw_predictions)) if len(labels) else None,
        'brier_score_calibrated': float(brier_score_loss(labels, predictions)) if len(labels) else None,
        'ece_raw': expected_calibration_error(labels, raw_predictions),
        'ece_calibrated': expected_calibration_error(labels, predictions),
    }
    if len(np.unique(labels)) < 2:
        result.update({
            'status': 'skipped_one_class_or_empty_split',
            'auroc': None,
            'average_precision': None,
        })
        return result
    result.update({
        'status': 'evaluated',
        'auroc': float(roc_auc_score(labels, predictions)),
        'average_precision': float(average_precision_score(labels, predictions)),
    })
    return result


def save_prediction_artifact(
    name: str,
    loader: DataLoader,
    labels: np.ndarray,
    raw_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    threshold: float,
    prediction_dir: Path,
    calibration: dict[str, Any],
) -> Path:
    metadata = loader.dataset.metadata
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
    prediction_path = prediction_dir / f'{name.lower().replace(" ", "_")}_predictions.csv'
    table.to_csv(prediction_path, index=False)
    return prediction_path


def _save_figure(figure, destination: Path) -> None:
    figure.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    figure.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def plot_training_curves(history: dict[str, list[float]], figure_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history['epoch'], history['loss'], color='#0072B2', label='Training loss')
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Loss')
    axes[0].grid(alpha=0.25)
    axes[1].plot(history['epoch'], history['auroc'], color='#D55E00', label='Validation AUROC')
    axes[1].plot(history['epoch'], history['f1'], color='#009E73', label='Validation F1')
    axes[1].set(title='Validation performance', xlabel='Epoch', ylabel='Score', ylim=(0, 1))
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
            if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE else None
        ),
        'hidden_channels': HIDDEN_CHANNELS,
        'use_chemberta': USE_CHEMBERTA,
        'pair_representation': (
            'embedding_sum + absolute_embedding_difference + toxicity_sum + absolute_toxicity_difference'
            if USE_TOXICITY_PAIR_FEATURES
            else 'embedding_sum + absolute_embedding_difference'
        ),
        'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
        'output_heads': ['interaction_risk', 'drug_a_toxicity', 'drug_b_toxicity'],
        'total_parameters': int(total_parameters),
        'trainable_parameters': int(trainable_parameters),
    }


def _prepare_positive_edges(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    exclusions_path = audit_dir / 'invalid_smiles_exclusions.csv'
    exclusions.to_csv(exclusions_path, index=False)
    curation_summary = save_counterion_curation_candidates(exclusions, audit_dir)
    sampled = clean_positives.sample(
        n=min(DATA_CAP, len(clean_positives)), random_state=SEED
    ).reset_index(drop=True)
    summary = {
        'raw_unique_positive_pairs': int(len(positives)),
        'excluded_positive_pairs': int(len(exclusions)),
        'clean_positive_pairs': int(len(clean_positives)),
        'sampled_positive_pairs': int(len(sampled)),
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
    set_reproducibility(SEED)
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
    splits = create_splits(
        full_dataset, drug_a_col='source', drug_b_col='target', seed=SEED
    )
    split_manifest = save_split_manifests(splits, RUN_ARTIFACTS_DIR)

    graph_cache = GraphCache(FEATURE_SCHEMA)
    train_loader = build_loader(
        splits['transductive_train'], toxicity_lookup, shuffle=True, graph_cache=graph_cache
    )
    validation_loader = build_loader(
        splits['validation'], toxicity_lookup, shuffle=False, graph_cache=graph_cache
    )
    test_loaders = {
        'Transductive': build_loader(
            splits['transductive_test'], toxicity_lookup, shuffle=False, graph_cache=graph_cache
        ),
        'S1': build_loader(
            splits['s1_test'], toxicity_lookup, shuffle=False, graph_cache=graph_cache
        ),
        'S2': build_loader(
            splits['s2_test'], toxicity_lookup, shuffle=False, graph_cache=graph_cache
        ),
    }
    all_loaders = {'train': train_loader, 'validation': validation_loader, **test_loaders}
    unexpected_skips = {
        name: int(loader.dataset.skipped_count)
        for name, loader in all_loaders.items()
        if loader.dataset.skipped_count
    }
    if unexpected_skips:
        raise RuntimeError(
            'Graph validation changed between the pre-split audit and dataset construction: '
            f'{unexpected_skips}. Review the input audit before training.'
        )
    validation_labels = np.asarray([record['label'] for record in validation_loader.dataset.metadata])
    if len(validation_labels) == 0 or len(np.unique(validation_labels)) < 2:
        raise ValueError('Validation split is unusable after SMILES validation; adjust the data split.')

    model = PxDDIModel(
        in_channels=INPUT_FEATURE_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        use_chemberta=USE_CHEMBERTA,
        architecture_version=MODEL_ARCHITECTURE,
        edge_feature_dim=(
            NUM_BOND_FEATURES
            if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE else None
        ),
        use_toxicity_pair_features=USE_TOXICITY_PAIR_FEATURES,
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
        'threshold': [],
        'learning_rate': [],
    }
    best_auroc = float('-inf')
    epochs_without_improvement = 0
    stopped_early = False

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
        history['threshold'].append(threshold)
        history['learning_rate'].append(float(optimizer.param_groups[0]['lr']))
        print(
            f'Epoch {epoch}/{EPOCHS}: loss={loss:.4f}; '
            f"validation AUROC={validation_metrics['auroc']:.4f}; "
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
                        if MODEL_ARCHITECTURE == MODEL_ARCHITECTURE_EDGE_AWARE else None
                    ),
                    'use_toxicity_pair_features': USE_TOXICITY_PAIR_FEATURES,
                    'toxicity_loss_weight': TOXICITY_LOSS_WEIGHT,
                    'use_chemberta': USE_CHEMBERTA,
                    'auroc': float(best_auroc),
                    'epoch': epoch,
                    'data_cap': DATA_CAP,
                    'seed': SEED,
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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    validation_true, validation_raw_predictions = collect_predictions(model, validation_loader)
    calibration = fit_platt_calibrator(validation_true, validation_raw_predictions)
    validation_calibrated_predictions = apply_calibrator(
        validation_raw_predictions, calibration
    )
    frozen_threshold = select_validation_threshold(
        validation_true, validation_calibrated_predictions
    )
    checkpoint['threshold'] = frozen_threshold
    checkpoint['calibration'] = calibration
    checkpoint_hash = safe_checkpoint_save(checkpoint, CHECKPOINT_PATH)

    plot_training_curves(history, figure_dir)
    history_summary = save_training_history(history, RUN_ARTIFACTS_DIR)
    results: dict[str, dict[str, Any]] = {}
    for name, loader in test_loaders.items():
        labels, raw_predictions = collect_predictions(model, loader)
        calibrated_predictions = apply_calibrator(raw_predictions, calibration)
        metrics = calculate_metrics(
            labels,
            calibrated_predictions,
            frozen_threshold,
            raw_predictions=raw_predictions,
        )
        prediction_path = save_prediction_artifact(
            name,
            loader,
            labels,
            raw_predictions,
            calibrated_predictions,
            frozen_threshold,
            prediction_dir,
            calibration,
        )
        metrics['prediction_path'] = str(prediction_path)
        metrics['prediction_sha256'] = get_file_hash(prediction_path)
        metrics['skipped_invalid_smiles'] = int(loader.dataset.skipped_count)
        results[name] = metrics
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
        'model_summary': model_summary(model),
        'training_history': history_summary,
        'early_stopping': {
            'enabled': EARLY_STOPPING_PATIENCE > 0,
            'patience': EARLY_STOPPING_PATIENCE,
            'minimum_epochs': EARLY_STOPPING_MIN_EPOCHS,
            'stopped_early': stopped_early,
            'completed_epochs': len(history['epoch']),
            'best_epoch': checkpoint['epoch'],
        },
        'toxicity_bridge': toxicity_summary,
        'input_quality': input_quality_summary,
        'dataset': dataset_summary,
        'graph_cache': graph_cache.summary(),
        'split_manifest': split_manifest,
        'results': results,
        'latest_results_directory': str(LATEST_RESULTS_DIR) if PUBLISH_LATEST_RESULTS else None,
    })
    write_json(RUN_ARTIFACTS_DIR / 'run_manifest.json', manifest)
    print(f'Run artifacts saved to: {RUN_ARTIFACTS_DIR}')
    if PUBLISH_LATEST_RESULTS:
        latest_results_dir = publish_latest_results(RUN_ARTIFACTS_DIR)
        print(f'Latest results refreshed at: {latest_results_dir}')


if __name__ == '__main__':
    main()
