"""Strict external-dataset evaluation for a reviewed PxDDI checkpoint.

This script intentionally requires dataset provenance metadata. It does not
download, invent, or relabel an external validation dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.build_dataloader import parse_binary_label
from src.data_prep.prepare_twosides import FEATURE_SCHEMA_LEGACY, smiles_to_graph
from src.data_prep.pubchem_bridge import canonicalize
from src.models.calibration import apply_calibrator
from src.models.ddi_model import (
    architecture_requires_fingerprint_features,
    architecture_requires_motif_features,
    model_from_checkpoint,
)
from src.models.neighbor_memory import AuditableNeighborMemory
from src.evaluation.ddi_metrics import bootstrap_confidence_intervals, calculate_binary_metrics


REQUIRED_METADATA_FIELDS = {
    'dataset_name',
    'source_url_or_doi',
    'data_version_or_date',
    'label_definition',
    'split_definition',
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_external_metadata(metadata: dict) -> None:
    missing = REQUIRED_METADATA_FIELDS.difference(metadata)
    empty = [field for field in REQUIRED_METADATA_FIELDS if not str(metadata.get(field, '')).strip()]
    if missing or empty:
        raise ValueError(
            'External evaluation metadata is incomplete. Missing or empty fields: '
            f'{sorted(set(missing).union(empty))}.'
        )


def _collate(records):
    graphs_a, graphs_b, labels, sources, targets = zip(*records)
    return (
        Batch.from_data_list(graphs_a),
        Batch.from_data_list(graphs_b),
        torch.tensor(labels, dtype=torch.float),
        list(sources),
        list(targets),
    )


def canonical_unordered_pair(source: object, target: object) -> tuple[str, str] | None:
    """Return one chemical-identity key for a pair, or ``None`` when invalid."""
    canonical_source = canonicalize(str(source))
    canonical_target = canonicalize(str(target))
    if canonical_source is None or canonical_target is None:
        return None
    return tuple(sorted((canonical_source, canonical_target)))


def load_verified_development_pair_keys(checkpoint: dict) -> tuple[set[tuple[str, str]], dict]:
    """Load hashed development splits and reject their pairs from external data."""
    artifacts = checkpoint.get('external_overlap_development_splits')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(
            'Checkpoint lacks verified development-split artifacts for external-overlap '
            'screening. Retrain with the current audited pipeline.'
        )
    keys: set[tuple[str, str]] = set()
    split_summaries: list[dict[str, object]] = []
    required = ('split_name', 'path', 'sha256', 'rows')
    for split_name, artifact in sorted(artifacts.items()):
        if not isinstance(artifact, dict):
            raise ValueError(
                f'Checkpoint development split {split_name!r} has invalid overlap metadata.'
            )
        missing = [
            key
            for key in required
            if key not in artifact
            or artifact[key] is None
            or (key != 'rows' and not str(artifact[key]).strip())
        ]
        if missing:
            raise ValueError(
                'Checkpoint development-split artifact is incomplete for external-overlap '
                f'screening; split={split_name!r}, missing: {missing}.'
            )
        if artifact['split_name'] != split_name:
            raise ValueError(
                'Checkpoint development-split artifact name disagrees with its manifest key: '
                f'{split_name!r}.'
            )
        path = Path(artifact['path'])
        if not path.is_file():
            raise FileNotFoundError(
                'Checkpoint development split is unavailable for external-overlap screening: '
                f'{path}. Restore the original run artifact directory.'
            )
        if file_sha256(path) != artifact['sha256']:
            raise ValueError(
                'Checkpoint development-split hash does not match its artifact; refusing '
                'to make an external-validation claim.'
            )
        development_pairs = pd.read_csv(path)
        required_columns = {'source', 'target'}
        missing_columns = required_columns.difference(development_pairs.columns)
        if missing_columns:
            raise ValueError(
                'Checkpoint development split lacks required pair columns: '
                f'{sorted(missing_columns)}.'
            )
        if len(development_pairs) != int(artifact['rows']):
            raise ValueError(
                'Checkpoint development-split row count does not match its provenance artifact.'
            )
        split_keys: set[tuple[str, str]] = set()
        for row in development_pairs[['source', 'target']].itertuples(index=False):
            key = canonical_unordered_pair(row.source, row.target)
            if key is None:
                raise ValueError(
                    'Checkpoint development split contains a graph-incompatible pair; '
                    'its external-overlap screen is not trustworthy.'
                )
            keys.add(key)
            split_keys.add(key)
        split_summaries.append({
            'split_name': split_name,
            'sha256': artifact['sha256'],
            'rows': int(artifact['rows']),
            'unique_canonical_pair_count': len(split_keys),
        })
    if not keys:
        raise ValueError('Checkpoint development splits have no usable pair keys.')
    return keys, {
        'development_split_count': len(split_summaries),
        'development_splits': split_summaries,
        'unique_canonical_pair_count': len(keys),
    }


def find_development_pair_overlaps(
    dataframe: pd.DataFrame, training_pair_keys: set[tuple[str, str]]
) -> pd.DataFrame:
    """Identify exact development pairs that would invalidate external evaluation."""
    required_columns = {'source', 'target'}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f'External dataset is missing required columns: {sorted(missing)}.')
    overlaps: list[dict[str, object]] = []
    for row_index, row in dataframe[['source', 'target']].iterrows():
        pair_key = canonical_unordered_pair(row['source'], row['target'])
        if pair_key in training_pair_keys:
            overlaps.append({'dataset_row_index': int(row_index)})
    return pd.DataFrame(overlaps, columns=['dataset_row_index'])


def attach_fingerprint_features(graph, smiles: str) -> None:
    """Attach the graph-fusion input expected by fingerprint candidate checkpoints."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Cannot make a fingerprint from an invalid external SMILES value.')
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=True
    )
    fingerprint = generator.GetFingerprint(molecule)
    values = np.zeros((1024,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    graph.fingerprint_features = torch.from_numpy(values).unsqueeze(0)


def build_external_records(
    dataframe: pd.DataFrame,
    feature_schema: str,
    include_motif_features: bool = False,
    include_fingerprint_features: bool = False,
):
    required = {'source', 'target', 'label'}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f'External dataset is missing required columns: {sorted(missing)}.')
    records, excluded = [], []
    for index, row in dataframe.iterrows():
        source, target = row['source'], row['target']
        graph_a = smiles_to_graph(
            source,
            feature_schema=feature_schema,
            include_motif_features=include_motif_features,
        )
        graph_b = smiles_to_graph(
            target,
            feature_schema=feature_schema,
            include_motif_features=include_motif_features,
        )
        if graph_a is None or graph_b is None:
            excluded.append({'dataset_row_index': index, 'source': source, 'target': target, 'reason': 'graph_incompatible'})
            continue
        if include_fingerprint_features:
            attach_fingerprint_features(graph_a, source)
            attach_fingerprint_features(graph_b, target)
        records.append((
            graph_a,
            graph_b,
            parse_binary_label(row['label'], 'label'),
            str(source),
            str(target),
        ))
    return records, pd.DataFrame(excluded, columns=['dataset_row_index', 'source', 'target', 'reason'])


def load_trained_model(checkpoint_path: Path, device: torch.device):
    """Load both the model architecture and its saved learned parameters."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_neighbor_memory(checkpoint: dict) -> AuditableNeighborMemory | None:
    """Restore required non-parametric state for memory-fusion candidates."""
    if not checkpoint.get('use_neighbor_memory', False):
        return None
    state = checkpoint.get('neighbor_memory_state')
    if not isinstance(state, dict):
        raise ValueError(
            'Checkpoint requires neighbor memory but has no auditable memory state.'
        )
    return AuditableNeighborMemory.from_state(state)


def memory_features_for_pairs(
    neighbor_memory: AuditableNeighborMemory,
    sources: list[str],
    targets: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Build one non-leaky memory-feature row for each external pair."""
    scores = [
        neighbor_memory.score_pair_memory(source, target)
        for source, target in zip(sources, targets)
    ]
    return torch.tensor(
        [
            [
                score['neighbor_density'],
                score['max_support'],
                score['structural_confidence'],
            ]
            for score in scores
        ],
        dtype=torch.float,
        device=device,
    )


def main() -> None:
    data_path = Path(os.environ['PXDDI_EXTERNAL_EDGES'])
    metadata_path = Path(os.environ['PXDDI_EXTERNAL_METADATA'])
    raw_checkpoint_path = os.environ.get('PXDDI_EXTERNAL_CHECKPOINT_PATH')
    if not raw_checkpoint_path:
        raise ValueError(
            'Set PXDDI_EXTERNAL_CHECKPOINT_PATH to the one reviewed checkpoint '
            'you intend to evaluate.'
        )
    checkpoint_path = Path(raw_checkpoint_path)
    output_dir = Path(os.environ.get(
        'PXDDI_EXTERNAL_ARTIFACTS_DIR',
        '/content/drive/MyDrive/pxddi-data/external_evaluations',
    )) / datetime.now(timezone.utc).strftime('external_%Y%m%dT%H%M%SZ')
    bootstrap_resamples = int(os.environ.get('PXDDI_EXTERNAL_BOOTSTRAP_RESAMPLES', '1000'))
    if bootstrap_resamples < 0:
        raise ValueError('PXDDI_EXTERNAL_BOOTSTRAP_RESAMPLES must be zero or positive.')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    validate_external_metadata(metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_trained_model(checkpoint_path, device)
    development_pair_keys, development_split_summary = load_verified_development_pair_keys(checkpoint)
    external_dataframe = pd.read_csv(data_path)
    overlaps = find_development_pair_overlaps(external_dataframe, development_pair_keys)
    if not overlaps.empty:
        raise ValueError(
            'External dataset contains '
            f'{len(overlaps)} exact canonical pair(s) from the checkpoint development splits. '
            'Remove the overlap before evaluating.'
        )
    feature_schema = checkpoint.get('feature_schema', FEATURE_SCHEMA_LEGACY)
    include_motif_features = architecture_requires_motif_features(
        checkpoint.get('architecture_version', 'legacy_gat_v1')
    )
    include_fingerprint_features = architecture_requires_fingerprint_features(
        checkpoint.get('architecture_version', 'legacy_gat_v1')
    )
    neighbor_memory = load_neighbor_memory(checkpoint)

    records, excluded = build_external_records(
        external_dataframe,
        feature_schema,
        include_motif_features,
        include_fingerprint_features,
    )
    if not records:
        raise ValueError('No graph-compatible rows remain in the external dataset.')
    output_dir.mkdir(parents=True, exist_ok=False)
    loader = DataLoader(records, batch_size=128, shuffle=False, collate_fn=_collate)
    raw_scores, labels = [], []
    with torch.no_grad():
        for graph_a, graph_b, batch_labels, sources, targets in loader:
            memory_features = None
            if neighbor_memory is not None:
                memory_features = memory_features_for_pairs(
                    neighbor_memory, sources, targets, device
                )
            risk, _, _ = model(
                graph_a.to(device),
                graph_b.to(device),
                memory_features=memory_features,
            )
            raw_scores.extend(torch.sigmoid(risk).cpu().numpy())
            labels.extend(batch_labels.numpy())
    labels_array = np.asarray(labels, dtype=int)
    raw_array = np.asarray(raw_scores, dtype=float)
    final_scores = apply_calibrator(raw_array, checkpoint.get('calibration'))
    threshold = float(checkpoint.get('threshold', 0.5))
    predicted_labels = (final_scores >= threshold).astype(int)
    metrics = calculate_binary_metrics(
        labels_array, final_scores, threshold, raw_predictions=raw_array
    )
    metrics['test_set_bootstrap_95ci'] = bootstrap_confidence_intervals(
        labels_array,
        final_scores,
        threshold,
        raw_predictions=raw_array,
        resamples=bootstrap_resamples,
        seed=0,
    )
    metrics['excluded_graph_incompatible_rows'] = int(len(excluded))
    excluded.to_csv(output_dir / 'graph_incompatible_exclusions.csv', index=False)
    pd.DataFrame({
        'label': labels_array,
        'raw_prediction_score': raw_array,
        'calibrated_prediction_score': final_scores,
        'predicted_label': predicted_labels,
    }).to_csv(output_dir / 'predictions.csv', index=False)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'external_dataset_metadata': metadata,
        'external_dataset_sha256': file_sha256(data_path),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': file_sha256(checkpoint_path),
        'checkpoint_calibration': checkpoint.get('calibration'),
        'development_overlap_screen': {
            'status': 'passed_no_exact_canonical_pair_overlap',
            **development_split_summary,
            'external_rows_checked': int(len(external_dataframe)),
            'overlapping_rows': 0,
        },
        'metrics': metrics,
        'warning': (
            'This report measures the supplied external dataset only. Its validity depends on '
            'the documented source, label definition, and absence of training-data overlap.'
        ),
    }
    (output_dir / 'external_evaluation.json').write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding='utf-8'
    )
    print(f'External evaluation saved to: {output_dir}')


if __name__ == '__main__':
    main()
