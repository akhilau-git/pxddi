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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.data_prep.build_dataloader import parse_binary_label
from src.data_prep.prepare_twosides import FEATURE_SCHEMA_LEGACY, smiles_to_graph
from src.models.calibration import apply_calibrator, expected_calibration_error
from src.models.ddi_model import architecture_requires_motif_features, model_from_checkpoint


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
    graphs_a, graphs_b, labels = zip(*records)
    return Batch.from_data_list(graphs_a), Batch.from_data_list(graphs_b), torch.tensor(labels, dtype=torch.float)


def build_external_records(
    dataframe: pd.DataFrame,
    feature_schema: str,
    include_motif_features: bool = False,
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
        records.append((graph_a, graph_b, parse_binary_label(row['label'], 'label')))
    return records, pd.DataFrame(excluded, columns=['dataset_row_index', 'source', 'target', 'reason'])


def load_trained_model(checkpoint_path: Path, device: torch.device):
    """Load both the model architecture and its saved learned parameters."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


def main() -> None:
    data_path = Path(os.environ['PXDDI_EXTERNAL_EDGES'])
    metadata_path = Path(os.environ['PXDDI_EXTERNAL_METADATA'])
    checkpoint_path = Path(os.environ.get(
        'PXDDI_EXTERNAL_CHECKPOINT_PATH',
        '/content/drive/MyDrive/pxddi-data/checkpoints/pxddi_model.pt',
    ))
    output_dir = Path(os.environ.get(
        'PXDDI_EXTERNAL_ARTIFACTS_DIR',
        '/content/drive/MyDrive/pxddi-data/external_evaluations',
    )) / datetime.now(timezone.utc).strftime('external_%Y%m%dT%H%M%SZ')
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    validate_external_metadata(metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_trained_model(checkpoint_path, device)
    feature_schema = checkpoint.get('feature_schema', FEATURE_SCHEMA_LEGACY)
    include_motif_features = architecture_requires_motif_features(
        checkpoint.get('architecture_version', 'legacy_gat_v1')
    )

    records, excluded = build_external_records(
        pd.read_csv(data_path), feature_schema, include_motif_features
    )
    if not records:
        raise ValueError('No graph-compatible rows remain in the external dataset.')
    loader = DataLoader(records, batch_size=128, shuffle=False, collate_fn=_collate)
    raw_scores, labels = [], []
    with torch.no_grad():
        for graph_a, graph_b, batch_labels in loader:
            risk, _, _ = model(graph_a.to(device), graph_b.to(device))
            raw_scores.extend(torch.sigmoid(risk).cpu().numpy())
            labels.extend(batch_labels.numpy())
    labels_array = np.asarray(labels, dtype=int)
    raw_array = np.asarray(raw_scores, dtype=float)
    final_scores = apply_calibrator(raw_array, checkpoint.get('calibration'))
    threshold = float(checkpoint.get('threshold', 0.5))
    predicted_labels = (final_scores >= threshold).astype(int)
    matrix = confusion_matrix(labels_array, predicted_labels, labels=[0, 1]).tolist()
    metrics = {
        'sample_count': int(len(labels_array)),
        'positive_count': int(labels_array.sum()),
        'negative_count': int((labels_array == 0).sum()),
        'auroc': float(roc_auc_score(labels_array, final_scores)) if len(np.unique(labels_array)) == 2 else None,
        'average_precision': float(average_precision_score(labels_array, final_scores)) if len(np.unique(labels_array)) == 2 else None,
        'brier_score_raw': float(brier_score_loss(labels_array, raw_array)),
        'brier_score_calibrated': float(brier_score_loss(labels_array, final_scores)),
        'ece_raw': expected_calibration_error(labels_array, raw_array),
        'ece_calibrated': expected_calibration_error(labels_array, final_scores),
        'threshold': threshold,
        'confusion_matrix': matrix,
        'excluded_graph_incompatible_rows': int(len(excluded)),
    }
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
