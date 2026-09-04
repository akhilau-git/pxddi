"""Tests for strict external-dataset provenance requirements."""

import pytest
import pandas as pd
import torch

from src.data_prep.molecular_motifs import MOTIF_FEATURE_DIM
from src.data_prep.prepare_twosides import FEATURE_SCHEMA_RICH
from src.models.ddi_model import PxDDIModel
from src.training.evaluate_external_dataset import (
    build_external_records,
    file_sha256,
    find_development_pair_overlaps,
    load_verified_development_pair_keys,
    load_trained_model,
    memory_features_for_pairs,
    validate_external_metadata,
)


def test_external_evaluation_requires_dataset_provenance():
    with pytest.raises(ValueError, match='incomplete'):
        validate_external_metadata({'dataset_name': 'Example'})


def test_external_evaluation_accepts_complete_provenance():
    validate_external_metadata({
        'dataset_name': 'Example external DDI set',
        'source_url_or_doi': 'https://example.org/dataset',
        'data_version_or_date': '2026-01-01',
        'label_definition': 'Binary reported interaction label',
        'split_definition': 'External held-out dataset',
    })


def test_external_evaluation_loads_the_checkpoint_weights(tmp_path):
    """External evaluation must not score records with a random model."""
    source_model = PxDDIModel(in_channels=13, hidden_channels=8).eval()
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.fill_(0.125)

    checkpoint_path = tmp_path / 'trained_model.pt'
    torch.save(
        {
            'model_state_dict': source_model.state_dict(),
            'in_channels': 13,
            'hidden_channels': 8,
            'use_chemberta': False,
        },
        checkpoint_path,
    )

    loaded_model, _ = load_trained_model(checkpoint_path, torch.device('cpu'))

    assert loaded_model.training is False
    for name, expected in source_model.state_dict().items():
        assert torch.equal(loaded_model.state_dict()[name], expected), name


def test_external_evaluation_builds_motif_graphs_for_a_motif_candidate():
    records, excluded = build_external_records(
        pd.DataFrame({
            'source': ['CCO'],
            'target': ['CCN'],
            'label': [1],
        }),
        FEATURE_SCHEMA_RICH,
        include_motif_features=True,
    )

    assert excluded.empty
    assert len(records) == 1
    assert records[0][0].motif_features.shape == (1, MOTIF_FEATURE_DIM)
    assert records[0][1].motif_features.shape == (1, MOTIF_FEATURE_DIM)


def test_external_evaluation_rejects_any_development_pair_overlap_using_verified_splits(tmp_path):
    training_split = tmp_path / 'transductive_train.csv'
    pd.DataFrame({
        'source': ['CCO'], 'target': ['CCN'], 'label': [1.0],
    }).to_csv(training_split, index=False)
    checkpoint = {
        'external_overlap_development_splits': {
            'transductive_train': {
                'split_name': 'transductive_train',
                'path': str(training_split),
                'sha256': file_sha256(training_split),
                'rows': 1,
            },
        }
    }
    keys, summary = load_verified_development_pair_keys(checkpoint)
    overlaps = find_development_pair_overlaps(
        pd.DataFrame({
            'source': ['NCC', 'CCC'],
            'target': ['OCC', 'CCCl'],
            'label': [1, 0],
        }),
        keys,
    )

    assert summary['development_split_count'] == 1
    assert summary['unique_canonical_pair_count'] == 1
    assert overlaps.to_dict(orient='records') == [{'dataset_row_index': 0}]


def test_external_evaluation_supports_fingerprint_and_memory_candidates():
    records, excluded = build_external_records(
        pd.DataFrame({
            'source': ['CCO'], 'target': ['CCN'], 'label': [1],
        }),
        FEATURE_SCHEMA_RICH,
        include_fingerprint_features=True,
    )
    assert excluded.empty
    assert records[0][0].fingerprint_features.shape == (1, 1024)
    assert records[0][1].fingerprint_features.shape == (1, 1024)

    from src.models.neighbor_memory import AuditableNeighborMemory

    memory = AuditableNeighborMemory(k_neighbors=1)
    memory.fit(['CCO'], ['CCN'], [1.0])
    features = memory_features_for_pairs(
        memory, ['CCC'], ['CCCl'], torch.device('cpu')
    )
    assert features.shape == (1, 3)
