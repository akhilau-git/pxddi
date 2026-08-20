"""Tests for strict external-dataset provenance requirements."""

import pytest
import pandas as pd
import torch

from src.data_prep.molecular_motifs import MOTIF_FEATURE_DIM
from src.data_prep.prepare_twosides import FEATURE_SCHEMA_RICH
from src.models.ddi_model import PxDDIModel
from src.training.evaluate_external_dataset import (
    build_external_records,
    load_trained_model,
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
