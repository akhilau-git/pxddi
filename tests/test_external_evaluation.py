"""Tests for strict external-dataset provenance requirements."""

import pytest

from src.training.evaluate_external_dataset import validate_external_metadata


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
