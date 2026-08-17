"""Tests for binary DDI labels in the reusable data-loader module."""

import pandas as pd
import pytest

from src.data_prep.build_dataloader import DDIPairDataset, parse_binary_label


def test_parse_binary_label_preserves_zero_and_one():
    assert parse_binary_label(0, 'label') == 0.0
    assert parse_binary_label(1, 'label') == 1.0
    assert parse_binary_label('0', 'label') == 0.0
    assert parse_binary_label('true', 'label') == 1.0


@pytest.mark.parametrize('value', [None, '', 'interaction', -1, 2, 0.5])
def test_parse_binary_label_rejects_missing_or_non_binary_values(value):
    with pytest.raises(ValueError):
        parse_binary_label(value, 'label')


def test_dataset_preserves_a_negative_label():
    dataframe = pd.DataFrame({
        'drug1_smiles': ['CCO', 'CCN'],
        'drug2_smiles': ['CCN', 'CCO'],
        'label': [0, 1],
    })

    dataset = DDIPairDataset(
        dataframe,
        'drug1_smiles',
        'drug2_smiles',
        'label',
    )

    assert [record[-1] for record in dataset.records] == [0.0, 1.0]
