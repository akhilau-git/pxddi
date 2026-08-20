"""Tests for the separate scaffold-disjoint PxDDI protocol."""

import pandas as pd
import pytest

from src.data_prep.scaffold_splits import (
    create_scaffold_disjoint_splits,
    murcko_scaffold_key,
)


def _complete_pair_frame(smiles):
    rows = []
    for left_index, left in enumerate(smiles):
        for right in smiles[left_index + 1:]:
            rows.append({'source': left, 'target': right, 'label': float((left_index + len(right)) % 2)})
    return pd.DataFrame(rows)


def test_murcko_key_distinguishes_acyclic_molecules_and_keeps_ring_scaffold():
    assert murcko_scaffold_key('CCO').startswith('acyclic_canonical:')
    assert murcko_scaffold_key('CCN').startswith('acyclic_canonical:')
    assert murcko_scaffold_key('CCO') != murcko_scaffold_key('CCN')
    assert murcko_scaffold_key('c1ccccc1O').startswith('murcko:')


def test_scaffold_disjoint_protocol_has_no_scaffold_leakage():
    smiles = [
        'c1ccccc1O', 'c1ccncc1', 'C1CCCCC1', 'C1CCNC1',
        'CCO', 'CCN', 'CCC', 'CCCl', 'CCBr', 'COC',
    ]
    splits, audit = create_scaffold_disjoint_splits(
        _complete_pair_frame(smiles), validation_fraction=0.2, test_fraction=0.3, seed=17
    )

    train_scaffolds = set(splits['scaffold_train']['_source_scaffold']) | set(splits['scaffold_train']['_target_scaffold'])
    validation_scaffolds = set(splits['scaffold_validation']['_source_scaffold']) | set(splits['scaffold_validation']['_target_scaffold'])
    test_scaffolds = set(splits['scaffold_test']['_source_scaffold']) | set(splits['scaffold_test']['_target_scaffold'])
    assert train_scaffolds.isdisjoint(validation_scaffolds)
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert validation_scaffolds.isdisjoint(test_scaffolds)
    assert audit['cross_partition_pairs_excluded'] == len(splits['scaffold_cross_partition_excluded'])
    assert audit['method'] == 'murcko_scaffold_disjoint_pair_partition_v1'


def test_scaffold_protocol_refuses_invalid_smiles_instead_of_silently_grouping_it():
    frame = pd.DataFrame({
        'source': ['CCO', 'not-a-smiles', 'CCN'],
        'target': ['CCN', 'CCO', 'CCC'],
        'label': [1.0, 0.0, 1.0],
    })
    with pytest.raises(ValueError, match='invalid SMILES'):
        create_scaffold_disjoint_splits(frame)
