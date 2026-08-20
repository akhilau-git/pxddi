"""Scaffold-disjoint DDI splitting for an explicitly separate evaluation run.

This module is intentionally not mixed into PxDDI's standard transductive/S1/S2
run.  A scaffold-disjoint result must train only on the scaffold-training
partition, otherwise the comparison would leak the held-out chemical framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .splits import deduplicate_unordered_pairs


SCAFFOLD_SPLIT_METHOD = 'murcko_scaffold_disjoint_pair_partition_v1'


def murcko_scaffold_key(smiles: str) -> str:
    """Return a deterministic scaffold key without collapsing acyclic drugs.

    Bemis-Murcko scaffolds are empty for acyclic molecules.  Treating every
    empty scaffold as one group would make most common small molecules unusable
    and would not distinguish their structures.  We therefore use canonical,
    isomeric SMILES as an explicit acyclic fallback.  This is conservative: an
    acyclic molecule cannot appear under the same fallback key in two splits.
    """
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f'Cannot derive a scaffold from invalid SMILES: {smiles!r}.')
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() > 0:
        return f'murcko:{Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)}'
    return f'acyclic_canonical:{Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)}'


def _partition_scaffolds(
    keys: list[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie strictly between zero and one.')
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must lie strictly between zero and one.')
    if validation_fraction + test_fraction >= 1:
        raise ValueError('validation_fraction + test_fraction must be less than one.')
    if len(keys) < 3:
        raise ValueError('At least three unique scaffolds are required.')
    ordered = np.asarray(sorted(keys), dtype=object)
    shuffled = np.random.default_rng(seed).permutation(ordered)
    test_count = max(1, int(round(test_fraction * len(shuffled))))
    validation_count = max(1, int(round(validation_fraction * len(shuffled))))
    if test_count + validation_count >= len(shuffled):
        raise ValueError('Not enough scaffold groups remain for a training partition.')
    assignment = {str(key): 'scaffold_test' for key in shuffled[:test_count]}
    assignment.update({
        str(key): 'scaffold_validation'
        for key in shuffled[test_count:test_count + validation_count]
    })
    assignment.update({
        str(key): 'scaffold_train'
        for key in shuffled[test_count + validation_count:]
    })
    return assignment


def create_scaffold_disjoint_splits(
    dataframe: pd.DataFrame,
    drug_a_col: str = 'source',
    drug_b_col: str = 'target',
    label_col: str = 'label',
    validation_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build train/validation/test pairs with no scaffold shared across roles.

    A pair is retained only if *both* drug scaffold keys belong to one role.
    Cross-role pairs are deliberately excluded and recorded: assigning them to
    training or test would break the scaffold-disjoint claim.  The function
    returns split tables plus an audit summary that must be saved with results.
    """
    deduplicated = deduplicate_unordered_pairs(
        dataframe, drug_a_col, drug_b_col, label_col
    ).copy()
    if deduplicated.empty:
        raise ValueError('Cannot create scaffold splits from an empty dataset.')
    scaffold_cache: dict[str, str] = {}

    def scaffold_for(smiles: str) -> str:
        value = str(smiles)
        if value not in scaffold_cache:
            scaffold_cache[value] = murcko_scaffold_key(value)
        return scaffold_cache[value]

    deduplicated['_source_scaffold'] = deduplicated[drug_a_col].map(scaffold_for)
    deduplicated['_target_scaffold'] = deduplicated[drug_b_col].map(scaffold_for)
    assignment = _partition_scaffolds(
        sorted(set(scaffold_cache.values())), validation_fraction, test_fraction, seed
    )
    deduplicated['_source_partition'] = deduplicated['_source_scaffold'].map(assignment)
    deduplicated['_target_partition'] = deduplicated['_target_scaffold'].map(assignment)
    same_partition = (
        deduplicated['_source_partition'] == deduplicated['_target_partition']
    )
    splits: dict[str, pd.DataFrame] = {}
    for name in ('scaffold_train', 'scaffold_validation', 'scaffold_test'):
        selected = deduplicated[
            same_partition & (deduplicated['_source_partition'] == name)
        ].drop(columns=['_source_partition', '_target_partition'])
        splits[name] = selected.reset_index(drop=True)
    excluded = deduplicated[~same_partition].copy().reset_index(drop=True)
    splits['scaffold_cross_partition_excluded'] = excluded

    empty_required = [
        name for name in ('scaffold_train', 'scaffold_validation', 'scaffold_test')
        if splits[name].empty
    ]
    if empty_required:
        counts = {name: int(len(frame)) for name, frame in splits.items()}
        raise ValueError(
            'Scaffold partitioning left a required role empty. Adjust the seed or '
            f'fractions; counts={counts}; empty={empty_required}.'
        )

    train_scaffolds = {
        *splits['scaffold_train']['_source_scaffold'],
        *splits['scaffold_train']['_target_scaffold'],
    }
    validation_scaffolds = {
        *splits['scaffold_validation']['_source_scaffold'],
        *splits['scaffold_validation']['_target_scaffold'],
    }
    test_scaffolds = {
        *splits['scaffold_test']['_source_scaffold'],
        *splits['scaffold_test']['_target_scaffold'],
    }
    if train_scaffolds & validation_scaffolds or train_scaffolds & test_scaffolds or validation_scaffolds & test_scaffolds:
        raise RuntimeError('Scaffold partition leakage detected while constructing splits.')
    audit = {
        'method': SCAFFOLD_SPLIT_METHOD,
        'seed': int(seed),
        'validation_fraction_target': float(validation_fraction),
        'test_fraction_target': float(test_fraction),
        'unique_drug_structures': int(len(scaffold_cache)),
        'unique_scaffold_groups': int(len(assignment)),
        'scaffold_group_counts': {
            partition: int(sum(value == partition for value in assignment.values()))
            for partition in ('scaffold_train', 'scaffold_validation', 'scaffold_test')
        },
        'pair_row_counts': {name: int(len(frame)) for name, frame in splits.items()},
        'cross_partition_pairs_excluded': int(len(excluded)),
        'acyclic_fallback': 'canonical_isomeric_smiles',
        'interpretation_warning': (
            'Pairs spanning scaffold roles are intentionally excluded. This is a '
            'more chemically stringent protocol than random split evaluation, but '
            'it remains internal benchmark evidence and not clinical validation.'
        ),
    }
    return splits, audit
