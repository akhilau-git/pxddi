"""Leakage-aware ChEMBL corpus selection for graph self-supervised pretraining.

This module deliberately treats ChEMBL as *unlabelled molecular structure*
data.  Before a cold-start experiment, it recreates the PxDDI split and removes
every molecule outside the transductive-training partition from the pretraining
corpus.  Consequently, no validation, S1-dev, S1-test, or S2-test molecule is
seen even without labels during pretraining.
"""

from __future__ import annotations

import hashlib
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .chembl_pipeline import iter_chembl_chemreps, sha256_file
from .prepare_twosides import graph_compatibility_reason
from .splits import build_binary_pair_dataset, create_splits, deduplicate_unordered_pairs


def canonicalize_smiles(value: object) -> str | None:
    """Return a canonical molecular identity or ``None`` for invalid SMILES."""
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def classify_smiles_for_graph(value: object) -> tuple[str | None, str | None]:
    """Canonicalise once and give any PxDDI graph-exclusion reason."""
    if not isinstance(value, str) or not value.strip():
        return None, 'missing_or_non_string_smiles'
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None, 'invalid_smiles'
    if molecule.GetNumBonds() == 0:
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            return None, 'counterion_or_inorganic_only_structure'
        return None, 'single_atom_or_disconnected_structure'
    return Chem.MolToSmiles(molecule, canonical=True), None


def stable_smiles_set_hash(smiles: set[str]) -> str:
    """Hash a set deterministically for a compact leakage audit record."""
    digest = hashlib.sha256()
    for smiles_value in sorted(smiles):
        digest.update(smiles_value.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def build_pretraining_exclusion_set(
    twosides_edges_path: str | Path,
    *,
    data_cap: int,
    split_seed: int,
    negative_sampling_strategy: str = 'degree_matched',
) -> tuple[set[str], dict[str, Any]]:
    """Recreate PxDDI's standard split and return all non-train structures.

    This is stricter than excluding only S1/S2 test molecules: it also excludes
    transductive validation/test and S1/S2 development molecules.  The model
    therefore receives direct ChEMBL structural exposure only for molecules in
    the final DDI training partition.
    """
    source = Path(twosides_edges_path)
    if not source.is_file():
        raise FileNotFoundError(f'TWOSIDES edge table was not found: {source}')
    if data_cap < 1:
        raise ValueError('data_cap must be positive.')

    edges = pd.read_csv(source, usecols=['source', 'target'], dtype=str, keep_default_na=False)
    positives = edges.copy()
    positives['label'] = 1.0
    positives = deduplicate_unordered_pairs(positives, 'source', 'target')
    source_reasons = positives['source'].map(graph_compatibility_reason)
    target_reasons = positives['target'].map(graph_compatibility_reason)
    graph_compatible = source_reasons.isna() & target_reasons.isna()
    clean_positives = positives.loc[graph_compatible, ['source', 'target', 'label']].copy()
    if clean_positives.empty:
        raise ValueError('No graph-compatible TWOSIDES positive pairs are available.')
    sampled = clean_positives.sample(
        n=min(data_cap, len(clean_positives)), random_state=split_seed
    ).reset_index(drop=True)
    full_dataset = build_binary_pair_dataset(
        sampled,
        source_col='source',
        target_col='target',
        neg_ratio=1.0,
        seed=split_seed,
        negative_sampling_strategy=negative_sampling_strategy,
    )
    splits = create_splits(full_dataset, drug_a_col='source', drug_b_col='target', seed=split_seed)
    excluded_frame = pd.concat(
        [frame for name, frame in splits.items() if name != 'transductive_train'],
        ignore_index=True,
    )
    excluded_smiles = {
        canonical
        for raw_smiles in excluded_frame[['source', 'target']].to_numpy().ravel()
        if (canonical := canonicalize_smiles(raw_smiles)) is not None
    }
    summary: dict[str, Any] = {
        'twosides_edges_path': str(source),
        'twosides_edges_sha256': sha256_file(source),
        'data_cap': data_cap,
        'split_seed': split_seed,
        'negative_sampling_strategy': negative_sampling_strategy,
        'pretraining_leakage_policy': 'exclude_all_non_train_twosides_structures_v1',
        'raw_unique_positive_pairs': int(len(positives)),
        'graph_compatible_positive_pairs': int(len(clean_positives)),
        'sampled_positive_pairs': int(len(sampled)),
        'split_rows': {name: int(len(frame)) for name, frame in splits.items()},
        'excluded_non_train_unique_structures': int(len(excluded_smiles)),
        'excluded_non_train_smiles_sha256': stable_smiles_set_hash(excluded_smiles),
    }
    return excluded_smiles, summary


def _priority(smiles: str, seed: int) -> int:
    """Create a source-order-independent stable sampling priority."""
    value = f'{seed}\0{smiles}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], byteorder='big')


def select_chembl_pretraining_corpus(
    chemreps_path: str | Path,
    *,
    excluded_smiles: set[str],
    maximum_molecules: int,
    seed: int,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose a deterministic, graph-compatible ChEMBL corpus without leakage.

    A bounded priority reservoir avoids loading 2.9M molecular rows into memory
    and produces the same selected structures for the same source hash, seed,
    exclusion set, and maximum size.
    """
    if maximum_molecules < 1:
        raise ValueError('maximum_molecules must be positive.')
    if chunksize < 1:
        raise ValueError('chunksize must be positive.')

    heap: list[tuple[int, str, str]] = []
    selected_smiles: set[str] = set()
    total_rows = valid_smiles_rows = graph_compatible_rows = excluded_rows = 0
    invalid_or_unsupported_rows = 0
    for chunk in iter_chembl_chemreps(chemreps_path, chunksize=chunksize):
        total_rows += len(chunk)
        for row in chunk[['chembl_id', 'canonical_smiles']].itertuples(index=False):
            canonical, exclusion_reason = classify_smiles_for_graph(row.canonical_smiles)
            if canonical is None:
                invalid_or_unsupported_rows += 1
                continue
            valid_smiles_rows += 1
            graph_compatible_rows += 1
            if canonical in excluded_smiles:
                excluded_rows += 1
                continue
            if canonical in selected_smiles:
                continue
            priority = _priority(canonical, seed)
            # Store negative priority so heap root is the current worst (largest)
            # retained priority.  A lower priority is selected deterministically.
            candidate = (-priority, canonical, str(row.chembl_id))
            if len(heap) < maximum_molecules:
                heapq.heappush(heap, candidate)
                selected_smiles.add(canonical)
            elif candidate[0] > heap[0][0]:
                removed = heapq.heapreplace(heap, candidate)
                selected_smiles.remove(removed[1])
                selected_smiles.add(canonical)

    ordered = sorted(
        ((-priority, canonical, chembl_id) for priority, canonical, chembl_id in heap),
        key=lambda item: (item[0], item[1], item[2]),
    )
    corpus = pd.DataFrame(ordered, columns=['sampling_priority', 'canonical_smiles', 'chembl_id'])
    summary: dict[str, Any] = {
        'chemreps_path': str(Path(chemreps_path)),
        'chemreps_sha256': sha256_file(chemreps_path),
        'selection_seed': seed,
        'maximum_molecules_requested': maximum_molecules,
        'source_rows': total_rows,
        'valid_smiles_rows': valid_smiles_rows,
        'graph_compatible_rows': graph_compatible_rows,
        'invalid_or_unsupported_rows': invalid_or_unsupported_rows,
        'rows_excluded_for_twosides_non_train_leakage': excluded_rows,
        'selected_unique_molecules': int(len(corpus)),
        'selected_smiles_sha256': stable_smiles_set_hash(set(corpus['canonical_smiles'])),
    }
    return corpus, summary
