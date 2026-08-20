"""Compare candidate occlusion explanations across independently trained seeds."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


EXPLANATION_STABILITY_METHOD = 'cross_seed_top_k_jaccard_v1'


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left.union(right)
    return 1.0 if not union else len(left.intersection(right)) / len(union)


def _example_key(example: dict[str, Any]) -> tuple[str, str, int]:
    return (str(example['source']), str(example['target']), int(example['label']))


def _top_atom_set(example: dict[str, Any], drug_key: str) -> set[int]:
    return {
        int(entry['atom_index'])
        for entry in example['explanation'][drug_key]['top_atom_occlusions']
    }


def _top_motif_set(example: dict[str, Any], drug_key: str) -> set[str]:
    return {
        str(entry['motif_name'])
        for entry in example['explanation'][drug_key]['top_motif_occlusions']
        if float(entry.get('input_count', 0)) > 0
    }


def _top_cross_motif_pair_set(example: dict[str, Any]) -> set[tuple[str, str]]:
    associations = example['explanation'].get('cross_drug_attention_associations', {})
    configured = associations.get('configured_motif_associations', {})
    return {
        (str(entry['source_motif']), str(entry['target_motif']))
        for entry in configured.get('drug_a_to_drug_b', [])
    }


def _index_examples(artifact: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed = {}
    for split in artifact.get('splits', {}).values():
        for example in split.get('examples', []):
            if 'explanation' not in example:
                continue
            key = _example_key(example)
            if key in indexed:
                raise ValueError(f'Explanation artifact contains a duplicate example: {key}.')
            indexed[key] = example
    return indexed


def compare_explanation_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """Report cross-seed local-explanation agreement for shared examples only."""
    if len(artifacts) < 2:
        raise ValueError('At least two explanation artifacts are required for stability analysis.')
    methods = {artifact.get('method') for artifact in artifacts}
    if methods != {'single_component_occlusion_v1'}:
        raise ValueError('All artifacts must use the same supported occlusion method.')
    architectures = {artifact.get('model_architecture') for artifact in artifacts}
    if len(architectures) != 1:
        raise ValueError('Explanation stability requires one shared model architecture.')
    model_seeds = [artifact.get('model_seed') for artifact in artifacts]
    split_seeds = {artifact.get('split_seed') for artifact in artifacts}
    if any(seed is None for seed in model_seeds):
        raise ValueError('Each artifact must record a model_seed for cross-seed stability.')
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError('Cross-seed stability requires distinct model_seed values.')
    if len(split_seeds) != 1 or None in split_seeds:
        raise ValueError('Cross-seed stability requires one shared recorded split_seed.')
    indexed = [_index_examples(artifact) for artifact in artifacts]
    shared_keys = sorted(set.intersection(*(set(items) for items in indexed)))
    comparison_rows = []
    for key in shared_keys:
        examples = [items[key] for items in indexed]
        for left_index, right_index in combinations(range(len(examples)), 2):
            left, right = examples[left_index], examples[right_index]
            comparison_rows.append({
                'source': key[0],
                'target': key[1],
                'label': key[2],
                'member_left_index': left_index,
                'member_right_index': right_index,
                'drug_a_top_atom_jaccard': _jaccard(
                    _top_atom_set(left, 'drug_a'), _top_atom_set(right, 'drug_a')
                ),
                'drug_b_top_atom_jaccard': _jaccard(
                    _top_atom_set(left, 'drug_b'), _top_atom_set(right, 'drug_b')
                ),
                'drug_a_top_motif_jaccard': _jaccard(
                    _top_motif_set(left, 'drug_a'), _top_motif_set(right, 'drug_a')
                ),
                'drug_b_top_motif_jaccard': _jaccard(
                    _top_motif_set(left, 'drug_b'), _top_motif_set(right, 'drug_b')
                ),
                'cross_motif_pair_jaccard': _jaccard(
                    _top_cross_motif_pair_set(left), _top_cross_motif_pair_set(right)
                ),
                'absolute_raw_probability_difference': abs(
                    float(left['explanation']['raw_probability'])
                    - float(right['explanation']['raw_probability'])
                ),
            })
    metric_names = (
        'drug_a_top_atom_jaccard', 'drug_b_top_atom_jaccard',
        'drug_a_top_motif_jaccard', 'drug_b_top_motif_jaccard',
        'cross_motif_pair_jaccard', 'absolute_raw_probability_difference',
    )
    summary = {
        metric: float(np.mean([row[metric] for row in comparison_rows]))
        if comparison_rows else None
        for metric in metric_names
    }
    return {
        'method': EXPLANATION_STABILITY_METHOD,
        'model_architecture': architectures.pop(),
        'model_seeds': [int(seed) for seed in model_seeds],
        'split_seed': int(next(iter(split_seeds))),
        'artifact_count': len(artifacts),
        'shared_explained_pair_count': len(shared_keys),
        'pairwise_comparison_count': len(comparison_rows),
        'mean_metrics': summary,
        'comparisons': comparison_rows,
        'interpretation_warning': (
            'This measures agreement among the selected local explanation outputs for '
            'shared examples. It does not prove explanation correctness, causality, or '
            'chemical plausibility. A low overlap should trigger investigation rather '
            'than post-hoc selection of a preferred seed.'
        ),
    }
