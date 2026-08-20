"""Safe combination helpers for fixed-split PxDDI research ensembles.

The ensemble is deliberately an offline research artifact.  It never replaces
the deployed checkpoint, and it refuses to average members that were evaluated
on different source data or different split manifests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ENSEMBLE_METHOD = 'fixed_split_probability_mean_v1'
ABSTENTION_LABEL = 'insufficient_evidence_for_reliable_unseen_drug_prediction'
PROVENANCE_COLUMNS = ('source', 'target', 'label', 'label_evidence')
SPLIT_NAMES = ('transductive_train', 'validation', 'transductive_test', 's1_test', 's2_test')


def _split_manifest_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove run-specific output paths before comparing split manifests."""
    split_manifest = manifest.get('split_manifest')
    if not isinstance(split_manifest, dict):
        raise ValueError('Ensemble manifest lacks a split_manifest.')
    missing = set(SPLIT_NAMES).difference(split_manifest)
    if missing:
        raise ValueError(f'Ensemble manifest lacks split evidence for {sorted(missing)}.')
    evidence = {
        name: {
            'sha256': split_manifest[name].get('sha256'),
            'rows': split_manifest[name].get('rows'),
            'label_counts': split_manifest[name].get('label_counts'),
        }
        for name in SPLIT_NAMES
    }
    if any(not details['sha256'] for details in evidence.values()):
        raise ValueError('Ensemble split evidence includes an unhashed split.')
    return evidence


def validate_ensemble_member_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Require 3–5 independently initialized members with identical evidence."""
    if not 3 <= len(manifests) <= 5:
        raise ValueError('An ensemble must contain between three and five completed members.')
    first = manifests[0]
    configuration = first.get('configuration', {})
    required_configuration_keys = (
        'model_architecture', 'feature_schema', 'use_toxicity_pair_features',
        'toxicity_loss_weight', 'data_cap', 'split_seed',
    )
    missing = [key for key in required_configuration_keys if key not in configuration]
    if missing:
        raise ValueError(f'First ensemble manifest lacks configuration fields: {missing}.')
    reference = {
        'twosides_input_sha256': first.get('input_sha256', {}).get('twosides_edges'),
        'split_manifest_evidence': _split_manifest_evidence(first),
        **{key: configuration[key] for key in required_configuration_keys},
    }
    if not reference['twosides_input_sha256']:
        raise ValueError('First ensemble manifest lacks source-data or split-manifest evidence.')
    member_seeds = []
    for member_index, manifest in enumerate(manifests):
        member_configuration = manifest.get('configuration', {})
        comparison = {
            'twosides_input_sha256': manifest.get('input_sha256', {}).get('twosides_edges'),
            'split_manifest_evidence': _split_manifest_evidence(manifest),
            **{
                key: member_configuration.get(key)
                for key in required_configuration_keys
            },
        }
        if comparison != reference:
            raise ValueError(
                f'Ensemble member {member_index} does not match data, split, or model configuration.'
            )
        model_seed = member_configuration.get('model_seed', manifest.get('model_seed'))
        if model_seed is None:
            raise ValueError(f'Ensemble member {member_index} does not record a model seed.')
        member_seeds.append(int(model_seed))
    if len(set(member_seeds)) != len(member_seeds):
        raise ValueError('Ensemble members must use distinct model seeds.')
    return {
        'method': ENSEMBLE_METHOD,
        'member_count': len(manifests),
        'member_model_seeds': member_seeds,
        'split_seed': int(reference['split_seed']),
        'model_architecture': reference['model_architecture'],
        'feature_schema': reference['feature_schema'],
        'use_toxicity_pair_features': bool(reference['use_toxicity_pair_features']),
        'toxicity_loss_weight': float(reference['toxicity_loss_weight']),
        'data_cap': int(reference['data_cap']),
        'twosides_input_sha256': reference['twosides_input_sha256'],
        'split_manifest': first['split_manifest'],
        'split_manifest_evidence': reference['split_manifest_evidence'],
    }


def combine_member_prediction_tables(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Align saved raw scores only after confirming row-level provenance."""
    if not tables:
        raise ValueError('At least one prediction table is required.')
    if any(not set(PROVENANCE_COLUMNS).issubset(table.columns) for table in tables):
        raise ValueError(f'Every member table requires {list(PROVENANCE_COLUMNS)} columns.')
    if any('raw_prediction_score' not in table.columns for table in tables):
        raise ValueError('Every member table requires raw_prediction_score.')
    base = tables[0].loc[:, list(PROVENANCE_COLUMNS)].copy().reset_index(drop=True)
    if base.duplicated().any():
        raise ValueError('Ensemble prediction provenance contains duplicate rows.')
    raw_member_scores = []
    for member_index, table in enumerate(tables):
        provenance = table.loc[:, list(PROVENANCE_COLUMNS)].copy().reset_index(drop=True)
        if not provenance.equals(base):
            raise ValueError(
                f'Prediction provenance differs for ensemble member {member_index}; refusing to average.'
            )
        scores = pd.to_numeric(table['raw_prediction_score'], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f'Ensemble member {member_index} has invalid raw probabilities.')
        raw_member_scores.append(scores)
    stacked = np.column_stack(raw_member_scores)
    result = base.copy()
    for member_index, scores in enumerate(raw_member_scores, start=1):
        result[f'member_{member_index}_raw_prediction_score'] = scores
    result['raw_prediction_score'] = stacked.mean(axis=1)
    result['ensemble_member_standard_deviation'] = stacked.std(axis=1, ddof=0)
    result['ensemble_member_count'] = stacked.shape[1]
    return result


def apply_safe_abstention(
    member_standard_deviation,
    conformal_abstain,
    structural_ood_flag,
    standard_deviation_threshold: float,
) -> dict[str, np.ndarray]:
    """Return an explicit research abstention instead of a forced risk label."""
    deviation = np.asarray(member_standard_deviation, dtype=float)
    conformal = np.asarray(conformal_abstain, dtype=bool)
    structural_ood = np.asarray(structural_ood_flag, dtype=bool)
    if not 0 <= standard_deviation_threshold <= 1:
        raise ValueError('standard_deviation_threshold must lie between zero and one.')
    if not (len(deviation) == len(conformal) == len(structural_ood)):
        raise ValueError('All abstention inputs must have equal lengths.')
    disagreement = deviation >= standard_deviation_threshold
    reason_rows = []
    for is_conformal, is_disagreement, is_ood in zip(conformal, disagreement, structural_ood):
        reasons = []
        if is_conformal:
            reasons.append('ambiguous_or_empty_conformal_prediction_set')
        if is_disagreement:
            reasons.append('high_ensemble_member_disagreement')
        if is_ood:
            reasons.append('outside_training_drug_structural_domain')
        reason_rows.append('|'.join(reasons) if reasons else 'none')
    abstain = conformal | disagreement | structural_ood
    return {
        'ensemble_high_disagreement': disagreement,
        'safe_abstain': abstain,
        'safe_abstention_reasons': np.asarray(reason_rows, dtype=object),
        'safe_prediction_status': np.where(
            abstain,
            ABSTENTION_LABEL,
            'single_label_within_internal_research_review_rules',
        ),
    }


def summarize_safe_abstention(abstention: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Report all abstention drivers so an aggregate rate is not misleading."""
    count = len(abstention['safe_abstain'])
    if count == 0:
        return {
            'sample_count': 0,
            'safe_abstention_rate': 0.0,
            'conformal_abstention_rate': 0.0,
            'high_disagreement_rate': 0.0,
            'structural_ood_rate': 0.0,
        }
    reason_rows = abstention['safe_abstention_reasons']
    return {
        'sample_count': int(count),
        'safe_abstention_rate': float(np.asarray(abstention['safe_abstain'], dtype=bool).mean()),
        'conformal_abstention_rate': float(
            np.asarray(['conformal' in value for value in reason_rows], dtype=bool).mean()
        ),
        'high_disagreement_rate': float(
            np.asarray(abstention['ensemble_high_disagreement'], dtype=bool).mean()
        ),
        'structural_ood_rate': float(
            np.asarray(['outside_training_drug' in value for value in reason_rows], dtype=bool).mean()
        ),
    }
