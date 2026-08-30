"""Leakage-safe construction of DDI training and cold-start evaluation splits."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Hashable, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def canonical_pair(drug_a: Hashable, drug_b: Hashable) -> tuple[str, str]:
    """Return a stable, order-independent identifier for a drug pair.

    PxDDI is order-independent, so ``A-B`` and ``B-A`` must never be treated
    as distinct observations during negative sampling or data splitting.
    """
    if pd.isna(cast(Any, drug_a)) or pd.isna(cast(Any, drug_b)):
        raise ValueError('Drug-pair values must not be missing.')
    left, right = str(drug_a), str(drug_b)
    if not left or not right:
        raise ValueError('Drug-pair values must not be empty.')
    return (left, right) if left <= right else (right, left)


def _canonicalize_pair_columns(
    dataframe: pd.DataFrame,
    drug_a_col: str,
    drug_b_col: str,
) -> pd.DataFrame:
    """Return a copy whose pair columns use one consistent order."""
    if drug_a_col not in dataframe or drug_b_col not in dataframe:
        raise ValueError(
            f"Expected pair columns '{drug_a_col}' and '{drug_b_col}', "
            f'but found {dataframe.columns.tolist()}.'
        )

    normalized = dataframe.copy()
    left = normalized[drug_a_col]
    right = normalized[drug_b_col]
    if (left.isna() | right.isna()).any():
        raise ValueError('Drug-pair values must not be missing.')

    # TWOSIDES uses molecular strings.  Converting whole columns and comparing
    # them at once avoids 200,000 Python-level ``canonical_pair`` calls during
    # the pre-split audit, which made Colab appear to hang before training.
    left_text = left.astype(str)
    right_text = right.astype(str)
    if (left_text.eq('') | right_text.eq('')).any():
        raise ValueError('Drug-pair values must not be empty.')

    left_first = left_text.le(right_text)
    normalized[drug_a_col] = left_text.where(left_first, right_text)
    normalized[drug_b_col] = right_text.where(left_first, left_text)
    return normalized


def deduplicate_unordered_pairs(
    dataframe: pd.DataFrame,
    drug_a_col: str,
    drug_b_col: str,
    label_col: str = 'label',
) -> pd.DataFrame:
    """Canonicalize pairs and remove duplicate observations safely.

    A pair with both a positive and a sampled-negative label is a data
    contradiction. Raising an error prevents an arbitrary row order from
    deciding its label.
    """
    if label_col not in dataframe:
        raise ValueError(
            f"Expected binary label column '{label_col}', but found "
            f'{dataframe.columns.tolist()}.'
        )

    normalized = _canonicalize_pair_columns(dataframe, drug_a_col, drug_b_col)
    labels_per_pair = normalized.groupby(
        [drug_a_col, drug_b_col], dropna=False
    )[label_col].nunique(dropna=False)
    conflicts = labels_per_pair[labels_per_pair > 1]
    if not conflicts.empty:
        examples = [tuple(pair) for pair in conflicts.index[:5]]
        raise ValueError(
            'Found unordered drug pairs with conflicting labels. '
            f'Examples: {examples}. Resolve these before training.'
        )

    return normalized.drop_duplicates(
        subset=[drug_a_col, drug_b_col], keep='first'
    ).reset_index(drop=True)


def build_binary_pair_dataset(
    positive_pairs: pd.DataFrame,
    source_col: str = 'source',
    target_col: str = 'target',
    neg_ratio: float = 1.0,
    seed: int = 42,
    max_attempt_multiplier: int = 50,
    negative_sampling_strategy: str = 'uniform',
) -> pd.DataFrame:
    """Create an auditable binary dataset from reported and unreported pairs.

    The generated zero labels mean only that the pair was not reported in the
    supplied TWOSIDES table. They are deliberately tagged as such and are not
    evidence that the pair is safe.
    """
    if neg_ratio < 0:
        raise ValueError('neg_ratio must be non-negative.')
    if max_attempt_multiplier <= 0:
        raise ValueError('max_attempt_multiplier must be positive.')
    if negative_sampling_strategy not in ('uniform', 'degree_matched'):
        raise ValueError(f'Unsupported negative sampling strategy: {negative_sampling_strategy}')

    positives = positive_pairs[[source_col, target_col]].copy()
    positives['label'] = 1.0
    positives['label_evidence'] = 'reported_twosides'
    positives = deduplicate_unordered_pairs(positives, source_col, target_col)
    if positives.empty:
        raise ValueError('At least one positive pair is required.')

    all_drugs = pd.unique(positives[[source_col, target_col]].to_numpy().ravel())
    if len(all_drugs) < 2:
        raise ValueError('At least two distinct drugs are required.')

    positive_keys = {
        canonical_pair(source, target)
        for source, target in zip(positives[source_col], positives[target_col])
    }
    target_negatives = round(len(positives) * neg_ratio)
    if target_negatives == 0:
        return positives.sample(frac=1, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    
    sampling_probs = None
    if negative_sampling_strategy == 'degree_matched':
        drug_counts = pd.concat([positives[source_col], positives[target_col]]).value_counts()
        sampling_probs = drug_counts.loc[all_drugs].to_numpy(dtype=float)
        sampling_probs /= sampling_probs.sum()

    negative_keys: set[tuple[str, str]] = set()
    max_attempts = max(target_negatives * max_attempt_multiplier, 1000)
    attempts = 0
    while len(negative_keys) < target_negatives and attempts < max_attempts:
        source, target = rng.choice(all_drugs, size=2, replace=False, p=sampling_probs)
        candidate = canonical_pair(source, target)
        if candidate not in positive_keys:
            negative_keys.add(candidate)
        attempts += 1

    if len(negative_keys) < target_negatives:
        possible_pairs = len(all_drugs) * (len(all_drugs) - 1) // 2
        available_pairs = possible_pairs - len(positive_keys)
        raise ValueError(
            'Could not sample the requested number of unique unreported pairs. '
            f'Requested {target_negatives}; sampled {len(negative_keys)}; '
            f'at most {available_pairs} are available.'
        )

    # Sorting removes Python hash-order variation before the seeded shuffle,
    # making the generated dataset repeatable across Colab processes.
    negatives = pd.DataFrame(
        [
            {source_col: source, target_col: target}
            for source, target in sorted(negative_keys)
        ]
    )
    negatives['label'] = 0.0
    negatives['label_evidence'] = 'unreported_twosides_sampled'
    combined = pd.concat([positives, negatives], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def _split_dataframe(
    dataframe: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labelled frame, stratifying only when it is valid."""
    if dataframe.empty:
        return dataframe.copy(), dataframe.copy()
    if len(dataframe) < 2:
        raise ValueError('At least two seen-drug pairs are required for a split.')

    labels = dataframe[label_col]
    can_stratify = labels.nunique(dropna=False) > 1 and labels.value_counts().min() >= 2
    return cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            dataframe,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=labels if can_stratify else None,
        )
    )


def _stable_string_set_hash(values: set[str]) -> str:
    """Hash a set of identities without exposing an order-dependent representation."""
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def _cold_start_groups(
    dataframe: pd.DataFrame,
    drug_a_col: str,
    drug_b_col: str,
    *,
    holdout_fraction: float,
    seed: int,
) -> dict[str, set[str]]:
    """Choose deterministic seen, S1-dev, and S1-test drug identities."""
    all_drugs = pd.unique(dataframe[[drug_a_col, drug_b_col]].to_numpy().ravel())
    if len(all_drugs) < 3:
        raise ValueError('At least three distinct drugs are required for cold-start splits.')
    holdout_count = max(1, round(holdout_fraction * len(all_drugs)))
    holdout_count = min(holdout_count, len(all_drugs) - 2)
    rng = np.random.default_rng(seed)
    holdout_drugs = rng.choice(all_drugs, size=holdout_count, replace=False)
    dev_holdout_count = len(holdout_drugs) // 2
    dev_holdouts = {str(drug) for drug in holdout_drugs[:dev_holdout_count]}
    test_holdouts = {str(drug) for drug in holdout_drugs[dev_holdout_count:]}
    all_drug_set = {str(drug) for drug in all_drugs}
    return {
        'seen': all_drug_set - dev_holdouts - test_holdouts,
        's1_dev': dev_holdouts,
        's1_test': test_holdouts,
    }


def _create_positive_cold_start_splits(
    dataframe: pd.DataFrame,
    drug_a_col: str,
    drug_b_col: str,
    label_col: str,
    *,
    holdout_fraction: float,
    validation_fraction_of_seen_train: float,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, set[str]]]:
    """Split already deduplicated labelled rows and retain the identity groups."""
    groups = _cold_start_groups(
        dataframe,
        drug_a_col,
        drug_b_col,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    source = dataframe[drug_a_col].astype(str)
    target = dataframe[drug_b_col].astype(str)
    source_seen, target_seen = source.isin(groups['seen']), target.isin(groups['seen'])
    source_dev, target_dev = source.isin(groups['s1_dev']), target.isin(groups['s1_dev'])
    source_test, target_test = source.isin(groups['s1_test']), target.isin(groups['s1_test'])

    s1_dev = dataframe[source_dev & target_dev].copy()
    s2_dev = dataframe[(source_dev & target_seen) | (source_seen & target_dev)].copy()
    s1_test = dataframe[source_test & target_test].copy()
    s2_test = dataframe[(source_test & target_seen) | (source_seen & target_test)].copy()
    seen = dataframe[source_seen & target_seen].copy()

    train_and_validation, transductive_test = _split_dataframe(
        seen, label_col, test_size=0.20, seed=seed
    )
    transductive_train, validation = _split_dataframe(
        train_and_validation,
        label_col,
        test_size=validation_fraction_of_seen_train,
        seed=seed + 1,
    )
    return {
        'transductive_train': transductive_train.reset_index(drop=True),
        'validation': validation.reset_index(drop=True),
        'transductive_test': transductive_test.reset_index(drop=True),
        's1_dev': s1_dev.reset_index(drop=True),
        's2_dev': s2_dev.reset_index(drop=True),
        's1_test': s1_test.reset_index(drop=True),
        's2_test': s2_test.reset_index(drop=True),
    }, groups


def _negative_pair_predicate(
    split_name: str,
    groups: dict[str, set[str]],
) -> Callable[[str, str], bool]:
    """Return the exact drug-identity condition for one evaluation partition."""
    seen, dev, test = groups['seen'], groups['s1_dev'], groups['s1_test']
    if split_name.startswith('transductive') or split_name == 'validation':
        return lambda first, second: first in seen and second in seen
    if split_name == 's1_dev':
        return lambda first, second: first in dev and second in dev
    if split_name == 's1_test':
        return lambda first, second: first in test and second in test
    if split_name == 's2_dev':
        return lambda first, second: (first in dev and second in seen) or (first in seen and second in dev)
    if split_name == 's2_test':
        return lambda first, second: (first in test and second in seen) or (first in seen and second in test)
    raise ValueError(f'Unknown split name for negative sampling: {split_name}.')


def _sample_partition_negatives(
    positive_frame: pd.DataFrame,
    *,
    candidate_drugs: set[str],
    is_eligible_pair: Callable[[str, str], bool],
    forbidden_keys: set[tuple[str, str]],
    source_col: str,
    target_col: str,
    neg_ratio: float,
    seed: int,
    negative_sampling_strategy: str,
    max_attempt_multiplier: int,
) -> tuple[pd.DataFrame, set[tuple[str, str]], dict[str, int]]:
    """Sample unique unreported negatives constrained to one split's identities."""
    if positive_frame.empty or neg_ratio == 0:
        return (
            pd.DataFrame(columns=[source_col, target_col, 'label', 'label_evidence']),
            set(),
            {'requested_negatives': 0, 'sampled_negatives': 0, 'attempts': 0},
        )
    target_negatives = round(len(positive_frame) * neg_ratio)
    if target_negatives == 0:
        return (
            pd.DataFrame(columns=[source_col, target_col, 'label', 'label_evidence']),
            set(),
            {'requested_negatives': 0, 'sampled_negatives': 0, 'attempts': 0},
        )
    eligible_drugs = sorted(candidate_drugs)
    if len(eligible_drugs) < 2:
        raise ValueError(
            'Cannot draw negatives for a non-empty split with fewer than two eligible drugs.'
        )
    probabilities = None
    if negative_sampling_strategy == 'degree_matched':
        observed_counts = pd.concat([
            positive_frame[source_col].astype(str), positive_frame[target_col].astype(str),
        ]).value_counts()
        weights = observed_counts.reindex(eligible_drugs, fill_value=0).to_numpy(dtype=float)
        if weights.sum() == 0:
            raise ValueError('No positive degrees were available for degree-matched sampling.')
        probabilities = weights / weights.sum()

    rng = np.random.default_rng(seed)
    sampled_keys: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(target_negatives * max_attempt_multiplier, 1000)
    while len(sampled_keys) < target_negatives and attempts < max_attempts:
        first, second = rng.choice(eligible_drugs, size=2, replace=False, p=probabilities)
        candidate = canonical_pair(first, second)
        attempts += 1
        if not is_eligible_pair(*candidate):
            continue
        if candidate not in forbidden_keys and candidate not in sampled_keys:
            sampled_keys.add(candidate)
    if len(sampled_keys) < target_negatives:
        raise ValueError(
            'Could not sample the requested number of split-aware unreported pairs. '
            f'Requested {target_negatives}; sampled {len(sampled_keys)}; '
            f'attempts {attempts}; strategy {negative_sampling_strategy}. '
            'Reduce the negative ratio or inspect the split-specific positive density.'
        )
    negatives = pd.DataFrame(sorted(sampled_keys), columns=[source_col, target_col])
    negatives['label'] = 0.0
    negatives['label_evidence'] = 'unreported_twosides_split_aware_sampled'
    return negatives, sampled_keys, {
        'requested_negatives': target_negatives,
        'sampled_negatives': len(sampled_keys),
        'attempts': attempts,
    }


def create_split_aware_binary_splits(
    positive_pairs: pd.DataFrame,
    *,
    known_reported_positive_pairs: pd.DataFrame | None = None,
    source_col: str = 'source',
    target_col: str = 'target',
    label_col: str = 'label',
    neg_ratio: float = 1.0,
    holdout_fraction: float = 0.15,
    validation_fraction_of_seen_train: float = 0.10,
    seed: int = 42,
    negative_sampling_strategy: str = 'degree_matched',
    max_attempt_multiplier: int = 50,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Construct cold-start splits before sampling their unreported negatives.

    Negatives are drawn only after the drug-identity partition is fixed. Every
    reported pair in ``known_reported_positive_pairs`` is forbidden even when a
    data cap omitted it from the fitting subset.
    """
    if neg_ratio < 0:
        raise ValueError('neg_ratio must be non-negative.')
    if max_attempt_multiplier < 1:
        raise ValueError('max_attempt_multiplier must be positive.')
    if negative_sampling_strategy not in {'uniform', 'degree_matched'}:
        raise ValueError(f'Unsupported negative sampling strategy: {negative_sampling_strategy}')
    positives = positive_pairs[[source_col, target_col]].copy()
    positives[label_col] = 1.0
    positives['label_evidence'] = 'reported_twosides'
    positives = deduplicate_unordered_pairs(positives, source_col, target_col, label_col)
    if positives.empty:
        raise ValueError('At least one positive pair is required.')
    known_frame = known_reported_positive_pairs if known_reported_positive_pairs is not None else positive_pairs
    known = known_frame[[source_col, target_col]].copy()
    known[label_col] = 1.0
    known = deduplicate_unordered_pairs(known, source_col, target_col, label_col)
    known_keys = {
        canonical_pair(source, target)
        for source, target in zip(known[source_col], known[target_col])
    }
    selected_keys = {
        canonical_pair(source, target)
        for source, target in zip(positives[source_col], positives[target_col])
    }
    if not selected_keys.issubset(known_keys):
        raise ValueError('known_reported_positive_pairs must contain every selected positive pair.')

    positive_splits, groups = _create_positive_cold_start_splits(
        positives,
        source_col,
        target_col,
        label_col,
        holdout_fraction=holdout_fraction,
        validation_fraction_of_seen_train=validation_fraction_of_seen_train,
        seed=seed,
    )
    used_negative_keys: set[tuple[str, str]] = set()
    splits: dict[str, pd.DataFrame] = {}
    per_split: dict[str, dict[str, int]] = {}
    candidate_drugs = groups['seen'] | groups['s1_dev'] | groups['s1_test']
    for position, (name, positive_split) in enumerate(positive_splits.items()):
        negatives, negative_keys, negative_audit = _sample_partition_negatives(
            positive_split,
            candidate_drugs=candidate_drugs,
            is_eligible_pair=_negative_pair_predicate(name, groups),
            forbidden_keys=known_keys | used_negative_keys,
            source_col=source_col,
            target_col=target_col,
            neg_ratio=neg_ratio,
            seed=seed + position + 1,
            negative_sampling_strategy=negative_sampling_strategy,
            max_attempt_multiplier=max_attempt_multiplier,
        )
        used_negative_keys.update(negative_keys)
        combined = (
            pd.concat([positive_split, negatives], ignore_index=True)
            if not negatives.empty else positive_split.copy()
        )
        splits[name] = combined.sample(frac=1, random_state=seed + position + 1).reset_index(drop=True)
        per_split[name] = {
            'reported_positive_pairs': int(len(positive_split)),
            **negative_audit,
            'total_rows': int(len(splits[name])),
        }
    audit: dict[str, Any] = {
        'protocol': 'split_aware_unreported_sampling_v1',
        'negative_label_meaning': 'unreported_twosides_split_aware_sampled',
        'negative_sampling_strategy': negative_sampling_strategy,
        'negative_ratio': neg_ratio,
        'split_seed': seed,
        'known_reported_positive_pairs_forbidden': int(len(known_keys)),
        'selected_reported_positive_pairs': int(len(selected_keys)),
        'unique_sampled_negative_pairs': int(len(used_negative_keys)),
        'identity_groups': {
            'seen_drugs': int(len(groups['seen'])),
            's1_dev_holdout_drugs': int(len(groups['s1_dev'])),
            's1_test_holdout_drugs': int(len(groups['s1_test'])),
            'seen_drugs_sha256': _stable_string_set_hash(groups['seen']),
            's1_dev_holdout_drugs_sha256': _stable_string_set_hash(groups['s1_dev']),
            's1_test_holdout_drugs_sha256': _stable_string_set_hash(groups['s1_test']),
        },
        'per_split': per_split,
    }
    return splits, audit


def create_splits(
    dataframe: pd.DataFrame,
    drug_a_col: str = 'drug1_id',
    drug_b_col: str = 'drug2_id',
    label_col: str = 'label',
    holdout_fraction: float = 0.15,
    validation_fraction_of_seen_train: float = 0.10,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create leakage-safe transductive, S1, and S2 DDI evaluation splits.

    S1 contains pairs where both drugs are held out from training. S2 contains
    pairs where exactly one drug is held out. The remaining seen-drug pairs are
    split into transductive train, validation, and test partitions.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must be between 0 and 1.')
    if not 0 < validation_fraction_of_seen_train < 1:
        raise ValueError('validation_fraction_of_seen_train must be between 0 and 1.')

    deduplicated = deduplicate_unordered_pairs(
        dataframe, drug_a_col, drug_b_col, label_col
    )
    splits, _ = _create_positive_cold_start_splits(
        deduplicated,
        drug_a_col,
        drug_b_col,
        label_col,
        holdout_fraction=holdout_fraction,
        validation_fraction_of_seen_train=validation_fraction_of_seen_train,
        seed=seed,
    )
    return splits
