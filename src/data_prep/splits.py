"""Leakage-safe construction of DDI training and cold-start evaluation splits."""

from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from typing import Any, cast

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
    all_drugs = pd.unique(
        deduplicated[[drug_a_col, drug_b_col]].to_numpy().ravel()
    )
    if len(all_drugs) < 3:
        raise ValueError('At least three distinct drugs are required for cold-start splits.')

    holdout_count = max(1, round(holdout_fraction * len(all_drugs)))
    holdout_count = min(holdout_count, len(all_drugs) - 2)
    rng = np.random.default_rng(seed)
    holdout_drugs = rng.choice(all_drugs, size=holdout_count, replace=False)
    dev_holdout_count = len(holdout_drugs) // 2
    dev_holdouts = set(holdout_drugs[:dev_holdout_count])
    test_holdouts = set(holdout_drugs[dev_holdout_count:])
    all_holdouts = dev_holdouts | test_holdouts

    drug_a_seen = ~deduplicated[drug_a_col].isin(all_holdouts)
    drug_b_seen = ~deduplicated[drug_b_col].isin(all_holdouts)
    dev_drug_a = deduplicated[drug_a_col].isin(dev_holdouts)
    dev_drug_b = deduplicated[drug_b_col].isin(dev_holdouts)
    test_drug_a = deduplicated[drug_a_col].isin(test_holdouts)
    test_drug_b = deduplicated[drug_b_col].isin(test_holdouts)

    s1_dev = deduplicated[dev_drug_a & dev_drug_b].copy()
    s2_dev = deduplicated[(dev_drug_a & drug_b_seen) | (drug_a_seen & dev_drug_b)].copy()
    
    s1_test = deduplicated[test_drug_a & test_drug_b].copy()
    s2_test = deduplicated[(test_drug_a & drug_b_seen) | (drug_a_seen & test_drug_b)].copy()

    seen = deduplicated[drug_a_seen & drug_b_seen].copy()

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
    }
