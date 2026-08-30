import pytest
import pandas as pd
import numpy as np

import src.data_prep.splits as splits_module

from src.data_prep.splits import (
    canonical_pair,
    deduplicate_unordered_pairs,
    build_binary_pair_dataset,
    create_split_aware_binary_splits,
    create_splits,
    _split_dataframe
)

def test_canonical_pair_ordering():
    """Test that reversed pairs are canonicalized correctly."""
    assert canonical_pair('DrugA', 'DrugB') == ('DrugA', 'DrugB')
    assert canonical_pair('DrugB', 'DrugA') == ('DrugA', 'DrugB')
    assert canonical_pair('A', 'A') == ('A', 'A')

    with pytest.raises(ValueError):
        canonical_pair(np.nan, 'DrugB')
        
    with pytest.raises(ValueError):
        canonical_pair('', 'DrugB')

def test_deduplicate_unordered_pairs_raises_on_conflict():
    """Test that conflicting labels for the same unordered pair raise an error."""
    df = pd.DataFrame({
        'source': ['DrugA', 'DrugB', 'DrugC'],
        'target': ['DrugB', 'DrugA', 'DrugD'],
        'label': [1.0, 0.0, 1.0] # DrugA-DrugB has both 1.0 and 0.0
    })
    
    with pytest.raises(ValueError, match="Found unordered drug pairs with conflicting labels"):
        deduplicate_unordered_pairs(df, 'source', 'target', 'label')

def test_deduplicate_unordered_pairs_keeps_first():
    """Test that identical labels for the same unordered pair are deduplicated."""
    df = pd.DataFrame({
        'source': ['DrugA', 'DrugB', 'DrugC'],
        'target': ['DrugB', 'DrugA', 'DrugD'],
        'label': [1.0, 1.0, 1.0] # DrugA-DrugB is consistently 1.0
    })
    dedup = deduplicate_unordered_pairs(df, 'source', 'target', 'label')
    assert len(dedup) == 2
    # The pairs should be canonicalized
    assert list(dedup['source'].values) == ['DrugA', 'DrugC']
    assert list(dedup['target'].values) == ['DrugB', 'DrugD']


def test_deduplicate_unordered_pairs_uses_vectorized_canonicalization(monkeypatch):
    """Large Colab inputs must not invoke the scalar helper once per row."""
    dataframe = pd.DataFrame({
        'source': ['DrugB', 'DrugA', 'DrugD'],
        'target': ['DrugA', 'DrugB', 'DrugC'],
        'label': [1.0, 1.0, 1.0],
    })

    def scalar_helper_must_not_run(*_args, **_kwargs):
        raise AssertionError('pair canonicalization must be vectorized')

    monkeypatch.setattr(splits_module, 'canonical_pair', scalar_helper_must_not_run)
    result = splits_module.deduplicate_unordered_pairs(dataframe, 'source', 'target')

    assert result.to_dict('records') == [
        {'source': 'DrugA', 'target': 'DrugB', 'label': 1.0},
        {'source': 'DrugC', 'target': 'DrugD', 'label': 1.0},
    ]

def test_build_binary_pair_dataset_negative_no_match():
    """Test that negative samples never match positive pairs in either direction."""
    positives = pd.DataFrame({
        'source': ['DrugA', 'DrugC'],
        'target': ['DrugB', 'DrugD']
    })
    
    dataset = build_binary_pair_dataset(positives, neg_ratio=2.0)
    
    # Extract canonical positive keys
    pos_keys = set(canonical_pair(row.source, row.target) for _, row in dataset[dataset['label'] == 1.0].iterrows())
    # Extract canonical negative keys
    neg_keys = set(canonical_pair(row.source, row.target) for _, row in dataset[dataset['label'] == 0.0].iterrows())
    
    # Ensure no overlap
    assert pos_keys.isdisjoint(neg_keys)
    # Ensure exact number of expected negatives
    assert len(neg_keys) == len(pos_keys) * 2

def test_create_splits_no_leakage_and_reversed_pairs():
    """Test that S1/S2 splits do not contain seen drugs, preventing leakage."""
    # Create a synthetic dataset with enough pairs
    drugs = [f'Drug{i}' for i in range(10)]
    pairs = []
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            pairs.append((drugs[i], drugs[j], 1.0))
            pairs.append((drugs[j], drugs[i], 1.0)) # Add reversed duplicates explicitly
            
    df = pd.DataFrame(pairs, columns=['drug1_id', 'drug2_id', 'label'])
    
    splits = create_splits(df, holdout_fraction=0.3)
    
    trans_train = splits['transductive_train']
    s1_test = splits['s1_test']
    s2_test = splits['s2_test']
    
    seen_drugs = set(trans_train['drug1_id']).union(set(trans_train['drug2_id']))
    
    s1_drugs_a = set(s1_test['drug1_id'])
    s1_drugs_b = set(s1_test['drug2_id'])
    s1_all_drugs = s1_drugs_a.union(s1_drugs_b)
    
    # S1 must only contain completely unseen drugs
    assert s1_all_drugs.isdisjoint(seen_drugs)
    
    # Check S2 - exactly one drug per pair must be unseen
    for _, row in s2_test.iterrows():
        a_seen = row['drug1_id'] in seen_drugs
        b_seen = row['drug2_id'] in seen_drugs
        assert a_seen != b_seen # XOR condition

def test_split_dataframe_safely_handles_empty_or_one_class():
    """Test that empty or one-class splits are reported safely rather than crashing."""
    empty_df = pd.DataFrame(columns=['source', 'target', 'label'])
    train, test = _split_dataframe(empty_df, 'label', test_size=0.2, seed=42)
    assert train.empty
    assert test.empty
    
    one_class_df = pd.DataFrame({
        'source': ['DrugA', 'DrugB'],
        'target': ['DrugC', 'DrugD'],
        'label': [1.0, 1.0]
    })
    
    # Should fallback to unstratified split instead of crashing
    train, test = _split_dataframe(one_class_df, 'label', test_size=0.5, seed=42)
    assert len(train) == 1
    assert len(test) == 1


def test_split_aware_negatives_are_disjoint_unreported_and_follow_cold_start_groups():
    drugs = [f'Drug{index}' for index in range(30)]
    generator = np.random.default_rng(7)
    positive_rows = [
        (drugs[first], drugs[second])
        for first in range(len(drugs))
        for second in range(first + 1, len(drugs))
        if generator.random() < 0.25
    ]
    positives = pd.DataFrame(positive_rows, columns=['source', 'target'])
    groups = splits_module._cold_start_groups(
        positives, 'source', 'target', holdout_fraction=0.60, seed=42
    )

    split_frames, audit = create_split_aware_binary_splits(
        positives,
        known_reported_positive_pairs=positives,
        source_col='source',
        target_col='target',
        holdout_fraction=0.60,
        seed=42,
        negative_sampling_strategy='uniform',
    )

    known_keys = {
        canonical_pair(row.source, row.target) for row in positives.itertuples(index=False)
    }
    all_negative_keys = set()
    for name, frame in split_frames.items():
        positives_in_split = frame[frame['label'] == 1.0]
        negatives_in_split = frame[frame['label'] == 0.0]
        assert len(positives_in_split) == len(negatives_in_split)
        negative_keys = {
            canonical_pair(row.source, row.target)
            for row in negatives_in_split.itertuples(index=False)
        }
        assert known_keys.isdisjoint(negative_keys)
        assert all_negative_keys.isdisjoint(negative_keys)
        all_negative_keys.update(negative_keys)
        assert set(negatives_in_split['label_evidence']) == {
            'unreported_twosides_split_aware_sampled'
        }
        if name == 's1_test':
            assert all(
                row.source in groups['s1_test'] and row.target in groups['s1_test']
                for row in negatives_in_split.itertuples(index=False)
            )
        if name == 's2_test':
            assert all(
                (row.source in groups['s1_test']) != (row.target in groups['s1_test'])
                and (row.source in groups['seen']) != (row.target in groups['seen'])
                for row in negatives_in_split.itertuples(index=False)
            )
    assert audit['protocol'] == 'split_aware_unreported_sampling_v1'
    assert audit['unique_sampled_negative_pairs'] == len(all_negative_keys)


def test_split_aware_sampler_forbids_reported_pairs_omitted_by_a_data_cap():
    positive_frame = pd.DataFrame({
        'source': ['DrugA'], 'target': ['DrugB'], 'label': [1.0],
    })
    negatives, keys, summary = splits_module._sample_partition_negatives(
        positive_frame,
        candidate_drugs={'DrugA', 'DrugB', 'DrugC', 'DrugD'},
        is_eligible_pair=lambda _first, _second: True,
        forbidden_keys={canonical_pair('DrugA', 'DrugB'), canonical_pair('DrugC', 'DrugD')},
        source_col='source',
        target_col='target',
        neg_ratio=4.0,
        seed=42,
        negative_sampling_strategy='uniform',
        max_attempt_multiplier=100,
    )

    assert len(negatives) == 4
    assert len(keys) == 4
    assert summary['sampled_negatives'] == 4
    assert canonical_pair('DrugC', 'DrugD') not in keys
