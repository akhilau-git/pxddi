import pytest
import os
import tempfile
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Mock matplotlib and seaborn since they are not installed locally but used in Colab
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['seaborn'] = MagicMock()

import torch
from unittest.mock import MagicMock, patch

# Mock torch.cuda.is_available to pass the assert on import
with patch('torch.cuda.is_available', return_value=True):
    from src.training.train_full_pipeline_v2 import (
        calculate_metrics,
        filter_graph_compatible_pairs,
        get_file_hash,
        publish_latest_results,
        runtime_environment,
        save_counterion_curation_candidates,
        safe_checkpoint_save,
        select_validation_threshold,
        should_stop_early,
    )
def test_get_file_hash():
    """Verify SHA-256 hash calculation."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"pxddi_test")
        tmp_path = tmp.name
        
    try:
        # SHA-256 of the exact bytes written above.
        h = get_file_hash(tmp_path)
        assert h == "8a46f52b4eb2d8bfadd3c6623b1534d3ccd424062343f6d4a0ff8852fb5bda30"
    finally:
        os.remove(tmp_path)

def test_safe_checkpoint_save_atomic_replace():
    """Verify safe_checkpoint_save writes to tmp, validates, then replaces."""
    
    # Dummy state
    state = {'test_key': torch.tensor([1, 2, 3])}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        final_path = os.path.join(tmpdir, "model.pt")
        
        # Test 1: Successful save
        chk_hash = safe_checkpoint_save(state, final_path)
        
        # Validate file was moved
        assert os.path.exists(final_path)
        assert not os.path.exists(final_path + ".tmp")
        
        # Validate hash matches written file
        assert chk_hash == get_file_hash(final_path)
        
        # Test 2: Validation failure (simulated by corrupting during save? We can mock torch.save)
        # Actually it's enough to know it creates .tmp and replaces.
        
        loaded_state = torch.load(final_path, map_location='cpu', weights_only=False)
        assert torch.equal(loaded_state['test_key'], state['test_key'])


def test_metrics_handle_evaluable_and_one_class_splits():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = select_validation_threshold(labels, predictions)

    metrics = calculate_metrics(labels, predictions, threshold)
    assert metrics['status'] == 'evaluated'
    assert metrics['auroc'] == 1.0
    assert metrics['average_precision'] == 1.0
    assert metrics['sample_count'] == 4

    one_class_metrics = calculate_metrics(
        np.array([0, 0]), np.array([0.1, 0.2]), threshold=0.5
    )
    assert one_class_metrics['status'] == 'skipped_one_class_or_empty_split'
    assert one_class_metrics['auroc'] is None


def test_early_stopping_waits_for_minimum_epochs_and_patience():
    assert not should_stop_early(39, 100, minimum_epochs=40, patience=30)
    assert not should_stop_early(40, 29, minimum_epochs=40, patience=30)
    assert should_stop_early(40, 30, minimum_epochs=40, patience=30)
    assert not should_stop_early(100, 100, minimum_epochs=40, patience=0)


def test_graph_incompatible_pairs_are_removed_before_splitting_with_an_audit():
    dataframe = pd.DataFrame({
        'source': ['CCO', 'C', 'CCN'],
        'target': ['CCN', 'CCO', 'not-a-smiles'],
        'label': [1.0, 1.0, 0.0],
    })

    clean, exclusions = filter_graph_compatible_pairs(dataframe)

    assert len(clean) == 1
    assert clean.iloc[0]['source'] == 'CCO'
    assert len(exclusions) == 2
    assert set(exclusions['dataset_row_index']) == {'1', '2'}
    statuses = set(exclusions['source_graph_status']).union(
        set(exclusions['target_graph_status'])
    )
    assert 'single_atom_or_disconnected_structure' in statuses
    assert 'invalid_smiles' in statuses


def test_runtime_environment_records_resolved_dependency_versions():
    environment = runtime_environment()

    assert environment['python_version']
    assert 'torch' in environment['package_versions']
    assert 'torch_geometric' in environment['package_versions']


def test_publish_latest_results_overwrites_the_easy_to_find_result_copy(tmp_path):
    run_dir = tmp_path / 'run_1'
    audit_dir = run_dir / 'audits'
    figure_dir = run_dir / 'figures'
    audit_dir.mkdir(parents=True)
    figure_dir.mkdir()
    for relative in (
        'run_manifest_initial.json',
        'run_manifest.json',
        'results_summary.json',
        'training_history.csv',
        'audits/toxicity_bridge_conflicts.csv',
        'audits/toxicity_bridge_summary.json',
        'audits/invalid_smiles_exclusions.csv',
        'audits/input_quality_summary.json',
        'audits/dataset_summary.json',
        'audits/counterion_curation_candidates.csv',
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'first:{relative}', encoding='utf-8')
    (figure_dir / 'training_curves.png').write_bytes(b'first figure')

    latest_dir = tmp_path / 'latest_results'
    publish_latest_results(run_dir, latest_dir)

    assert (latest_dir / 'figures/training_curves.png').read_bytes() == b'first figure'
    (figure_dir / 'training_curves.png').write_bytes(b'new figure')
    publish_latest_results(run_dir, latest_dir)

    assert (latest_dir / 'figures/training_curves.png').read_bytes() == b'new figure'
    assert (latest_dir / 'latest_run.json').is_file()


def test_counterion_curation_queue_requires_manual_review(tmp_path):
    exclusions = pd.DataFrame({
        'source': ['[Na+].[Cl-]', 'CCO'],
        'target': ['CCN', '[Ca+2]'],
        'source_graph_status': ['counterion_or_inorganic_only_structure', 'graph_compatible'],
        'target_graph_status': ['graph_compatible', 'counterion_or_inorganic_only_structure'],
    })

    summary = save_counterion_curation_candidates(exclusions, tmp_path)
    candidates = pd.read_csv(tmp_path / 'counterion_curation_candidates.csv')

    assert summary['automatic_parent_mapping_applied'] is False
    assert summary['unique_counterion_or_inorganic_structures'] == 2
    assert set(candidates['curation_decision']) == {'pending_manual_review'}


def test_counterion_curation_merges_the_same_structure_seen_on_both_sides(tmp_path):
    exclusions = pd.DataFrame({
        'source': ['[Na+].[Cl-]', 'CCO'],
        'target': ['CCN', '[Na+].[Cl-]'],
        'source_graph_status': ['counterion_or_inorganic_only_structure', 'graph_compatible'],
        'target_graph_status': ['graph_compatible', 'counterion_or_inorganic_only_structure'],
    })

    summary = save_counterion_curation_candidates(exclusions, tmp_path)
    candidates = pd.read_csv(tmp_path / 'counterion_curation_candidates.csv')

    assert summary['unique_counterion_or_inorganic_structures'] == 1
    assert candidates.iloc[0]['occurrence_count'] == 2
    assert candidates.iloc[0]['observed_as'] == 'source|target'
