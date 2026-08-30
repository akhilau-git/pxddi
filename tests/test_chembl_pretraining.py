import gzip

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from src.data_prep.chembl_pretraining import (
    build_pretraining_exclusion_set,
    classify_smiles_for_graph,
    select_chembl_pretraining_corpus,
)
from src.data_prep.prepare_twosides import (
    FEATURE_SCHEMA_RICH,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    smiles_to_graph,
)
from src.models.encoder import EdgeAwareMolecularEncoder
from src.models.encoder_pretraining import (
    PRETRAINING_ARTIFACT_TYPE,
    EdgeAwareContrastivePretrainer,
    augment_edge_aware_batch,
    bidirectional_nt_xent_loss,
    load_pretrained_edge_aware_encoder,
)


def _write_chemreps(path):
    content = (
        'chembl_id\tcanonical_smiles\tstandard_inchi\tstandard_inchi_key\n'
        'CHEMBL1\tCCO\tInChI=1S/C2H6O\tKEY1\n'
        'CHEMBL2\tCCN\tInChI=1S/C2H7N\tKEY2\n'
        'CHEMBL3\t[Na+]\tInChI=1S/Na/q+1\tKEY3\n'
    )
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write(content)


def test_pretraining_corpus_excludes_non_train_structures_and_counterions(tmp_path):
    chemreps = tmp_path / 'chemreps.txt.gz'
    _write_chemreps(chemreps)

    corpus, summary = select_chembl_pretraining_corpus(
        chemreps,
        excluded_smiles={'CCO'},
        maximum_molecules=10,
        seed=7,
    )

    assert corpus['canonical_smiles'].tolist() == ['CCN']
    assert summary['rows_excluded_for_twosides_non_train_leakage'] == 1
    assert summary['selected_unique_molecules'] == 1
    assert summary['invalid_or_unsupported_rows'] == 1


def test_pretraining_smiles_classifier_reports_invalid_and_counterion_structures():
    assert classify_smiles_for_graph('not a smiles') == (None, 'invalid_smiles')
    assert classify_smiles_for_graph('[Na+]') == (
        None, 'counterion_or_inorganic_only_structure'
    )


def test_pretraining_exclusion_set_recreates_the_ddi_split(tmp_path):
    drugs = ['C' * length for length in range(2, 14)]
    edges = pd.DataFrame({
        'source': drugs,
        'target': drugs[1:] + drugs[:1],
    })
    edges_path = tmp_path / 'drug_drug_edges.csv'
    edges.to_csv(edges_path, index=False)

    excluded, summary = build_pretraining_exclusion_set(
        edges_path,
        data_cap=12,
        split_seed=42,
        negative_sampling_strategy='uniform',
    )

    assert summary['pretraining_leakage_policy'] == 'exclude_all_non_train_twosides_structures_v1'
    assert summary['split_rows']['transductive_train'] > 0
    assert summary['excluded_non_train_unique_structures'] == len(excluded)
    assert excluded


def test_edge_aware_contrastive_objective_backpropagates():
    graph_a = smiles_to_graph('CCO', feature_schema=FEATURE_SCHEMA_RICH)
    graph_b = smiles_to_graph('CCN', feature_schema=FEATURE_SCHEMA_RICH)
    assert graph_a is not None and graph_b is not None
    batch = Batch.from_data_list([graph_a, graph_b])
    model = EdgeAwareContrastivePretrainer(
        RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, hidden_channels=8
    )

    first_view = augment_edge_aware_batch(
        batch, atom_feature_mask_rate=0.2, bond_feature_mask_rate=0.2
    )
    second_view = augment_edge_aware_batch(
        batch, atom_feature_mask_rate=0.2, bond_feature_mask_rate=0.2
    )
    loss = bidirectional_nt_xent_loss(model(first_view), model(second_view))
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_contrastive_objective_rejects_a_singleton_batch():
    embeddings = torch.randn(1, 8)
    with pytest.raises(ValueError, match='batch size at least two'):
        bidirectional_nt_xent_loss(embeddings, embeddings)


def test_checked_pretraining_checkpoint_initializes_encoder_only(tmp_path):
    source = EdgeAwareMolecularEncoder(RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8)
    checkpoint = tmp_path / 'pretrained_encoder.pt'
    torch.save({
        'artifact_type': PRETRAINING_ARTIFACT_TYPE,
        'encoder_state_dict': source.state_dict(),
        'encoder_configuration': {
            'in_channels': RICH_NUM_ATOM_FEATURES,
            'edge_feature_dim': NUM_BOND_FEATURES,
            'hidden_channels': 8,
        },
        'pretraining_leakage_policy': 'exclude_all_non_train_twosides_structures_v1',
        'source_corpus': {'selected_unique_molecules': 2},
        'pretraining_split_audit': {'split_seed': 42},
    }, checkpoint)
    restored = EdgeAwareMolecularEncoder(RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8)

    loaded = load_pretrained_edge_aware_encoder(
        restored,
        checkpoint,
        expected_in_channels=RICH_NUM_ATOM_FEATURES,
        expected_edge_feature_dim=NUM_BOND_FEATURES,
        expected_hidden_channels=8,
        expected_split_audit={'split_seed': 42},
    )

    assert loaded['artifact_type'] == PRETRAINING_ARTIFACT_TYPE
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value), name


def test_pretraining_checkpoint_rejects_missing_leakage_contract(tmp_path):
    checkpoint = tmp_path / 'unsafe.pt'
    torch.save({
        'artifact_type': PRETRAINING_ARTIFACT_TYPE,
        'encoder_state_dict': EdgeAwareMolecularEncoder(
            RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8
        ).state_dict(),
        'encoder_configuration': {
            'in_channels': RICH_NUM_ATOM_FEATURES,
            'edge_feature_dim': NUM_BOND_FEATURES,
            'hidden_channels': 8,
        },
    }, checkpoint)
    restored = EdgeAwareMolecularEncoder(RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8)

    with pytest.raises(ValueError, match='strict leakage-exclusion policy'):
        load_pretrained_edge_aware_encoder(
            restored,
            checkpoint,
            expected_in_channels=RICH_NUM_ATOM_FEATURES,
            expected_edge_feature_dim=NUM_BOND_FEATURES,
            expected_hidden_channels=8,
        )


def test_pretraining_checkpoint_rejects_a_different_ddi_split_contract(tmp_path):
    checkpoint = tmp_path / 'different_split.pt'
    torch.save({
        'artifact_type': PRETRAINING_ARTIFACT_TYPE,
        'encoder_state_dict': EdgeAwareMolecularEncoder(
            RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8
        ).state_dict(),
        'encoder_configuration': {
            'in_channels': RICH_NUM_ATOM_FEATURES,
            'edge_feature_dim': NUM_BOND_FEATURES,
            'hidden_channels': 8,
        },
        'pretraining_leakage_policy': 'exclude_all_non_train_twosides_structures_v1',
        'pretraining_split_audit': {'split_seed': 42},
    }, checkpoint)
    restored = EdgeAwareMolecularEncoder(RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 8)

    with pytest.raises(ValueError, match='split contract'):
        load_pretrained_edge_aware_encoder(
            restored,
            checkpoint,
            expected_in_channels=RICH_NUM_ATOM_FEATURES,
            expected_edge_feature_dim=NUM_BOND_FEATURES,
            expected_hidden_channels=8,
            expected_split_audit={'split_seed': 99},
        )
