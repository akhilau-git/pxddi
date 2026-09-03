"""Unit tests for AuditDDI: Auditable Neighbor Interaction Memory and GNN Fusion."""

import pytest
import torch
import numpy as np
from rdkit import Chem

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.ddi_model import (
    AuditDDIModel,
    MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
    architecture_requires_fingerprint_features,
    architecture_uses_edge_features,
)
from models.neighbor_memory import AuditableNeighborMemory
from data_prep.prepare_twosides import smiles_to_graph, FEATURE_SCHEMA_RICH


def test_auditddi_architecture_constants():
    assert architecture_uses_edge_features(MODEL_ARCHITECTURE_AUDITDDI_MEMORY)
    assert architecture_requires_fingerprint_features(MODEL_ARCHITECTURE_AUDITDDI_MEMORY)


def test_auditable_neighbor_memory_retrieval():
    # Training drugs: Aspirin, Ibuprofen, Acetaminophen, Caffeine
    train_a = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CC1=CC=C(C=C1)C(C)C(=O)O']
    train_b = ['CC(=O)NC1=CC=C(C=C1)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
    labels = [1.0, 0.0]

    mem = AuditableNeighborMemory(k_neighbors=2)
    fit_summary = mem.fit(train_a, train_b, labels)
    assert fit_summary['unique_training_drugs'] == 4
    assert fit_summary['k_neighbors'] == 2

    # Novel drug query: Naproxen + Acetaminophen
    naproxen = 'CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O'
    acetaminophen = 'CC(=O)NC1=CC=C(C=C1)O'

    scores = mem.score_pair_memory(naproxen, acetaminophen)
    assert 0.0 <= scores['neighbor_density'] <= 1.0
    assert 0.0 <= scores['max_support'] <= 1.0
    assert 0.0 <= scores['structural_confidence'] <= 1.0
    assert len(scores['audit_trail']) > 0

    # Batch extraction
    batch_features = mem.score_batch([naproxen], [acetaminophen])
    assert batch_features.shape == (1, 3)


def test_auditddi_model_forward_and_symmetry():
    smiles_a = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Aspirin
    smiles_b = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen

    ga = smiles_to_graph(
        smiles_a,
        feature_schema=FEATURE_SCHEMA_RICH,
        include_fingerprint_features=True,
    )
    gb = smiles_to_graph(
        smiles_b,
        feature_schema=FEATURE_SCHEMA_RICH,
        include_fingerprint_features=True,
    )
    assert ga is not None and gb is not None

    # Synthetic memory feature [density, max_support, confidence]
    mem_features = torch.tensor([[0.75, 0.60, 0.85]], dtype=torch.float)

    model = AuditDDIModel(
        in_channels=ga.x.size(-1),
        edge_feature_dim=ga.edge_attr.size(-1),
        hidden_channels=32,
        architecture_version=MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
        use_toxicity_pair_features=True,
        use_neighbor_memory=True,
    )

    model.eval()
    with torch.no_grad():
        risk_ab, _, _ = model(ga, gb, memory_features=mem_features)
        risk_ba, _, _ = model(gb, ga, memory_features=mem_features)

    # Symmetry check: f(A, B) must equal f(B, A)
    assert torch.allclose(risk_ab, risk_ba, atol=1e-5), f"Symmetry broken: {risk_ab} vs {risk_ba}"


def test_auditddi_backward_pass():
    smiles_a = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    smiles_b = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
    ga = smiles_to_graph(smiles_a, feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)
    gb = smiles_to_graph(smiles_b, feature_schema=FEATURE_SCHEMA_RICH, include_fingerprint_features=True)

    mem_features = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.float)

    model = AuditDDIModel(
        in_channels=ga.x.size(-1),
        edge_feature_dim=ga.edge_attr.size(-1),
        hidden_channels=32,
        architecture_version=MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
        use_neighbor_memory=True,
    )
    model.train()

    risk, _, _ = model(ga, gb, memory_features=mem_features)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(risk, torch.tensor([1.0]))
    loss.backward()

    # Verify gradients flowed into risk classifier and GNN encoder
    assert model.risk_classifier[0].weight.grad is not None
    assert any(p.grad is not None for p in model.encoder.parameters())
