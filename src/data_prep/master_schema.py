"""Master schema definition and strict validation for the Unified AuditDDI Graph.

Defines standardized data models for:
- Drug nodes (canonical SMILES, InChIKey, multi-modal attributes)
- Interaction edges (polypharmacy DDI events, source provenance, split assignment)
- Future expansion modules (BindingDB target affinities, GEO transcriptomics)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase

INCHIKEY_PATTERN = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')


def canonicalize_smiles(smiles: str | None) -> str | None:
    """Return RDKit canonical isomeric SMILES or None if malformed/empty."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_inchikey(smiles: str | None) -> str | None:
    """Return standard InChIKey or None if malformed/empty."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


@dataclass(frozen=True)
class DrugNode:
    """Standardized multi-modal drug node."""

    drug_id: str  # RDKit Canonical SMILES (Primary Key)
    inchikey: str  # Standardized 27-character InChIKey
    display_name: str | None = None
    synonyms: list[str] = field(default_factory=list)
    source_ids: dict[str, str] = field(default_factory=dict)
    drug_type: str = 'small_molecule'

    # Multi-Modal Node Attributes
    gene_symbols: list[str] = field(default_factory=list)  # PharmGKB linked genes/enzymes
    gene_vector_multihot: list[int] = field(default_factory=list)  # Binary presence vector
    toxicity_score: float | None = None  # FAERS clinical severity score
    n_faers_reports: int | None = None  # Number of adverse event reports

    # Future Expansion Modules (Retained as inactive/empty to avoid schema changes)
    bindingdb_targets: dict[str, float] = field(default_factory=dict)
    geo_expression_signatures: dict[str, float] = field(default_factory=dict)
    is_bindingdb_active: bool = False
    is_geo_active: bool = False

    def __post_init__(self) -> None:
        """Enforce strict validation rules on drug node creation."""
        if not self.drug_id or not isinstance(self.drug_id, str):
            raise ValueError('drug_id must be a non-empty canonical SMILES string.')
        canonical = canonicalize_smiles(self.drug_id)
        if canonical is None or canonical != self.drug_id:
            raise ValueError(
                f'drug_id must be an exact RDKit canonical SMILES. '
                f'Provided: {self.drug_id!r}, Canonical: {canonical!r}'
            )

        if not self.inchikey or not isinstance(self.inchikey, str):
            raise ValueError('inchikey must be a non-empty string.')
        if not INCHIKEY_PATTERN.match(self.inchikey):
            raise ValueError(f'Invalid InChIKey format: {self.inchikey!r}')

        if self.toxicity_score is not None:
            if not isinstance(self.toxicity_score, (int, float)):
                raise ValueError(f'toxicity_score must be a numeric float, got: {type(self.toxicity_score)}')

        if self.n_faers_reports is not None and self.n_faers_reports < 0:
            raise ValueError(f'n_faers_reports cannot be negative: {self.n_faers_reports}')

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary with JSON-friendly attributes."""
        res = asdict(self)
        res['synonyms_json'] = json.dumps(self.synonyms)
        res['source_ids_json'] = json.dumps(self.source_ids)
        res['gene_symbols_json'] = json.dumps(self.gene_symbols)
        res['gene_vector_json'] = json.dumps(self.gene_vector_multihot)
        return res


@dataclass(frozen=True)
class DDIEdge:
    """Standardized drug-drug interaction edge."""

    drug_a_id: str  # Canonical SMILES
    drug_b_id: str  # Canonical SMILES
    interaction_type: str | int  # Polypharmacy side effect class or term
    interaction_source: str = 'TWOSIDES'  # TWOSIDES, FAERS, etc.
    evidence_count: int = 1
    confidence_score: float | None = None
    split_group: str | None = None  # train, val, test_transductive, test_s1_cold, test_s2_semi

    def __post_init__(self) -> None:
        """Enforce strict edge validation rules."""
        if not self.drug_a_id or not isinstance(self.drug_a_id, str):
            raise ValueError('drug_a_id must be a non-empty canonical SMILES.')
        if not self.drug_b_id or not isinstance(self.drug_b_id, str):
            raise ValueError('drug_b_id must be a non-empty canonical SMILES.')
        if self.drug_a_id == self.drug_b_id:
            raise ValueError(f'Self-loops are forbidden in DDI graph: {self.drug_a_id}')
        if self.evidence_count < 0:
            raise ValueError(f'evidence_count cannot be negative: {self.evidence_count}')
        if self.split_group is not None:
            valid_splits = {'train', 'val', 'test_transductive', 'test_s1_cold', 'test_s2_semi', 'unassigned'}
            if self.split_group not in valid_splits:
                raise ValueError(f'Invalid split_group: {self.split_group!r}. Must be one of {valid_splits}')

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge to dictionary."""
        return asdict(self)


class MasterGraphCatalog:
    """In-memory collection and validator for the unified heterogeneous graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, DrugNode] = {}
        self.edges: list[DDIEdge] = []

    def add_node(self, node: DrugNode) -> None:
        """Add or update a drug node."""
        if not isinstance(node, DrugNode):
            raise TypeError(f'Expected DrugNode, got {type(node)}')
        self.nodes[node.drug_id] = node

    def add_edge(self, edge: DDIEdge) -> None:
        """Add an interaction edge after ensuring endpoint nodes exist."""
        if not isinstance(edge, DDIEdge):
            raise TypeError(f'Expected DDIEdge, got {type(edge)}')
        if edge.drug_a_id not in self.nodes:
            raise KeyError(f'drug_a_id {edge.drug_a_id} not registered in graph nodes.')
        if edge.drug_b_id not in self.nodes:
            raise KeyError(f'drug_b_id {edge.drug_b_id} not registered in graph nodes.')
        self.edges.append(edge)

    def summary(self) -> dict[str, Any]:
        """Return diagnostic metrics of the catalog."""
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        nodes_with_genes = sum(1 for n in self.nodes.values() if n.gene_symbols)
        nodes_with_tox = sum(1 for n in self.nodes.values() if n.toxicity_score is not None)
        return {
            'total_nodes': n_nodes,
            'total_edges': n_edges,
            'nodes_with_pharmgkb_genes': nodes_with_genes,
            'pharmgkb_coverage_pct': (nodes_with_genes / n_nodes * 100.0) if n_nodes else 0.0,
            'nodes_with_faers_toxicity': nodes_with_tox,
            'faers_coverage_pct': (nodes_with_tox / n_nodes * 100.0) if n_nodes else 0.0,
            'bindingdb_module_status': 'inactive_expansion_module',
            'geo_module_status': 'inactive_expansion_module',
        }

    def export_tables(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Save nodes and edges to validated CSV files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        nodes_df = pd.DataFrame([n.to_dict() for n in self.nodes.values()])
        edges_df = pd.DataFrame([e.to_dict() for e in self.edges])

        nodes_path = out_dir / 'master_drug_nodes.csv'
        edges_path = out_dir / 'master_ddi_edges.csv'

        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        return nodes_path, edges_path
