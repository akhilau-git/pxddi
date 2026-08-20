"""SMILES-to-graph conversion with legacy and edge-aware feature schemas."""

from __future__ import annotations

from rdkit import Chem
import torch
from torch_geometric.data import Data

from .molecular_motifs import motif_count_vector


FEATURE_SCHEMA_LEGACY = 'legacy_v1'
FEATURE_SCHEMA_RICH = 'rich_v2'

# The legacy schema is retained for the already-audited deployed checkpoint.
LEGACY_ATOM_LIST = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'H']
LEGACY_NUM_ATOM_FEATURES = len(LEGACY_ATOM_LIST) + 3

# The richer schema exposes chemically meaningful atom, bond, and stereo data
# to a new edge-aware GNN. An explicit OTHER bucket avoids silently treating
# unfamiliar chemistry as an all-zero vector.
RICH_ATOM_SYMBOLS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
RICH_DEGREES = [0, 1, 2, 3, 4, 5]
RICH_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
RICH_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
RICH_HYDROGEN_COUNTS = [0, 1, 2, 3, 4]
RICH_CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
RICH_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
RICH_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def _one_hot_with_other(value, choices) -> list[float]:
    """One-hot encode a categorical chemistry value with an explicit OTHER bin."""
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


RICH_NUM_ATOM_FEATURES = (
    len(RICH_ATOM_SYMBOLS) + 1
    + len(RICH_DEGREES) + 1
    + len(RICH_FORMAL_CHARGES) + 1
    + len(RICH_HYBRIDIZATIONS) + 1
    + len(RICH_HYDROGEN_COUNTS) + 1
    + len(RICH_CHIRAL_TAGS) + 1
    + 2  # aromatic and ring membership flags
)
NUM_BOND_FEATURES = len(RICH_BOND_TYPES) + 1 + len(RICH_BOND_STEREOS) + 1 + 2

# Backward-compatible public constant for legacy callers and checkpoints.
NUM_ATOM_FEATURES = LEGACY_NUM_ATOM_FEATURES


def atom_features(atom, feature_schema: str = FEATURE_SCHEMA_LEGACY) -> list[float]:
    """Create feature vectors for one RDKit atom under a named schema."""
    if feature_schema == FEATURE_SCHEMA_LEGACY:
        one_hot = [float(atom.GetSymbol() == symbol) for symbol in LEGACY_ATOM_LIST]
        return one_hot + [atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic())]
    if feature_schema != FEATURE_SCHEMA_RICH:
        raise ValueError(f'Unknown molecular feature schema: {feature_schema}.')

    features = []
    features += _one_hot_with_other(atom.GetSymbol(), RICH_ATOM_SYMBOLS)
    features += _one_hot_with_other(atom.GetDegree(), RICH_DEGREES)
    features += _one_hot_with_other(atom.GetFormalCharge(), RICH_FORMAL_CHARGES)
    features += _one_hot_with_other(atom.GetHybridization(), RICH_HYBRIDIZATIONS)
    features += _one_hot_with_other(atom.GetTotalNumHs(includeNeighbors=True), RICH_HYDROGEN_COUNTS)
    features += _one_hot_with_other(atom.GetChiralTag(), RICH_CHIRAL_TAGS)
    features += [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    return features


def bond_features(bond) -> list[float]:
    """Create an edge feature vector for one RDKit bond."""
    features = []
    features += _one_hot_with_other(bond.GetBondType(), RICH_BOND_TYPES)
    features += _one_hot_with_other(bond.GetStereo(), RICH_BOND_STEREOS)
    features += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    return features


def graph_compatibility_reason(smiles) -> str | None:
    """Explain why a SMILES cannot be represented as a molecular graph."""
    if not isinstance(smiles, str) or not smiles.strip():
        return 'missing_or_non_string_smiles'
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return 'invalid_smiles'
    if molecule.GetNumAtoms() == 0:
        return 'empty_molecule'
    if molecule.GetNumBonds() == 0:
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            return 'counterion_or_inorganic_only_structure'
        return 'single_atom_or_disconnected_structure'
    return None


def smiles_to_graph(
    smiles,
    feature_schema: str = FEATURE_SCHEMA_LEGACY,
    include_motif_features: bool = False,
):
    """Convert a graph-compatible SMILES into a PyG graph under one schema."""
    reason = graph_compatibility_reason(smiles)
    if reason is not None:
        return None
    normalized_smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(normalized_smiles)
    if molecule is None:  # Defensive: compatibility check above already parsed it.
        return None

    node_features = [atom_features(atom, feature_schema) for atom in molecule.GetAtoms()]
    edge_pairs = []
    edge_features = []
    for bond in molecule.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_pairs.extend([[source, target], [target, source]])
        if feature_schema == FEATURE_SCHEMA_RICH:
            feature = bond_features(bond)
            edge_features.extend([feature, feature])

    if not edge_pairs:
        return None
    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
        smiles=normalized_smiles,
    )
    if feature_schema == FEATURE_SCHEMA_RICH:
        graph.edge_attr = torch.tensor(edge_features, dtype=torch.float)
    if include_motif_features:
        graph.motif_features = torch.tensor(
            motif_count_vector(molecule), dtype=torch.float
        ).unsqueeze(0)
    return graph
