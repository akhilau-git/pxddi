"""Small, explicit SMARTS motif vocabulary for an experimental PxDDI view.

These descriptors are chemical-prior features, not curated causal DDI
mechanisms.  They are kept separate from atom/bond graphs so the motif-fusion
candidate can be ablated fairly against the existing edge-aware GATv2 model.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem


MOTIF_SCHEMA_SMARTS_COUNTS_V1 = 'smarts_counts_v1'
MOTIF_COUNT_CLIP = 3

# Every SMARTS rule is compiled at import time below.  Rules overlap by design:
# for example, an ester contributes to both ``ester`` and ``carbonyl``.  The
# neural model can learn whether the additional chemical-prior signal helps.
MOTIF_SMARTS: tuple[tuple[str, str], ...] = (
    ('carboxylic_acid', '[CX3](=[OX1])[OX2H1]'),
    ('ester', '[CX3](=[OX1])[OX2][#6]'),
    ('amide', '[NX3][CX3](=[OX1])'),
    ('primary_secondary_amine', '[NX3;H1,H2;!$(N[C,S]=O)]'),
    ('tertiary_amine', '[NX3;H0;!$(N[C,S]=O)]'),
    ('alcohol_or_phenol', '[OX2H][#6;!$([#6]=[OX1])]'),
    ('ether', '[OD2]([#6])[#6]'),
    ('aromatic_ring', 'a1aaaaa1'),
    ('sulfonamide', '[SX4](=[OX1])(=[OX1])[NX3]'),
    ('sulfone', '[SX4](=[OX1])(=[OX1])([#6])[#6]'),
    ('nitrile', '[CX2]#[NX1]'),
    ('nitro', '[N+](=[OX1])[O-]'),
    ('carbon_halogen', '[#6][F,Cl,Br,I]'),
    ('thiol', '[SX2H]'),
    ('thioether', '[SX2]([#6])[#6]'),
    ('phosphate', '[PX4](=[OX1])([OX2])([OX2])'),
    ('carbonyl', '[CX3]=[OX1]'),
)

MOTIF_FEATURE_NAMES = tuple(name for name, _ in MOTIF_SMARTS)
MOTIF_FEATURE_DIM = len(MOTIF_FEATURE_NAMES)


def _compile_smarts() -> tuple[Chem.Mol, ...]:
    patterns = []
    for name, smarts in MOTIF_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise RuntimeError(f'Invalid SMARTS rule for motif {name!r}: {smarts!r}.')
        patterns.append(pattern)
    return tuple(patterns)


MOTIF_PATTERNS = _compile_smarts()


def motif_count_vector(
    molecule: Chem.Mol,
    count_clip: int = MOTIF_COUNT_CLIP,
) -> np.ndarray:
    """Return clipped substructure-match counts in the fixed motif order."""
    if molecule is None:
        raise ValueError('A valid RDKit molecule is required for motif extraction.')
    if count_clip <= 0:
        raise ValueError('count_clip must be positive.')
    return np.asarray(
        [
            min(len(molecule.GetSubstructMatches(pattern, uniquify=True)), count_clip)
            for pattern in MOTIF_PATTERNS
        ],
        dtype=np.float32,
    )


def motif_count_vector_from_smiles(
    smiles: str,
    count_clip: int = MOTIF_COUNT_CLIP,
) -> np.ndarray:
    """Parse a SMILES string and return its fixed-order motif-count vector."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('SMILES must be a non-empty string for motif extraction.')
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(f'Invalid SMILES for motif extraction: {smiles!r}.')
    return motif_count_vector(molecule, count_clip=count_clip)


def motif_substructure_matches(
    smiles: str,
) -> dict[str, list[list[int]]]:
    """Return atom-index matches for the fixed motif vocabulary.

    This helper is deliberately descriptive only.  It lets an explanation
    artifact say which configured SMARTS motifs overlap an important atom; it
    does *not* establish that a motif causes an interaction or toxicity event.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('SMILES must be a non-empty string for motif matching.')
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError(f'Invalid SMILES for motif matching: {smiles!r}.')
    return {
        name: [list(match) for match in molecule.GetSubstructMatches(pattern, uniquify=True)]
        for name, pattern in zip(MOTIF_FEATURE_NAMES, MOTIF_PATTERNS)
    }


def motif_metadata() -> dict[str, object]:
    """Return the auditable vocabulary metadata stored with a candidate run."""
    return {
        'schema': MOTIF_SCHEMA_SMARTS_COUNTS_V1,
        'count_clip': MOTIF_COUNT_CLIP,
        'feature_dimension': MOTIF_FEATURE_DIM,
        'feature_names': list(MOTIF_FEATURE_NAMES),
        'smarts_rules': {name: smarts for name, smarts in MOTIF_SMARTS},
        'interpretation_warning': (
            'These SMARTS counts are experimental chemical-prior features. '
            'They are not verified causal mechanisms or validated explanations.'
        ),
    }
