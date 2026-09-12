"""Biophysical and Pharmacokinetic Engine for AuditDDI.

Grounds cold-start DDI and polypharmacy evaluation in first-principles
medicinal chemistry, human Cytochrome P450 (CYP) pharmacophores,
Site-of-Metabolism (SoM) rules, and competitive enzyme clearance collisions.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------------------------------------------------------------------
# CYP Enzyme Nomenclature and Pharmacophore SMARTS
# ---------------------------------------------------------------------------
CYP_ENZYMES: tuple[str, ...] = ("CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP2C19")
BIOPHYSICAL_DIM: int = 12

# Validated medicinal chemistry pharmacophore rules
_SMARTS_RULES: dict[str, list[tuple[str, str, float]]] = {
    "CYP2D6": [
        # Basic aliphatic amines, piperidines, tertiary amines with aromatic anchor
        ("basic_amine", "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(Na)]", 1.5),
        ("tertiary_amine", "[NX3]([#6])([#6])[#6]", 1.2),
        ("piperidine_or_pyrrolidine", "[NX3;R]1[#6][#6][#6][#6]1", 1.3),
        ("aromatic_ring", "a1aaaaa1", 0.6),
    ],
    "CYP2C9": [
        # Weakly acidic lipophiles: carboxylic acids, sulfonamides, acidic phenols
        ("carboxylic_acid", "[CX3](=[OX1])[OX2H1]", 2.0),
        ("sulfonamide", "[SX4](=[OX1])(=[OX1])[NX3]", 1.8),
        ("acidic_phenol", "c[OX2H]", 1.2),
        ("tetrazole", "c1nnnn1", 1.5),
    ],
    "CYP1A2": [
        # Planar, flat heteroaromatic systems, purines, xanthines
        ("purine_like", "n1cnc2n1cnc2", 2.0),
        ("quinoline_isoquinoline", "c1cccc2ncccc12", 1.5),
        ("indole", "c1ccc2[nH]ccc2c1", 1.3),
        ("aromatic_ring", "a1aaaaa1", 0.8),
    ],
    "CYP2C19": [
        # Intermediate lipophilic amides, benzimidazoles, lactams
        ("amide", "[NX3][CX3](=[OX1])", 1.2),
        ("benzimidazole", "c1ccc2[nH]cnc2c1", 1.6),
        ("sulfoxide", "[SX3]=[OX1]", 1.4),
        ("lactam", "[NX3;R][CX3;R](=[OX1])", 1.3),
    ],
    "CYP3A4": [
        # Bulky, lipophilic multi-ring systems, macrolides, azoles
        ("imidazole_triazole", "c1c[nH,n]cn1", 1.8),
        ("steroid_or_macroring", "[#6;R]1~[#6;R]~[#6;R]~[#6;R]~[#6;R]~[#6;R]1", 1.0),
        ("ester_or_ether_bulk", "[OD2]([#6])[#6]", 0.8),
    ],
}

_COMPILED_SMARTS: dict[str, list[tuple[str, Chem.Mol, float]]] = {}
for _cyp, _rules in _SMARTS_RULES.items():
    _compiled = []
    for _name, _smarts, _weight in _rules:
        _mol = Chem.MolFromSmarts(_smarts)
        if _mol is not None:
            _compiled.append((_name, _mol, _weight))
    _COMPILED_SMARTS[_cyp] = _compiled

# Site-of-metabolism oxidation motifs
_SOM_SMARTS: list[tuple[str, Chem.Mol]] = [
    ("n_dealkylation", Chem.MolFromSmarts("[NX3]([CH3,CH2])[#6]")),
    ("o_dealkylation", Chem.MolFromSmarts("[OX2][CH3,CH2]")),
    ("aliphatic_hydroxylation", Chem.MolFromSmarts("[CH2,CH3;!$(C=O);!$(C#N)]")),
    ("benzylic_oxidation", Chem.MolFromSmarts("c[CH2,CH3]")),
    ("aromatic_hydroxylation", Chem.MolFromSmarts("c1ccccc1")),
]
_SOM_SMARTS = [(name, mol) for name, mol in _SOM_SMARTS if mol is not None]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


def compute_cyp_affinities(mol: Chem.Mol) -> dict[str, float]:
    """Compute continuous liability/substrate probability for top human CYP enzymes."""
    if mol is None:
        return {cyp: 0.2 for cyp in CYP_ENZYMES}

    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0])
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    scores: dict[str, float] = {}

    # 1. CYP3A4 Score: bulky, lipophilic, large molecular surface, multiple rings
    cyp3a4_raw = 0.0
    if mw > 350.0:
        cyp3a4_raw += min((mw - 350.0) / 150.0, 2.0)
    if logp > 2.0:
        cyp3a4_raw += min((logp - 2.0) / 1.5, 1.5)
    if aromatic_rings >= 2:
        cyp3a4_raw += 1.0
    if rotb >= 4:
        cyp3a4_raw += 0.8
    for _, smarts_mol, weight in _COMPILED_SMARTS.get("CYP3A4", []):
        if mol.HasSubstructMatch(smarts_mol):
            cyp3a4_raw += weight
    scores["CYP3A4"] = _sigmoid(cyp3a4_raw - 1.5)

    # 2. CYP2D6 Score: basic nitrogen, aromatic anchor, low/moderate MW
    cyp2d6_raw = 0.0
    has_basic_n = False
    for name, smarts_mol, weight in _COMPILED_SMARTS.get("CYP2D6", []):
        if mol.HasSubstructMatch(smarts_mol):
            cyp2d6_raw += weight
            if "amine" in name:
                has_basic_n = True
    if has_basic_n and aromatic_rings >= 1:
        cyp2d6_raw += 1.5
    if mw < 450.0:
        cyp2d6_raw += 0.5
    scores["CYP2D6"] = _sigmoid(cyp2d6_raw - 1.8)

    # 3. CYP2C9 Score: acidic functional groups, moderate LogP
    cyp2c9_raw = 0.0
    for _, smarts_mol, weight in _COMPILED_SMARTS.get("CYP2C9", []):
        if mol.HasSubstructMatch(smarts_mol):
            cyp2c9_raw += weight
    if 1.0 <= logp <= 4.5:
        cyp2c9_raw += 0.8
    if hbd >= 1:
        cyp2c9_raw += 0.5
    scores["CYP2C9"] = _sigmoid(cyp2c9_raw - 1.5)

    # 4. CYP1A2 Score: flat, planar heteroaromatic rings, low Csp3, lower MW
    cyp1a2_raw = 0.0
    for _, smarts_mol, weight in _COMPILED_SMARTS.get("CYP1A2", []):
        if mol.HasSubstructMatch(smarts_mol):
            cyp1a2_raw += weight
    if fsp3 < 0.35:
        cyp1a2_raw += 1.2
    if aromatic_rings >= 2 and mw < 400.0:
        cyp1a2_raw += 1.0
    if tpsa < 80.0:
        cyp1a2_raw += 0.6
    scores["CYP1A2"] = _sigmoid(cyp1a2_raw - 1.6)

    # 5. CYP2C19 Score: moderate lipophilicity, amides/sulfoxides, lactams
    cyp2c19_raw = 0.0
    for _, smarts_mol, weight in _COMPILED_SMARTS.get("CYP2C19", []):
        if mol.HasSubstructMatch(smarts_mol):
            cyp2c19_raw += weight
    if 1.0 <= logp <= 3.5:
        cyp2c19_raw += 0.7
    if hba >= 2:
        cyp2c19_raw += 0.5
    scores["CYP2C19"] = _sigmoid(cyp2c19_raw - 1.5)

    return scores


def compute_admet_pharmacokinetics(mol: Chem.Mol) -> dict[str, float]:
    """In silico pharmacokinetic parameter estimation for absorption, distribution, and clearance."""
    if mol is None:
        return {
            "p_eff": 0.5,
            "f_unbound": 0.1,
            "logp_norm": 0.5,
            "tpsa_norm": 0.5,
            "mw_norm": 0.5,
            "rotb_norm": 0.5,
            "hepatic_clearance_bias": 0.5,
        }

    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0])
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # 1. Intestinal Permeability proxy (Caco-2 / Papp in silico):
    # Favored by moderate lipophilicity (LogP 1-3), low TPSA (< 100), low H-bond donors (< 3)
    p_eff_raw = (
        (logp * 0.4)
        - (max(0.0, tpsa - 60.0) / 70.0)
        - (max(0.0, hbd - 2.0) * 0.5)
        - (max(0.0, rotb - 5.0) * 0.1)
    )
    p_eff = _sigmoid(p_eff_raw)

    # 2. Unbound Fraction in Plasma (fu):
    # Highly lipophilic, large molecules bind strongly to albumin (fu is low, < 0.05).
    # Small polar molecules have higher free unbound fraction.
    binding_raw = (logp * 0.7) + (mw / 300.0) - (tpsa / 100.0)
    f_unbound = 1.0 - _sigmoid(binding_raw - 1.0)
    f_unbound = max(0.01, min(0.99, f_unbound))

    # 3. Hepatic vs. Renal Clearance Bias:
    # Lipophilic, bulky compounds are primarily cleared by hepatic metabolism (CYPs).
    # Hydrophilic, polar, small compounds (LogP < 1, TPSA > 100) are cleared renally.
    hepatic_bias = _sigmoid((logp * 0.6) + (mw / 400.0) - (tpsa / 80.0))

    return {
        "p_eff": p_eff,
        "f_unbound": f_unbound,
        "logp_norm": float(np.clip((logp + 2.0) / 8.0, 0.0, 1.0)),
        "tpsa_norm": float(np.clip(tpsa / 200.0, 0.0, 1.0)),
        "mw_norm": float(np.clip(mw / 800.0, 0.0, 1.0)),
        "rotb_norm": float(np.clip(rotb / 15.0, 0.0, 1.0)),
        "hepatic_clearance_bias": hepatic_bias,
    }


def compute_biophysical_vector(smiles: str) -> np.ndarray:
    """Compute complete 12-dimensional biophysical pharmacokinetic descriptor vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(BIOPHYSICAL_DIM, dtype=np.float32)

    cyp_scores = compute_cyp_affinities(mol)
    admet = compute_admet_pharmacokinetics(mol)

    vec = np.array([
        cyp_scores["CYP3A4"],
        cyp_scores["CYP2D6"],
        cyp_scores["CYP2C9"],
        cyp_scores["CYP1A2"],
        cyp_scores["CYP2C19"],
        admet["p_eff"],
        admet["f_unbound"],
        admet["logp_norm"],
        admet["tpsa_norm"],
        admet["mw_norm"],
        admet["rotb_norm"],
        admet["hepatic_clearance_bias"],
    ], dtype=np.float32)

    return vec


def identify_site_of_metabolism(smiles: str) -> list[dict[str, Any]]:
    """Identify the primary labile atomic positions vulnerable to hepatic oxidation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    som_sites: list[dict[str, Any]] = []
    for motif_name, smarts_mol in _SOM_SMARTS:
        matches = mol.GetSubstructMatches(smarts_mol)
        for match in matches:
            som_sites.append({
                "motif": motif_name,
                "atom_indices": list(match),
                "primary_atom_symbol": mol.GetAtomWithIdx(match[0]).GetSymbol(),
            })
    return som_sites


def compute_metabolic_collision_score(smiles_a: str, smiles_b: str) -> dict[str, Any]:
    """Compute symmetric competitive metabolic clearance collision between two drugs."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    cyp_a = compute_cyp_affinities(mol_a)
    cyp_b = compute_cyp_affinities(mol_b)
    pk_a = compute_admet_pharmacokinetics(mol_a)
    pk_b = compute_admet_pharmacokinetics(mol_b)

    # 1. CYP Shared Substrate Overlap
    # Dot product of CYP probability profiles
    cyp_overlap = 0.0
    enzyme_breakdown: dict[str, float] = {}
    for cyp in CYP_ENZYMES:
        score = cyp_a[cyp] * cyp_b[cyp]
        enzyme_breakdown[cyp] = score
        cyp_overlap += score
    cyp_overlap_normalized = float(np.clip(cyp_overlap / 1.5, 0.0, 1.0))

    # 2. High-Binding Displacement Collision
    # If both drugs bind heavily to plasma proteins (low fu), one displaces the other
    displacement_risk = (1.0 - pk_a["f_unbound"]) * (1.0 - pk_b["f_unbound"])

    # 3. Combined Clearance Collision Index
    collision_index = float(np.clip(
        0.65 * cyp_overlap_normalized + 0.35 * displacement_risk,
        0.0,
        1.0,
    ))

    # Determine dominant colliding enzyme
    dominant_cyp = max(enzyme_breakdown.keys(), key=lambda k: enzyme_breakdown[k])

    return {
        "collision_index": collision_index,
        "cyp_overlap": cyp_overlap_normalized,
        "displacement_risk": displacement_risk,
        "dominant_cyp": dominant_cyp,
        "dominant_cyp_score": enzyme_breakdown[dominant_cyp],
        "enzyme_breakdown": enzyme_breakdown,
        "cyp_a": cyp_a,
        "cyp_b": cyp_b,
        "pk_a": pk_a,
        "pk_b": pk_b,
    }
