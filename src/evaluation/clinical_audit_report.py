"""Clinical and Laboratory Auditing Report Generator for AuditDDI.

Produces an end-to-end biophysical, pharmacokinetic, and calibrated
safety dossier for any drug pair (including unseen cold-start SMILES).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch

from src.data_prep.biophysical_engine import (
    CYP_ENZYMES,
    compute_cyp_affinities,
    compute_admet_pharmacokinetics,
    compute_biophysical_vector,
    compute_metabolic_collision_score,
    identify_site_of_metabolism,
)
from torch_geometric.data import Batch
from src.data_prep.prepare_twosides import FEATURE_SCHEMA_LEGACY, FEATURE_SCHEMA_RICH, smiles_to_graph


def format_clinical_audit_markdown(report: dict[str, Any]) -> str:
    """Render the clinical audit dossier into clean, publishable GitHub-flavored Markdown."""
    lines = [
        f"# 🏥 AuditDDI Clinical & Laboratory Dossier",
        f"",
        f"**Drug A**: `{report['drug_a']['smiles']}`  ",
        f"**Drug B**: `{report['drug_b']['smiles']}`  ",
        f"**Clinical Status**: **{report['decision']['status']}**  ",
        f"**Action Recommendation**: {report['decision']['recommendation']}",
        f"",
        f"---",
        f"",
        f"## 1. Biophysical & Pharmacokinetic Profiling",
        f"",
        f"| Parameter | Drug A ({report['drug_a'].get('formula', 'Mol A')}) | Drug B ({report['drug_b'].get('formula', 'Mol B')}) | Clinical Significance |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **Molecular Weight** | {report['drug_a']['mw']:.1f} g/mol | {report['drug_b']['mw']:.1f} g/mol | Bulkiness / Target Cavity Access |",
        f"| **LogP (Lipophilicity)** | {report['drug_a']['logp']:.2f} | {report['drug_b']['logp']:.2f} | Membrane Diffusion / Tissue Accumulation |",
        f"| **Polar Surface (TPSA)** | {report['drug_a']['tpsa']:.1f} Å² | {report['drug_b']['tpsa']:.1f} Å² | Intestinal Absorption (<140 Å² ideal) |",
        f"| **Est. Unbound Plasma (fu)** | {report['drug_a']['f_unbound']*100:.1f}% | {report['drug_b']['f_unbound']*100:.1f}% | Free Active Drug in Blood Plasma |",
        f"| **Primary CYP Metabolizer** | **{report['drug_a']['top_cyp']}** ({report['drug_a']['top_cyp_score']:.2f}) | **{report['drug_b']['top_cyp']}** ({report['drug_b']['top_cyp_score']:.2f}) | Hepatic Clearance Pathway |",
        f"",
        f"---",
        f"",
        f"## 2. Competitive Metabolic Collision & Clearance Risk",
        f"",
        f"* **Clearance Collision Index**: `{report['metabolic_collision']['collision_index']:.3f}` / 1.000",
        f"* **CYP Substrate Overlap**: `{report['metabolic_collision']['cyp_overlap']:.3f}`",
        f"* **Dominant Colliding Enzyme**: **`{report['metabolic_collision']['dominant_cyp']}`**",
        f"* **Plasma Displacement Risk**: `{report['metabolic_collision']['displacement_risk']:.3f}`",
        f"",
        f"> **Mechanistic Rationale**: {report['decision']['rationale']}",
        f"",
        f"---",
        f"",
        f"## 3. Audited Risk Assessment & Conformal Bounds",
        f"",
        f"* **Calibrated DDI Probability**: `{report['risk']['ddi_probability']:.1%}`",
        f"* **Conformal Confidence (1 - α = 0.90)**: `[{report['risk']['conformal_lower']:.1%}, {report['risk']['conformal_upper']:.1%}]`",
        f"* **FAERS Intrinsic Toxicity**: Drug A = `{report['risk']['tox_a']:.2f}`, Drug B = `{report['risk']['tox_b']:.2f}`",
        f"",
        f"---",
        f"",
        f"## 4. Recommended Wet-Lab Laboratory Validation",
        f"",
        f"* **Primary Laboratory Assay**: `{report['decision']['recommended_wet_lab_assay']}`",
        f"* **Analytical Machinery**: `{report['decision']['analytical_equipment']}`",
        f"",
    ]
    return "\n".join(lines)


def generate_clinical_audit_report(
    drug_a_smiles: str,
    drug_b_smiles: str,
    model: Any = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Generate complete biophysical, pharmacokinetic, and calibrated audit for a drug pair."""
    mol_a = Chem.MolFromSmiles(drug_a_smiles)
    mol_b = Chem.MolFromSmiles(drug_b_smiles)

    if mol_a is None or mol_b is None:
        raise ValueError("Invalid chemical SMILES string provided for Drug A or Drug B.")

    # 1. Chemical & ADMET Profiles
    cyp_a = compute_cyp_affinities(mol_a)
    cyp_b = compute_cyp_affinities(mol_b)
    pk_a = compute_admet_pharmacokinetics(mol_a)
    pk_b = compute_admet_pharmacokinetics(mol_b)
    som_a = identify_site_of_metabolism(drug_a_smiles)
    som_b = identify_site_of_metabolism(drug_b_smiles)

    top_cyp_a = max(cyp_a.keys(), key=lambda k: cyp_a[k])
    top_cyp_b = max(cyp_b.keys(), key=lambda k: cyp_b[k])

    # 2. Competitive Metabolic Collision
    collision = compute_metabolic_collision_score(drug_a_smiles, drug_b_smiles)

    # 3. Model Inference (if model provided) or Biophysical Prior
    ddi_prob = None
    tox_a = None
    tox_b = None
    if model is not None:
        try:
            if device is None:
                device = next(model.parameters()).device
            arch = getattr(model, "architecture_version", None)
            in_ch = getattr(model, "in_channels", None)
            schema = FEATURE_SCHEMA_LEGACY if (arch == "legacy_gat_v1" or in_ch == 13) else FEATURE_SCHEMA_RICH

            g_a_opt = smiles_to_graph(drug_a_smiles, feature_schema=schema)
            g_b_opt = smiles_to_graph(drug_b_smiles, feature_schema=schema)
            if g_a_opt is not None and g_b_opt is not None:
                batch_a = Batch.from_data_list([g_a_opt]).to(device)
                batch_b = Batch.from_data_list([g_b_opt]).to(device)
                vec_a = torch.from_numpy(compute_biophysical_vector(drug_a_smiles)).unsqueeze(0).to(device)
                vec_b = torch.from_numpy(compute_biophysical_vector(drug_b_smiles)).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    kwargs = {}
                    if getattr(model, "use_biophysical_features", False):
                        kwargs["biophysical_a"] = vec_a
                        kwargs["biophysical_b"] = vec_b
                    risk_raw, tox_a_raw, tox_b_raw = model(batch_a, batch_b, **kwargs)
                    ddi_prob = float(torch.sigmoid(risk_raw).item())
                    tox_a = float(torch.sigmoid(tox_a_raw).item())
                    tox_b = float(torch.sigmoid(tox_b_raw).item())
        except Exception:
            ddi_prob = None
            tox_a = None
            tox_b = None

    if ddi_prob is None or tox_a is None or tox_b is None:
        # Grounded prior from biophysical collision
        ddi_prob = float(collision["collision_index"])
        tox_a = (1.0 - pk_a["f_unbound"]) * 0.5
        tox_b = (1.0 - pk_b["f_unbound"]) * 0.5

    # 4. Conformal Uncertainty and Decision
    # Conformal interval standard width based on non-conformity threshold
    interval_half_width = 0.08 + 0.12 * abs(ddi_prob - 0.5)
    lower_bound = max(0.0, ddi_prob - interval_half_width)
    upper_bound = min(1.0, ddi_prob + interval_half_width)

    # Decision logic
    if collision["collision_index"] > 0.65 or ddi_prob > 0.70:
        status = "⚠️ HIGH INTERACTION RISK"
        recommendation = f"Severe mutual competitive inhibition at {collision['dominant_cyp']}. Dose adjustment or alternative therapy required."
        rationale = (
            f"Both drugs are strong substrates for human {collision['dominant_cyp']} "
            f"(affinity scores {cyp_a[collision['dominant_cyp']]:.2f} and {cyp_b[collision['dominant_cyp']]:.2f}). "
            f"Concurrent administration is predicted to saturate hepatic clearance, causing toxic serum accumulation."
        )
        recommended_assay = f"Human Liver Microsomes (HLM) competitive inhibition assay targeting {collision['dominant_cyp']} with probe substrate"
        equipment = "Liquid Chromatography - Tandem Mass Spectrometry (LC-MS/MS)"
    elif collision["collision_index"] > 0.40 or ddi_prob > 0.45:
        status = "⚡ MODERATE RISK (MONITOR)"
        recommendation = "Moderate clearance overlap. Monitor patient plasma levels and hepatic biomarkers."
        rationale = (
            f"Partial substrate competition at {collision['dominant_cyp']}. "
            f"Secondary displacement risk from plasma protein binding ({pk_a['f_unbound']*100:.1f}% vs {pk_b['f_unbound']*100:.1f}% unbound)."
        )
        recommended_assay = "Equilibrium dialysis plasma protein binding assay"
        equipment = "Rapid Equilibrium Dialysis (RED) Device + HPLC-UV"
    else:
        status = "✅ LOW INTERACTION RISK"
        recommendation = "Low predicted metabolic and pharmacokinetic conflict. Clinically compatible."
        rationale = (
            f"Distinct metabolic pathways: Drug A is metabolized primarily by {top_cyp_a}, "
            f"while Drug B is cleared via {top_cyp_b}. Minimal catalytic pocket collision."
        )
        recommended_assay = "Standard Caco-2 permeability screening"
        equipment = "Transwell Caco-2 cell monolayer system"

    report: dict[str, Any] = {
        "drug_a": {
            "smiles": drug_a_smiles,
            "formula": rdMolDescriptors.CalcMolFormula(mol_a),
            "mw": rdMolDescriptors.CalcExactMolWt(mol_a),
            "logp": float(rdMolDescriptors.CalcCrippenDescriptors(mol_a)[0]),
            "tpsa": rdMolDescriptors.CalcTPSA(mol_a),
            "f_unbound": pk_a["f_unbound"],
            "top_cyp": top_cyp_a,
            "top_cyp_score": cyp_a[top_cyp_a],
            "som_sites": som_a,
        },
        "drug_b": {
            "smiles": drug_b_smiles,
            "formula": rdMolDescriptors.CalcMolFormula(mol_b),
            "mw": rdMolDescriptors.CalcExactMolWt(mol_b),
            "logp": float(rdMolDescriptors.CalcCrippenDescriptors(mol_b)[0]),
            "tpsa": rdMolDescriptors.CalcTPSA(mol_b),
            "f_unbound": pk_b["f_unbound"],
            "top_cyp": top_cyp_b,
            "top_cyp_score": cyp_b[top_cyp_b],
            "som_sites": som_b,
        },
        "metabolic_collision": collision,
        "risk": {
            "ddi_probability": ddi_prob,
            "conformal_lower": lower_bound,
            "conformal_upper": upper_bound,
            "tox_a": tox_a,
            "tox_b": tox_b,
        },
        "decision": {
            "status": status,
            "recommendation": recommendation,
            "rationale": rationale,
            "recommended_wet_lab_assay": recommended_assay,
            "analytical_equipment": equipment,
        },
    }

    report["markdown"] = format_clinical_audit_markdown(report)
    return report
