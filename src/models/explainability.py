import torch
from torch_geometric.explain import Explainer, GNNExplainer
from rdkit import Chem

RISKY_GROUPS = {"imidazole":"c1cnc[nH]1","triazole":"c1ncnn1","sulfonamide":"S(=O)(=O)N",
                 "carboxylic_acid":"C(=O)[OH]","tertiary_amine":"[NX3](C)(C)C"}

def run_gnn_explainer(model, drug_graph):
    explainer = Explainer(model=model.encoder, algorithm=GNNExplainer(epochs=100),
        explanation_type='model', node_mask_type='attributes', edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'))
    exp = explainer(drug_graph.x, drug_graph.edge_index, batch=drug_graph.batch)
    return exp.edge_mask, exp.node_mask

def literature_match_check(smiles, important_atoms):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return {"error": "invalid SMILES"}
    matches = {}
    for name, smarts in RISKY_GROUPS.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None: continue
        hits = mol.GetSubstructMatches(pat)
        flat = set(i for h in hits for i in h)
        overlap = flat & set(important_atoms)
        if overlap: matches[name] = list(overlap)
    hit = len(matches) > 0
    return {"literature_match": hit, "matched_groups": matches,
            "explanation": f"Matched: {list(matches.keys())}" if hit else "No known-group overlap — flag for review"}

def full_explanation_pipeline(model, drug_a_graph, drug_a_smiles, drug_b_graph, drug_b_smiles):
    """
    FIXED: now explains BOTH drugs correctly — each drug's important
    atoms are checked against THAT SAME drug's SMILES, not the other one.
    """
    # Explain drug A
    _, node_mask_a = run_gnn_explainer(model, drug_a_graph)
    top_atoms_a = torch.topk(node_mask_a.sum(dim=1), k=min(5, node_mask_a.shape[0])).indices.tolist()
    check_a = literature_match_check(drug_a_smiles, top_atoms_a)

    # Explain drug B
    _, node_mask_b = run_gnn_explainer(model, drug_b_graph)
    top_atoms_b = torch.topk(node_mask_b.sum(dim=1), k=min(5, node_mask_b.shape[0])).indices.tolist()
    check_b = literature_match_check(drug_b_smiles, top_atoms_b)

    return {
        "drug_a": {
            "important_atom_indices": top_atoms_a,
            "literature_validated": check_a["literature_match"],
            "matched_functional_groups": check_a["matched_groups"],
            "human_readable_explanation": check_a["explanation"]
        },
        "drug_b": {
            "important_atom_indices": top_atoms_b,
            "literature_validated": check_b["literature_match"],
            "matched_functional_groups": check_b["matched_groups"],
            "human_readable_explanation": check_b["explanation"]
        }
    }
