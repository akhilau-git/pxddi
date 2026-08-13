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

def full_explanation_pipeline(model, drug_a_graph, drug_b_smiles):
    _, node_mask = run_gnn_explainer(model, drug_a_graph)
    top = torch.topk(node_mask.sum(dim=1), k=min(5, node_mask.shape[0])).indices.tolist()
    check = literature_match_check(drug_b_smiles, top)
    return {"important_atom_indices": top, "literature_validated": check["literature_match"],
            "matched_functional_groups": check["matched_groups"],
            "human_readable_explanation": check["explanation"]}
