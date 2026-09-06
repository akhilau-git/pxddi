"""Offline perturbation explanations for experimental PxDDI candidates.

The deployed legacy API explanation remains separate.  This module is designed
for a small, auditable set of candidate-evaluation examples: it measures how a
candidate's *raw* DDI score changes when an atom, bond feature, or motif input
is masked.  It is deliberately not a clinical explanation system and does not
turn an internal attention weight into chemical evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch, Data

try:  # Training runs import ``models`` from src; tests import ``src.models``.
    from data_prep.molecular_motifs import motif_substructure_matches
except ModuleNotFoundError:  # pragma: no cover - import style depends on caller.
    from ..data_prep.molecular_motifs import motif_substructure_matches


EXPLANATION_METHOD = 'single_component_occlusion_v1'
EXPLANATION_LIMITATIONS = [
    'Attribution measures local sensitivity of this trained model, not chemical causality.',
    'Masked feature vectors can be out-of-distribution molecular inputs; do not interpret them as synthesizable interventions.',
    'Attention weights, when available, are model-internal associations and are not validated interaction mechanisms.',
    'These artifacts use raw model probabilities so calibration does not hide attribution changes.',
]


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _single_graph_batch(graph: Data, device: torch.device) -> Batch:
    return Batch.from_data_list([graph.clone()]).to(device)


def raw_pair_probability(model, graph_a: Data, graph_b: Data) -> float:
    """Return the candidate's uncalibrated probability for one graph pair."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch_a = _single_graph_batch(graph_a, _model_device(model))
            batch_b = _single_graph_batch(graph_b, _model_device(model))
            risk_logit, _, _ = model(batch_a, batch_b)
            return float(torch.sigmoid(risk_logit).reshape(-1)[0].cpu().item())
    finally:
        model.train(was_training)


def _mask_atoms(graph: Data, atom_indices: list[int]) -> Data:
    masked = graph.clone()
    masked.x = masked.x.clone()
    masked.x[atom_indices] = 0.0
    return masked


def _retain_only_atoms(graph: Data, atom_indices: list[int]) -> Data:
    retained = graph.clone()
    retained.x = retained.x.clone()
    keep = torch.zeros(retained.x.shape[0], dtype=torch.bool, device=retained.x.device)
    keep[atom_indices] = True
    retained.x[~keep] = 0.0
    return retained


def _unique_undirected_bonds(graph: Data) -> list[tuple[int, int]]:
    if not hasattr(graph, 'edge_attr'):
        return []
    edge_pairs = graph.edge_index.detach().cpu().t().tolist()
    return sorted({tuple(sorted((int(source), int(target)))) for source, target in edge_pairs})


def _mask_bonds(graph: Data, bonds: list[tuple[int, int]]) -> Data:
    if not hasattr(graph, 'edge_attr'):
        raise ValueError('Bond occlusion requires a graph with edge_attr.')
    masked = graph.clone()
    masked.edge_attr = masked.edge_attr.clone()
    requested = {tuple(sorted(bond)) for bond in bonds}
    edges = masked.edge_index.detach().cpu().t().tolist()
    edge_mask = torch.tensor(
        [tuple(sorted((int(source), int(target)))) in requested for source, target in edges],
        dtype=torch.bool,
        device=masked.edge_attr.device,
    )
    masked.edge_attr[edge_mask] = 0.0
    return masked


def _mask_motifs(graph: Data, motif_indices: list[int]) -> Data:
    if not hasattr(graph, 'motif_features'):
        raise ValueError('Motif occlusion requires graph.motif_features.')
    masked = graph.clone()
    masked.motif_features = masked.motif_features.clone()
    masked.motif_features[:, motif_indices] = 0.0
    return masked


def _atom_symbols(smiles: str, atom_count: int) -> list[str | None]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() != atom_count:
        return [None] * atom_count
    return [atom.GetSymbol() for atom in molecule.GetAtoms()]


def _bond_descriptions(smiles: str, bonds: list[tuple[int, int]]) -> dict[tuple[int, int], str | None]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {bond: None for bond in bonds}
    descriptions: dict[tuple[int, int], str | None] = {}
    for begin, end in bonds:
        bond = molecule.GetBondBetweenAtoms(begin, end)
        descriptions[(begin, end)] = str(bond.GetBondType()) if bond is not None else None
    return descriptions


def _rank_by_absolute_change(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda entry: (-abs(float(entry['raw_probability_change'])), entry['component_index']),
    )


def _atom_occlusion(
    model,
    graph_a: Data,
    graph_b: Data,
    smiles: str,
    side: str,
    baseline: float,
) -> list[dict[str, Any]]:
    graph = graph_a if side == 'a' else graph_b
    symbols = _atom_symbols(smiles, int(graph.x.shape[0]))
    entries = []
    for atom_index in range(graph.x.shape[0]):
        masked_a, masked_b = (
            (_mask_atoms(graph_a, [atom_index]), graph_b)
            if side == 'a'
            else (graph_a, _mask_atoms(graph_b, [atom_index]))
        )
        score_after_mask = raw_pair_probability(model, masked_a, masked_b)
        entries.append({
            'component_index': int(atom_index),
            'atom_index': int(atom_index),
            'atom_symbol': symbols[atom_index],
            'raw_probability_after_mask': score_after_mask,
            'raw_probability_change': baseline - score_after_mask,
        })
    return _rank_by_absolute_change(entries)


def _bond_occlusion(
    model,
    graph_a: Data,
    graph_b: Data,
    smiles: str,
    side: str,
    baseline: float,
) -> list[dict[str, Any]]:
    graph = graph_a if side == 'a' else graph_b
    bonds = _unique_undirected_bonds(graph)
    descriptions = _bond_descriptions(smiles, bonds)
    entries = []
    for bond_index, bond in enumerate(bonds):
        masked_a, masked_b = (
            (_mask_bonds(graph_a, [bond]), graph_b)
            if side == 'a'
            else (graph_a, _mask_bonds(graph_b, [bond]))
        )
        score_after_mask = raw_pair_probability(model, masked_a, masked_b)
        entries.append({
            'component_index': int(bond_index),
            'bond_atom_indices': [int(bond[0]), int(bond[1])],
            'bond_type': descriptions[bond],
            'raw_probability_after_mask': score_after_mask,
            'raw_probability_change': baseline - score_after_mask,
        })
    return _rank_by_absolute_change(entries)


def _motif_occlusion(
    model,
    graph_a: Data,
    graph_b: Data,
    motif_names: list[str],
    side: str,
    baseline: float,
) -> list[dict[str, Any]]:
    graph = graph_a if side == 'a' else graph_b
    if not hasattr(graph, 'motif_features'):
        return []
    values = graph.motif_features.detach().cpu().reshape(-1).tolist()
    entries = []
    for motif_index, value in enumerate(values):
        masked_a, masked_b = (
            (_mask_motifs(graph_a, [motif_index]), graph_b)
            if side == 'a'
            else (graph_a, _mask_motifs(graph_b, [motif_index]))
        )
        score_after_mask = raw_pair_probability(model, masked_a, masked_b)
        entries.append({
            'component_index': int(motif_index),
            'motif_name': motif_names[motif_index],
            'input_count': float(value),
            'raw_probability_after_mask': score_after_mask,
            'raw_probability_change': baseline - score_after_mask,
        })
    return _rank_by_absolute_change(entries)


def _functional_group_context(smiles: str, important_atoms: list[int]) -> list[dict[str, Any]]:
    important = set(important_atoms)
    context = []
    for name, matches in motif_substructure_matches(smiles).items():
        overlapping = [match for match in matches if important.intersection(match)]
        if overlapping:
            context.append({'motif_name': name, 'atom_index_matches': overlapping})
    return context


def _top_cross_attention_pairs(
    weights: torch.Tensor,
    source_symbols: list[str | None],
    target_symbols: list[str | None],
    top_k: int,
) -> list[dict[str, Any]]:
    flattened = weights.detach().cpu().reshape(-1)
    take = min(top_k, int(flattened.numel()))
    if take == 0:
        return []
    values, indices = torch.topk(flattened, k=take)
    target_count = weights.shape[1]
    return [
        {
            'source_atom_index': int(index.item() // target_count),
            'source_atom_symbol': source_symbols[int(index.item() // target_count)],
            'target_atom_index': int(index.item() % target_count),
            'target_atom_symbol': target_symbols[int(index.item() % target_count)],
            'attention_weight': float(value.item()),
        }
        for value, index in zip(values, indices)
    ]


def _motif_atom_sets(smiles: str) -> dict[str, list[int]]:
    """Return the union of atoms for every configured motif present in a drug."""
    return {
        name: sorted({atom_index for match in matches for atom_index in match})
        for name, matches in motif_substructure_matches(smiles).items()
        if matches
    }


def _top_cross_attention_motif_pairs(
    weights: torch.Tensor,
    source_smiles: str,
    target_smiles: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Aggregate atom attention over configured SMARTS motif pairs.

    The result summarizes the candidate's internal atom-attention matrix over
    named motif atom sets.  It is an association view for audit purposes, not
    an observed pharmacological or causal drug-drug mechanism.
    """
    source_motifs = _motif_atom_sets(source_smiles)
    target_motifs = _motif_atom_sets(target_smiles)
    associations = []
    for source_name, source_atoms in source_motifs.items():
        for target_name, target_atoms in target_motifs.items():
            association_weights = weights[source_atoms][:, target_atoms]
            associations.append({
                'source_motif': source_name,
                'source_atom_indices': source_atoms,
                'target_motif': target_name,
                'target_atom_indices': target_atoms,
                'mean_attention_weight': float(association_weights.mean().item()),
                'total_attention_weight': float(association_weights.sum().item()),
            })
    return sorted(
        associations,
        key=lambda entry: (-entry['mean_attention_weight'], entry['source_motif'], entry['target_motif']),
    )[:top_k]


def _cross_attention_associations(
    model,
    graph_a: Data,
    graph_b: Data,
    smiles_a: str,
    smiles_b: str,
    top_k: int,
) -> dict[str, Any]:
    if getattr(model, 'cross_drug_attention', None) is None:
        return {
            'available': False,
            'reason': 'Model architecture has no cross-drug attention layer.',
        }
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            batch_a = _single_graph_batch(graph_a, _model_device(model))
            batch_b = _single_graph_batch(graph_b, _model_device(model))
            a_to_b, b_to_a = model.cross_drug_attention_maps(batch_a, batch_b)
    finally:
        model.train(was_training)
    symbols_a = _atom_symbols(smiles_a, int(graph_a.x.shape[0]))
    symbols_b = _atom_symbols(smiles_b, int(graph_b.x.shape[0]))
    return {
        'available': True,
        'interpretation_warning': (
            'Attention weights are model-internal associations only; they are not '
            'validated atom-to-atom interaction mechanisms.'
        ),
        'drug_a_to_drug_b': _top_cross_attention_pairs(
            a_to_b[0], symbols_a, symbols_b, top_k
        ),
        'drug_b_to_drug_a': _top_cross_attention_pairs(
            b_to_a[0], symbols_b, symbols_a, top_k
        ),
        'configured_motif_associations': {
            'available': True,
            'aggregation': (
                'Mean and total of the pair-isolated atom-attention weights over '
                'atoms matched by each configured SMARTS motif.'
            ),
            'interpretation_warning': (
                'Configured motif associations summarize internal attention only. '
                'They are not validated DDI mechanisms or chemical plausibility evidence.'
            ),
            'drug_a_to_drug_b': _top_cross_attention_motif_pairs(
                a_to_b[0], smiles_a, smiles_b, top_k
            ),
            'drug_b_to_drug_a': _top_cross_attention_motif_pairs(
                b_to_a[0], smiles_b, smiles_a, top_k
            ),
        },
    }


def _canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def render_occlusion_svg(
    smiles: str,
    atom_occlusions: list[dict[str, Any]],
    bond_occlusions: list[dict[str, Any]],
    destination: str | Path,
) -> None:
    """Render a self-contained SVG of the selected local occlusion results.

    Orange means masking locally lowered the raw model score; blue means
    masking locally raised it.  Colours indicate score sensitivity, never a
    toxicophore or a causal pharmacological mechanism.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Cannot draw invalid SMILES {smiles!r}.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atom_colours: dict[int, tuple[float, float, float]] = {}
    for entry in atom_occlusions:
        index = int(entry['atom_index'])
        change = float(entry['raw_probability_change'])
        atom_colours[index] = (0.90, 0.45, 0.05) if change >= 0 else (0.10, 0.45, 0.75)
    bond_colours: dict[int, tuple[float, float, float]] = {}
    for entry in bond_occlusions:
        first, second = entry['bond_atom_indices']
        bond = molecule.GetBondBetweenAtoms(int(first), int(second))
        if bond is None:
            continue
        change = float(entry['raw_probability_change'])
        bond_colours[bond.GetIdx()] = (
            (0.90, 0.45, 0.05) if change >= 0 else (0.10, 0.45, 0.75)
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(560, 320)
    options = drawer.drawOptions()
    options.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=list(atom_colours),
        highlightAtomColors=atom_colours,
        highlightBonds=list(bond_colours),
        highlightBondColors=bond_colours,
    )
    drawer.FinishDrawing()
    destination.write_text(drawer.GetDrawingText(), encoding='utf-8')


def _canonical_reencoding_stability(
    model,
    smiles_a: str,
    smiles_b: str,
    baseline: float,
    graph_builder: Callable[[str], Data | None] | None,
) -> dict[str, Any]:
    if graph_builder is None:
        return {'status': 'not_run', 'reason': 'No graph builder was supplied.'}
    canonical_a, canonical_b = _canonical_smiles(smiles_a), _canonical_smiles(smiles_b)
    if canonical_a is None or canonical_b is None:
        return {'status': 'not_run', 'reason': 'SMILES could not be canonicalized.'}
    rebuilt_a, rebuilt_b = graph_builder(canonical_a), graph_builder(canonical_b)
    if rebuilt_a is None or rebuilt_b is None:
        return {'status': 'not_run', 'reason': 'Canonical SMILES could not be represented.'}
    canonical_score = raw_pair_probability(model, rebuilt_a, rebuilt_b)
    return {
        'status': 'evaluated',
        'canonical_smiles_a': canonical_a,
        'canonical_smiles_b': canonical_b,
        'raw_probability_after_canonical_reencoding': canonical_score,
        'absolute_raw_probability_difference': abs(baseline - canonical_score),
    }


def explain_pair_with_occlusion(
    model,
    graph_a: Data,
    graph_b: Data,
    smiles_a: str,
    smiles_b: str,
    motif_names: list[str] | tuple[str, ...] = (),
    top_k: int = 5,
    graph_builder: Callable[[str], Data | None] | None = None,
) -> dict[str, Any]:
    """Explain one pair using score perturbations and candidate-only audits."""
    if top_k <= 0:
        raise ValueError('top_k must be positive.')
    baseline = raw_pair_probability(model, graph_a, graph_b)
    atoms_a = _atom_occlusion(model, graph_a, graph_b, smiles_a, 'a', baseline)
    atoms_b = _atom_occlusion(model, graph_a, graph_b, smiles_b, 'b', baseline)
    bonds_a = _bond_occlusion(model, graph_a, graph_b, smiles_a, 'a', baseline)
    bonds_b = _bond_occlusion(model, graph_a, graph_b, smiles_b, 'b', baseline)
    motifs_a = _motif_occlusion(model, graph_a, graph_b, list(motif_names), 'a', baseline)
    motifs_b = _motif_occlusion(model, graph_a, graph_b, list(motif_names), 'b', baseline)
    top_atoms_a = [entry['atom_index'] for entry in atoms_a[:top_k]]
    top_atoms_b = [entry['atom_index'] for entry in atoms_b[:top_k]]
    masked_score = raw_pair_probability(
        model, _mask_atoms(graph_a, top_atoms_a), _mask_atoms(graph_b, top_atoms_b)
    )
    retained_score = raw_pair_probability(
        model,
        _retain_only_atoms(graph_a, top_atoms_a),
        _retain_only_atoms(graph_b, top_atoms_b),
    )
    swapped_score = raw_pair_probability(model, graph_b, graph_a)
    return {
        'method': EXPLANATION_METHOD,
        'scope_limitations': EXPLANATION_LIMITATIONS,
        'raw_probability': baseline,
        'symmetry_check': {
            'raw_probability_after_swapping_drug_order': swapped_score,
            'absolute_raw_probability_difference': abs(baseline - swapped_score),
        },
        'canonical_reencoding_stability': _canonical_reencoding_stability(
            model, smiles_a, smiles_b, baseline, graph_builder
        ),
        'atom_attribution_quality': {
            'top_k_per_drug': top_k,
            'fidelity_mask_top_atoms_raw_probability': masked_score,
            'fidelity_raw_probability_change': baseline - masked_score,
            'sufficiency_retain_top_atoms_raw_probability': retained_score,
            'sufficiency_raw_probability_change': baseline - retained_score,
            'interpretation_warning': (
                'Fidelity and sufficiency are local perturbation checks, not proof of '
                'explanation correctness or chemical plausibility.'
            ),
        },
        'drug_a': {
            'smiles': smiles_a,
            'top_atom_occlusions': atoms_a[:top_k],
            'top_bond_occlusions': bonds_a[:top_k],
            'top_motif_occlusions': motifs_a[:top_k],
            'overlapping_configured_motifs': _functional_group_context(smiles_a, top_atoms_a),
        },
        'drug_b': {
            'smiles': smiles_b,
            'top_atom_occlusions': atoms_b[:top_k],
            'top_bond_occlusions': bonds_b[:top_k],
            'top_motif_occlusions': motifs_b[:top_k],
            'overlapping_configured_motifs': _functional_group_context(smiles_b, top_atoms_b),
        },
        'cross_drug_attention_associations': _cross_attention_associations(
            model, graph_a, graph_b, smiles_a, smiles_b, top_k
        ),
    }


def select_representative_indices(
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    maximum_examples: int,
) -> list[int]:
    """Select a deterministic mix of correct and incorrect evaluation cases."""
    if maximum_examples <= 0:
        return []
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=float)
    if len(labels) != len(predictions):
        raise ValueError('labels and predictions must have equal length.')
    predicted = predictions >= threshold
    groups = [
        np.where((labels == 1) & ~predicted)[0],  # false negatives
        np.where((labels == 0) & predicted)[0],   # false positives
        np.where((labels == 1) & predicted)[0],   # true positives
        np.where((labels == 0) & ~predicted)[0],  # true negatives
    ]
    selected: list[int] = []
    for group in groups:
        for index in sorted(group.tolist(), key=lambda item: (-abs(predictions[item] - threshold), item)):
            if len(selected) >= maximum_examples:
                return selected
            selected.append(int(index))
            break
    remaining = sorted(
        set(range(len(labels))) - set(selected),
        key=lambda item: (-abs(predictions[item] - threshold), item),
    )
    return selected + [int(index) for index in remaining[: maximum_examples - len(selected)]]


def explain_multimodal_pair(
    model: Any,
    graph_a: Data | None = None,
    graph_b: Data | None = None,
    smiles_a: str | None = None,
    smiles_b: str | None = None,
    fp_a: torch.Tensor | None = None,
    fp_b: torch.Tensor | None = None,
    gene_a: torch.Tensor | None = None,
    gene_b: torch.Tensor | None = None,
    gene_mask_a: torch.Tensor | None = None,
    gene_mask_b: torch.Tensor | None = None,
    tox_a: torch.Tensor | None = None,
    tox_b: torch.Tensor | None = None,
    gene_vocabulary: list[str] | None = None,
    calibrator: dict[str, Any] | None = None,
    cache: Any = None,
    drug_a_smiles: str | None = None,
    drug_b_smiles: str | None = None,
    gene_names: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Provide structured clinical and chemical attribution for a multimodal pair.

    Quantifies:
    1. Overall predicted raw and calibrated interaction probability.
    2. Modality contribution ablations (Graph vs +ECFP vs +Genes vs +Toxicity).
    3. Active metabolizing enzymes (CYP450 / Transporters) present on each drug.
    """
    from src.models.calibration import apply_calibrator

    # Support cache-based drug loading
    if cache is not None:
        if drug_a_smiles is not None:
            smiles_a = drug_a_smiles
        if drug_b_smiles is not None:
            smiles_b = drug_b_smiles
        if smiles_a is None or smiles_b is None:
            raise ValueError('smiles_a and smiles_b (or drug_a_smiles, drug_b_smiles) must be provided when using cache.')
        cache.register_drug(smiles_a)
        cache.register_drug(smiles_b)
        graph_a = cache.graphs[smiles_a]
        graph_b = cache.graphs[smiles_b]
        fp_a = cache.fingerprints.get(smiles_a)
        fp_b = cache.fingerprints.get(smiles_b)
        gene_a = cache.gene_vectors.get(smiles_a)
        gene_b = cache.gene_vectors.get(smiles_b)
        gene_mask_a = cache.gene_masks.get(smiles_a)
        gene_mask_b = cache.gene_masks.get(smiles_b)
        tox_a = cache.toxicity_scalars.get(smiles_a)
        tox_b = cache.toxicity_scalars.get(smiles_b)

    if gene_names is not None and gene_vocabulary is None:
        gene_vocabulary = gene_names

    if graph_a is None or graph_b is None or smiles_a is None or smiles_b is None:
        raise ValueError('graph_a, graph_b, smiles_a, and smiles_b are required.')

    device = _model_device(model)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            batch_a = _single_graph_batch(graph_a, device)
            batch_b = _single_graph_batch(graph_b, device)

            fp_t_a = fp_a.unsqueeze(0).to(device) if fp_a is not None else None
            fp_t_b = fp_b.unsqueeze(0).to(device) if fp_b is not None else None
            gene_t_a = gene_a.unsqueeze(0).to(device) if gene_a is not None else None
            gene_t_b = gene_b.unsqueeze(0).to(device) if gene_b is not None else None
            gmask_t_a = gene_mask_a.unsqueeze(0).to(device) if gene_mask_a is not None else None
            gmask_t_b = gene_mask_b.unsqueeze(0).to(device) if gene_mask_b is not None else None
            tox_t_a = tox_a.unsqueeze(0).to(device) if tox_a is not None else None
            tox_t_b = tox_b.unsqueeze(0).to(device) if tox_b is not None else None

            # 1. Full Multimodal Prediction
            logits_full, _, _ = model(
                batch_a, batch_b,
                fp_a=fp_t_a, fp_b=fp_t_b,
                gene_a=gene_t_a, gene_b=gene_t_b,
                gene_mask_a=gmask_t_a, gene_mask_b=gmask_t_b,
                clinical_tox_a=tox_t_a, clinical_tox_b=tox_t_b,
            )
            p_full = float(torch.sigmoid(logits_full).reshape(-1)[0].cpu().item())

            # 2. Ablations to isolate modality contribution
            logits_no_tox, _, _ = model(
                batch_a, batch_b,
                fp_a=fp_t_a, fp_b=fp_t_b,
                gene_a=gene_t_a, gene_b=gene_t_b,
                gene_mask_a=gmask_t_a, gene_mask_b=gmask_t_b,
                clinical_tox_a=None, clinical_tox_b=None,
            )
            p_no_tox = float(torch.sigmoid(logits_no_tox).reshape(-1)[0].cpu().item())

            logits_no_gene, _, _ = model(
                batch_a, batch_b,
                fp_a=fp_t_a, fp_b=fp_t_b,
                gene_a=None, gene_b=None,
                clinical_tox_a=tox_t_a, clinical_tox_b=tox_t_b,
            )
            p_no_gene = float(torch.sigmoid(logits_no_gene).reshape(-1)[0].cpu().item())

            logits_graph_only, _, _ = model(batch_a, batch_b)
            p_graph_only = float(torch.sigmoid(logits_graph_only).reshape(-1)[0].cpu().item())

    finally:
        model.train(was_training)

    # 3. Active Gene / Enzyme extraction
    active_genes_a: list[str] = []
    active_genes_b: list[str] = []
    if gene_vocabulary is not None and gene_a is not None and gene_b is not None:
        active_genes_a = [gene_vocabulary[i] for i, val in enumerate(gene_a.cpu().numpy().ravel()) if val > 0.5 and i < len(gene_vocabulary)]
        active_genes_b = [gene_vocabulary[i] for i, val in enumerate(gene_b.cpu().numpy().ravel()) if val > 0.5 and i < len(gene_vocabulary)]

    shared_enzymes = sorted(set(active_genes_a).intersection(set(active_genes_b)))

    # 4. Calibration
    calibrated_prob = float(apply_calibrator(np.array([p_full]), calibrator)[0]) if calibrator else p_full

    shared_gene_hotspots = []
    if gene_vocabulary is not None and gene_a is not None and gene_b is not None:
        vocab_map = {g: idx for idx, g in enumerate(gene_vocabulary)}
        for enzyme in shared_enzymes:
            idx = vocab_map.get(enzyme)
            if idx is not None and idx < gene_a.numel() and idx < gene_b.numel():
                sig = float((gene_a[idx].item() + gene_b[idx].item()) / 2.0)
            else:
                sig = 1.0
            shared_gene_hotspots.append({'gene': enzyme, 'combined_signal': sig})
    else:
        shared_gene_hotspots = [{'gene': enzyme, 'combined_signal': 1.0} for enzyme in shared_enzymes]

    return {
        'smiles_a': smiles_a,
        'smiles_b': smiles_b,
        'predicted_raw_probability': p_full,
        'predicted_calibrated_probability': calibrated_prob,
        'overall_risk_score': calibrated_prob,
        'risk_level': 'High Risk' if calibrated_prob >= 0.65 else ('Moderate Risk' if calibrated_prob >= 0.35 else 'Low Risk'),
        'modality_marginal_contributions': {
            'molecular_graph_baseline': p_graph_only,
            'delta_faers_clinical_toxicity': p_full - p_no_tox,
            'delta_pharmgkb_pharmacogenomics': p_full - p_no_gene,
            'delta_combined_external_knowledge': p_full - p_graph_only,
        },
        'modality_contributions': {
            'molecular_graph_baseline': p_graph_only,
            'faers_clinical_toxicity': p_full - p_no_tox,
            'pharmgkb_pharmacogenomics': p_full - p_no_gene,
            'combined_external_signal': p_full - p_graph_only,
        },
        'pharmacogenomic_context': {
            'drug_a_enzymes': active_genes_a,
            'drug_b_enzymes': active_genes_b,
            'shared_cyp_competition': shared_enzymes,
            'potential_metabolic_bottleneck': len(shared_enzymes) > 0,
        },
        'shared_pharmacogenomic_genes': shared_gene_hotspots,
        'clinical_toxicity_context': {
            'drug_a_faers_score': float(tox_a.item()) if tox_a is not None else None,
            'drug_b_faers_score': float(tox_b.item()) if tox_b is not None else None,
        },
    }

