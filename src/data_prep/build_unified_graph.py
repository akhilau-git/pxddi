"""Build and export the Unified Heterogeneous Graph for AuditDDI.

Integrates:
1. TWOSIDES: Polypharmacy DDI interactions and canonical molecular graph nodes.
2. PharmGKB: Multi-hot binary pharmacogenomic gene/enzyme vectors (CYP450, transporters).
3. FAERS: Clinical adverse-event toxicity scores and reporting volume.
4. Future Expansion: Inactive schemas for BindingDB targets and GEO transcriptomics.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .master_schema import DDIEdge, DrugNode, MasterGraphCatalog, canonicalize_smiles, smiles_to_inchikey
from .pharmgkb_pipeline import normalise_drug_name

DEFAULT_TOP_GENES = 50


def extract_top_gene_vocabulary(
    gene_profiles_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_GENES,
) -> list[str]:
    """Return the top-K most prevalent genes/enzymes across all profiled drugs."""
    counter: Counter[str] = Counter()
    for raw in gene_profiles_df['genes_list'].dropna():
        try:
            genes = json.loads(raw) if isinstance(raw, str) else list(raw)
            for g in genes:
                clean_g = str(g).strip().upper()
                if clean_g:
                    counter[clean_g] += 1
        except Exception:
            continue
    return [gene for gene, _ in counter.most_common(top_k)]


def encode_multihot_gene_vector(
    genes: list[str],
    vocabulary: list[str],
) -> list[int]:
    """Encode a drug's gene associations into a binary multi-hot vector."""
    gene_set = {str(g).strip().upper() for g in genes}
    return [1 if vocab_gene in gene_set else 0 for vocab_gene in vocabulary]


def build_unified_graph(
    twosides_edges_path: str | Path,
    pharmgkb_profiles_path: str | Path | None = None,
    faers_bridge_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_genes: int = DEFAULT_TOP_GENES,
) -> tuple[MasterGraphCatalog, dict[str, Any]]:
    """Construct and strictly validate the unified heterogeneous graph."""
    edges_source = Path(twosides_edges_path)
    if not edges_source.is_file():
        raise FileNotFoundError(f'TWOSIDES edges file not found: {edges_source}')

    print(f'Loading TWOSIDES edges from: {edges_source}')
    df_edges = pd.read_csv(edges_source, usecols=['source', 'target', 'interaction_type'], low_memory=False)

    # 1. Standardize and Canonicalize all TWOSIDES drugs
    print('Canonicalizing drug structures...')
    raw_structures = set(df_edges['source']).union(set(df_edges['target']))
    smiles_map: dict[str, tuple[str, str]] = {}  # raw -> (canonical, inchikey)

    for raw in raw_structures:
        can = canonicalize_smiles(str(raw))
        if can:
            ikey = smiles_to_inchikey(can)
            if ikey:
                smiles_map[str(raw)] = (can, ikey)

    print(f'Total unique raw structures: {len(raw_structures):,}; Valid canonical: {len(set(smiles_map.values())):,}')

    # 2. Load PharmGKB Profiles if available
    gene_lookup: dict[str, list[str]] = {}
    gene_vocab: list[str] = []

    if pharmgkb_profiles_path and Path(pharmgkb_profiles_path).is_file():
        print(f'Loading PharmGKB gene profiles from: {pharmgkb_profiles_path}')
        df_genes = pd.read_csv(pharmgkb_profiles_path)
        gene_vocab = extract_top_gene_vocabulary(df_genes, top_k=top_k_genes)
        print(f'Top {len(gene_vocab)} Gene/Enzyme Vocabulary: {gene_vocab[:10]}...')

        for _, row in df_genes.iterrows():
            can = canonicalize_smiles(str(row.get('canonical_smiles')))
            if can and pd.notna(row.get('genes_list')):
                try:
                    genes = json.loads(row['genes_list']) if isinstance(row['genes_list'], str) else list(row['genes_list'])
                    gene_lookup[can] = [str(g).strip().upper() for g in genes]
                except Exception:
                    pass

    # 3. Load FAERS Toxicity Bridge if available
    faers_lookup: dict[str, tuple[float, int]] = {}
    if faers_bridge_path and Path(faers_bridge_path).is_file():
        print(f'Loading FAERS toxicity bridge from: {faers_bridge_path}')
        df_faers = pd.read_csv(faers_bridge_path)
        for _, row in df_faers.iterrows():
            can = canonicalize_smiles(str(row.get('canonical_smiles')))
            if can and pd.notna(row.get('toxicity_score')):
                score = float(row['toxicity_score'])
                n_rep = int(row['n_reports']) if pd.notna(row.get('n_reports')) else 0
                faers_lookup[can] = (score, n_rep)

    # 4. Construct Master Nodes
    catalog = MasterGraphCatalog()
    unique_canonical_drugs = set(smiles_map.values())

    for can_smi, ikey in unique_canonical_drugs:
        genes = gene_lookup.get(can_smi, [])
        gene_vec = encode_multihot_gene_vector(genes, gene_vocab) if gene_vocab else []
        faers_data = faers_lookup.get(can_smi)
        tox_score = faers_data[0] if faers_data else None
        n_reports = faers_data[1] if faers_data else None

        node = DrugNode(
            drug_id=can_smi,
            inchikey=ikey,
            gene_symbols=genes,
            gene_vector_multihot=gene_vec,
            toxicity_score=tox_score,
            n_faers_reports=n_reports,
            is_bindingdb_active=False,
            is_geo_active=False,
        )
        catalog.add_node(node)

    # 5. Construct Master Edges
    print('Constructing interaction edges...')
    edge_seen = set()
    skipped_self_loops = 0

    for _, row in df_edges.iterrows():
        raw_a, raw_b = str(row['source']), str(row['target'])
        if raw_a not in smiles_map or raw_b not in smiles_map:
            continue
        can_a, _ = smiles_map[raw_a]
        can_b, _ = smiles_map[raw_b]

        if can_a == can_b:
            skipped_self_loops += 1
            continue

        edge_key = (min(can_a, can_b), max(can_a, can_b), str(row['interaction_type']))
        if edge_key in edge_seen:
            continue
        edge_seen.add(edge_key)

        edge = DDIEdge(
            drug_a_id=can_a,
            drug_b_id=can_b,
            interaction_type=str(row['interaction_type']),
            interaction_source='TWOSIDES',
            evidence_count=1,
            split_group='unassigned',
        )
        catalog.add_edge(edge)

    summary = catalog.summary()
    summary['top_gene_vocabulary'] = gene_vocab
    summary['skipped_self_loops'] = skipped_self_loops
    print(f'Graph Build Complete: {summary["total_nodes"]} nodes, {summary["total_edges"]} edges.')
    print(f'PharmGKB Gene Coverage: {summary["nodes_with_pharmgkb_genes"]} / {summary["total_nodes"]} ({summary["pharmgkb_coverage_pct"]:.1f}%)')
    print(f'FAERS Toxicity Coverage: {summary["nodes_with_faers_toxicity"]} / {summary["total_nodes"]} ({summary["faers_coverage_pct"]:.1f}%)')

    if output_dir:
        nodes_p, edges_p = catalog.export_tables(output_dir)
        summary['exported_nodes_path'] = str(nodes_p)
        summary['exported_edges_path'] = str(edges_p)
        vocab_path = Path(output_dir) / 'gene_vocabulary.json'
        vocab_path.write_text(json.dumps(gene_vocab, indent=2), encoding='utf-8')
        summary['exported_gene_vocab_path'] = str(vocab_path)

    return catalog, summary
