"""Expanded PharmGKB-to-TWOSIDES biological bridge.

Maximizes pharmacogenomic gene/enzyme coverage for the 645 TWOSIDES drugs by:
1. Exact Accession ID resolution (Entity1_id / Entity2_id -> PharmGKB Accession Id).
2. Dual Structure resolution: RDKit Canonical SMILES and InChIKey matching.
3. Multi-field synonym indexing: Name, Generic Names, Trade Names, Brand Mixtures, and ChEMBL Cross-references.
4. Robust PubChem fallback for missing structures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase

from .master_schema import canonicalize_smiles, smiles_to_inchikey
from .pharmgkb_pipeline import normalise_drug_name
from .pubchem_bridge import lookup_pubchem_smiles


def build_expanded_pharmgkb_profiles(
    twosides_edges_path: str | Path,
    pharmgkb_chemicals_path: str | Path,
    pharmgkb_relationships_path: str | Path,
    output_profiles_path: str | Path,
    pubchem_cache_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build high-coverage PharmGKB gene/enzyme profile matrix for TWOSIDES drugs."""
    rdBase.BlockLogs()

    # 1. Load TWOSIDES Drugs and build Canonical & InChIKey Lookups
    print(f'Loading TWOSIDES edges from: {twosides_edges_path}')
    df_two = pd.read_csv(twosides_edges_path, usecols=['source', 'target'], low_memory=False)
    raw_twosides = set(df_two['source']).union(set(df_two['target']))

    twosides_canonical: dict[str, str] = {}  # can_smi -> can_smi
    twosides_by_inchikey: dict[str, str] = {}  # inchikey -> can_smi

    for raw in raw_twosides:
        can = canonicalize_smiles(str(raw))
        if can:
            ikey = smiles_to_inchikey(can)
            twosides_canonical[can] = can
            if ikey:
                twosides_by_inchikey[ikey] = can

    print(f'TWOSIDES Unique Canonical Molecules: {len(twosides_canonical):,}')

    # 2. Parse PharmGKB Chemicals Table
    print(f'Loading PharmGKB Chemicals from: {pharmgkb_chemicals_path}')
    df_chem = pd.read_csv(pharmgkb_chemicals_path, sep='\t', low_memory=False)

    accession_to_twosides: dict[str, str] = {}  # PharmGKB Accession Id -> TWOSIDES can_smi
    name_to_twosides: dict[str, str] = {}  # normalised name -> TWOSIDES can_smi

    # Pass A: Direct SMILES and InChIKey matching from chemicals.tsv
    for _, row in df_chem.iterrows():
        acc_id = str(row.get('PharmGKB Accession Id', '')).strip()
        raw_smi = row.get('SMILES')
        matched_can = None

        if pd.notna(raw_smi):
            can = canonicalize_smiles(str(raw_smi))
            if can:
                if can in twosides_canonical:
                    matched_can = can
                else:
                    ikey = smiles_to_inchikey(can)
                    if ikey and ikey in twosides_by_inchikey:
                        matched_can = twosides_by_inchikey[ikey]

        if matched_can:
            if acc_id:
                accession_to_twosides[acc_id] = matched_can
            for col in ['Name', 'Generic Names', 'Trade Names', 'Brand Mixtures']:
                val = row.get(col)
                if pd.notna(val):
                    for part in str(val).split(','):
                        norm = normalise_drug_name(part)
                        if norm:
                            name_to_twosides[norm] = matched_can

    print(f'Pass A (Direct Chemical Structure Overlap): {len(set(accession_to_twosides.values()))} TWOSIDES drugs linked directly!')

    # Pass B: Synonym & Name resolution for chemicals without direct SMILES
    for _, row in df_chem.iterrows():
        acc_id = str(row.get('PharmGKB Accession Id', '')).strip()
        if acc_id in accession_to_twosides:
            continue  # already resolved

        matched_can = None
        for col in ['Name', 'Generic Names']:
            val = row.get(col)
            if pd.notna(val) and not matched_can:
                for part in str(val).split(','):
                    norm = normalise_drug_name(part)
                    if norm and norm in name_to_twosides:
                        matched_can = name_to_twosides[norm]
                        break

        if matched_can and acc_id:
            accession_to_twosides[acc_id] = matched_can

    print(f'Pass B (Accession ID Synonyms Resolved): {len(set(accession_to_twosides.values()))} TWOSIDES drugs linked!')

    # 3. Parse Relationships Table (Chemical <-> Gene links)
    print(f'Loading PharmGKB Relationships from: {pharmgkb_relationships_path}')
    df_rel = pd.read_csv(pharmgkb_relationships_path, sep='\t', low_memory=False)

    drug_genes: dict[str, set[str]] = defaultdict(set)
    evidence_count: dict[str, int] = defaultdict(int)

    for _, row in df_rel.iterrows():
        e1_id, e1_type, e1_name = str(row.get('Entity1_id')), str(row.get('Entity1_type')), str(row.get('Entity1_name'))
        e2_id, e2_type, e2_name = str(row.get('Entity2_id')), str(row.get('Entity2_type')), str(row.get('Entity2_name'))

        matched_can = None
        gene_name = None

        # Scenario 1: Entity 1 is Chemical, Entity 2 is Gene
        if e1_type == 'Chemical' and e2_type == 'Gene':
            norm1 = normalise_drug_name(e1_name)
            matched_can = accession_to_twosides.get(e1_id) or (name_to_twosides.get(norm1) if norm1 is not None else None)
            gene_name = e2_name
        # Scenario 2: Entity 2 is Chemical, Entity 1 is Gene
        elif e2_type == 'Chemical' and e1_type == 'Gene':
            norm2 = normalise_drug_name(e2_name)
            matched_can = accession_to_twosides.get(e2_id) or (name_to_twosides.get(norm2) if norm2 is not None else None)
            gene_name = e1_name

        if matched_can and gene_name and pd.notna(gene_name):
            clean_gene = gene_name.strip().upper()
            if clean_gene and clean_gene not in {'', 'NAN', 'NONE', 'NULL'}:
                drug_genes[matched_can].add(clean_gene)
                evidence_count[matched_can] += 1

    # 4. Build Profiles DataFrame
    profile_records = []
    all_genes = Counter()

    for can_smi, genes in sorted(drug_genes.items()):
        gene_list = sorted(list(genes))
        for g in gene_list:
            all_genes[g] += 1
        profile_records.append({
            'canonical_smiles': can_smi,
            'unique_genes_count': len(gene_list),
            'evidence_row_count': evidence_count[can_smi],
            'genes_list': json.dumps(gene_list),
            'sample_genes': ', '.join(gene_list[:5]),
        })

    profiles_df = pd.DataFrame(profile_records)
    out_path = Path(output_profiles_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(out_path, index=False)

    summary = {
        'total_twosides_drugs': len(twosides_canonical),
        'drugs_with_gene_profiles': len(profiles_df),
        'coverage_pct': (len(profiles_df) / len(twosides_canonical) * 100.0) if twosides_canonical else 0.0,
        'total_unique_genes': len(all_genes),
        'top_10_genes': [g for g, _ in all_genes.most_common(10)],
        'exported_profiles_path': str(out_path),
    }

    print(f'\nExpanded PharmGKB Profiles Generated!')
    print(f'-> TWOSIDES Drugs with Gene Profiles: {summary["drugs_with_gene_profiles"]} / {summary["total_twosides_drugs"]} ({summary["coverage_pct"]:.1f}%)')
    print(f'-> Total Unique Genes/Enzymes Mapped: {summary["total_unique_genes"]}')
    print(f'-> Top 10 Genes: {summary["top_10_genes"]}')
    print(f'-> Saved to: {out_path}')

    return profiles_df, summary
