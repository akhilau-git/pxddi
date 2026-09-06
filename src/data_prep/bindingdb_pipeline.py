"""BindingDB Target Affinity Pipeline for AuditDDI.

Parses and integrates BindingDB drug-target binding data (target proteins,
kinases, receptors, and affinity profiles) to augment drug nodes with
biological target interaction vectors.

Supports:
1. Parsing `bindingdb_drugs.csv`, `bindingdb_targets.csv`, and `drug_target_edges.csv`.
2. Resolving drug structures to RDKit canonical SMILES and standard InChIKeys.
3. Extracting the top-K most prevalent biomedical targets across all drugs.
4. Generating multi-hot target presence/affinity vectors for multimodal fusion.
5. Updating `master_drug_nodes.csv` in-place or exporting to a new path.
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

from .master_schema import canonicalize_smiles, smiles_to_inchikey

DEFAULT_TOP_TARGETS = 50

# Candidate column names for resilient loading across different exports
DRUG_SMILES_COLUMNS = ('canonical_smiles', 'smiles', 'structure', 'isomeric_smiles', 'drug_smiles')
DRUG_ID_COLUMNS = ('bindingdb_drug_id', 'drug_id', 'compound_id', 'id', 'source_id')
DRUG_NAME_COLUMNS = ('drug_name', 'name', 'chemical_name', 'display_name')

TARGET_ID_COLUMNS = ('bindingdb_target_id', 'target_id', 'id', 'uniprot_id')
TARGET_NAME_COLUMNS = ('target_name', 'name', 'protein_name', 'gene_name', 'target', 'gene_symbol')

EDGE_DRUG_COLUMNS = ('bindingdb_drug_id', 'drug_id', 'source', 'compound_id', 'drug')
EDGE_TARGET_COLUMNS = ('bindingdb_target_id', 'target_id', 'target', 'protein_id', 'uniprot_id')
EDGE_AFFINITY_COLUMNS = ('affinity_nm', 'ic50_nm', 'ki_nm', 'kd_nm', 'affinity', 'score', 'activity_value')


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Find the first matching column name from candidates (case-insensitive)."""
    col_map = {str(c).lower().strip(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in col_map:
            return col_map[cand.lower().strip()]
    return None


def extract_top_target_vocabulary(
    target_lists: list[list[str]],
    top_k: int = DEFAULT_TOP_TARGETS,
) -> list[str]:
    """Return the top-K most prevalent targets across all profiled drugs."""
    counter: Counter[str] = Counter()
    for targets in target_lists:
        for t in targets:
            clean_t = str(t).strip().upper()
            if clean_t:
                counter[clean_t] += 1
    return [target for target, _ in counter.most_common(top_k)]


def encode_multihot_target_vector(
    drug_targets: list[str],
    vocabulary: list[str],
) -> list[int]:
    """Encode a drug's target associations into a binary multi-hot vector."""
    target_set = {str(t).strip().upper() for t in drug_targets}
    return [1 if vocab_target in target_set else 0 for vocab_target in vocabulary]


def parse_bindingdb_directory(
    bindingdb_dir: str | Path,
    master_nodes_path: str | Path | None = None,
    top_k_targets: int = DEFAULT_TOP_TARGETS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw BindingDB directory and generate standardized drug target profiles.

    Parameters
    ----------
    bindingdb_dir : str | Path
        Directory containing `bindingdb_drugs.csv`, `bindingdb_targets.csv`,
        and `drug_target_edges.csv`.
    master_nodes_path : str | Path | None
        Optional path to `master_drug_nodes.csv` to restrict profiles to
        drugs relevant to AuditDDI and align canonical SMILES.
    top_k_targets : int
        Number of top target proteins to include in the multi-hot feature vector.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A DataFrame of drug-target profiles and a summary dictionary.
    """
    bdir = Path(bindingdb_dir)
    if not bdir.is_dir():
        raise FileNotFoundError(f'BindingDB directory not found: {bdir}')

    # 1. Locate files
    drugs_file = bdir / 'bindingdb_drugs.csv'
    targets_file = bdir / 'bindingdb_targets.csv'
    edges_file = bdir / 'drug_target_edges.csv'

    for f, name in [(drugs_file, 'bindingdb_drugs.csv'), (targets_file, 'bindingdb_targets.csv'), (edges_file, 'drug_target_edges.csv')]:
        if not f.is_file():
            # Try recursive search if not at root of bdir
            matches = list(bdir.rglob(f'*{f.name}*'))
            if matches:
                if name == 'bindingdb_drugs.csv':
                    drugs_file = matches[0]
                elif name == 'bindingdb_targets.csv':
                    targets_file = matches[0]
                else:
                    edges_file = matches[0]
            else:
                raise FileNotFoundError(f'Required BindingDB file {name} not found in {bdir}')

    # 2. Read targets metadata
    df_targets = pd.read_csv(targets_file)
    tid_col = _find_column(df_targets, TARGET_ID_COLUMNS) or df_targets.columns[0]
    tname_col = _find_column(df_targets, TARGET_NAME_COLUMNS) or tid_col

    target_name_map: dict[str, str] = {}
    for _, row in df_targets.iterrows():
        raw_id = str(row[tid_col]).strip()
        name = str(row[tname_col]).strip() if pd.notna(row.get(tname_col)) else raw_id
        target_name_map[raw_id] = name

    # 3. Read drugs and canonicalize SMILES
    df_drugs = pd.read_csv(drugs_file)
    did_col = _find_column(df_drugs, DRUG_ID_COLUMNS) or df_drugs.columns[0]
    dsmi_col = _find_column(df_drugs, DRUG_SMILES_COLUMNS)

    drug_id_to_smiles: dict[str, str] = {}
    drug_id_to_inchikey: dict[str, str] = {}

    if dsmi_col:
        for _, row in df_drugs.iterrows():
            raw_id = str(row[did_col]).strip()
            raw_smi = row.get(dsmi_col)
            if pd.notna(raw_smi):
                can = canonicalize_smiles(str(raw_smi))
                if can:
                    ikey = smiles_to_inchikey(can)
                    drug_id_to_smiles[raw_id] = can
                    if ikey:
                        drug_id_to_inchikey[raw_id] = ikey

    # 4. Optional Master Nodes alignment
    valid_master_smiles: set[str] = set()
    valid_master_inchikeys: dict[str, str] = {}  # inchikey -> can_smiles
    if master_nodes_path and Path(master_nodes_path).is_file():
        df_master = pd.read_csv(master_nodes_path)
        m_id_col = 'drug_id' if 'drug_id' in df_master.columns else ('canonical_smiles' if 'canonical_smiles' in df_master.columns else df_master.columns[0])
        for _, row in df_master.iterrows():
            can = canonicalize_smiles(str(row[m_id_col]))
            if can:
                valid_master_smiles.add(can)
                ikey = smiles_to_inchikey(can)
                if ikey:
                    valid_master_inchikeys[ikey] = can

    # 5. Read Drug-Target Edges
    df_edges = pd.read_csv(edges_file)
    e_drug_col = _find_column(df_edges, EDGE_DRUG_COLUMNS) or df_edges.columns[0]
    e_target_col = _find_column(df_edges, EDGE_TARGET_COLUMNS) or df_edges.columns[1]
    e_aff_col = _find_column(df_edges, EDGE_AFFINITY_COLUMNS)

    # Accumulate drug -> list of targets and affinities
    drug_targets_acc: dict[str, dict[str, float]] = {}  # canonical_smiles -> {target_name: affinity}

    for _, row in df_edges.iterrows():
        raw_did = str(row[e_drug_col]).strip()
        raw_tid = str(row[e_target_col]).strip()

        # Resolve SMILES
        can_smi = drug_id_to_smiles.get(raw_did)
        if not can_smi and raw_did in valid_master_smiles:
            can_smi = raw_did

        # Check inchikey fallback
        if not can_smi and raw_did in drug_id_to_inchikey:
            ikey = drug_id_to_inchikey[raw_did]
            can_smi = valid_master_inchikeys.get(ikey)

        if not can_smi:
            continue

        # Target name
        tname = target_name_map.get(raw_tid, raw_tid).strip().upper()
        if not tname:
            continue

        # Affinity
        aff_val = 1.0
        if e_aff_col and pd.notna(row.get(e_aff_col)):
            try:
                aff_val = float(row[e_aff_col])
            except (ValueError, TypeError):
                aff_val = 1.0

        if can_smi not in drug_targets_acc:
            drug_targets_acc[can_smi] = {}
        # Keep highest or lowest affinity / presence
        drug_targets_acc[can_smi][tname] = aff_val

    # 6. Extract Top-K Vocabulary across profiled drugs
    all_target_lists = [list(targets.keys()) for targets in drug_targets_acc.values()]
    vocab = extract_top_target_vocabulary(all_target_lists, top_k=top_k_targets)

    # 7. Construct Result DataFrame
    rows = []
    for can_smi, targets_dict in drug_targets_acc.items():
        targets_list = sorted(list(targets_dict.keys()))
        target_vec = encode_multihot_target_vector(targets_list, vocab)
        ikey = smiles_to_inchikey(can_smi) or ''
        rows.append({
            'canonical_smiles': can_smi,
            'inchikey': ikey,
            'targets_list': json.dumps(targets_list),
            'targets_dict': json.dumps(targets_dict),
            'target_vector_multihot': json.dumps(target_vec),
            'n_targets': len(targets_list),
            'is_bindingdb_active': True,
        })

    df_profiles = pd.DataFrame(rows)
    summary = {
        'total_profiled_drugs': len(df_profiles),
        'unique_targets_observed': len({t for targets in all_target_lists for t in targets}),
        'target_vocabulary_size': len(vocab),
        'top_10_targets': vocab[:10],
    }
    return df_profiles, summary


def update_master_nodes_with_bindingdb(
    master_nodes_path: str | Path,
    bindingdb_dir_or_profiles: str | Path | pd.DataFrame,
    output_path: str | Path | None = None,
    top_k_targets: int = DEFAULT_TOP_TARGETS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Update master_drug_nodes.csv with BindingDB target vectors in seconds.

    Sets `is_bindingdb_active = True` for profiled nodes and enriches them
    with `bindingdb_targets_json` and `bindingdb_target_vector`.
    """
    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f'Master nodes file not found: {nodes_p}')

    df_nodes = pd.read_csv(nodes_p)

    if isinstance(bindingdb_dir_or_profiles, pd.DataFrame):
        df_profiles = bindingdb_dir_or_profiles
        vocab = extract_top_target_vocabulary(
            [json.loads(x) if isinstance(x, str) else list(x) for x in df_profiles['targets_list']],
            top_k=top_k_targets,
        )
    elif Path(bindingdb_dir_or_profiles).is_dir():
        df_profiles, _ = parse_bindingdb_directory(
            bindingdb_dir_or_profiles,
            master_nodes_path=nodes_p,
            top_k_targets=top_k_targets,
        )
        vocab = extract_top_target_vocabulary(
            [json.loads(x) if isinstance(x, str) else list(x) for x in df_profiles['targets_list']],
            top_k=top_k_targets,
        )
    elif Path(bindingdb_dir_or_profiles).is_file():
        df_profiles = pd.read_csv(bindingdb_dir_or_profiles)
        vocab = extract_top_target_vocabulary(
            [json.loads(x) if isinstance(x, str) else list(x) for x in df_profiles['targets_list']],
            top_k=top_k_targets,
        )
    else:
        raise ValueError(f'Invalid bindingdb source: {bindingdb_dir_or_profiles}')

    # Build lookup map: canonical_smiles -> (targets_dict, target_vec)
    lookup: dict[str, tuple[dict[str, float], list[int]]] = {}
    for _, row in df_profiles.iterrows():
        smi = canonicalize_smiles(str(row.get('canonical_smiles')))
        if not smi:
            continue
        t_dict = json.loads(row['targets_dict']) if isinstance(row.get('targets_dict'), str) else {}
        t_vec = json.loads(row['target_vector_multihot']) if isinstance(row.get('target_vector_multihot'), str) else []
        lookup[smi] = (t_dict, t_vec)

    node_id_col = 'drug_id' if 'drug_id' in df_nodes.columns else ('canonical_smiles' if 'canonical_smiles' in df_nodes.columns else df_nodes.columns[0])

    updated_targets_json: list[str] = []
    updated_vec_json: list[str] = []
    updated_active_flags: list[bool] = []

    matched = 0
    zero_vec = [0] * len(vocab)

    for _, row in df_nodes.iterrows():
        can = canonicalize_smiles(str(row[node_id_col]))
        if can and can in lookup:
            tdict, tvec = lookup[can]
            updated_targets_json.append(json.dumps(tdict))
            updated_vec_json.append(json.dumps(tvec))
            updated_active_flags.append(True)
            matched += 1
        else:
            updated_targets_json.append(json.dumps({}))
            updated_vec_json.append(json.dumps(zero_vec))
            updated_active_flags.append(False)

    df_nodes['bindingdb_targets_json'] = updated_targets_json
    df_nodes['bindingdb_target_vector'] = updated_vec_json
    df_nodes['is_bindingdb_active'] = updated_active_flags

    target_out = Path(output_path) if output_path else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_out, index=False)

    summary = {
        'total_nodes': len(df_nodes),
        'nodes_with_bindingdb_targets': matched,
        'bindingdb_coverage_pct': (matched / len(df_nodes) * 100.0) if len(df_nodes) else 0.0,
        'target_vocabulary_size': len(vocab),
        'target_vocabulary': vocab,
        'exported_to': str(target_out),
    }
    print(f'BindingDB synchronization complete: {matched} / {len(df_nodes)} drugs covered ({summary["bindingdb_coverage_pct"]:.1f}%).')
    return df_nodes, summary
