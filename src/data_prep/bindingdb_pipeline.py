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
import re
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .master_schema import canonicalize_smiles, smiles_to_inchikey
from .pharmgkb_pipeline import normalise_drug_name

DEFAULT_TOP_TARGETS = 50

# Candidate column names for resilient loading across different exports
DRUG_SMILES_COLUMNS = (
    'canonical_smiles', 'smiles', 'structure', 'isomeric_smiles', 'drug_smiles',
    'ligand_smiles', 'ligand smiles', 'canonical smiles',
)
DRUG_ID_COLUMNS = (
    'bindingdb_drug_id', 'drug_id', 'compound_id', 'id', 'source_id',
    'monomerid', 'ligand_id', 'chembl_id', 'pubchem_cid',
)
DRUG_NAME_COLUMNS = (
    'drug_name', 'name', 'chemical_name', 'display_name', 'compound_name',
    'ligand_name', 'bindingdb_name', 'synonyms', 'pref_name', 'compound name',
)
DRUG_INCHIKEY_COLUMNS = (
    'inchikey', 'inchi_key', 'ligand_inchikey', 'std_inchikey',
)

TARGET_ID_COLUMNS = ('bindingdb_target_id', 'target_id', 'id', 'uniprot_id')
TARGET_NAME_COLUMNS = (
    'target_name', 'name', 'protein_name', 'gene_name', 'target',
    'gene_symbol', 'target name', 'uniprot_id', 'target source organism',
)

EDGE_DRUG_COLUMNS = ('bindingdb_drug_id', 'drug_id', 'source', 'compound_id', 'drug')
EDGE_TARGET_COLUMNS = ('bindingdb_target_id', 'target_id', 'target', 'protein_id', 'uniprot_id')
EDGE_AFFINITY_COLUMNS = ('affinity_nm', 'ic50_nm', 'ki_nm', 'kd_nm', 'affinity', 'score', 'activity_value')

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def extract_parent_structure(smiles: str | None) -> tuple[str | None, str | None, str | None]:
    """Return (canonical_smiles, inchikey, skeleton_14char) after stripping salts."""
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return None, None, None
    can = canonicalize_smiles(smiles.strip())
    if not can:
        return None, None, None
    parent_can = can
    if '.' in can:
        frags = can.split('.')
        largest = max(frags, key=len)
        cand_parent = canonicalize_smiles(largest)
        if cand_parent:
            parent_can = cand_parent
    ikey = smiles_to_inchikey(parent_can)
    skel = ikey.split('-')[0] if (ikey and '-' in ikey) else ikey
    return parent_can, ikey, skel


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

    Uses a multi-tier matching strategy:
    - Tier 1: Exact Canonical SMILES
    - Tier 2: Salt-stripped Parent Canonical SMILES
    - Tier 3: Standard InChIKey (27 chars)
    - Tier 4: InChIKey skeleton (14-char connectivity block)
    - Tier 5: Normalized drug name and synonyms
    """
    bdir = Path(bindingdb_dir)
    if bdir.is_file():
        bdir = bdir.parent
    if not bdir.is_dir():
        raise FileNotFoundError(f'BindingDB directory not found: {bdir}')

    # 1. Locate files
    drugs_file = bdir / 'bindingdb_drugs.csv'
    targets_file = bdir / 'bindingdb_targets.csv'
    edges_file = bdir / 'drug_target_edges.csv'

    for f, name in [(drugs_file, 'bindingdb_drugs.csv'), (targets_file, 'bindingdb_targets.csv'), (edges_file, 'drug_target_edges.csv')]:
        if not f.is_file():
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

    # 3. Build Master Nodes Index (if provided)
    master_lookup_smiles: dict[str, str] = {}
    master_lookup_parent: dict[str, str] = {}
    master_lookup_inchikey: dict[str, str] = {}
    master_lookup_skel: dict[str, str] = {}
    master_lookup_names: dict[str, str] = {}

    if master_nodes_path and Path(master_nodes_path).is_file():
        df_master = pd.read_csv(master_nodes_path)
        m_id_col = 'drug_id' if 'drug_id' in df_master.columns else ('canonical_smiles' if 'canonical_smiles' in df_master.columns else df_master.columns[0])
        for _, row in df_master.iterrows():
            m_smi = str(row[m_id_col]).strip()
            parent_can, ikey, skel = extract_parent_structure(m_smi)
            can = canonicalize_smiles(m_smi) or parent_can
            if not can:
                continue
            master_lookup_smiles[can] = can
            if parent_can:
                master_lookup_parent[parent_can] = can
            if ikey:
                master_lookup_inchikey[ikey] = can
            if skel:
                master_lookup_skel[skel] = can

            # Names and synonyms
            dname = row.get('display_name')
            if pd.notna(dname):
                norm_dname = normalise_drug_name(str(dname))
                if norm_dname:
                    master_lookup_names[norm_dname] = can

            syns_raw = row.get('synonyms_json')
            if pd.notna(syns_raw):
                try:
                    syn_list = json.loads(syns_raw) if isinstance(syns_raw, str) else list(syns_raw)
                    for syn in syn_list:
                        norm_syn = normalise_drug_name(str(syn))
                        if norm_syn:
                            master_lookup_names[norm_syn] = can
                except Exception:
                    pass

    # 4. Read drugs and map BindingDB drug IDs to canonical structures
    df_drugs = pd.read_csv(drugs_file)
    did_col = _find_column(df_drugs, DRUG_ID_COLUMNS) or df_drugs.columns[0]
    dsmi_col = _find_column(df_drugs, DRUG_SMILES_COLUMNS)
    dname_col = _find_column(df_drugs, DRUG_NAME_COLUMNS)
    dikey_col = _find_column(df_drugs, DRUG_INCHIKEY_COLUMNS)

    bindingdb_id_to_master: dict[str, str] = {}
    bindingdb_id_to_smiles: dict[str, str] = {}
    bindingdb_id_to_names: dict[str, str] = {}

    for _, row in df_drugs.iterrows():
        raw_id = str(row[did_col]).strip()
        raw_smi = row.get(dsmi_col) if dsmi_col else None
        raw_name = row.get(dname_col) if dname_col else None
        raw_ikey = row.get(dikey_col) if dikey_col else None

        can, ikey, skel = extract_parent_structure(str(raw_smi)) if pd.notna(raw_smi) else (None, None, None)
        if raw_ikey and pd.notna(raw_ikey):
            ikey = str(raw_ikey).strip()
            skel = ikey.split('-')[0] if '-' in ikey else ikey

        norm_name = normalise_drug_name(str(raw_name)) if pd.notna(raw_name) else None

        if can:
            bindingdb_id_to_smiles[raw_id] = can
        if norm_name:
            bindingdb_id_to_names[raw_id] = norm_name

        # Resolve to master drug if master lookup exists
        resolved_master = None
        if can and can in master_lookup_smiles:
            resolved_master = master_lookup_smiles[can]
        elif can and can in master_lookup_parent:
            resolved_master = master_lookup_parent[can]
        elif ikey and ikey in master_lookup_inchikey:
            resolved_master = master_lookup_inchikey[ikey]
        elif skel and skel in master_lookup_skel:
            resolved_master = master_lookup_skel[skel]
        elif norm_name and norm_name in master_lookup_names:
            resolved_master = master_lookup_names[norm_name]

        if resolved_master:
            bindingdb_id_to_master[raw_id] = resolved_master

    # 5. Read Drug-Target Edges
    df_edges = pd.read_csv(edges_file)
    e_drug_col = _find_column(df_edges, EDGE_DRUG_COLUMNS) or df_edges.columns[0]
    e_target_col = _find_column(df_edges, EDGE_TARGET_COLUMNS) or df_edges.columns[1]
    e_aff_col = _find_column(df_edges, EDGE_AFFINITY_COLUMNS)

    # Accumulate drug -> list of targets and affinities
    drug_targets_acc: dict[str, dict[str, float]] = {}  # target_canonical_smiles -> {target_name: affinity}

    for _, row in df_edges.iterrows():
        raw_did = str(row[e_drug_col]).strip()
        raw_tid = str(row[e_target_col]).strip()

        # Multi-tier drug resolution
        target_smi = bindingdb_id_to_master.get(raw_did)
        if not target_smi:
            target_smi = bindingdb_id_to_smiles.get(raw_did)
        if not target_smi and raw_did in master_lookup_smiles:
            target_smi = master_lookup_smiles[raw_did]
        if not target_smi and raw_did in master_lookup_names:
            target_smi = master_lookup_names[raw_did]
        if not target_smi:
            norm_did = normalise_drug_name(raw_did)
            if norm_did and norm_did in master_lookup_names:
                target_smi = master_lookup_names[norm_did]

        if not target_smi:
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

        if target_smi not in drug_targets_acc:
            drug_targets_acc[target_smi] = {}
        drug_targets_acc[target_smi][tname] = aff_val

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
    master_nodes_path: str | Path | None = None,
    bindingdb_dir_or_profiles: str | Path | pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    top_k_targets: int = DEFAULT_TOP_TARGETS,
    impute_by_tanimoto: bool = True,
    tanimoto_threshold: float = 0.15,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Update master_drug_nodes.csv with BindingDB target vectors with multi-tier matching and structural imputation.

    Sets `is_bindingdb_active = True` for profiled nodes and enriches them
    with `bindingdb_targets_json` and `bindingdb_target_vector`.
    Supports keyword aliases: master_nodes_csv, bindingdb_tsv_path, output_csv.
    """
    if master_nodes_path is None:
        master_nodes_path = kwargs.get('master_nodes_csv')
    if master_nodes_path is None:
        raise ValueError('master_nodes_path (or master_nodes_csv) must be provided')

    if bindingdb_dir_or_profiles is None:
        bindingdb_dir_or_profiles = kwargs.get('bindingdb_tsv_path') or kwargs.get('bindingdb_dir')
    if bindingdb_dir_or_profiles is None:
        raise ValueError('bindingdb_dir_or_profiles (or bindingdb_tsv_path / bindingdb_dir) must be provided')

    if output_path is None:
        output_path = kwargs.get('output_csv')

    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f'Master nodes file not found: {nodes_p}')

    df_nodes = pd.read_csv(nodes_p)

    if isinstance(bindingdb_dir_or_profiles, pd.DataFrame):
        df_profiles = bindingdb_dir_or_profiles
    elif Path(bindingdb_dir_or_profiles).is_dir():
        df_profiles, _ = parse_bindingdb_directory(
            bindingdb_dir_or_profiles,
            master_nodes_path=nodes_p,
            top_k_targets=top_k_targets,
        )
    elif Path(bindingdb_dir_or_profiles).is_file():
        p = Path(bindingdb_dir_or_profiles)
        if (p.parent / 'bindingdb_drugs.csv').is_file() or p.suffix in ['.tsv', '.txt']:
            df_profiles, _ = parse_bindingdb_directory(
                p.parent,
                master_nodes_path=nodes_p,
                top_k_targets=top_k_targets,
            )
        else:
            df_profiles = pd.read_csv(p)
    else:
        raise ValueError(f'Invalid bindingdb source: {bindingdb_dir_or_profiles}')

    vocab = extract_top_target_vocabulary(
        [json.loads(x) if isinstance(x, str) else list(x) for x in df_profiles['targets_list']],
        top_k=top_k_targets,
    )

    # Multi-tier index on df_profiles:
    # 1. exact canonical smiles -> (tdict, tvec)
    # 2. parent canonical smiles -> (tdict, tvec)
    # 3. full inchikey -> (tdict, tvec)
    # 4. 14-char skeleton -> (tdict, tvec)
    exact_lookup: dict[str, tuple[dict[str, float], list[int]]] = {}
    parent_lookup: dict[str, tuple[dict[str, float], list[int]]] = {}
    inchikey_lookup: dict[str, tuple[dict[str, float], list[int]]] = {}
    skel_lookup: dict[str, tuple[dict[str, float], list[int]]] = {}

    profiled_fps: list[Any] = []
    profiled_profile_data: list[tuple[dict[str, float], list[int]]] = []

    for _, row in df_profiles.iterrows():
        smi = str(row.get('canonical_smiles')).strip()
        parent_can, ikey, skel = extract_parent_structure(smi)
        can = canonicalize_smiles(smi) or parent_can
        if not can:
            continue
        tdict = json.loads(row['targets_dict']) if isinstance(row.get('targets_dict'), str) else {}
        tvec = json.loads(row['target_vector_multihot']) if isinstance(row.get('target_vector_multihot'), str) else []

        exact_lookup[can] = (tdict, tvec)
        if parent_can:
            parent_lookup[parent_can] = (tdict, tvec)
        if ikey:
            inchikey_lookup[ikey] = (tdict, tvec)
        if skel:
            skel_lookup[skel] = (tdict, tvec)

        # Precompute fingerprint for similarity imputation
        if impute_by_tanimoto:
            mol = Chem.MolFromSmiles(can)
            if mol is not None:
                fp = _MORGAN_GEN.GetFingerprint(mol)
                profiled_fps.append(fp)
                profiled_profile_data.append((tdict, tvec))

    node_id_col = 'drug_id' if 'drug_id' in df_nodes.columns else ('canonical_smiles' if 'canonical_smiles' in df_nodes.columns else df_nodes.columns[0])

    updated_targets_json: list[str] = []
    updated_vec_json: list[str] = []
    updated_active_flags: list[bool] = []

    matched_exact = 0
    matched_parent = 0
    matched_skel = 0
    matched_similarity = 0
    zero_vec = [0] * len(vocab)

    for _, row in df_nodes.iterrows():
        raw_smi = str(row[node_id_col]).strip()
        parent_can, ikey, skel = extract_parent_structure(raw_smi)
        can = canonicalize_smiles(raw_smi) or parent_can

        tdict_found: dict[str, float] | None = None
        tvec_found: list[int] | None = None

        if can and can in exact_lookup:
            tdict_found, tvec_found = exact_lookup[can]
            matched_exact += 1
        elif parent_can and parent_can in parent_lookup:
            tdict_found, tvec_found = parent_lookup[parent_can]
            matched_parent += 1
        elif ikey and ikey in inchikey_lookup:
            tdict_found, tvec_found = inchikey_lookup[ikey]
            matched_exact += 1
        elif skel and skel in skel_lookup:
            tdict_found, tvec_found = skel_lookup[skel]
            matched_skel += 1
        elif impute_by_tanimoto and profiled_fps and can:
            # Impute from nearest structural neighbor
            mol = Chem.MolFromSmiles(can)
            if mol is not None:
                fp = _MORGAN_GEN.GetFingerprint(mol)
                sims = DataStructs.BulkTanimotoSimilarity(fp, profiled_fps)
                max_sim_idx = int(np.argmax(sims))
                max_sim = float(sims[max_sim_idx])
                if max_sim >= tanimoto_threshold:
                    base_dict, base_vec = profiled_profile_data[max_sim_idx]
                    tdict_found = {k: round(v * max_sim, 3) for k, v in base_dict.items()}
                    tvec_found = base_vec
                    matched_similarity += 1

        if tdict_found is not None and tvec_found is not None:
            updated_targets_json.append(json.dumps(tdict_found))
            updated_vec_json.append(json.dumps(tvec_found))
            updated_active_flags.append(True)
        else:
            updated_targets_json.append(json.dumps({}))
            updated_vec_json.append(json.dumps(zero_vec))
            updated_active_flags.append(False)

    df_nodes['bindingdb_targets_json'] = updated_targets_json
    df_nodes['bindingdb_target_vector'] = updated_vec_json
    df_nodes['is_bindingdb_active'] = updated_active_flags

    total_matched = sum(updated_active_flags)
    target_out = Path(output_path) if output_path else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_out, index=False)

    summary = {
        'total_nodes': len(df_nodes),
        'nodes_with_bindingdb_targets': total_matched,
        'matched_exact_smiles': matched_exact,
        'matched_parent_salt_stripped': matched_parent,
        'matched_inchikey_skeleton': matched_skel,
        'matched_tanimoto_similarity': matched_similarity,
        'bindingdb_coverage_pct': (total_matched / len(df_nodes) * 100.0) if len(df_nodes) else 0.0,
        'target_vocabulary_size': len(vocab),
        'target_vocabulary': vocab,
        'exported_to': str(target_out),
    }
    print(
        f'BindingDB synchronization complete: {total_matched} / {len(df_nodes)} drugs covered ({summary["bindingdb_coverage_pct"]:.1f}%).\n'
        f'  (Exact/InChIKey: {matched_exact}, Parent: {matched_parent}, Skeleton: {matched_skel}, Chemical Analogs: {matched_similarity})'
    )
    return df_nodes, summary

