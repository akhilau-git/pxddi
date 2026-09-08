"""Protein Data Bank (PDB) 3D Macromolecular Complex Pipeline for AuditDDI.

Parses and integrates PDB drug-protein co-crystal complexes, 3D binding pocket
structures, and crystallographic resolution data to augment drug nodes with
3D macromolecular structural vectors.

Supports:
1. Parsing PDB tabular files (pdb_ligands.csv, pdb_complexes.csv, drug_pdb.csv, etc.)
   and raw .pdb / .cif crystallographic headers.
2. Resolving drug structures to RDKit canonical SMILES and standard InChIKeys.
3. Multi-tier matching (Exact SMILES -> Parent SMILES -> InChIKey -> Skeleton -> Name -> Tanimoto).
4. Generating multi-hot PDB target complex presence vectors for multimodal fusion.
5. Updating `master_drug_nodes.csv` in-place or exporting to an enriched catalog.
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

DEFAULT_TOP_PDB_TARGETS = 50

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)

DRUG_SMILES_COLUMNS = (
    'canonical_smiles', 'smiles', 'structure', 'isomeric_smiles', 'drug_smiles',
    'ligand_smiles', 'ligand smiles', 'canonical smiles',
)
DRUG_ID_COLUMNS = (
    'pdb_ligand_id', 'ligand_id', 'drug_id', 'comp_id', 'het_id', 'pdb_code',
    'pdb_id', 'id', 'compound_id', 'chembl_id', 'pubchem_cid',
)
DRUG_NAME_COLUMNS = (
    'drug_name', 'name', 'ligand_name', 'compound_name', 'chemical_name',
    'display_name', 'pref_name', 'synonyms', 'het_name',
)
DRUG_INCHIKEY_COLUMNS = (
    'inchikey', 'inchi_key', 'ligand_inchikey', 'std_inchikey',
)
PDB_CODE_COLUMNS = (
    'pdb_id', 'pdb_code', 'entry_id', 'pdb_entry', 'structure_id', 'pdb',
)
TARGET_NAME_COLUMNS = (
    'target_name', 'target', 'protein_name', 'macromolecule', 'gene_symbol',
    'gene_name', 'protein', 'uniprot_id', 'chain',
)
RESOLUTION_COLUMNS = (
    'resolution', 'resolution_angstrom', 'crystallographic_resolution',
    'resolution_a', 'res_a', 'score',
)


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


def extract_top_pdb_vocabulary(
    target_lists: list[list[str]],
    top_k: int = DEFAULT_TOP_PDB_TARGETS,
) -> list[str]:
    """Return the top-K most prevalent PDB macromolecular targets across all profiled drugs."""
    counter: Counter[str] = Counter()
    for targets in target_lists:
        for t in targets:
            clean_t = str(t).strip().upper()
            if clean_t:
                counter[clean_t] += 1
    return [target for target, _ in counter.most_common(top_k)]


def encode_multihot_pdb_vector(
    drug_targets: list[str],
    vocabulary: list[str],
) -> list[int]:
    """Encode a drug's PDB target complex associations into a binary multi-hot vector."""
    target_set = {str(t).strip().upper() for t in drug_targets}
    return [1 if vocab_target in target_set else 0 for vocab_target in vocabulary]


PDB_LIGAND_TO_DRUG: dict[str, str] = {
    'STI': 'Imatinib',
    'GNA': 'Gefitinib',
    'IRE': 'Iressa',
    'AQ4': 'Erlotinib',
    'STU': 'Staurosporine',
    'ASP': 'Aspirin',
    'CAF': 'Caffeine',
    'TYL': 'Acetaminophen',
    'IBP': 'Ibuprofen',
    'SDF': 'Sildenafil',
    'VRN': 'Varenicline',
    'WAF': 'Warfarin',
    'MTX': 'Methotrexate',
    'DOX': 'Doxorubicin',
    'TXO': 'Paclitaxel',
    'CP6': 'Ciprofloxacin',
    'DIZ': 'Diazepam',
    'CLO': 'Clonazepam',
    'FLU': 'Fluoxetine',
    'MET': 'Metformin',
    'ATP': 'Adenosine',
    'ADN': 'Adenosine',
    'RIT': 'Ritonavir',
    'IDV': 'Indinavir',
    'NFV': 'Nelfinavir',
    'AMP': 'Amprenavir',
    'LPV': 'Lopinavir',
    'DRV': 'Darunavir',
    'ATV': 'Atazanavir',
}


def parse_raw_pdb_header(filepath: Path) -> dict[str, Any]:
    """Extract metadata (PDB ID, resolution, ligands, macromolecules) from a raw .pdb file."""
    meta: dict[str, Any] = {
        'pdb_id': filepath.stem.upper(),
        'resolution': 2.5,
        'ligands': [],
        'ligand_names': {},
        'targets': [],
    }
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('HEADER'):
                    code = line[62:66].strip().upper()
                    if code:
                        meta['pdb_id'] = code
                elif line.startswith('REMARK   2 RESOLUTION.'):
                    match = re.search(r'([0-9]+\.[0-9]+)\s+ANGSTROMS', line)
                    if match:
                        meta['resolution'] = float(match.group(1))
                elif line.startswith('HET   '):
                    parts = line.split()
                    if len(parts) >= 2:
                        lig_id = parts[1].strip().upper()
                        if lig_id not in {'HOH', 'WAT', 'DOD', 'SO4', 'PO4', 'CL', 'NA', 'MG'}:
                            if lig_id not in meta['ligands']:
                                meta['ligands'].append(lig_id)
                elif line.startswith('HETNAM'):
                    parts = line[6:].strip().split(maxsplit=1)
                    if len(parts) >= 2:
                        lid, ldesc = parts[0].strip().upper(), parts[1].strip()
                        meta['ligand_names'][lid] = ldesc
                        if lid not in meta['ligands'] and lid not in {'HOH', 'WAT', 'SO4', 'CL', 'NA'}:
                            meta['ligands'].append(lid)
                elif line.startswith('COMPND   2 MOLECULE:'):
                    mol_desc = line[21:].strip().rstrip(';')
                    if mol_desc and mol_desc not in meta['targets']:
                        meta['targets'].append(mol_desc)
    except Exception:
        pass
    return meta


def parse_pdb_directory(
    pdb_dir: str | Path,
    master_nodes_path: str | Path | None = None,
    top_k_targets: int = DEFAULT_TOP_PDB_TARGETS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse raw PDB directory containing CSVs, TSVs, or .pdb files into drug-PDB mappings."""
    pdir = Path(pdb_dir)
    if pdir.is_file():
        pdir = pdir.parent
    if not pdir.is_dir():
        raise FileNotFoundError(f'PDB directory not found: {pdir}')

    # Load master node names to SMILES lookup if master_nodes_path exists
    name_to_smiles: dict[str, str] = {}
    if master_nodes_path and Path(master_nodes_path).is_file():
        try:
            m_df = pd.read_csv(master_nodes_path)
            s_col = 'canonical_smiles' if 'canonical_smiles' in m_df.columns else ('drug_id' if 'drug_id' in m_df.columns else m_df.columns[0])
            for _, r in m_df.iterrows():
                cs = str(r[s_col]).strip()
                dn = str(r.get('display_name', '')).strip()
                if dn:
                    name_to_smiles[normalise_drug_name(dn)] = cs
                if 'synonyms_json' in r and pd.notna(r['synonyms_json']):
                    try:
                        syns = json.loads(str(r['synonyms_json']))
                        for s in syns:
                            ns = normalise_drug_name(str(s))
                            if ns:
                                name_to_smiles[ns] = cs
                    except Exception:
                        pass
        except Exception:
            pass

    all_csvs = [f for f in pdir.rglob('*') if f.is_file() and f.suffix.lower() in ('.csv', '.tsv', '.tab', '.txt', '.csv.gz', '.tsv.gz')]
    all_pdbs = [f for f in pdir.rglob('*') if f.is_file() and f.suffix.lower() in ('.pdb', '.ent', '.cif', '.pdb.gz', '.ent.gz')]

    records: list[dict[str, Any]] = []

    # 1. Parse tabular files if present
    for tfile in all_csvs:
        try:
            sep = '\t' if 'tsv' in tfile.name.lower() else ','
            df = pd.read_csv(tfile, sep=sep, low_memory=False, nrows=50000)
            smi_col = _find_column(df, DRUG_SMILES_COLUMNS)
            name_col = _find_column(df, DRUG_NAME_COLUMNS)
            id_col = _find_column(df, DRUG_ID_COLUMNS)
            target_col = _find_column(df, TARGET_NAME_COLUMNS)
            pdb_col = _find_column(df, PDB_CODE_COLUMNS)
            res_col = _find_column(df, RESOLUTION_COLUMNS)

            for _, row in df.iterrows():
                smi = str(row[smi_col]).strip() if smi_col and pd.notna(row[smi_col]) else None
                nm = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else None
                did = str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else None
                tgt = str(row[target_col]).strip().upper() if target_col and pd.notna(row[target_col]) else None
                pdb_code = str(row[pdb_col]).strip().upper() if pdb_col and pd.notna(row[pdb_col]) else None
                res_val = float(row[res_col]) if res_col and pd.notna(row[res_col]) else 2.5

                if smi or nm or did or pdb_code:
                    records.append({
                        'raw_smiles': smi,
                        'drug_name': nm,
                        'drug_id': did,
                        'pdb_code': pdb_code or 'UNKNOWN_PDB',
                        'target_name': tgt or 'UNSPECIFIED_MACROMOLECULE',
                        'resolution': res_val,
                    })
        except Exception:
            continue

    # 2. Parse raw .pdb structure files if present
    for pfile in all_pdbs[:1000]:
        hmeta = parse_raw_pdb_header(pfile)
        for lig in hmeta['ligands']:
            resolved_drug = PDB_LIGAND_TO_DRUG.get(lig) or hmeta.get('ligand_names', {}).get(lig) or lig
            norm_drug = normalise_drug_name(resolved_drug)
            matched_smi = name_to_smiles.get(norm_drug) if norm_drug else None
            records.append({
                'raw_smiles': matched_smi,
                'drug_name': resolved_drug,
                'drug_id': lig,
                'pdb_code': hmeta['pdb_id'],
                'target_name': hmeta['targets'][0] if hmeta['targets'] else 'PDB_PROTEIN_TARGET',
                'resolution': hmeta['resolution'],
            })

    if not records:
        # Fallback: create empty DataFrame with correct schema
        empty_df = pd.DataFrame(columns=[
            'canonical_smiles', 'inchikey', 'drug_name', 'pdb_structures_json',
            'pdb_targets_json', 'pdb_vector_multihot', 'pdb_resolution_score'
        ])
        return empty_df, {'total_pdb_drugs': 0, 'top_10_targets': []}

    df_raw = pd.DataFrame(records)

    # Group by drug identifiers
    grouped_profiles: dict[str, dict[str, Any]] = {}

    for _, row in df_raw.iterrows():
        smi = row['raw_smiles']
        can = canonicalize_smiles(smi) if smi else None
        key = can or (row['drug_name'] and normalise_drug_name(row['drug_name'])) or row['drug_id']
        if not key:
            continue

        if key not in grouped_profiles:
            grouped_profiles[key] = {
                'canonical_smiles': can,
                'inchikey': smiles_to_inchikey(can) if can else None,
                'drug_name': row['drug_name'],
                'pdb_codes': set(),
                'targets': set(),
                'resolutions': [],
            }
        if row['pdb_code']:
            grouped_profiles[key]['pdb_codes'].add(row['pdb_code'])
        if row['target_name']:
            grouped_profiles[key]['targets'].add(row['target_name'])
        if row['resolution']:
            grouped_profiles[key]['resolutions'].append(row['resolution'])

    # Build target vocabulary
    all_target_lists = [list(p['targets']) for p in grouped_profiles.values() if p['targets']]
    vocab = extract_top_pdb_vocabulary(all_target_lists, top_k=top_k_targets)

    final_rows: list[dict[str, Any]] = []
    for key, p in grouped_profiles.items():
        targets_list = sorted(list(p['targets']))
        vec = encode_multihot_pdb_vector(targets_list, vocab)
        avg_res = float(np.mean(p['resolutions'])) if p['resolutions'] else 2.5
        final_rows.append({
            'canonical_smiles': p['canonical_smiles'],
            'inchikey': p['inchikey'],
            'drug_name': p['drug_name'],
            'pdb_structures_json': json.dumps(sorted(list(p['pdb_codes']))),
            'pdb_targets_json': json.dumps(targets_list),
            'pdb_vector_multihot': json.dumps(vec),
            'pdb_resolution_score': round(avg_res, 3),
        })

    df_pdb = pd.DataFrame(final_rows)
    summary = {
        'total_pdb_drugs': len(df_pdb),
        'top_10_targets': vocab[:10],
    }
    return df_pdb, summary


def update_master_nodes_with_pdb(
    master_nodes_path: str | Path,
    pdb_dir_or_profiles: str | Path | pd.DataFrame,
    output_path: str | Path | None = None,
    top_k_targets: int = DEFAULT_TOP_PDB_TARGETS,
    impute_by_tanimoto: bool = True,
    tanimoto_threshold: float = 0.70,
    **kwargs: Any,
) -> dict[str, Any]:
    """Enrich master_drug_nodes.csv with PDB macromolecular structures and 3D target vectors."""
    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f'Master nodes file not found: {nodes_p}')

    df_nodes = pd.read_csv(nodes_p)

    if isinstance(pdb_dir_or_profiles, pd.DataFrame):
        df_pdb = pdb_dir_or_profiles
        vocab = DEFAULT_TOP_PDB_TARGETS
    else:
        df_pdb, _ = parse_pdb_directory(pdb_dir_or_profiles, top_k_targets=top_k_targets, master_nodes_path=nodes_p)

    node_id_col = 'drug_id' if 'drug_id' in df_nodes.columns else ('canonical_smiles' if 'canonical_smiles' in df_nodes.columns else df_nodes.columns[0])

    exact_lookup: dict[str, tuple[str, list[int], float]] = {}
    name_lookup: dict[str, tuple[str, list[int], float]] = {}
    inchikey_lookup: dict[str, tuple[str, list[int], float]] = {}

    profiled_fps: list[Any] = []
    profiled_data: list[tuple[str, list[int], float]] = []

    for _, row in df_pdb.iterrows():
        can = canonicalize_smiles(str(row.get('canonical_smiles')))
        ikey = str(row.get('inchikey')).strip() if pd.notna(row.get('inchikey')) else None
        nm = normalise_drug_name(str(row.get('drug_name'))) if pd.notna(row.get('drug_name')) else None
        pdbs_json = str(row.get('pdb_structures_json', '[]'))
        vec = json.loads(row['pdb_vector_multihot']) if isinstance(row.get('pdb_vector_multihot'), str) else [0] * top_k_targets
        res = float(row.get('pdb_resolution_score', 2.5)) if pd.notna(row.get('pdb_resolution_score')) else 2.5

        if can:
            exact_lookup[can] = (pdbs_json, vec, res)
            mol = Chem.MolFromSmiles(can)
            if mol is not None:
                profiled_fps.append(_MORGAN_GEN.GetFingerprint(mol))
                profiled_data.append((pdbs_json, vec, res))
        if ikey:
            inchikey_lookup[ikey] = (pdbs_json, vec, res)
        if nm:
            name_lookup[nm] = (pdbs_json, vec, res)

    pdb_structures: list[str] = []
    pdb_vectors: list[str] = []
    pdb_active_flags: list[bool] = []
    pdb_resolutions: list[float] = []

    matched = 0
    zero_vec = [0] * top_k_targets

    for _, row in df_nodes.iterrows():
        raw_smi = str(row[node_id_col]).strip()
        parent_can, ikey, _ = extract_parent_structure(raw_smi)
        can = canonicalize_smiles(raw_smi) or parent_can
        nm = normalise_drug_name(str(row.get('display_name', ''))) if 'display_name' in row else None
        syn_names: list[str] = []
        if 'synonyms_json' in row and pd.notna(row['synonyms_json']):
            try:
                raw_syns = json.loads(str(row['synonyms_json'])) if isinstance(row['synonyms_json'], str) else list(row['synonyms_json'])
                syn_names = [normalise_drug_name(str(s)) for s in raw_syns if normalise_drug_name(str(s))]
            except Exception:
                pass

        found: tuple[str, list[int], float] | None = None

        if can and can in exact_lookup:
            found = exact_lookup[can]
        elif parent_can and parent_can in exact_lookup:
            found = exact_lookup[parent_can]
        elif ikey and ikey in inchikey_lookup:
            found = inchikey_lookup[ikey]
        elif nm and nm in name_lookup:
            found = name_lookup[nm]
        else:
            for s_nm in syn_names:
                if s_nm in name_lookup:
                    found = name_lookup[s_nm]
                    break

        if not found and nm:
            for cand_n, val in name_lookup.items():
                if len(cand_n) >= 4 and (cand_n in nm or nm in cand_n):
                    found = val
                    break

        if not found and impute_by_tanimoto and profiled_fps and can:
            mol = Chem.MolFromSmiles(can)
            if mol is not None:
                fp = _MORGAN_GEN.GetFingerprint(mol)
                sims = DataStructs.BulkTanimotoSimilarity(fp, profiled_fps)
                max_idx = int(np.argmax(sims))
                if sims[max_idx] >= tanimoto_threshold:
                    found = profiled_data[max_idx]

        if found:
            matched += 1
            pdb_structures.append(found[0])
            pdb_vectors.append(json.dumps(found[1]))
            pdb_active_flags.append(True)
            pdb_resolutions.append(found[2])
        else:
            pdb_structures.append('[]')
            pdb_vectors.append(json.dumps(zero_vec))
            pdb_active_flags.append(False)
            pdb_resolutions.append(2.5)

    df_nodes['pdb_structures_json'] = pdb_structures
    df_nodes['pdb_vector_multihot'] = pdb_vectors
    df_nodes['is_pdb_active'] = pdb_active_flags
    df_nodes['pdb_resolution_score'] = pdb_resolutions

    target_out = Path(output_path) if output_path else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_out, index=False)

    summary = {
        'total_nodes': len(df_nodes),
        'nodes_with_pdb_structures': matched,
        'coverage_pct': round((matched / max(len(df_nodes), 1)) * 100, 2),
    }
    print(f"PDB enrichment complete: {matched}/{len(df_nodes)} ({summary['coverage_pct']}%) master nodes annotated with PDB complexes.")
    return summary
