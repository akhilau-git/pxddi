"""GEO Disease Transcriptomics Pipeline for AuditDDI.

Parses and integrates Gene Expression Omnibus (GEO) disease gene expression series:
- `Brain_Alzheimers_GSE5281.txt.gz`: Neurodegenerative CNS signature.
- `Cardiovascular_HF_GSE57338.txt.gz`: Heart Failure Cardiovascular signature.

Maps disease-perturbed expression signatures to drug biological profiles
(genes/enzymes/targets) to generate biological context vectors for unseen drugs.

Supports:
1. Resilient streaming and parsing of `.txt.gz` and `.txt` expression matrices.
2. Extracting disease signature vectors across high-variance disease genes.
3. Scoring drug-disease transcriptomic interaction profiles.
4. Updating `master_drug_nodes.csv` in-place or exporting to a new path.
"""

from __future__ import annotations

import gzip
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

DEFAULT_TOP_SIGNATURE_GENES = 50
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def _open_geo_file(file_path: Path):
    """Open a plain text or gzip-compressed file transparently."""
    if file_path.suffix == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='replace')
    return open(file_path, 'r', encoding='utf-8', errors='replace')


TARGET_TO_GENE_MAP = {
    'ADENOSINE RECEPTOR A2A': 'ADORA2A',
    'ADENOSINE A2A': 'ADORA2A',
    'CYCLOOXYGENASE-1': 'PTGS1',
    'CYCLOOXYGENASE-2': 'PTGS2',
    'COX-1': 'PTGS1',
    'COX-2': 'PTGS2',
    'CYTOCHROME P450 3A4': 'CYP3A4',
    'CYTOCHROME P450 2D6': 'CYP2D6',
    'CYTOCHROME P450 2C9': 'CYP2C9',
    'CYTOCHROME P450 2C19': 'CYP2C19',
    'CYTOCHROME P450 1A2': 'CYP1A2',
    'DOPAMINE D2 RECEPTOR': 'DRD2',
    'SEROTONIN 2A RECEPTOR': 'HTR2A',
    '5-HT2A': 'HTR2A',
    'BETA-1 ADRENERGIC RECEPTOR': 'ADRB1',
    'BETA-2 ADRENERGIC RECEPTOR': 'ADRB2',
    'ANGIOTENSIN-CONVERTING ENZYME': 'ACE',
    'HMG-COA REDUCTASE': 'HMGCR',
    'P-GLYCOPROTEIN': 'ABCB1',
    'HER2': 'ERBB2',
    'EGFR': 'EGFR',
}


def parse_disease_expression_file(
    file_path: str | Path,
    max_genes: int = 5000,
) -> dict[str, float]:
    """Parse a GEO disease expression file and extract top perturbed gene scores.

    Handles SOFT series matrix format, tab-delimited tables, and CSVs by detecting
    probe/gene symbol headers and computing mean expression variance or perturbation.
    """
    fpath = Path(file_path)
    if not fpath.is_file():
        raise FileNotFoundError(f'GEO expression file not found: {fpath}')

    raw_lines: list[str] = []
    with _open_geo_file(fpath) as f:
        in_table = False
        has_table_marker = False

        # Read first pass to detect if SOFT table marker exists
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if '!series_matrix_table_begin' in stripped:
                has_table_marker = True
                in_table = True
                continue
            if '!series_matrix_table_end' in stripped:
                break
            if has_table_marker and not in_table:
                continue
            if not has_table_marker and stripped.startswith(('!', '^', '#')):
                continue
            raw_lines.append(stripped)
            if len(raw_lines) >= 25000:
                break

    if not raw_lines:
        return {}

    # Header parsing
    first_line = raw_lines[0]
    delimiter = '\t' if '\t' in first_line else (',' if ',' in first_line else None)
    header = [col.strip().strip('"').upper() for col in (first_line.split(delimiter) if delimiter else first_line.split())]

    # Prioritize gene symbol column
    gene_col_idx = None
    for idx, col in enumerate(header):
        if any(term == col or term in col for term in ['GENE_SYMBOL', 'GENE SYMBOL', 'GENE_NAME', 'GENENAME', 'SYMBOL', 'GENE']):
            gene_col_idx = idx
            break

    if gene_col_idx is None:
        for idx, col in enumerate(header):
            if any(term in col for term in ['NAME', 'ID_REF', 'PROBE', 'ID', 'IDENTIFIER']):
                gene_col_idx = idx
                break

    if gene_col_idx is None:
        gene_col_idx = 0

    gene_scores: dict[str, float] = {}

    for line in raw_lines[1:]:
        parts = line.split(delimiter) if delimiter else line.split()
        if len(parts) <= gene_col_idx:
            continue
        raw_gene = parts[gene_col_idx].strip().strip('"').upper()
        # Clean multi-gene annotations (e.g. 'CYP2D6 /// CYP2D7')
        gene = raw_gene.split('///')[0].strip().split('//')[0].strip().split(' ')[0].strip()
        if not gene or len(gene) < 2:
            continue

        # Extract numeric expression values
        numeric_vals = []
        for idx, val in enumerate(parts):
            if idx == gene_col_idx:
                continue
            try:
                cleaned_val = val.strip().strip('"')
                if cleaned_val and cleaned_val.lower() not in {'null', 'na', 'nan'}:
                    numeric_vals.append(float(cleaned_val))
            except (ValueError, TypeError):
                continue

        if numeric_vals:
            # Score as variance or magnitude of expression perturbation
            score = float(np.std(numeric_vals) if len(numeric_vals) > 1 else abs(numeric_vals[0]))
            if gene in gene_scores:
                gene_scores[gene] = max(gene_scores[gene], score)
            else:
                gene_scores[gene] = score

        if len(gene_scores) >= max_genes * 2:
            break

    if not gene_scores:
        return {}

    # Sort and retain top perturbed genes
    sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:max_genes]
    max_val = max((val for _, val in sorted_genes), default=1.0) or 1.0
    return {gene: float(val / max_val) for gene, val in sorted_genes}


def parse_geo_directory(
    geo_dir: str | Path,
    max_genes_per_disease: int = 5000,
) -> dict[str, dict[str, float]]:
    """Parse all GEO disease datasets in directory.

    Returns mapping: disease_series_name -> {gene_symbol: normalized_score}.
    """
    gdir = Path(geo_dir)
    if not gdir.is_dir():
        raise FileNotFoundError(f'GEO directory not found: {gdir}')

    disease_signatures: dict[str, dict[str, float]] = {}
    candidates = list(gdir.glob('*.txt*')) + list(gdir.glob('*.csv*'))

    for file_path in candidates:
        name = file_path.stem.replace('.txt', '')
        print(f'Parsing GEO disease signature from: {file_path.name}...')
        try:
            sig = parse_disease_expression_file(file_path, max_genes=max_genes_per_disease)
            if sig:
                disease_signatures[name] = sig
                print(f'  -> Extracted {len(sig)} disease signature genes for {name}.')
            else:
                print(f'  -> Warning: 0 signature genes extracted for {file_path.name}.')
        except Exception as exc:
            print(f'  -> Warning: failed to parse {file_path.name} ({exc}), skipping.')

    return disease_signatures


def update_master_nodes_with_geo(
    master_nodes_path: str | Path | None = None,
    geo_dir_or_signatures: str | Path | dict[str, dict[str, float]] | None = None,
    output_path: str | Path | None = None,
    impute_by_tanimoto: bool = True,
    tanimoto_threshold: float = 0.15,
    max_genes: int = 5000,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enrich master_drug_nodes.csv with GEO disease transcriptomic signatures.

    Scores each drug against disease signatures using its PharmGKB genes and
    BindingDB targets. For unprofiled drugs, imputes disease context via Morgan
    ECFP chemical similarity. Sets `is_geo_active = True` for profiled nodes.
    Supports keyword aliases: master_nodes_csv, geo_dir, output_csv, max_genes.
    """
    if master_nodes_path is None:
        master_nodes_path = kwargs.get('master_nodes_csv')
    if master_nodes_path is None:
        raise ValueError('master_nodes_path (or master_nodes_csv) must be provided')

    if geo_dir_or_signatures is None:
        geo_dir_or_signatures = kwargs.get('geo_dir')
    if geo_dir_or_signatures is None:
        raise ValueError('geo_dir_or_signatures (or geo_dir) must be provided')

    if output_path is None:
        output_path = kwargs.get('output_csv')

    max_genes = kwargs.get('max_genes', max_genes)

    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f'Master nodes file not found: {nodes_p}')

    df_nodes = pd.read_csv(nodes_p)

    if isinstance(geo_dir_or_signatures, dict):
        disease_signatures = geo_dir_or_signatures
    elif Path(geo_dir_or_signatures).is_dir():
        disease_signatures = parse_geo_directory(geo_dir_or_signatures, max_genes_per_disease=max_genes)
    else:
        raise ValueError(f'Invalid GEO source: {geo_dir_or_signatures}')

    disease_names = sorted(list(disease_signatures.keys()))

    node_id_col = 'drug_id' if 'drug_id' in df_nodes.columns else ('canonical_smiles' if 'canonical_smiles' in df_nodes.columns else df_nodes.columns[0])

    # Pass 1: Direct Target Overlap Scoring
    drug_direct_scores: list[tuple[dict[str, float], list[float], bool]] = []
    profiled_fps: list[Any] = []
    profiled_vecs: list[list[float]] = []

    for _, row in df_nodes.iterrows():
        drug_genes: set[str] = set()

        # 1. From PharmGKB
        if 'gene_symbols_json' in row and pd.notna(row['gene_symbols_json']):
            try:
                genes = json.loads(row['gene_symbols_json']) if isinstance(row['gene_symbols_json'], str) else list(row['gene_symbols_json'])
                drug_genes.update(str(g).strip().upper() for g in genes if g)
            except Exception:
                pass
        elif 'gene_symbols' in row and pd.notna(row['gene_symbols']):
            try:
                val = row['gene_symbols']
                genes = json.loads(val) if (isinstance(val, str) and val.startswith('[')) else str(val).split(';')
                drug_genes.update(str(g).strip().upper() for g in genes if g)
            except Exception:
                pass

        # 2. From BindingDB targets
        if 'bindingdb_targets_json' in row and pd.notna(row['bindingdb_targets_json']):
            try:
                targets = json.loads(row['bindingdb_targets_json']) if isinstance(row['bindingdb_targets_json'], str) else dict(row['bindingdb_targets_json'])
                drug_genes.update(str(t).strip().upper() for t in targets.keys())
            except Exception:
                pass

        # Map common target aliases to HGNC symbols
        for t in list(drug_genes):
            for pattern, sym in TARGET_TO_GENE_MAP.items():
                if pattern in t:
                    drug_genes.add(sym)

        scores: dict[str, float] = {}
        vec: list[float] = []

        if drug_genes and disease_signatures:
            for dname in disease_names:
                sig = disease_signatures[dname]
                overlapping = [sig[gene] for gene in drug_genes if gene in sig]
                score = float(np.mean(overlapping)) if overlapping else 0.0
                scores[dname] = round(score, 4)
                vec.append(round(score, 4))
            has_sig = any(v > 0 for v in vec)
        else:
            for dname in disease_names:
                scores[dname] = 0.0
                vec.append(0.0)
            has_sig = False

        drug_direct_scores.append((scores, vec, has_sig))

        if has_sig and impute_by_tanimoto:
            can = canonicalize_smiles(str(row[node_id_col]))
            if can:
                mol = Chem.MolFromSmiles(can)
                if mol is not None:
                    profiled_fps.append(_MORGAN_GEN.GetFingerprint(mol))
                    profiled_vecs.append(vec)

    # Pass 2: Impute for remaining drugs via Morgan ECFP Chemical Similarity
    geo_signatures_json: list[str] = []
    geo_vectors_json: list[str] = []
    geo_active_flags: list[bool] = []

    matched_direct = 0
    matched_similarity = 0

    for idx, row in df_nodes.iterrows():
        scores, vec, has_sig = drug_direct_scores[idx]

        if has_sig:
            matched_direct += 1
            geo_signatures_json.append(json.dumps(scores))
            geo_vectors_json.append(json.dumps(vec))
            geo_active_flags.append(True)
        elif impute_by_tanimoto and profiled_fps:
            can = canonicalize_smiles(str(row[node_id_col]))
            imputed = False
            if can:
                mol = Chem.MolFromSmiles(can)
                if mol is not None:
                    fp = _MORGAN_GEN.GetFingerprint(mol)
                    sims = DataStructs.BulkTanimotoSimilarity(fp, profiled_fps)
                    max_sim_idx = int(np.argmax(sims))
                    max_sim = float(sims[max_sim_idx])
                    if max_sim >= tanimoto_threshold:
                        base_vec = profiled_vecs[max_sim_idx]
                        imputed_vec = [round(v * max_sim, 4) for v in base_vec]
                        imputed_scores = {dname: imputed_vec[i] for i, dname in enumerate(disease_names)}
                        geo_signatures_json.append(json.dumps(imputed_scores))
                        geo_vectors_json.append(json.dumps(imputed_vec))
                        geo_active_flags.append(True)
                        matched_similarity += 1
                        imputed = True

            if not imputed:
                zero_vec = [0.0] * len(disease_names)
                zero_scores = {dname: 0.0 for dname in disease_names}
                geo_signatures_json.append(json.dumps(zero_scores))
                geo_vectors_json.append(json.dumps(zero_vec))
                geo_active_flags.append(False)
        else:
            zero_vec = [0.0] * len(disease_names)
            zero_scores = {dname: 0.0 for dname in disease_names}
            geo_signatures_json.append(json.dumps(zero_scores))
            geo_vectors_json.append(json.dumps(zero_vec))
            geo_active_flags.append(False)

    df_nodes['geo_expression_signatures_json'] = geo_signatures_json
    df_nodes['geo_signature_vector'] = geo_vectors_json
    df_nodes['is_geo_active'] = geo_active_flags

    total_matched = sum(geo_active_flags)
    target_out = Path(output_path) if output_path else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_out, index=False)

    summary = {
        'total_nodes': len(df_nodes),
        'nodes_with_geo_signatures': total_matched,
        'matched_direct_overlap': matched_direct,
        'matched_chemical_analogs': matched_similarity,
        'geo_coverage_pct': (total_matched / len(df_nodes) * 100.0) if len(df_nodes) else 0.0,
        'disease_signatures_loaded': disease_names,
        'exported_to': str(target_out),
    }
    print(
        f'GEO synchronization complete: {total_matched} / {len(df_nodes)} drugs covered ({summary["geo_coverage_pct"]:.1f}%).\n'
        f'  (Direct Biological Overlap: {matched_direct}, Chemical Analog Imputed: {matched_similarity})'
    )
    return df_nodes, summary
