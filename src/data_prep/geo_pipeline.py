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
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .master_schema import canonicalize_smiles, smiles_to_inchikey

DEFAULT_TOP_SIGNATURE_GENES = 50


def _open_geo_file(file_path: Path):
    """Open a plain text or gzip-compressed file transparently."""
    if file_path.suffix == '.gz':
        return gzip.open(file_path, 'rt', encoding='utf-8', errors='replace')
    return open(file_path, 'r', encoding='utf-8', errors='replace')


def parse_disease_expression_file(
    file_path: str | Path,
    max_genes: int = 500,
) -> dict[str, float]:
    """Parse a GEO disease expression file and extract top perturbed gene scores.

    Handles series matrix formats and tabular expression tables by detecting
    probe/gene symbol headers and computing mean expression or log fold change.
    """
    fpath = Path(file_path)
    if not fpath.is_file():
        raise FileNotFoundError(f'GEO expression file not found: {fpath}')

    gene_scores: dict[str, float] = {}
    with _open_geo_file(fpath) as f:
        # Skip comment lines (e.g. '^', '!', '#' in SOFT format)
        lines = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(('!', '^', '#')):
                continue
            lines.append(stripped)
            if len(lines) >= 10000:  # Sample up to 10k lines for signature extraction
                break

    if not lines:
        return {}

    # Header parsing
    delimiter = '\t' if '\t' in lines[0] else ','
    header = [col.strip().strip('"').upper() for col in lines[0].split(delimiter)]

    # Look for gene symbol column
    gene_col_idx = 0
    for idx, col in enumerate(header):
        if any(term in col for term in ['GENE', 'SYMBOL', 'ID', 'PROBE', 'NAME']):
            gene_col_idx = idx
            break

    for line in lines[1:]:
        parts = line.split(delimiter)
        if len(parts) <= gene_col_idx:
            continue
        raw_gene = parts[gene_col_idx].strip().strip('"').upper()
        # Clean multi-gene annotations (e.g. 'CYP2D6 /// CYP2D7')
        gene = raw_gene.split('///')[0].strip().split('//')[0].strip().split(' ')[0].strip()
        if not gene or len(gene) < 2 or gene.isdigit():
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
            # Score as variance or magnitude of expression
            score = float(np.std(numeric_vals) if len(numeric_vals) > 1 else abs(numeric_vals[0]))
            if gene in gene_scores:
                gene_scores[gene] = max(gene_scores[gene], score)
            else:
                gene_scores[gene] = score

        if len(gene_scores) >= max_genes * 2:
            break

    # Sort and retain top perturbed genes
    sorted_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:max_genes]
    max_val = max((val for _, val in sorted_genes), default=1.0) or 1.0
    # Normalize scores to [0, 1]
    return {gene: float(val / max_val) for gene, val in sorted_genes}


def parse_geo_directory(
    geo_dir: str | Path,
    max_genes_per_disease: int = 200,
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
        except Exception as exc:
            print(f'  -> Warning: failed to parse {file_path.name} ({exc}), skipping.')

    return disease_signatures


def update_master_nodes_with_geo(
    master_nodes_path: str | Path,
    geo_dir_or_signatures: str | Path | dict[str, dict[str, float]],
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enrich master_drug_nodes.csv with GEO disease transcriptomic signatures.

    Scores each drug against disease signatures using its PharmGKB genes and
    BindingDB targets. Sets `is_geo_active = True` for profiled nodes.
    """
    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f'Master nodes file not found: {nodes_p}')

    df_nodes = pd.read_csv(nodes_p)

    if isinstance(geo_dir_or_signatures, dict):
        disease_signatures = geo_dir_or_signatures
    elif Path(geo_dir_or_signatures).is_dir():
        disease_signatures = parse_geo_directory(geo_dir_or_signatures)
    else:
        raise ValueError(f'Invalid GEO source: {geo_dir_or_signatures}')

    disease_names = sorted(list(disease_signatures.keys()))

    geo_signatures_json: list[str] = []
    geo_vectors_json: list[str] = []
    geo_active_flags: list[bool] = []

    matched = 0

    for _, row in df_nodes.iterrows():
        # Collect all biological targets/genes associated with this drug
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

        # Score overlap with each disease
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
            if has_sig:
                matched += 1
                geo_active_flags.append(True)
            else:
                geo_active_flags.append(False)
        else:
            for dname in disease_names:
                scores[dname] = 0.0
                vec.append(0.0)
            geo_active_flags.append(False)

        geo_signatures_json.append(json.dumps(scores))
        geo_vectors_json.append(json.dumps(vec))

    df_nodes['geo_expression_signatures_json'] = geo_signatures_json
    df_nodes['geo_signature_vector'] = geo_vectors_json
    df_nodes['is_geo_active'] = geo_active_flags

    target_out = Path(output_path) if output_path else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(target_out, index=False)

    summary = {
        'total_nodes': len(df_nodes),
        'nodes_with_geo_signatures': matched,
        'geo_coverage_pct': (matched / len(df_nodes) * 100.0) if len(df_nodes) else 0.0,
        'disease_signatures_loaded': disease_names,
        'exported_to': str(target_out),
    }
    print(f'GEO synchronization complete: {matched} / {len(df_nodes)} drugs covered ({summary["geo_coverage_pct"]:.1f}%).')
    return df_nodes, summary
