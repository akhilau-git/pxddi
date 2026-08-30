"""Leakage-safe parsing of supplied PharmGKB chemical--gene/pathway evidence.

PharmGKB is not a labelled drug--drug interaction dataset. Its records are
therefore never appended to TWOSIDES positives or treated as external DDI test
labels. This module produces auditable chemical--gene evidence and only maps
names by exact normalised matching; ambiguous or unmatched names stay unmapped.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import unicodedata
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase


RELATIONSHIP_COLUMNS = {
    'Entity1_id', 'Entity1_name', 'Entity1_type',
    'Entity2_id', 'Entity2_name', 'Entity2_type',
    'Evidence', 'Association', 'PK', 'PD', 'PMIDs',
}
EVIDENCE_COLUMNS = [
    'drug_name', 'gene_symbol', 'evidence_source', 'source_file',
    'association', 'evidence', 'pk', 'pd', 'pmids',
]
SMILES_COLUMN_CANDIDATES = (
    'canonical_smiles', 'smiles', 'drug_smiles', 'structure', 'isomeric_smiles',
)
NAME_COLUMN_CANDIDATES = ('drug_name', 'drugname', 'name', 'drug', 'drug_label')


def normalise_drug_name(value: object) -> str | None:
    """Return a conservative key for exact-name matching, never fuzzy matching."""
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize('NFKC', value).casefold().strip()
    normalized = re.sub(r'[^\w]+', ' ', normalized, flags=re.UNICODE)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized or None


def _canonicalize_smiles(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None


def _read_tsv(path: str | Path, required: set[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'PharmGKB file was not found: {source}')
    frame = pd.read_csv(source, sep='\t', dtype=str, keep_default_na=False, low_memory=False)
    if required is not None:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f'PharmGKB file {source} is missing required columns {sorted(missing)}. '
                f'Observed columns: {frame.columns.tolist()}.'
            )
    return frame


def _empty_evidence() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def load_relationship_chemical_gene_evidence(path: str | Path) -> pd.DataFrame:
    """Extract only Chemical↔Gene records from ``relationships.tsv``."""
    source = Path(path)
    relationships = _read_tsv(source, RELATIONSHIP_COLUMNS)
    records: list[dict[str, str]] = []
    for row in relationships.to_dict(orient='records'):
        left_type = str(row['Entity1_type']).strip().casefold()
        right_type = str(row['Entity2_type']).strip().casefold()
        if left_type == 'chemical' and right_type == 'gene':
            drug_name, gene_symbol = row['Entity1_name'], row['Entity2_name']
        elif left_type == 'gene' and right_type == 'chemical':
            drug_name, gene_symbol = row['Entity2_name'], row['Entity1_name']
        else:
            continue
        if not normalise_drug_name(drug_name) or not str(gene_symbol).strip():
            continue
        records.append({
            'drug_name': str(drug_name).strip(),
            'gene_symbol': str(gene_symbol).strip(),
            'evidence_source': 'relationships_chemical_gene',
            'source_file': str(source),
            'association': str(row['Association']).strip(),
            'evidence': str(row['Evidence']).strip(),
            'pk': str(row['PK']).strip(),
            'pd': str(row['PD']).strip(),
            'pmids': str(row['PMIDs']).strip(),
        })
    return pd.DataFrame(records, columns=EVIDENCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def _split_multi_value(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in re.split(r'[,;|]', value) if item.strip()]


def load_pathway_chemical_gene_evidence(path: str | Path) -> pd.DataFrame:
    """Extract explicit ``Drugs`` × ``Genes`` pathway annotations recursively."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f'PharmGKB pathway directory was not found: {root}')
    records: list[dict[str, str]] = []
    for source in sorted(root.rglob('*.tsv')):
        if source.name.casefold() == 'relationships.tsv':
            continue
        try:
            frame = _read_tsv(source)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        if not {'Drugs', 'Genes'}.issubset(frame.columns):
            continue
        for row in frame.to_dict(orient='records'):
            drugs = _split_multi_value(row.get('Drugs', ''))
            genes = _split_multi_value(row.get('Genes', ''))
            for drug_name in drugs:
                if not normalise_drug_name(drug_name):
                    continue
                for gene_symbol in genes:
                    if not gene_symbol.strip():
                        continue
                    records.append({
                        'drug_name': drug_name,
                        'gene_symbol': gene_symbol.strip(),
                        'evidence_source': 'pathway_drug_gene',
                        'source_file': str(source),
                        'association': '',
                        'evidence': '',
                        'pk': '',
                        'pd': '',
                        'pmids': str(row.get('PMIDs', '')).strip(),
                    })
    return pd.DataFrame(records, columns=EVIDENCE_COLUMNS).drop_duplicates().reset_index(drop=True)


def load_pharmgkb_chemical_gene_evidence(
    relationships_path: str | Path,
    pharmgkb_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine valid PharmGKB chemical--gene evidence with provenance counts."""
    relationship_evidence = load_relationship_chemical_gene_evidence(relationships_path)
    pathway_evidence = load_pathway_chemical_gene_evidence(pharmgkb_root)
    combined = pd.concat([relationship_evidence, pathway_evidence], ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True) if not combined.empty else _empty_evidence()
    summary: dict[str, Any] = {
        'relationship_chemical_gene_rows': int(len(relationship_evidence)),
        'pathway_chemical_gene_rows': int(len(pathway_evidence)),
        'combined_rows': int(len(combined)),
        'unique_pharmgkb_chemical_names': int(combined['drug_name'].nunique()) if not combined.empty else 0,
        'unique_gene_symbols': int(combined['gene_symbol'].nunique()) if not combined.empty else 0,
        'purpose': 'candidate_biological_auxiliary_evidence_only',
        'not_evidence_of': ['DDI_label', 'external_DDI_validation', 'clinical_outcome'],
    }
    return combined, summary


def load_twosides_drug_catalog(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a name+SMILES catalogue required for a conservative PharmGKB join."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'TWOSIDES drug catalogue was not found: {source}')
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, low_memory=False)
    lower_to_actual = {str(column).casefold(): str(column) for column in frame.columns}
    smiles_column = next((lower_to_actual.get(name) for name in SMILES_COLUMN_CANDIDATES if name in lower_to_actual), None)
    name_column = next((lower_to_actual.get(name) for name in NAME_COLUMN_CANDIDATES if name in lower_to_actual), None)
    metadata: dict[str, Any] = {
        'source_file': str(source),
        'source_columns': frame.columns.tolist(),
        'selected_name_column': name_column,
        'selected_smiles_column': smiles_column,
    }
    if name_column is None or smiles_column is None:
        metadata['mapping_ready'] = False
        metadata['reason'] = (
            'No recognised name+SMILES column pair was found. Configure an explicit '
            'catalogue adapter after inspecting source_columns; no fuzzy mapping was attempted.'
        )
        return pd.DataFrame(columns=['source_drug_name', 'normalised_drug_name', 'canonical_smiles']), metadata
    catalog = pd.DataFrame({
        'source_drug_name': frame[name_column],
        'normalised_drug_name': frame[name_column].map(normalise_drug_name),
        'canonical_smiles': frame[smiles_column].map(_canonicalize_smiles),
    })
    catalog = catalog.dropna(subset=['normalised_drug_name', 'canonical_smiles']).drop_duplicates().reset_index(drop=True)
    metadata.update({
        'mapping_ready': True,
        'valid_catalog_rows': int(len(catalog)),
        'unique_normalised_names': int(catalog['normalised_drug_name'].nunique()),
        'unique_canonical_smiles': int(catalog['canonical_smiles'].nunique()),
    })
    return catalog, metadata


def resolve_pharmgkb_evidence_to_twosides(
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map chemical names only when exactly one canonical structure is possible."""
    required_evidence = {'drug_name', 'gene_symbol'}
    required_catalog = {'normalised_drug_name', 'canonical_smiles'}
    if missing := required_evidence.difference(evidence.columns):
        raise ValueError(f'PharmGKB evidence is missing required columns: {sorted(missing)}.')
    if missing := required_catalog.difference(catalog.columns):
        raise ValueError(f'TWOSIDES catalog is missing required columns: {sorted(missing)}.')
    structures_by_name: dict[str, set[str]] = defaultdict(set)
    for row in catalog[['normalised_drug_name', 'canonical_smiles']].itertuples(index=False):
        structures_by_name[row.normalised_drug_name].add(row.canonical_smiles)

    resolved = evidence.copy()
    resolved['normalised_drug_name'] = resolved['drug_name'].map(normalise_drug_name)
    statuses: list[str] = []
    mapped_smiles: list[str | None] = []
    for key in resolved['normalised_drug_name']:
        candidates = structures_by_name.get(key, set()) if key is not None else set()
        if not candidates:
            statuses.append('unmatched_exact_name')
            mapped_smiles.append(None)
        elif len(candidates) > 1:
            statuses.append('ambiguous_exact_name')
            mapped_smiles.append(None)
        else:
            statuses.append('matched_exact_name')
            mapped_smiles.append(next(iter(candidates)))
    resolved['mapping_status'] = statuses
    resolved['canonical_smiles'] = mapped_smiles
    resolved = resolved.sort_values(['mapping_status', 'drug_name', 'gene_symbol'], ignore_index=True)
    summary: dict[str, Any] = {
        'evidence_rows': int(len(resolved)),
        'matched_exact_name_rows': int((resolved['mapping_status'] == 'matched_exact_name').sum()),
        'ambiguous_exact_name_rows': int((resolved['mapping_status'] == 'ambiguous_exact_name').sum()),
        'unmatched_exact_name_rows': int((resolved['mapping_status'] == 'unmatched_exact_name').sum()),
        'matched_unique_structures': int(resolved.loc[
            resolved['mapping_status'] == 'matched_exact_name', 'canonical_smiles'
        ].nunique()),
        'mapping_policy': 'exact_normalised_name_only_no_fuzzy_matching',
    }
    return resolved, summary
