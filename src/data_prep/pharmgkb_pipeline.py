"""Leakage-safe parsing of supplied PharmGKB chemical--gene/pathway evidence.

PharmGKB is not a labelled drug--drug interaction dataset. Its records are
therefore never appended to TWOSIDES positives or treated as external DDI test
labels. This module produces auditable chemical--gene evidence and only maps
names by exact normalised matching; ambiguous or unmatched names stay unmapped.
"""

from __future__ import annotations

from collections import defaultdict
import json
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
PHARMGKB_CHEMICAL_ID_COLUMN_CANDIDATES = (
    'pharmgkb accession id', 'pharmgkb_accession_id', 'chemical_id', 'id',
)
PHARMGKB_CHEMICAL_NAME_COLUMN_CANDIDATES = ('name', 'chemical_name', 'drug_name')
PHARMGKB_CHEMICAL_SMILES_COLUMN_CANDIDATES = (
    'smiles', 'canonical_smiles', 'isomeric_smiles',
)


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


def _select_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first recognised header without guessing a semantic field."""
    lower_to_actual = {str(column).casefold(): str(column) for column in frame.columns}
    return next(
        (lower_to_actual.get(candidate) for candidate in candidates if candidate in lower_to_actual),
        None,
    )


def load_pharmgkb_chemical_catalog(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read direct PharmGKB chemical structures for an exact identity audit.

    This deliberately accepts only an explicit name plus SMILES column from a
    PharmGKB export. Generic names, brand names, cross references, and fuzzy
    name matching are not used because they can silently assign evidence to the
    wrong molecule.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'PharmGKB chemical catalogue was not found: {source}')
    separator = '\t' if source.suffix.casefold() in {'.tsv', '.txt'} else ','
    frame = pd.read_csv(source, sep=separator, dtype=str, keep_default_na=False, low_memory=False)
    id_column = _select_column(frame, PHARMGKB_CHEMICAL_ID_COLUMN_CANDIDATES)
    name_column = _select_column(frame, PHARMGKB_CHEMICAL_NAME_COLUMN_CANDIDATES)
    smiles_column = _select_column(frame, PHARMGKB_CHEMICAL_SMILES_COLUMN_CANDIDATES)
    summary: dict[str, Any] = {
        'source_file': str(source),
        'source_columns': frame.columns.tolist(),
        'selected_id_column': id_column,
        'selected_name_column': name_column,
        'selected_smiles_column': smiles_column,
        'mapping_policy': 'explicit_pharmgkb_name_and_smiles_only',
    }
    if name_column is None or smiles_column is None:
        summary.update({
            'mapping_ready': False,
            'reason': (
                'No recognised direct PharmGKB name+SMILES column pair was found. '
                'Do not substitute generic names, brand names, or fuzzy matching.'
            ),
        })
        return pd.DataFrame(columns=[
            'pharmgkb_chemical_id', 'source_chemical_name',
            'normalised_drug_name', 'canonical_smiles',
        ]), summary
    catalog = pd.DataFrame({
        'pharmgkb_chemical_id': (
            frame[id_column].astype(str).str.strip() if id_column is not None
            else pd.Series('', index=frame.index, dtype=str)
        ),
        'source_chemical_name': frame[name_column].astype(str).str.strip(),
        'normalised_drug_name': frame[name_column].map(normalise_drug_name),
        'canonical_smiles': frame[smiles_column].map(_canonicalize_smiles),
    })
    catalog = catalog.dropna(
        subset=['normalised_drug_name', 'canonical_smiles']
    ).drop_duplicates().reset_index(drop=True)
    summary.update({
        'mapping_ready': True,
        'valid_catalog_rows': int(len(catalog)),
        'unique_normalised_names': int(catalog['normalised_drug_name'].nunique()),
        'unique_canonical_smiles': int(catalog['canonical_smiles'].nunique()),
    })
    return catalog, summary


def build_twosides_structure_catalog(edges_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical-SMILES identity catalogue directly from DDI edges.

    The supplied ``twosides_drugs.csv`` has identifiers only, but the edge
    table itself contains molecular SMILES. This is sufficient for an exact
    structure match and avoids inventing a name mapping.
    """
    source = Path(edges_path)
    if not source.is_file():
        raise FileNotFoundError(f'TWOSIDES pair table was not found: {source}')
    edges = pd.read_csv(source, usecols=['source', 'target'], dtype=str, keep_default_na=False)
    raw_smiles = pd.unique(edges[['source', 'target']].to_numpy().ravel())
    canonical = [
        canonical_smiles for value in raw_smiles
        if (canonical_smiles := _canonicalize_smiles(value)) is not None
    ]
    catalog = pd.DataFrame({'canonical_smiles': sorted(set(canonical))})
    summary: dict[str, Any] = {
        'source_file': str(source),
        'raw_unique_smiles_values': int(len(raw_smiles)),
        'canonical_structure_rows': int(len(catalog)),
        'purpose': 'exact_structure_identity_catalogue_only',
    }
    return catalog, summary


def resolve_pharmgkb_evidence_to_chemical_catalog(
    evidence: pd.DataFrame,
    chemical_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve evidence names to one direct PharmGKB chemical structure only."""
    required_evidence = {'drug_name', 'gene_symbol'}
    required_catalog = {
        'pharmgkb_chemical_id', 'normalised_drug_name', 'canonical_smiles',
    }
    if missing := required_evidence.difference(evidence.columns):
        raise ValueError(f'PharmGKB evidence is missing required columns: {sorted(missing)}.')
    if missing := required_catalog.difference(chemical_catalog.columns):
        raise ValueError(f'PharmGKB chemical catalogue is missing required columns: {sorted(missing)}.')
    structures_by_name: dict[str, set[str]] = defaultdict(set)
    identifiers_by_name: dict[str, set[str]] = defaultdict(set)
    for row in chemical_catalog[
        ['normalised_drug_name', 'canonical_smiles', 'pharmgkb_chemical_id']
    ].itertuples(index=False):
        structures_by_name[row.normalised_drug_name].add(row.canonical_smiles)
        if row.pharmgkb_chemical_id:
            identifiers_by_name[row.normalised_drug_name].add(row.pharmgkb_chemical_id)
    resolved = evidence.copy()
    resolved['normalised_drug_name'] = resolved['drug_name'].map(normalise_drug_name)
    statuses: list[str] = []
    smiles: list[str | None] = []
    identifiers: list[str | None] = []
    for key in resolved['normalised_drug_name']:
        structures = structures_by_name.get(key, set()) if key is not None else set()
        ids = identifiers_by_name.get(key, set()) if key is not None else set()
        if not structures:
            statuses.append('unmatched_exact_name_to_pharmgkb_catalog')
            smiles.append(None)
            identifiers.append(None)
        elif len(structures) > 1:
            statuses.append('ambiguous_exact_name_to_pharmgkb_catalog')
            smiles.append(None)
            identifiers.append(None)
        else:
            statuses.append('matched_exact_name_to_pharmgkb_catalog')
            smiles.append(next(iter(structures)))
            identifiers.append(next(iter(ids)) if len(ids) == 1 else None)
    resolved['pharmgkb_catalog_mapping_status'] = statuses
    resolved['pharmgkb_chemical_id'] = identifiers
    resolved['canonical_smiles'] = smiles
    resolved = resolved.sort_values(
        ['pharmgkb_catalog_mapping_status', 'drug_name', 'gene_symbol'], ignore_index=True
    )
    summary = {
        'evidence_rows': int(len(resolved)),
        'matched_exact_name_rows': int((
            resolved['pharmgkb_catalog_mapping_status'] ==
            'matched_exact_name_to_pharmgkb_catalog'
        ).sum()),
        'ambiguous_exact_name_rows': int((
            resolved['pharmgkb_catalog_mapping_status'] ==
            'ambiguous_exact_name_to_pharmgkb_catalog'
        ).sum()),
        'unmatched_exact_name_rows': int((
            resolved['pharmgkb_catalog_mapping_status'] ==
            'unmatched_exact_name_to_pharmgkb_catalog'
        ).sum()),
        'mapping_policy': 'exact_normalised_name_to_direct_pharmgkb_structure_only',
    }
    return resolved, summary


def resolve_pharmgkb_catalog_to_twosides_structures(
    resolved_evidence: pd.DataFrame,
    twosides_structures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only direct canonical-SMILES PharmGKB↔TWOSIDES matches."""
    required_evidence = {'canonical_smiles', 'pharmgkb_catalog_mapping_status'}
    if missing := required_evidence.difference(resolved_evidence.columns):
        raise ValueError(f'Resolved PharmGKB evidence is missing columns: {sorted(missing)}.')
    if 'canonical_smiles' not in twosides_structures:
        raise ValueError('TWOSIDES structure catalogue is missing canonical_smiles.')
    twosides_smiles = set(twosides_structures['canonical_smiles'].dropna().astype(str))
    resolved = resolved_evidence.copy()
    catalog_matched = resolved['pharmgkb_catalog_mapping_status'].eq(
        'matched_exact_name_to_pharmgkb_catalog'
    )
    twosides_matched = catalog_matched & resolved['canonical_smiles'].isin(twosides_smiles)
    resolved['twosides_structure_mapping_status'] = 'not_resolved_to_pharmgkb_structure'
    resolved.loc[catalog_matched, 'twosides_structure_mapping_status'] = (
        'not_present_in_twosides_structure_catalogue'
    )
    resolved.loc[twosides_matched, 'twosides_structure_mapping_status'] = (
        'matched_exact_canonical_smiles'
    )
    resolved = resolved.sort_values(
        ['twosides_structure_mapping_status', 'drug_name', 'gene_symbol'], ignore_index=True
    )
    matched = resolved['twosides_structure_mapping_status'].eq('matched_exact_canonical_smiles')
    summary = {
        'evidence_rows': int(len(resolved)),
        'matched_exact_canonical_smiles_rows': int(matched.sum()),
        'matched_unique_twosides_structures': int(resolved.loc[matched, 'canonical_smiles'].nunique()),
        'mapping_policy': 'exact_canonical_smiles_only',
    }
    return resolved, summary


def aggregate_twosides_pharmgkb_gene_profiles(
    resolved_evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate matched PharmGKB gene evidence per TWOSIDES structure.

    This is an auditable candidate feature table, never an interaction label.
    Gene symbols are JSON-encoded so a later model can construct a vocabulary
    without reparsing the original PharmGKB evidence.
    """
    required = {'canonical_smiles', 'gene_symbol', 'twosides_structure_mapping_status'}
    if missing := required.difference(resolved_evidence.columns):
        raise ValueError(f'Resolved PharmGKB evidence is missing columns: {sorted(missing)}.')
    matched = resolved_evidence.loc[
        resolved_evidence['twosides_structure_mapping_status'].eq(
            'matched_exact_canonical_smiles'
        ),
        ['canonical_smiles', 'gene_symbol'],
    ].copy()
    if matched.empty:
        empty = pd.DataFrame(columns=[
            'canonical_smiles', 'gene_symbols_json', 'unique_gene_count', 'evidence_row_count',
        ])
        return empty, {
            'matched_unique_twosides_structures': 0,
            'unique_gene_symbols': 0,
            'purpose': 'candidate_biological_auxiliary_features_only',
            'not_evidence_of': ['DDI_label', 'external_DDI_validation', 'clinical_outcome'],
        }
    profile_rows: list[dict[str, Any]] = []
    for smiles, group in matched.groupby('canonical_smiles', sort=True):
        genes = sorted({str(value).strip() for value in group['gene_symbol'] if str(value).strip()})
        profile_rows.append({
            'canonical_smiles': smiles,
            'gene_symbols_json': json.dumps(genes),
            'unique_gene_count': len(genes),
            'evidence_row_count': int(len(group)),
        })
    profiles = pd.DataFrame(profile_rows)
    summary = {
        'matched_unique_twosides_structures': int(len(profiles)),
        'unique_gene_symbols': int(matched['gene_symbol'].nunique()),
        'purpose': 'candidate_biological_auxiliary_features_only',
        'not_evidence_of': ['DDI_label', 'external_DDI_validation', 'clinical_outcome'],
    }
    return profiles, summary


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
