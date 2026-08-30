"""Schema-validated utilities for the supplied ChEMBL structure files.

The current PxDDI Drive folder contains ``chembl_37_chemreps.txt.gz`` and a
UniProt-to-ChEMBL *target metadata* mapping. Neither file links a molecule to
a target. This module therefore supports structural-overlap auditing and a
future self-supervised molecular pretraining corpus only. It deliberately does
not manufacture drug-target features from unrelated files.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, rdBase


CHEMREPS_REQUIRED_COLUMNS = {
    'chembl_id',
    'canonical_smiles',
    'standard_inchi',
    'standard_inchi_key',
}
CHEMREPS_USE_COLUMNS = [
    'chembl_id',
    'canonical_smiles',
    'standard_inchi',
    'standard_inchi_key',
]


def _canonicalize_smiles(value: object) -> str | None:
    """Canonicalise one SMILES value without accepting malformed chemistry."""
    if not isinstance(value, str) or not value.strip():
        return None
    # Large public chemical corpora contain a small number of malformed or
    # unsupported records.  They are counted and excluded by callers; blocking
    # RDKit's per-row diagnostic avoids burying the actual audit result.
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a large dataset at once."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_chemreps_columns(columns: Iterable[object], source: Path) -> None:
    observed = {str(column) for column in columns}
    missing = CHEMREPS_REQUIRED_COLUMNS.difference(observed)
    if missing:
        raise ValueError(
            f'ChEMBL chemical-representation file {source} is missing required '
            f'columns {sorted(missing)}. Observed columns: {sorted(observed)}.'
        )


def iter_chembl_chemreps(
    path: str | Path,
    *,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Yield validated chunks from ChEMBL's tab-delimited ``chemreps`` export."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'ChEMBL chemical-representation file was not found: {source}')
    if chunksize < 1:
        raise ValueError('chunksize must be positive.')

    reader = pd.read_csv(
        source,
        sep='\t',
        compression='infer',
        usecols=lambda column: str(column) in CHEMREPS_USE_COLUMNS,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    saw_chunk = False
    for chunk in reader:
        if not saw_chunk:
            _validate_chemreps_columns(chunk.columns, source)
            saw_chunk = True
        yield chunk[CHEMREPS_USE_COLUMNS].copy()
    if not saw_chunk:
        raise ValueError(f'ChEMBL chemical-representation file is empty: {source}')


def load_chembl_chemreps(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load a validated, optionally limited ChEMBL molecular structure table."""
    if nrows is not None and nrows < 1:
        raise ValueError('nrows must be positive when supplied.')
    chunks: list[pd.DataFrame] = []
    remaining = nrows
    for chunk in iter_chembl_chemreps(path):
        if remaining is not None:
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        chunks.append(chunk)
        if remaining == 0:
            break
    return pd.concat(chunks, ignore_index=True)


def audit_chembl_structure_overlap(
    chemreps_path: str | Path,
    pxddi_smiles: Iterable[object],
    *,
    chunksize: int = 100_000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit exact canonical-SMILES overlap without retaining all ChEMBL rows.

    The returned overlap is evidence of structural corpus coverage. It is not
    evidence of drug-target activity and must not be interpreted as a DDI label.
    """
    pxddi_canonical = {
        canonical
        for raw_smiles in pxddi_smiles
        if (canonical := _canonicalize_smiles(raw_smiles)) is not None
    }
    matched_records: list[dict[str, str]] = []
    source_rows = valid_structures = matched_rows = 0
    for chunk in iter_chembl_chemreps(chemreps_path, chunksize=chunksize):
        source_rows += len(chunk)
        canonical_smiles = chunk['canonical_smiles'].map(_canonicalize_smiles)
        valid_mask = canonical_smiles.notna()
        valid_structures += int(valid_mask.sum())
        matches = chunk.loc[valid_mask & canonical_smiles.isin(pxddi_canonical)].copy()
        if not matches.empty:
            matches['canonical_smiles'] = canonical_smiles.loc[matches.index].values
            matched_rows += len(matches)
            matched_records.extend(
                matches[['chembl_id', 'canonical_smiles', 'standard_inchi_key']]
                .to_dict(orient='records')
            )
    overlap = pd.DataFrame(
        matched_records,
        columns=['chembl_id', 'canonical_smiles', 'standard_inchi_key'],
    ).drop_duplicates(['chembl_id', 'canonical_smiles']).sort_values(
        ['canonical_smiles', 'chembl_id'], ignore_index=True
    )
    summary: dict[str, Any] = {
        'source_file': str(Path(chemreps_path)),
        'source_sha256': sha256_file(chemreps_path),
        'source_rows': source_rows,
        'valid_canonical_smiles_rows': valid_structures,
        'pxddi_unique_valid_structures': len(pxddi_canonical),
        'matched_chembl_rows': matched_rows,
        'matched_unique_pxddi_structures': int(overlap['canonical_smiles'].nunique()),
        'purpose': 'structure_corpus_coverage_and_future_pretraining_only',
        'not_evidence_of': ['molecule_target_activity', 'DDI_label', 'external_DDI_validation'],
    }
    return summary, overlap


def load_chembl_uniprot_target_metadata(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load supplied UniProt mapping as target metadata, never as activity."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'ChEMBL UniProt mapping file was not found: {source}')
    frame = pd.read_csv(
        source,
        sep='\t',
        header=None,
        comment='#',
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if frame.empty or frame.shape[1] < 2:
        raise ValueError(
            f'ChEMBL UniProt mapping {source} must contain at least two tab-delimited columns.'
        )
    columns = ['uniprot_accession', 'chembl_target_id', 'target_name'] + [
        f'extra_column_{index}' for index in range(4, frame.shape[1] + 1)
    ]
    frame.columns = columns[:frame.shape[1]]
    frame = frame.apply(lambda column: column.str.strip())
    frame = frame[(frame['uniprot_accession'] != '') & (frame['chembl_target_id'] != '')]
    summary: dict[str, Any] = {
        'source_file': str(source),
        'source_sha256': sha256_file(source),
        'rows': int(len(frame)),
        'unique_uniprot_accessions': int(frame['uniprot_accession'].nunique()),
        'unique_chembl_target_ids': int(frame['chembl_target_id'].nunique()),
        'purpose': 'target_metadata_only',
        'not_evidence_of': ['molecule_target_activity', 'drug_target_mapping'],
    }
    return frame.reset_index(drop=True), summary


def load_chembl_targets(*_args: object, **_kwargs: object) -> pd.DataFrame:
    """Reject old unsafe API instead of silently inventing target records."""
    raise RuntimeError(
        'The supplied ChEMBL files do not contain molecule-to-target activity rows. '
        'Use load_chembl_chemreps() for structure-corpus auditing, or provide a '
        'documented ChEMBL activity export before building target features.'
    )
