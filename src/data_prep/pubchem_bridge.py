"""Auditable mapping of FAERS drug names to PubChem molecular structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import time

import pandas as pd
import requests
from rdkit import Chem


PUBCHEM_URL = (
    'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/'
    'CanonicalSMILES/TXT'
)
REQUIRED_BRIDGE_COLUMNS = {
    'drugname',
    'raw_smiles',
    'canonical_smiles',
    'toxicity_score',
    'n_reports',
}


@dataclass(frozen=True)
class PubChemLookupResult:
    """The result and provenance of one PubChem name query."""

    smiles: str | None
    status: str
    attempts: int
    query_url: str


def pubchem_query_url(drug_name: str) -> str:
    """Build a path-safe PubChem URL for a drug name."""
    if not isinstance(drug_name, str) or not drug_name.strip():
        raise ValueError('drug_name must be a non-empty string.')
    return PUBCHEM_URL.format(quote(drug_name.strip(), safe=''))


def lookup_pubchem_smiles(
    drug_name: str | None,
    max_retries: int = 3,
    timeout_seconds: int = 10,
    request_get=requests.get,
    sleep=time.sleep,
    lookup_cache: dict[str, PubChemLookupResult] | None = None,
) -> PubChemLookupResult:
    """Query PubChem with bounded retries and record the outcome.

    A 404 is a valid unresolved name and is not retried. Transient HTTP errors
    and network failures are retried so a short outage is not silently treated
    as a chemical mapping failure. NaN, None, or empty names are safely skipped.
    """
    if drug_name is None or not isinstance(drug_name, str) or not drug_name.strip() or drug_name.strip().lower() in {'nan', 'none', 'null'}:
        return PubChemLookupResult(None, 'invalid_or_nan_name', 0, '')

    cleaned_name = drug_name.strip().upper()
    if lookup_cache is not None and cleaned_name in lookup_cache:
        return lookup_cache[cleaned_name]

    if max_retries < 1:
        raise ValueError('max_retries must be at least 1.')
    if timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive.')

    query_url = pubchem_query_url(cleaned_name)
    for attempt in range(1, max_retries + 1):
        try:
            response = request_get(query_url, timeout=timeout_seconds)
        except requests.exceptions.RequestException:
            if attempt == max_retries:
                res = PubChemLookupResult(None, 'network_error', attempt, query_url)
                if lookup_cache is not None:
                    lookup_cache[cleaned_name] = res
                return res
            sleep(attempt)
            continue

        if response.status_code == 200:
            smiles = response.text.strip()
            status = 'matched' if smiles else 'empty_response'
            res = PubChemLookupResult(smiles or None, status, attempt, query_url)
            if lookup_cache is not None:
                lookup_cache[cleaned_name] = res
            return res
        if response.status_code in {400, 404}:
            res = PubChemLookupResult(None, f'not_found_http_{response.status_code}', attempt, query_url)
            if lookup_cache is not None:
                lookup_cache[cleaned_name] = res
            return res
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < max_retries:
                sleep(attempt)
                continue
        res = PubChemLookupResult(
            None,
            f'http_{response.status_code}',
            attempt,
            query_url,
        )
        if lookup_cache is not None:
            lookup_cache[cleaned_name] = res
        return res

    raise AssertionError('PubChem retry loop exited unexpectedly.')


def fetch_smiles_from_pubchem(
    drug_name: str,
    max_retries: int = 3,
    timeout_seconds: int = 10,
) -> str | None:
    """Backward-compatible wrapper returning only a matched SMILES string."""
    return lookup_pubchem_smiles(
        drug_name,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    ).smiles


def canonicalize(smiles: str | None) -> str | None:
    """Return RDKit's canonical SMILES representation, or ``None`` if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def validate_bridge_dataframe(bridge: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimum schema required for a reusable toxicity bridge."""
    missing = REQUIRED_BRIDGE_COLUMNS.difference(bridge.columns)
    if missing:
        raise ValueError(
            'Toxicity bridge is missing required columns: '
            f'{sorted(missing)}.'
        )
    validated = bridge.copy()
    validated['drugname'] = validated['drugname'].astype(str).str.strip().str.upper()
    validated['toxicity_score'] = pd.to_numeric(
        validated['toxicity_score'], errors='raise'
    )
    validated['n_reports'] = pd.to_numeric(validated['n_reports'], errors='raise')
    if (validated['n_reports'] < 0).any():
        raise ValueError('Toxicity bridge contains negative n_reports values.')
    return validated


def audit_toxicity_bridge(bridge: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """Return coverage counts and unresolved duplicate-structure conflicts.

    This function intentionally does not resolve conflicting labels. A later
    documented scientific policy must decide whether a duplicated structure is
    a salt, combination product, synonym, or a mapping error.
    """
    validated = validate_bridge_dataframe(bridge)
    matched = validated.dropna(subset=['canonical_smiles']).copy()
    matched['canonical_smiles'] = matched['canonical_smiles'].astype(str).str.strip()
    matched = matched[matched['canonical_smiles'] != '']

    grouped = matched.groupby('canonical_smiles', sort=True).agg(
        source_rows=('canonical_smiles', 'size'),
        unique_drug_names=('drugname', 'nunique'),
        unique_scores=('toxicity_score', 'nunique'),
        drug_names=('drugname', lambda values: ' | '.join(sorted(set(values)))),
        toxicity_scores=(
            'toxicity_score',
            lambda values: ' | '.join(f'{value:.12g}' for value in sorted(set(values))),
        ),
        report_counts=(
            'n_reports',
            lambda values: ' | '.join(str(int(value)) for value in sorted(set(values))),
        ),
    ).reset_index()
    duplicates = grouped[grouped['source_rows'] > 1].copy()
    conflicts = grouped[
        (grouped['source_rows'] > 1) & (grouped['unique_scores'] > 1)
    ].copy()
    summary = {
        'source_rows': int(len(validated)),
        'rows_with_canonical_smiles': int(len(matched)),
        'unique_canonical_structures': int(len(grouped)),
        'duplicate_canonical_structures': int(len(duplicates)),
        'conflicting_canonical_structures': int(len(conflicts)),
    }
    return summary, conflicts


def resolve_toxicity_bridge(bridge: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Apply the conservative duplicate policy for toxicity supervision.

    Exact duplicate labels collapse to a single representative structure.
    Structures with conflicting source scores are excluded from supervision;
    callers receive the conflict table and must save it as a run artifact for
    review. No choice is made based on report count or CSV row order.
    """
    validated = validate_bridge_dataframe(bridge)
    summary, conflicts = audit_toxicity_bridge(validated)
    matched = validated.dropna(subset=['canonical_smiles']).copy()
    matched['canonical_smiles'] = matched['canonical_smiles'].astype(str).str.strip()
    matched = matched[matched['canonical_smiles'] != '']
    conflicting_structures = set(conflicts['canonical_smiles'])
    resolved = matched[
        ~matched['canonical_smiles'].isin(conflicting_structures)
    ].sort_values(
        ['canonical_smiles', 'n_reports', 'drugname'],
        ascending=[True, False, True],
    ).drop_duplicates(subset='canonical_smiles', keep='first')
    summary['resolved_unique_canonical_structures'] = int(len(resolved))
    summary['excluded_conflicting_structures'] = int(len(conflicts))
    return resolved.reset_index(drop=True), summary, conflicts


def load_validated_bridge_cache(cache_path: str | Path) -> pd.DataFrame:
    """Load a cached bridge only after confirming its required schema."""
    path = Path(cache_path)
    return validate_bridge_dataframe(pd.read_csv(path))


def _load_or_create_lookup_cache(cache_file: Path | None) -> dict[str, PubChemLookupResult]:
    """Load persistent disk lookup cache if available."""
    if cache_file is None or not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding='utf-8'))
        return {
            k: PubChemLookupResult(
                smiles=v.get('smiles'),
                status=v.get('status', 'cached'),
                attempts=v.get('attempts', 1),
                query_url=v.get('query_url', ''),
            )
            for k, v in data.items()
        }
    except Exception as exc:
        print(f'Warning: could not read PubChem lookup cache {cache_file}: {exc}')
        return {}


def _save_lookup_cache(cache_file: Path | None, cache: dict[str, PubChemLookupResult]) -> None:
    """Save persistent disk lookup cache."""
    if cache_file is None:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: {
                'smiles': v.smiles,
                'status': v.status,
                'attempts': v.attempts,
                'query_url': v.query_url,
            }
            for k, v in cache.items()
        }
        cache_file.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
    except Exception as exc:
        print(f'Warning: could not save PubChem lookup cache {cache_file}: {exc}')


def build_name_to_smiles_bridge(
    tox_labels_df: pd.DataFrame,
    cache_path: str | Path,
    top_n: int = 500,
    delay: float = 0.25,
    lookup_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build or validate an auditable FAERS-to-PubChem toxicity bridge."""
    if top_n < 1:
        raise ValueError('top_n must be at least 1.')
    if delay < 0:
        raise ValueError('delay must be non-negative.')
    required_labels = {'drugname', 'toxicity_score', 'n_reports'}
    missing = required_labels.difference(tox_labels_df.columns)
    if missing:
        raise ValueError(f'Toxicity labels are missing required columns: {sorted(missing)}.')

    cache = Path(cache_path)
    if cache.exists():
        print(f'Loading and validating cached bridge from {cache}')
        return load_validated_bridge_cache(cache)

    lookup_cache_file = Path(lookup_cache_path) if lookup_cache_path else cache.parent / 'pubchem_lookup_cache.json'
    lookup_cache = _load_or_create_lookup_cache(lookup_cache_file)
    print(f'Loaded {len(lookup_cache)} cached PubChem queries from {lookup_cache_file}')

    # Filter out NaNs, blanks, and invalid drugnames
    valid_drugs = tox_labels_df.dropna(subset=['drugname']).copy()
    valid_drugs['drugname_clean'] = valid_drugs['drugname'].astype(str).str.strip()
    valid_drugs = valid_drugs[
        ~valid_drugs['drugname_clean'].str.lower().isin({'', 'nan', 'none', 'null'})
    ]

    top_drugs = valid_drugs.sort_values('n_reports', ascending=False).head(top_n)
    print(f'Querying PubChem for top {len(top_drugs)} most-reported drugs...')
    fetched_at = datetime.now(timezone.utc).isoformat()
    results = []

    for index, row in enumerate(top_drugs.itertuples(index=False), 1):
        raw_name = str(row.drugname).strip().upper()
        lookup = lookup_pubchem_smiles(raw_name, lookup_cache=lookup_cache)
        canonical = canonicalize(lookup.smiles)
        results.append({
            'drugname': raw_name,
            'raw_smiles': lookup.smiles,
            'canonical_smiles': canonical,
            'toxicity_score': float(row.toxicity_score),
            'n_reports': int(row.n_reports),
            'query_name': raw_name,
            'pubchem_query_url': lookup.query_url,
            'pubchem_lookup_status': lookup.status,
            'pubchem_lookup_attempts': lookup.attempts,
            'pubchem_fetched_at_utc': fetched_at,
        })
        if index % 50 == 0:
            matched = sum(result['canonical_smiles'] is not None for result in results)
            print(f'  Progress: {index}/{len(top_drugs)}; {matched} structures matched so far')
            _save_lookup_cache(lookup_cache_file, lookup_cache)
        if delay and lookup.status not in {'cached', 'invalid_or_nan_name'}:
            time.sleep(delay)

    _save_lookup_cache(lookup_cache_file, lookup_cache)
    bridge = validate_bridge_dataframe(pd.DataFrame(results))
    summary, conflicts = audit_toxicity_bridge(bridge)
    print(
        'BRIDGE COMPLETE: '
        f"{summary['rows_with_canonical_smiles']}/{len(bridge)} rows mapped; "
        f"{summary['unique_canonical_structures']} unique structures; "
        f"{summary['conflicting_canonical_structures']} unresolved score conflicts."
    )
    if not conflicts.empty:
        print('Conflict report must be reviewed before toxicity-model retraining.')

    cache.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(cache, index=False)
    print(f'Cached validated bridge at {cache}')
    return bridge


def rebuild_faers_toxicity_bridge_from_ascii(
    faers_ascii_dir: str | Path,
    output_bridge_path: str | Path,
    top_n: int = 500,
    min_reports: int = 5,
    delay: float = 0.25,
) -> pd.DataFrame:
    """End-to-end pipeline: Parse FAERS ASCII tables, aggregate severity, and build PubChem bridge."""
    from .faers_pipeline import build_toxicity_labels

    print(f'Parsing FAERS ASCII files from {faers_ascii_dir}...')
    tox_labels = build_toxicity_labels(str(faers_ascii_dir), min_reports=min_reports)
    print(f'Built toxicity labels for {len(tox_labels)} drugs.')
    return build_name_to_smiles_bridge(tox_labels, cache_path=output_bridge_path, top_n=top_n, delay=delay)

