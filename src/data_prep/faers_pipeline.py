"""
faers_pipeline.py
Builds REAL (non-synthetic) toxicity labels (Module C) and patient
context labels (Module D) from your actual FAERS files.

Joins: DRUG (drugname) + OUTC (severity) + DEMO (age/sex) via primaryid.
"""

import pandas as pd

SEVERE_OUTCOMES = {'DE', 'HO', 'LT', 'DS'}  # Death, Hospitalization, Life-threatening, Disability


def aggregate_toxicity_labels(
    drug: pd.DataFrame,
    outc: pd.DataFrame,
    min_reports: int = 5,
    missing_outcome_policy: str = 'exclude',
) -> pd.DataFrame:
    """Aggregate one severe-outcome flag per report before grouping by drug.

    FAERS can contain multiple DRUG and OUTC rows for one ``primaryid``. A
    direct table join creates a cartesian multiplication and biases toxicity
    labels. This function first reduces each report to its maximum observed
    severity, then counts each drug-report combination once.

    ``missing_outcome_policy='exclude'`` preserves the conservative historical
    interpretation: reports with no OUTC record are not assumed to be
    non-severe. Use ``'non_severe'`` only when that study decision is explicit.
    """
    if min_reports < 1:
        raise ValueError('min_reports must be at least 1.')
    if missing_outcome_policy not in {'exclude', 'non_severe'}:
        raise ValueError("missing_outcome_policy must be 'exclude' or 'non_severe'.")
    for name, frame, columns in (
        ('DRUG', drug, {'primaryid', 'drugname'}),
        ('OUTC', outc, {'primaryid', 'outc_cod'}),
    ):
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f'{name} data is missing required columns: {sorted(missing)}.')

    drug_reports = drug[['primaryid', 'drugname']].copy()
    drug_reports = drug_reports.dropna(subset=['primaryid', 'drugname'])
    drug_reports['drugname'] = drug_reports['drugname'].astype(str).str.strip().str.upper()
    drug_reports = drug_reports[drug_reports['drugname'] != '']
    drug_reports = drug_reports.drop_duplicates(subset=['primaryid', 'drugname'])

    outcomes = outc[['primaryid', 'outc_cod']].copy()
    outcomes = outcomes.dropna(subset=['primaryid'])
    outcomes['is_severe'] = outcomes['outc_cod'].isin(SEVERE_OUTCOMES).astype(int)
    report_severity = outcomes.groupby('primaryid', as_index=False)['is_severe'].max()

    merged = drug_reports.merge(
        report_severity,
        on='primaryid',
        how='left' if missing_outcome_policy == 'non_severe' else 'inner',
    )
    if missing_outcome_policy == 'non_severe':
        merged['is_severe'] = merged['is_severe'].fillna(0).astype(int)

    toxicity = merged.groupby('drugname', as_index=False).agg(
        toxicity_score=('is_severe', 'mean'),
        n_reports=('primaryid', 'nunique'),
    )
    toxicity = toxicity[toxicity['n_reports'] >= min_reports]
    return toxicity.sort_values(['n_reports', 'drugname'], ascending=[False, True]).reset_index(drop=True)


def build_toxicity_labels(
    faers_base_path: str,
    min_reports: int = 5,
    missing_outcome_policy: str = 'exclude',
):
    """
    Returns a DataFrame: drugname -> toxicity_score (0-1)
    toxicity_score = fraction of reports for that drug with a severe outcome.
    """
    from pathlib import Path
    base = Path(faers_base_path)
    drug_files = sorted(list(base.glob('*[Dd][Rr][Uu][Gg]*.txt')) + list(base.glob('*[Dd][Rr][Uu][Gg]*.TXT')))
    outc_files = sorted(list(base.glob('*[Oo][Uu][Tt][Cc]*.txt')) + list(base.glob('*[Oo][Uu][Tt][Cc]*.TXT')))
    drug_path = drug_files[0] if drug_files else base / 'DRUG23Q4.txt'
    outc_path = outc_files[0] if outc_files else base / 'OUTC23Q4.txt'

    print(f"Loading FAERS DRUG file ({drug_path.name})...")
    drug = pd.read_csv(drug_path, sep='$',
                        usecols=['primaryid', 'drugname'], low_memory=False)
    print(f"Loading FAERS OUTC file ({outc_path.name})...")
    outc = pd.read_csv(outc_path, sep='$',
                        usecols=['primaryid', 'outc_cod'], low_memory=False)
    print("Aggregating one outcome flag per report before grouping by drug...")
    tox_scores = aggregate_toxicity_labels(
        drug,
        outc,
        min_reports=min_reports,
        missing_outcome_policy=missing_outcome_policy,
    )
    print(f"Built toxicity labels for {len(tox_scores)} drugs (min {min_reports} reports each)")
    return tox_scores


def build_patient_context(faers_base_path: str, sample_size: int = 50000):
    """
    Returns real (primaryid, age, sex) rows from FAERS DEMO —
    genuine patient demographics, not synthetic.
    """
    print("Loading FAERS DEMO file...")
    demo = pd.read_csv(f'{faers_base_path}/DEMO23Q4.txt', sep='$',
                        usecols=['primaryid', 'age', 'age_cod', 'sex'],
                        low_memory=False, nrows=sample_size)

    def normalize_age(row):
        if pd.isna(row['age']):
            return None
        unit = row['age_cod']
        if unit == 'YR' or pd.isna(unit):
            return row['age']
        elif unit == 'MON':
            return row['age'] / 12
        elif unit == 'DEC':
            return row['age'] * 10
        return None

    demo['age_years'] = demo.apply(normalize_age, axis=1)
    demo['sex_code'] = demo['sex'].map({'M': 0, 'F': 1})
    demo = demo.dropna(subset=['age_years', 'sex_code'])

    print(f"Built patient context for {len(demo)} real FAERS reports")
    return demo[['primaryid', 'age_years', 'sex_code']]


def match_drugname_to_smiles(drugname: str, chembl_synonym_map: dict):
    """Bridges FAERS text drug names to structural SMILES (exact match)."""
    return chembl_synonym_map.get(drugname.strip().upper())
