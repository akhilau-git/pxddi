"""Create auditable ChEMBL/PharmGKB coverage reports in Colab.

Run this before enabling biological features. It only reads source datasets and
writes audit artifacts; it does not modify TWOSIDES labels or checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = PROJECT_ROOT / 'src'
for candidate in (PROJECT_ROOT, REPOSITORY_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from data_prep.chembl_pipeline import (
    audit_chembl_structure_overlap,
    load_chembl_uniprot_target_metadata,
)
from data_prep.pharmgkb_pipeline import (
    load_pharmgkb_chemical_gene_evidence,
    load_twosides_drug_catalog,
    resolve_pharmgkb_evidence_to_twosides,
)


DATA_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
RESULTS_BASE = Path(os.environ.get('PXDDI_RESULTS_BASE', '/content/drive/MyDrive/pxddi-results'))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')


def _load_pxddi_smiles(edges_path: Path) -> list[object]:
    if not edges_path.is_file():
        raise FileNotFoundError(f'TWOSIDES pair table was not found: {edges_path}')
    edges = pd.read_csv(edges_path, usecols=['source', 'target'], dtype=str, keep_default_na=False)
    return edges[['source', 'target']].to_numpy().ravel().tolist()


def main() -> None:
    chembl_dir = DATA_BASE / 'chembl'
    pharmgkb_dir = DATA_BASE / 'pharmgkb'
    twosides_dir = DATA_BASE / 'twosides'
    chemreps_path = chembl_dir / 'chembl_37_chemreps.txt.gz'
    target_metadata_path = chembl_dir / 'chembl_uniprot_mapping.txt'
    relationships_path = pharmgkb_dir / 'relationships.tsv'
    edges_path = twosides_dir / 'drug_drug_edges.csv'
    catalog_path = twosides_dir / 'twosides_drugs.csv'

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = RESULTS_BASE / 'external_knowledge_audits' / f'audit_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, object] = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_base': str(DATA_BASE),
        'purpose': (
            'Audit external structural and biological evidence before any new '
            'candidate feature is trained. No DDI labels were changed.'
        ),
    }

    pxddi_smiles = _load_pxddi_smiles(edges_path)
    chembl_summary, overlap = audit_chembl_structure_overlap(chemreps_path, pxddi_smiles)
    overlap_path = output_dir / 'chembl_structure_overlap.csv'
    overlap.to_csv(overlap_path, index=False)
    chembl_summary['overlap_csv'] = str(overlap_path)
    summary['chembl_structure_corpus'] = chembl_summary

    _, target_metadata_summary = load_chembl_uniprot_target_metadata(target_metadata_path)
    summary['chembl_target_metadata'] = target_metadata_summary

    evidence, pharmgkb_summary = load_pharmgkb_chemical_gene_evidence(
        relationships_path, pharmgkb_dir
    )
    evidence_path = output_dir / 'pharmgkb_chemical_gene_evidence.csv'
    evidence.to_csv(evidence_path, index=False)
    pharmgkb_summary['evidence_csv'] = str(evidence_path)
    summary['pharmgkb_chemical_gene_evidence'] = pharmgkb_summary

    catalog, catalog_summary = load_twosides_drug_catalog(catalog_path)
    summary['twosides_catalog'] = catalog_summary
    if catalog_summary['mapping_ready']:
        resolved, mapping_summary = resolve_pharmgkb_evidence_to_twosides(evidence, catalog)
        resolution_path = output_dir / 'pharmgkb_to_twosides_name_resolution.csv'
        resolved.to_csv(resolution_path, index=False)
        mapping_summary['resolution_csv'] = str(resolution_path)
        summary['pharmgkb_to_twosides_mapping'] = mapping_summary
    else:
        summary['pharmgkb_to_twosides_mapping'] = {
            'status': 'not_attempted',
            'reason': catalog_summary['reason'],
        }

    summary['next_gate'] = (
        'Enable a biological-feature candidate only if the exact-name resolution '
        'coverage is sufficient and its mapping audit is manually accepted. The '
        'current ChEMBL files still require a molecule-target activity export for '
        'true target features.'
    )
    summary_path = output_dir / 'external_knowledge_audit_summary.json'
    _write_json(summary_path, summary)
    print(f'External-knowledge audit saved to: {output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
