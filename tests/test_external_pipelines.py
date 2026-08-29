import gzip

import pandas as pd
import pytest

from src.data_prep.chembl_pipeline import (
    audit_chembl_structure_overlap,
    load_chembl_chemreps,
    load_chembl_targets,
)
from src.data_prep.pharmgkb_pipeline import (
    load_pharmgkb_chemical_gene_evidence,
    load_twosides_drug_catalog,
    resolve_pharmgkb_evidence_to_twosides,
)


def test_chembl_chemreps_audit_is_structure_only(tmp_path):
    path = tmp_path / 'chembl_chemreps.txt.gz'
    content = (
        'chembl_id\tcanonical_smiles\tstandard_inchi\tstandard_inchi_key\n'
        'CHEMBL1\tCCO\tInChI=1S/C2H6O\tKEY1\n'
        'CHEMBL2\tCCN\tInChI=1S/C2H7N\tKEY2\n'
    )
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        handle.write(content)

    frame = load_chembl_chemreps(path)
    summary, overlap = audit_chembl_structure_overlap(path, ['OCC', 'CCCl'])

    assert frame.columns.tolist() == [
        'chembl_id', 'canonical_smiles', 'standard_inchi', 'standard_inchi_key'
    ]
    assert summary['matched_unique_pxddi_structures'] == 1
    assert summary['not_evidence_of'] == ['molecule_target_activity', 'DDI_label', 'external_DDI_validation']
    assert overlap.to_dict(orient='records') == [{
        'chembl_id': 'CHEMBL1', 'canonical_smiles': 'CCO', 'standard_inchi_key': 'KEY1'
    }]
    with pytest.raises(RuntimeError, match='do not contain molecule-to-target'):
        load_chembl_targets(path)


def test_pharmgkb_exact_mapping_excludes_ambiguous_and_nonchemical_rows(tmp_path):
    relationships = tmp_path / 'relationships.tsv'
    relationships.write_text(
        'Entity1_id\tEntity1_name\tEntity1_type\tEntity2_id\tEntity2_name\tEntity2_type\tEvidence\tAssociation\tPK\tPD\tPMIDs\n'
        'PA1\tDrug A\tChemical\tPA2\tCYP2D6\tGene\t1A\tassociated\tYes\tNo\t1\n'
        'PA3\tDrug B\tChemical\tPA4\tDrug C\tChemical\t1A\tassociated\tNo\tNo\t2\n'
    )
    pathway = tmp_path / 'pathways' / 'example.tsv'
    pathway.parent.mkdir()
    pathway.write_text('Drugs\tGenes\tPMIDs\nDrug A; Missing drug\tCYP3A4, ABCB1\t3\n')
    catalog_path = tmp_path / 'twosides_drugs.csv'
    pd.DataFrame({
        'drug_name': ['drug a', 'Drug A', 'drug b'],
        'smiles': ['CCO', 'CCO', 'CCC'],
    }).to_csv(catalog_path, index=False)

    evidence, evidence_summary = load_pharmgkb_chemical_gene_evidence(relationships, pathway.parent)
    catalog, catalog_summary = load_twosides_drug_catalog(catalog_path)
    resolved, mapping_summary = resolve_pharmgkb_evidence_to_twosides(evidence, catalog)

    assert evidence_summary['relationship_chemical_gene_rows'] == 1
    assert evidence_summary['pathway_chemical_gene_rows'] == 4
    assert catalog_summary['mapping_ready'] is True
    assert mapping_summary['matched_exact_name_rows'] == 3
    assert mapping_summary['unmatched_exact_name_rows'] == 2
    assert set(resolved['mapping_status']) == {'matched_exact_name', 'unmatched_exact_name'}


def test_catalog_without_recognised_columns_is_reported_not_guessed(tmp_path):
    path = tmp_path / 'unknown_catalog.csv'
    pd.DataFrame({'compound': ['Drug A'], 'chemical': ['CCO']}).to_csv(path, index=False)
    catalog, summary = load_twosides_drug_catalog(path)
    assert catalog.empty
    assert summary['mapping_ready'] is False
    assert 'no fuzzy mapping' in summary['reason'].lower()
