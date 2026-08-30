import gzip

import pandas as pd
import pytest

from src.data_prep.chembl_pipeline import (
    audit_chembl_structure_overlap,
    load_chembl_chemreps,
    load_chembl_targets,
)
from src.data_prep.pharmgkb_pipeline import (
    aggregate_twosides_pharmgkb_gene_profiles,
    build_twosides_structure_catalog,
    load_pharmgkb_chemical_catalog,
    load_pharmgkb_chemical_gene_evidence,
    load_twosides_drug_catalog,
    resolve_pharmgkb_catalog_to_twosides_structures,
    resolve_pharmgkb_evidence_to_chemical_catalog,
    resolve_pharmgkb_evidence_to_twosides,
)
from src.training.audit_external_knowledge import (
    resolve_data_base,
    resolve_pharmgkb_chemical_catalog_path,
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


def test_pharmgkb_direct_structure_mapping_builds_auditable_gene_profiles(tmp_path):
    relationships = tmp_path / 'relationships.tsv'
    relationships.write_text(
        'Entity1_id\tEntity1_name\tEntity1_type\tEntity2_id\tEntity2_name\tEntity2_type\tEvidence\tAssociation\tPK\tPD\tPMIDs\n'
        'PA1\tDrug A\tChemical\tPA2\tCYP2D6\tGene\t1A\tassociated\tYes\tNo\t1\n'
        'PA3\tDrug B\tChemical\tPA4\tCYP3A4\tGene\t1A\tassociated\tNo\tNo\t2\n'
    )
    evidence, _ = load_pharmgkb_chemical_gene_evidence(relationships, tmp_path)
    chemical_catalog_path = tmp_path / 'chemicals.tsv'
    chemical_catalog_path.write_text(
        'PharmGKB Accession Id\tName\tSMILES\n'
        'PA1\tDrug A\tOCC\n'
        'PA3\tDrug B\tCCC\n'
        'PA5\tDrug B\tCCN\n'
    )
    chemical_catalog, catalog_summary = load_pharmgkb_chemical_catalog(chemical_catalog_path)
    assert catalog_summary['mapping_ready'] is True

    named_evidence, name_summary = resolve_pharmgkb_evidence_to_chemical_catalog(
        evidence, chemical_catalog
    )
    assert name_summary['matched_exact_name_rows'] == 1
    assert name_summary['ambiguous_exact_name_rows'] == 1

    edges_path = tmp_path / 'drug_drug_edges.csv'
    pd.DataFrame({'source': ['CCO'], 'target': ['CCCC']}).to_csv(edges_path, index=False)
    twosides_structures, structure_catalog_summary = build_twosides_structure_catalog(edges_path)
    assert structure_catalog_summary['canonical_structure_rows'] == 2
    fully_resolved, structure_summary = resolve_pharmgkb_catalog_to_twosides_structures(
        named_evidence, twosides_structures
    )
    assert structure_summary['matched_exact_canonical_smiles_rows'] == 1
    assert structure_summary['matched_unique_twosides_structures'] == 1

    profiles, profile_summary = aggregate_twosides_pharmgkb_gene_profiles(fully_resolved)
    assert profile_summary['matched_unique_twosides_structures'] == 1
    assert profile_summary['unique_gene_symbols'] == 1
    assert profiles.iloc[0]['canonical_smiles'] == 'CCO'
    assert profiles.iloc[0]['gene_symbols_json'] == '["CYP2D6"]'


def test_pharmgkb_catalog_path_is_optional_but_never_guessed_from_random_files(tmp_path, monkeypatch):
    assert resolve_pharmgkb_chemical_catalog_path(tmp_path) is None
    chemicals = tmp_path / 'chemicals.tsv'
    chemicals.write_text('Name\tSMILES\nDrug A\tCCO\n')
    assert resolve_pharmgkb_chemical_catalog_path(tmp_path) == chemicals
    monkeypatch.setenv('PXDDI_PHARMGKB_CHEMICAL_CATALOG', str(tmp_path / 'missing.tsv'))
    with pytest.raises(FileNotFoundError, match='PXDDI_PHARMGKB_CHEMICAL_CATALOG'):
        resolve_pharmgkb_chemical_catalog_path(tmp_path)


def test_external_audit_data_root_must_be_an_existing_directory(tmp_path):
    assert resolve_data_base(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError, match='PXDDI_DATA_BASE'):
        resolve_data_base(tmp_path / 'missing')
