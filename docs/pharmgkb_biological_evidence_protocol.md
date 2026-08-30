# PharmGKB biological-evidence protocol

PharmGKB is used only as external chemical--gene and pathway evidence. It is
not a drug--drug interaction label source, an external DDI test set, or a
clinical-outcome dataset. This protocol must never append PharmGKB records to
TWOSIDES positive pairs.

## What the pipeline does

The audit takes the following conservative route:

1. extracts only PharmGKB Chemical↔Gene and pathway Drugs×Genes evidence;
2. resolves each evidence chemical name to a direct PharmGKB chemical-catalogue
   SMILES by exact normalised name only;
3. resolves that canonical SMILES to a structure observed in a TWOSIDES pair;
4. writes per-structure gene-profile rows only for exact matches.

It does not use fuzzy names, synonyms, brand names, PubChem lookups, or an
assumed drug catalogue. Ambiguous and unmatched records remain unmapped.

The earlier `twosides_drugs.csv` name route is expected to be unavailable in
this data release because it contains only `drug_id`. Matching against the
canonical SMILES in `drug_drug_edges.csv` is the correct alternative, but it
requires a direct PharmGKB chemical catalogue containing both a name and a
SMILES column.

## Run the audit in Colab

Run this after pulling the source code and after an active ChEMBL pretraining
job has finished. It does not change a checkpoint, but it scans large Drive
datasets and would unnecessarily compete with pretraining for CPU and Drive
throughput.

```python
%cd /content/drive/MyDrive/pxddi-data/pxddi
!git pull origin main

%env PXDDI_DATA_BASE=/content/drive/.shortcut-targets-by-id/1EK5SEg3iwEAEUBzwrCOsj_Y0huxGZklA/pxddi-data
%env PXDDI_RESULTS_BASE=/content/drive/MyDrive/pxddi-results

!python src/training/audit_external_knowledge.py
```

The script looks first for either of these files below
`$PXDDI_DATA_BASE/pharmgkb/`:

- `chemicals.tsv`
- `chemicals.csv`

If your direct PharmGKB chemical catalogue has another name, set its exact
path before running the audit:

```python
%env PXDDI_PHARMGKB_CHEMICAL_CATALOG=/content/drive/.shortcut-targets-by-id/1EK5SEg3iwEAEUBzwrCOsj_Y0huxGZklA/pxddi-data/pharmgkb/PASTE_THE_ACTUAL_FILE_NAME.tsv
```

## Required audit outputs

The timestamped output folder contains:

- `pharmgkb_chemical_gene_evidence.csv`: source evidence, never labels;
- `pharmgkb_to_twosides_structure_resolution.csv`: every match, ambiguity, and
  non-match with its status;
- `pharmgkb_twosides_gene_profiles.csv`: only exact structure matches, with a
  JSON list of gene symbols;
- `external_knowledge_audit_summary.json`: source paths, row counts, coverage,
  and mapping policy.

Do not enable a biological-feature model merely because these files exist.
First review the number and proportion of exact TWOSIDES structure matches,
the ambiguity count, and a sample of the resolved chemical identities. The
subsequent candidate must be trained and tested with the same S1/S2 split
protocol as all other models; it cannot claim an external validation result.
