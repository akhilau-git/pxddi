# AuditDDI Next-Phase Research & Engineering Roadmap

This document outlines the end-to-end plan for executing **Option 1 (UniProt / ESM-2 Protein Modeling)**, **Option 2 (Murcko Scaffold-Disjoint Study)**, **Option 3 (Model Card & Documentation Overhaul)**, and **Option 4 (Production API & Interactive UI Serving)**. Each item contains **What to do**, **Why to do it**, **How to do it**, and verifiable tick-boxes (`- [ ]` / `- [x]`).

---

## 🧬 Special Guide: How UniProt Datasets Are Collected and Used

### 1. What is UniProt?
**UniProt (Universal Protein Resource)** is the world's most comprehensive, authoritative catalog of protein sequences and functional annotation. In drug discovery, drugs exert their therapeutic or adverse effects by binding to macromolecular protein targets (receptors, enzymes like CYP450, ion channels, kinases).

### 2. Where Do We Collect UniProt Data From?
There are three standard sources used in AuditDDI:

1. **Direct UniProt REST API (Automated Online Fetching)**:
   - **Endpoint**: `https://rest.uniprot.org/uniprotkb/{accession}.fasta` (e.g., `https://rest.uniprot.org/uniprotkb/P08684.fasta` for Human CYP3A4).
   - **Batch Endpoint**: `https://rest.uniprot.org/uniprotkb/accessions?accessions=P08684,P00533,P23219&format=fasta`
   - Returns exact primary amino acid sequences (e.g., `MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFK...`).

2. **ChEMBL-to-UniProt Target Mapping (`chembl_uniprot_mapping.txt`)**:
   - Location in dataset: `pxddi-data/chembl/chembl_uniprot_mapping.txt`.
   - Links ChEMBL target identifiers (`CHEMBL203`, `CHEMBL240`) to standardized UniProt accessions (`P08684`, `P00533`).

3. **BindingDB Target Cross-References**:
   - Tabular export in `pxddi-data/BindingDB` contains explicit columns `uniprot_id` (`P23219`, `P29274`, `P04150`).

### 3. How UniProt Data Powers the "Cold-Target / Protein" Scenario:
- **The Problem**: Currently, target information is represented as a fixed 50-dimensional multi-hot presence vector. If an existing drug or new compound is tested against a protein not in the predefined 50-gene vocabulary, the model receives a zero vector, capping performance at ~0.65 AUC.
- **The Solution**:
  1. Map each target gene/protein to its **UniProt Accession**.
  2. Download or cache the primary **FASTA amino acid sequence**.
  3. Pass the sequence through **ESM-2** (Meta's Evolutionary Scale Modeling protein language model: `facebook/esm2_t6_8M_UR50D`), producing a rich 320-dimensional contextualized embedding.
  4. Perform cross-modal attention between the molecular graph of the drug and the ESM-2 sequence embedding of the protein.
  5. Result: The model generalizes to **completely novel protein targets** (Cold-Target AUC improves to the literature target of **0.73–0.87**).

---

## 📋 Comprehensive Execution Checklist

### Phase 1: Option 1 — UniProt Dataset Pipeline & ESM-2 Target Sequence Modeling
- [x] **1.1 UniProt Sequence Ingestion Pipeline (`src/data_prep/uniprot_pipeline.py`)**
  - **What**: Build a dedicated pipeline to resolve target gene symbols / BindingDB targets to UniProt accessions and cache amino acid FASTA sequences.
  - **Why**: Provides the primary biological sequence data required for sequence-level protein embeddings.
  - **How**:
    - Implemented `fetch_uniprot_sequence(accession_or_gene, cache_dir)` using the UniProt REST API with local disk caching (`.fasta` and JSON index).
    - Mapped top CYP enzymes, receptors, and kinases (`CYP3A4` -> `P08684`, `CYP2D6` -> `P10635`, `EGFR` -> `P00533`, etc.).
    - Exported `target_sequences.json` mapping target IDs to amino acid strings.
    - Verified with `tests/test_uniprot_pipeline.py` (3 passing unit tests).

- [x] **1.2 ESM-2 Target Sequence Encoder (`src/models/protein_target_encoder.py`)**
  - **What**: Implement a lightweight protein sequence encoder using `esm2_t6_8M_UR50D` (8M parameters, fast execution on CPU/GPU) with fallback learned residue 1D CNN.
  - **Why**: Transforms variable-length amino acid sequences into fixed-dimensional contextualized biological embeddings (dim=320 or projected to dim=64).
  - **How**:
    - Pretrained ESM-2 tokenizer and model integration with frozen weights.
    - Truncates sequences and pools residue embeddings; projects to `target_hidden_channels` (64).
    - Verified with `tests/test_protein_target_encoder.py` (1 passing unit test).

- [x] **1.3 Multimodal Model Integration (`src/models/ddi_model.py` & `src/data_prep/cached_graph_loader.py`)**
  - **What**: Connect ESM-2 target sequence embeddings into `CrossModalBioAttention`.
  - **Why**: Allows molecular graphs of novel compounds to attend to amino-acid binding domains.
  - **How**:
    - Added `use_protein_sequence_encoder: bool = False` to `PxDDIModel` and `model_from_checkpoint`.
    - Integrated `target_sequences` into `MolecularCache`, `CachedDDIPairDataset`, and `multimodal_collate_fn`.
    - Fully regression tested with 218 passing automated tests.

---

### Phase 2: Option 2 — Murcko Scaffold-Disjoint Study Benchmark Runner
- [x] **2.1 Scaffold-Disjoint Study Script (`src/training/benchmark_scaffold_study.py`)**
  - **What**: Implement a dedicated training launcher that executes Murcko scaffold-disjoint splits.
  - **Why**: Random splits share identical core chemical scaffolds across train and test. Scaffold-disjoint splits prove whether the model generalizes to completely distinct chemical families.
  - **How**:
    - Generated Bemis-Murcko scaffold clusters (`scaffold_train.csv`, `scaffold_val.csv`, `scaffold_test.csv`).
    - Trains and compares: (a) Baseline 2D GNN, and (b) Full Multimodal AuditDDI.
    - Measures AUROC, AUPRC, Recall, Specificity, Brier score, and ECE across scaffold holdouts.

- [x] **2.2 Statistical Hypothesis Testing & Export**
  - **What**: Compute paired bootstrap confidence intervals and Wilcoxon signed-rank tests across scaffold splits.
  - **Why**: Required by `MODEL_CARD.md` Phase 7 for publishing rigorous comparative claims.
  - **How**:
    - Exports `scaffold_benchmark_results.json`, `scaffold_benchmark_summary.csv`, and `scaffold_performance_comparison.md`.
    - Verified with `tests/test_benchmark_scaffold_study.py` (2 passing unit tests).

---

### Phase 3: Option 3 — Model Card & Research Documentation Overhaul
- [x] **3.1 Update `MODEL_CARD.md`**
  - **What**: Overhaul the model card to reflect the clean, audited, and calibrated state of the repository.
  - **Why**: The existing model card warned of historical uncalibrated test metrics and test leakage that have now been completely remediated.
  - **How**:
    - Documented all 12 completed audit remediations.
    - Recorded honest validation AUROC (0.9516) and out-of-sample calibrated test performance.
    - Detailed the 8 ingested multimodal data sources (PDB 421 complexes, PharmGKB 552, BindingDB 597, GEO 639, FAERS 639, UniProt sequences).

- [x] **3.2 Update `README.md` and Research Roadmap**
  - **What**: Update `README.md` with the verified cold-start benchmark table and Colab reproduction instructions.
  - **Why**: Gives collaborators and reviewers transparent, reproducible evidence.

---

### Phase 4: Option 4 — Production FastAPI Serving & Interactive UI
- [x] **4.1 Hot-Load Multimodal Checkpoint in `backend/main.py`**
  - **What**: Enable the FastAPI server to serve predictions from `auditddi_multimodal_v1` checkpoints.
  - **Why**: Transitions the repository from research-only scripts to a live interactive web deployment.
  - **How**:
    - Supported `MODEL_ARCHITECTURE_MULTIMODAL` in `backend/main.py`.
    - Returns interaction probability, calibrated risk category, conformal prediction sets, and OOD status.

- [x] **4.2 Interactive Frontend Polish (`frontend/app.js`, `frontend/index.html`)**
  - **What**: Update the web interface to display multi-modal biological evidence and conformal prediction sets.
  - **Why**: Allows clinicians and researchers to visualize uncertainty and evidence for why two drugs are predicted to interact.
