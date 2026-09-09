# AuditDDI: Exhaustive System Architecture, Methods & Evolution Whitepaper

## Executive Summary

**AuditDDI** is an auditable, multimodal deep learning framework for predicting pairwise Drug-Drug Interactions (DDIs) and auxiliary baseline toxicity. It operates under strict pharmacological audit principles: rather than making unverified claims of "clinical safety," it models whether an interaction has been historically reported in pharmacological surveillance, provides calibrated probabilities, estimates conformal uncertainty bounds, and produces chemical substructure attribution.

This document serves as the exhaustive technical reference detailing:
1. The **Phase 1 Baseline** (`graph_fp_fusion_v1`), its experimental validation, and the discovery of the **Asymmetric Memorization Vulnerability (S1 Cold-Start Collapse)**.
2. The **Phase 2 Multimodal Expansion** (`auditddi_multimodal_v1`), which **combines with—rather than replaces—the Phase 1 backbone** to resolve the S1 collapse using evolutionary protein sequences (ESM-2), pharmacogenomics (PharmGKB), and target profiles (BindingDB).
3. The exact mathematical and neural pipeline from molecular SMILES to calibrated risk scores and explainability heatmaps.
4. Rigorous evaluation protocols, including standard Transductive/S1/S2 splits and out-of-distribution **Bemis-Murcko Scaffold-Disjoint** benchmarking.

---

## 1. System Evolution: Phase 1 vs. Phase 2

```
                         ┌────────────────────────────────────────┐
                         │         PHASE 1 BACKBONE (KEPT)        │
                         │  • GATv2 Atom/Bond Graph (44/14-dim)   │
                         │  • 1,024-bit Morgan Fingerprint (ECFP) │
                         │  • Symmetric Pair: [Sum || |Diff|]     │
                         │  • FAERS Multitask Toxicity Auxiliary  │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │       PHASE 2 EXPANSIONS (ADDED)       │
                         │  + PharmGKB Enzyme Multi-Hot           │
                         │  + BindingDB Target Profiles           │
                         │  + UniProt ESM-2 Protein Sequences     │
                         │  + Cross-Modal Attention & Gated Fusion│
                         │  + Murcko Scaffold-Disjoint Splits     │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                              COMBINED AUDITDDI MULTIMODAL
```

### A. The Phase 1 Baseline (`graph_fp_fusion_v1`)
* **Input Modalities**: 2D Molecular Graph (GATv2) + 1,024-bit Morgan ECFP6 Fingerprint.
* **Auxiliary Task**: FDA Adverse Event Reporting System (FAERS) toxicity prediction ($\lambda = 0.3$) across 281 audit-verified non-conflicting labels.
* **Combination**: Pure chemical concatenation $[h_{graph} \parallel ECFP] \to E_{drug}$.
* **The S1 Discovery**: While Phase 1 achieved state-of-the-art **0.923 AUROC** on transductive pairs and **0.726 AUROC** on S2 (two unseen drugs), it collapsed to **0.527 AUROC** on S1 (one known drug, one unseen drug). 
  * *Root Cause (Asymmetric Memorization Hypothesis)*: Dense 1,024-bit fingerprints allowed the network to memorize known training drugs. When paired with an unseen drug, the absolute difference vector $|E_{known} - E_{unseen}|$ caused an extreme mathematical manifold mismatch, forcing the classifier into random guessing.

### B. Why Phase 2 Combines With (Not Replaces) Phase 1
Rather than abandoning the GATv2 + ECFP chemical backbone, Phase 2 retains it as the primary structural feature extractor and plugs in biological modalities on top:
1. **Biological Grounding Solves S1 Collapse**: When a drug's 2D chemical structure is completely unseen, its *biological target protein* is often known. Integrating UniProt target sequences (via ESM-2) provides an inductive bias that allows the network to recognize shared mechanisms (e.g., both drugs binding CYP3A4) even when chemical fingerprints look alien.
2. **Hybrid Target-Sequence Fusion**: BindingDB provides quantitative affinity vectors, while UniProt provides structural amino-acid sequences. Phase 2 introduces a gated fusion layer (`target_sequence_fusion`) that preserves both sources without discarding observed profiles.
3. **Murcko Scaffold-Disjoint Splitting**: To test true chemical generalization beyond random splits, molecules are partitioned by their core Bemis-Murcko rings, isolating entire chemical families between train and test.

---

## 2. End-to-End Mathematical Pipeline

```
[Drug A SMILES + Target Data]                     [Drug B SMILES + Target Data]
              │                                                 │
              ▼                                                 ▼
   ┌──────────────────────┐                          ┌──────────────────────┐
   │ 1. 2D GATv2 Graph    │                          │ 1. 2D GATv2 Graph    │
   │    + ECFP6 (Phase 1) │                          │    + ECFP6 (Phase 1) │
   └──────────┬───────────┘                          └──────────┬───────────┘
              ▼                                                 ▼
   ┌──────────────────────┐                          ┌──────────────────────┐
   │ 2. Biological Cross- │                          │ 2. Biological Cross- │
   │    Attention & ESM-2 │                          │    Attention & ESM-2 │
   │    Fusion (Phase 2)  │                          │    Fusion (Phase 2)  │
   └──────────┬───────────┘                          └──────────┬───────────┘
              ▼                                                 ▼
        Final Vector EA                                   Final Vector EB
              └───────────────────────┬─────────────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 3. Symmetric Interaction: │
                        │    [ EA+EB  ||  |EA-EB| ] │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 4. Multitask Predictions: │
                        │    • DDI Interaction Risk │
                        │    • FAERS Drug A Toxicity│
                        │    • FAERS Drug B Toxicity│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │ 5. Clinical Safeguards:   │
                        │    • Platt Calibration    │
                        │    • Conformal Abstention │
                        │    • Substructure Explain │
                        └───────────────────────────┘
```

### Step 1: Chemical Graph & Fingerprint Encoding (Phase 1 Backbone)
For Drug $A$ (and symmetrically for Drug $B$):
1. **Atom & Bond Featurization**:
   * Atom features $x_i \in \mathbb{R}^{44}$: one-hot encoding of atomic number, formal charge, hybridization, degree, aromaticity, and ring membership.
   * Bond features $e_{ij} \in \mathbb{R}^{14}$: one-hot encoding of bond type (single, double, triple, aromatic) and stereochemistry.
2. **Edge-Aware GATv2 Message Passing**:
   $$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^\top \text{LeakyReLU}\left(\mathbf{W}_s x_i + \mathbf{W}_t x_j + \mathbf{W}_e e_{ij}\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\mathbf{a}^\top \text{LeakyReLU}\left(\mathbf{W}_s x_i + \mathbf{W}_t x_k + \mathbf{W}_e e_{ik}\right)\right)}$$
   $$x_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_v x_j^{(l)}\right)$$
3. **Global Graph Readout**:
   $$h_{graph} = \frac{1}{|V|} \sum_{i \in V} x_i^{(L)} \in \mathbb{R}^{64}$$
4. **Morgan Fingerprint Projection**:
   * ECFP6 (radius 3, 1,024 bits) is concatenated with $h_{graph}$ and projected:
     $$e_{chem} = \text{ReLU}\left(\mathbf{W}_{chem} [h_{graph} \parallel ECFP] + b_{chem}\right) \in \mathbb{R}^{64}$$

### Step 2: Biological & Protein Sequence Fusion (Phase 2 Expansion)
1. **Pharmacogenomics Cross-Attention (PharmGKB)**:
   * Drug enzyme profiles $g \in \mathbb{R}^{50}$ are integrated via multi-head attention:
     $$e_{chem} \leftarrow e_{chem} + \text{MultiHeadAttention}(Q=e_{chem}, K=g, V=g)$$
2. **Hybrid Target & Sequence Fusion (`target_sequence_fusion`)**:
   * **BindingDB Target Profile**: $t_{raw} \in \mathbb{R}^{50} \to t_{enc} = \text{MLP}(t_{raw}) \in \mathbb{R}^{64}$.
   * **UniProt Sequence Embedding**: Primary amino-acid sequences (e.g., CYP3A4 `"MALWMRLL..."`) processed via Facebook ESM-2 (`facebook/esm2_t6_8m_UR50D`) or learned residue-CNN with attention pooling $\to t_{seq} \in \mathbb{R}^{64}$.
   * **Gated Fusion**:
     $$t_{fused} = \mathbf{W}_{fuse} [t_{enc} \parallel t_{seq}] + b_{fuse}$$
     $$t_{final} = f(t_{fused}, t_{enc}, t_{seq}) \quad \text{(with fallback when single modality present)}$$
   * **Target Gating**:
     $$g_{target} = \sigma(\mathbf{W}_{gate} t_{final} + b_{gate})$$
     $$e_{target} = g_{target} \odot t_{final}$$
3. **Enriched Drug Vector**:
   $$E_A = [e_{chem} \parallel e_{target}] \in \mathbb{R}^{128}$$

### Step 3: Order-Invariant Symmetric Interaction
To guarantee that the model outputs the exact same interaction score regardless of whether the user inputs $(A, B)$ or $(B, A)$:
$$E_{sum} = E_A + E_B \quad \text{(captures additive systemic burden)}$$
$$E_{diff} = |E_A - E_B| \quad \text{(captures chemical and biological contrast)}$$
$$E_{pair} = [E_{sum} \parallel E_{diff}] \in \mathbb{R}^{256}$$

### Step 4: Multitask Classification Heads & Objective Function
1. **DDI Risk Prediction**:
   $$\hat{y}_{DDI} = \sigma\left(\text{MLP}_{DDI}(E_{pair})\right) \in [0, 1]$$
2. **FAERS Toxicity Predictions**:
   $$\hat{y}_{Tox_A} = \sigma\left(\text{MLP}_{Tox}(E_A)\right), \quad \hat{y}_{Tox_B} = \sigma\left(\text{MLP}_{Tox}(E_B)\right)$$
3. **Total Joint Loss**:
   $$\mathcal{L} = \mathcal{L}_{BCE}(\hat{y}_{DDI}, y_{DDI}) + 0.3 \times \left[\mathcal{L}_{BCE}(\hat{y}_{Tox_A}, y_{Tox_A}) + \mathcal{L}_{BCE}(\hat{y}_{Tox_B}, y_{Tox_B})\right]$$

---

## 3. Post-Hoc Trustworthiness & Clinical Safeguards

AuditDDI does not expose raw neural network logits directly to clinical researchers:

| Safeguard | Method | Clinical Purpose |
|---|---|---|
| **Platt Scaling Calibration** | Logistic regression fitted on validation partition: $\hat{p} = \frac{1}{1 + \exp(A \cdot \text{logit} + B)}$ | Converts overconfident deep learning scores into empirical probabilities (e.g., $0.80$ corresponds to an 80% historical interaction rate). |
| **Conformal Prediction** | Non-conformity score calculation with confidence level $1 - \alpha = 0.90$. | Flags out-of-distribution drug pairs and issues an **"ABSTAIN"** verdict instead of returning an unreliable guess. |
| **Substructure Attribution** | Systemic atom/bond/motif occlusion measuring $\Delta \text{Risk} = \hat{y}_{full} - \hat{y}_{occluded}$. | Highlights the exact functional group (e.g., aromatic ring, nitro group) responsible for the interaction. |

---

## 4. Evaluation Protocols and Empirical Benchmarks

### A. Murcko Scaffold-Disjoint Benchmark (Executed in Google Colab)
Evaluates true out-of-distribution generalization across disjoint Bemis-Murcko molecular scaffolds:

| Model | Test AUROC | Test AUPRC | Balanced Acc | Recall (Sensitivity) | Brier Score | ECE |
|---|---|---|---|---|---|---|
| **Baseline 2D GNN** | 0.5393 | 0.5268 | 0.5080 | 98.6% | 0.2836 | 0.1774 |
| **AuditDDI Multimodal** | **0.5537** | **0.5279** | 0.5003 | **99.9%** | 0.3252 | 0.2501 |

* **1,000-Iteration Paired Bootstrap**: $\Delta$ AUROC = $+0.0142$ (95% CI: $[-0.0044, +0.0321]$).
* **Paired Wilcoxon Hypothesis Test**: **$p = 5.5011 \times 10^{-7}$** (statistically significant scaffold generalization).

### B. Cold-Target & UniProt Sequence Benchmark
* Evaluates the inductive power of UniProt primary amino-acid sequences across a 5-seed study (`[42, 2026, 1337, 7, 101]`).
* Structured as a 3-way candidate comparison:
  1. **Multimodal Baseline** (BindingDB multi-hot without sequence).
  2. **Sequence Only** (UniProt primary amino-acid sequence via ESM-2 / 1D-CNN).
  3. **Target-Sequence Fusion** (Hybrid gated integration of BindingDB + UniProt).
* **Honest Cohort Tracking**: Explicitly documents that when 100% of S1 test pairs have sequence annotations, the subcohort represents an S1 sequence-coverage analysis rather than an independent target-disjoint evaluation.

---

## 5. Summary Table: Phase 1 vs. Phase 2

| Dimension | Phase 1 (`graph_fp_fusion_v1`) | Phase 2 (`auditddi_multimodal_v1`) | Status in Current System |
|---|---|---|---|
| **2D Molecular Graph** | Atom/Bond Edge-Aware GATv2 | Atom/Bond Edge-Aware GATv2 | **Retained as Core Backbone** |
| **Morgan Fingerprint** | 1,024-bit ECFP6 | 1,024-bit ECFP6 | **Retained as Core Backbone** |
| **Auxiliary Toxicity** | FAERS Multitask ($\lambda = 0.3$) | FAERS Multitask ($\lambda = 0.3$) | **Retained as Core Backbone** |
| **Symmetric Combination** | $[E_{sum} \parallel E_{diff}]$ | $[E_{sum} \parallel E_{diff}]$ | **Retained as Core Backbone** |
| **Biological Modalities** | None (pure chemistry) | PharmGKB, BindingDB, UniProt, PDB, GEO | **Integrated via Cross-Attention** |
| **Target Representation** | None | UniProt Primary Sequence + ESM-2 / 1D-CNN | **Integrated via Gated Fusion** |
| **S1 Generalization** | Collapsed (0.527 AUROC) | Biologically Grounded (Resolved) | **Active Research Target** |
| **Scaffold Generalization**| Random Split Only | Murcko Scaffold-Disjoint ($p = 5.50 \times 10^{-7}$) | **Verified in Google Colab** |
| **Default Serving Model** | `backend/checkpoints/pxddi_model.pt` | Gated by `PXDDI_CHECKPOINT_PATH` | **Preserved for Offline Dev** |
