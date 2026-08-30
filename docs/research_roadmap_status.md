# PxDDI research roadmap status

This status separates implemented source code from evidence that still requires
a Colab run or an independently sourced dataset. A feature is not a result
until its audited artifacts have been generated and reviewed.

| Original phase | Current status | What is implemented | What remains before a claim |
| --- | --- | --- | --- |
| 1. Correctness | Source complete; historical candidates require rerun | Version-aware checkpoint loading, candidate graph-schema selection, corrected auxiliary toxicity logits loss, disjoint early-stopping/post-hoc validation roles, corrected external evaluator, Colab dependency bounds, early stopping, tests, and split-aware unreported-negative sampling that forbids every known reported pair. | Rerun every multi-task candidate with the current `split_aware_standard_v1` protocol; run the selected candidate in Colab; Docker execution remains separately unverified. |
| 2. Fair baselines | Partial | Symmetric ECFP/Morgan + SGD logistic baseline, a fixed-split experiment launcher that separates model seeds from split seed, and repeated-seed comparison support. | XGBoost, GCN/GIN, directed MPNN, and actual matched baseline runs. |
| 3A. Motif view | Source complete | Auditable 17-SMARTS motif features and motif-edge-aware ablations. | Repeated matched-seed S1/S2 evidence. |
| 3B. Cross-drug layer | Source complete | Pair-isolated symmetric atom-level cross attention and atom/motif association audit. | Repeated matched-seed S1/S2 and efficiency evidence. |
| 3C. Contrastive pretraining | Source complete; execution pending | Schema-validated ChEMBL structure parser, deterministic corpus selection, strict exclusion of every non-training PxDDI structure, edge-aware masked-feature NT-Xent pretraining, encoder-only checkpoint provenance, and a separate fine-tuning candidate. | Run the pretraining candidate in Colab, then compare its S1/S2 results against the identical-split edge-aware model across matched seeds. |
| 4. Ablations | Source partial | Legacy, edge-aware, motif, and cross-attention DDI-only/multitask variants in a controlled suite. | Contrastive/full-model ablations and executed repeated-seed studies. |
| 5. Explainability | Source complete for local audit | Bounded atom/bond/motif occlusion, fidelity/sufficiency, canonical-SMILES stability, pair symmetry, attention/motif associations, SVG figures, conformal/OOD fields, and cross-seed stability analyzer. | Run the cross-seed audit and obtain a real curated mechanism reference for chemical-plausibility evaluation. |
| 6. Ensemble and abstention | Source complete; unrun | Fixed-split 3–5 member ensemble, score mean/std, fresh calibration/conformal analysis, structural-domain flag, explicit abstention. | Colab ensemble run and an honest S1/S2 analysis; it is not deployable by default. |
| 7. Strong evaluation | Source complete; execution pending | Standard Transductive/S1/S2 and separate Murcko scaffold-disjoint protocol; stratified test bootstrap CIs; five-seed paired bootstrap + Wilcoxon/Holm comparison; calibration and abstention diagnostics; confidence-ranked error files; structural-novelty slices; speed/memory recording; provenance-enforced external evaluator. | Execute normal, scaffold-disjoint, and five-seed studies; review error files; then evaluate a genuinely independent external dataset. |
| 8. Patient context | Intentionally disabled | API states that accepted patient fields are not used. | Real linked patient–drug–outcome data and a separate approved study. |
| 9. External biological knowledge | Data-audit source complete; feature candidate gated | PharmGKB Chemical↔Gene/pathway parser, exact-name-only TWOSIDES mapping audit, and explicit prohibition on adding PharmGKB records as DDI labels. | Run and review mapping coverage. A molecule--target activity export is still required before claiming ChEMBL target features; enable a biological candidate only after the audit is accepted. |

## Non-negotiable limits

- A sampled unreported TWOSIDES pair is not a known-safe pair.
- S1 results remain the key generalization test; the historical S1 result was near random.
- FAERS toxicity is observational and the 58 conflicting mapped structures remain excluded.
- Older multi-task artifacts used before the recorded toxicity-logits loss contract, validation-role separation, and split-aware unreported-negative protocol are historical only; do not compare or ensemble them with new runs.
- Explanations, attention, calibration, conformal sets, OOD flags, and abstention are research aids—not clinical evidence.
- No candidate or ensemble may overwrite `backend/checkpoints/pxddi_model.pt` without a documented review and explicit approval.
- The supplied ChEMBL `chemreps` and UniProt mapping files are a structure corpus and target metadata, respectively; they do not contain molecule--target activity labels.
- PharmGKB chemical--gene/pathway evidence must never be appended to TWOSIDES positive DDI labels or described as external DDI validation.
- ChEMBL contrastive pretraining is an encoder warm start only. Its corpus excludes direct structures outside the DDI training split and its checkpoint cannot replace the deployed model automatically.
