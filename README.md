# PxDDI: Patient-Context Drug-Drug Interaction System

Research prototype implementing patient-context-conditioned, structure-based
DDI risk prediction, connecting drug interaction, toxicity, and patient
context signals — addressing a gap identified across a 26-paper literature
review (see /docs/literature_review.md).

## Status: Research Prototype — NOT for clinical use.

## Production model
GNN-based dual-view encoder, trained on 200,000 real TWOSIDES pairs.
- Transductive AUROC: 0.9735
- S1 (unseen drug pairs) AUROC: 0.5021 — known limitation, see Model Card
- S2 (one unseen drug) AUROC: 0.7474

## Setup
1. `pip install -r requirements.txt`
2. Place trained checkpoint at `backend/checkpoints/pxddi_model.pt`
3. Place toxicity bridge at `backend/checkpoints/toxicity_smiles_bridge.csv`
4. `cd backend && uvicorn main:app --reload`
5. Visit `http://127.0.0.1:8000/docs`

## Training
Run `src/training/train_full_pipeline_v2.py` in Google Colab (GPU required).
See `/notebooks/pxddi_training_run.ipynb` for the exact executed run.

## Known limitations
See MODEL_CARD.md for full disclosure of dataset, negative-sampling,
and generalization limitations.
