from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import torch
import sys
import os

# Works both locally (uvicorn run from backend/) and inside Docker
# (where backend/ and src/ are siblings under /app)
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.append(SRC_PATH)
from models.ddi_model import PxDDIModel
from models.explainability import full_explanation_pipeline
from data_prep.prepare_twosides import smiles_to_graph, NUM_ATOM_FEATURES
from torch_geometric.data import Batch
from toxicity_lookup import KNOWN_TOXICITY_SMILES

app = FastAPI(title="PxDDI API")

# Fix 6: CORS, so the frontend (port 3000) can actually call this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend domain before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

checkpoint = torch.load('checkpoints/pxddi_model.pt', map_location='cpu', weights_only=False)
model = PxDDIModel(
    in_channels=checkpoint['in_channels'],
    hidden_channels=checkpoint['hidden_channels'],
    use_chemberta=checkpoint.get('use_chemberta', False)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
DECISION_THRESHOLD = checkpoint.get('threshold', 0.5)
print(f"Loaded model. AUROC={checkpoint.get('auroc')}, threshold={DECISION_THRESHOLD}")

from typing import Optional, List

class DDIRequest(BaseModel):
    smiles_a: str
    smiles_b: str
    age_band: Optional[int] = None      # 0-9, representing decades (0=0-9yrs, 9=90+)
    sex: Optional[int] = None           # 0=male, 1=female
    comorbidities: Optional[List[int]] = None  # multi-hot list of length 10, e.g. [0,1,0,...]

    @field_validator('age_band')
    @classmethod
    def validate_age_band(cls, v):
        if v is not None and not (0 <= v <= 9):
            raise ValueError('age_band must be between 0 and 9')
        return v

    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v is not None and v not in (0, 1):
            raise ValueError('sex must be 0 or 1')
        return v

    @field_validator('comorbidities')
    @classmethod
    def validate_comorbidities(cls, v):
        if v is not None and len(v) != 10:
            raise ValueError('comorbidities must be a list of exactly 10 values')
        return v

@app.post("/predict")
def predict_ddi(req: DDIRequest):
    graph_a = smiles_to_graph(req.smiles_a)
    graph_b = smiles_to_graph(req.smiles_b)
    if graph_a is None or graph_b is None:
        # Fix (from review): don't return 200 with a hidden error — return a real error status
        raise HTTPException(status_code=422, detail="Invalid SMILES string for one or both drugs.")

    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    # HONEST LIMITATION: patient-context module exists architecturally but
    # has not been trained on linked patient-outcome data (no dataset in
    # this project links a specific patient to a specific drug-pair
    # outcome). Applying it now would run UNTRAINED random weights and
    # silently bias results. Disabled until real training data exists.
    patient = None
    patient_context_note = (
        "Patient context fields were accepted but NOT applied — the "
        "patient-conditioning module is not yet trained on real linked "
        "patient-outcome data. This will be enabled once available."
    )

    with torch.no_grad():
        risk, tox_a, tox_b = model(batch_a, batch_b, patient=None)

    risk_score = float(torch.sigmoid(risk))

    return {
        "disclaimer": "Research prototype output. Not clinical advice. Not FDA/regulatory reviewed.",
        "interaction_risk_estimate": risk_score,
        "interaction_predicted": risk_score >= DECISION_THRESHOLD,
        "decision_threshold_used": DECISION_THRESHOLD,
        "patient_context_applied": False,
        "patient_context_note": patient_context_note,
        "drug_a_toxicity": {"score": float(tox_a), "known": req.smiles_a in KNOWN_TOXICITY_SMILES},
        "drug_b_toxicity": {"score": float(tox_b), "known": req.smiles_b in KNOWN_TOXICITY_SMILES},
        "explanation_available_at": "/explain (separate, slower endpoint)"
    }


@app.post("/explain")
def explain_ddi(req: DDIRequest):
    """Separate, slower endpoint — GNNExplainer is expensive (100 epochs
    x2), so it's decoupled from the fast /predict path per review feedback."""
    graph_a = smiles_to_graph(req.smiles_a)
    graph_b = smiles_to_graph(req.smiles_b)
    if graph_a is None or graph_b is None:
        raise HTTPException(status_code=422, detail="Invalid SMILES string for one or both drugs.")
    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])
    explanation = full_explanation_pipeline(model, batch_a, req.smiles_a, batch_b, req.smiles_b)
    return {
        "disclaimer": "Explanation identifies which atoms most influenced this molecule's learned embedding, cross-checked against a small curated list of known-risk functional groups. This is not a validated literature review.",
        "explanation": explanation
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": "GNN" if not checkpoint.get('use_chemberta', False) else "ChemBERTa",
        "model_auroc": checkpoint.get('auroc'),
        "toxicity_bridge_loaded": len(KNOWN_TOXICITY_SMILES) > 0,
        "toxicity_bridge_size": len(KNOWN_TOXICITY_SMILES),
        "decision_threshold": DECISION_THRESHOLD,
    }
