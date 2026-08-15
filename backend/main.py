from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import sys
sys.path.append('/app/src')
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

model = PxDDIModel(in_channels=NUM_ATOM_FEATURES)
model.load_state_dict(torch.load('checkpoints/pxddi_model.pt', map_location='cpu'))
model.eval()

class DDIRequest(BaseModel):
    smiles_a: str
    smiles_b: str
    age_band: int = None      # 0-9, representing decades (0=0-9yrs, 9=90+)
    sex: int = None           # 0=male, 1=female
    comorbidities: list = None  # multi-hot list of length 10, e.g. [0,1,0,...]

@app.post("/predict")
def predict_ddi(req: DDIRequest):
    graph_a = smiles_to_graph(req.smiles_a)
    graph_b = smiles_to_graph(req.smiles_b)
    if graph_a is None or graph_b is None:
        # Fix (from review): don't return 200 with a hidden error — return a real error status
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Invalid SMILES string for one or both drugs.")

    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    # FIX: actually build and pass the patient dict, instead of ignoring the input
    patient = None
    if req.age_band is not None and req.sex is not None and req.comorbidities is not None:
        patient = {
            'age_band': torch.tensor([float(req.age_band)]),
            'sex': torch.tensor([float(req.sex)]),
            'comorbidities': torch.tensor([[float(x) for x in req.comorbidities]])
        }

    with torch.no_grad():
        risk, tox_a, tox_b = model(batch_a, batch_b, patient=patient)

    explanation = full_explanation_pipeline(model, batch_a, req.smiles_a, batch_b, req.smiles_b)

    return {
        # Honest framing (from review) — not a clinical probability
        "disclaimer": "Research prototype output. Not clinical advice. Not FDA/regulatory reviewed.",
        "interaction_risk_estimate": float(torch.sigmoid(risk)),
        "patient_context_applied": patient is not None,
        "drug_a_toxicity": {
            "score": float(tox_a),
            "known": req.smiles_a in KNOWN_TOXICITY_SMILES  # Fix 3, defined below
        },
        "drug_b_toxicity": {
            "score": float(tox_b),
            "known": req.smiles_b in KNOWN_TOXICITY_SMILES
        },
        "explanation": explanation
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}
