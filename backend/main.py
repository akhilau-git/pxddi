from fastapi import FastAPI
from pydantic import BaseModel
import torch, sys
sys.path.append('../src')
from models.ddi_model import PxDDIModel
from models.explainability import full_explanation_pipeline
from data_prep.prepare_twosides import smiles_to_graph, NUM_ATOM_FEATURES
from torch_geometric.data import Batch

app = FastAPI(title="PxDDI API")
model = PxDDIModel(in_channels=NUM_ATOM_FEATURES)
model.load_state_dict(torch.load('checkpoints/pxddi_model.pt', map_location='cpu'))
model.eval()

class DDIRequest(BaseModel):
    smiles_a: str; smiles_b: str
    age_band: int = None; sex: int = None; comorbidities: list = None

@app.post("/predict")
def predict_ddi(req: DDIRequest):
    ga, gb = smiles_to_graph(req.smiles_a), smiles_to_graph(req.smiles_b)
    if ga is None or gb is None: return {"error": "Invalid SMILES"}
    ba, bb = Batch.from_data_list([ga]), Batch.from_data_list([gb])
    with torch.no_grad(): risk, ta, tb = model(ba, bb)
    explanation = full_explanation_pipeline(model, ba, req.smiles_b)
    return {"interaction_risk": float(torch.sigmoid(risk)), "drug_a_toxicity": float(ta),
            "drug_b_toxicity": float(tb), "explanation": explanation}

@app.get("/health")
def health(): return {"status": "ok"}
