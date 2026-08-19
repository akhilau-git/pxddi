"""FastAPI service for the research-only PxDDI model."""

from pathlib import Path
from typing import List, Optional
import hashlib
import os
import sys
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import torch
from torch_geometric.data import Batch


BACKEND_DIR = Path(__file__).resolve().parent
SRC_PATH = BACKEND_DIR.parent / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from models.ddi_model import MODEL_ARCHITECTURE_EDGE_AWARE, model_from_checkpoint
from models.calibration import apply_calibrator
from models.explainability import full_explanation_pipeline
from data_prep.prepare_twosides import smiles_to_graph

if __package__:
    from .toxicity_lookup import (
        KNOWN_TOXICITY_SMILES,
        TOXICITY_BRIDGE_ERROR,
        TOXICITY_BRIDGE_SUMMARY,
        is_toxicity_known,
    )
else:
    from toxicity_lookup import (
        KNOWN_TOXICITY_SMILES,
        TOXICITY_BRIDGE_ERROR,
        TOXICITY_BRIDGE_SUMMARY,
        is_toxicity_known,
    )


def positive_integer_from_environment(name: str, default: int) -> int:
    """Read a positive integer setting and fail early on an invalid config."""
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise RuntimeError(f'{name} must be a positive integer')
    return value


def configured_origins():
    default_origins = 'http://localhost:3000,http://127.0.0.1:3000'
    raw_origins = os.environ.get('PXDDI_ALLOWED_ORIGINS', default_origins)
    origins = [origin.strip() for origin in raw_origins.split(',') if origin.strip()]
    if not origins:
        raise RuntimeError('PXDDI_ALLOWED_ORIGINS must contain at least one origin')
    return origins


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


MAX_SMILES_LENGTH = positive_integer_from_environment('PXDDI_MAX_SMILES_LENGTH', 1000)
MAX_MOLECULE_ATOMS = positive_integer_from_environment('PXDDI_MAX_MOLECULE_ATOMS', 200)
MAX_MOLECULE_BONDS = positive_integer_from_environment('PXDDI_MAX_MOLECULE_BONDS', 250)
MAX_CONCURRENT_EXPLANATIONS = positive_integer_from_environment(
    'PXDDI_MAX_CONCURRENT_EXPLANATIONS', 1
)
ALLOWED_ORIGINS = configured_origins()

app = FastAPI(title='PxDDI API')

# The default permits only the local frontend. Deployments must explicitly set
# PXDDI_ALLOWED_ORIGINS to their own comma-separated frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)

CHECKPOINT_PATH = BACKEND_DIR / 'checkpoints' / 'pxddi_model.pt'
# Checkpoint metadata contains only safe built-in types and tensors.
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
model = model_from_checkpoint(checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

DECISION_THRESHOLD = float(checkpoint.get('threshold', 0.5))
CALIBRATION = checkpoint.get('calibration')
CHECKPOINT_SHA256 = file_sha256(CHECKPOINT_PATH)
EXPLANATION_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_EXPLANATIONS)

print(
    'Loaded model. '
    f"AUROC={checkpoint.get('auroc')}, threshold={DECISION_THRESHOLD}, "
    f'checkpoint_sha256={CHECKPOINT_SHA256}'
)


class DDIRequest(BaseModel):
    smiles_a: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    smiles_b: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    age_band: Optional[int] = None
    sex: Optional[int] = None
    comorbidities: Optional[List[int]] = None

    @field_validator('smiles_a', 'smiles_b', mode='before')
    @classmethod
    def strip_smiles(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator('age_band')
    @classmethod
    def validate_age_band(cls, value):
        if value is not None and not (0 <= value <= 9):
            raise ValueError('age_band must be between 0 and 9')
        return value

    @field_validator('sex')
    @classmethod
    def validate_sex(cls, value):
        if value is not None and value not in (0, 1):
            raise ValueError('sex must be 0 or 1')
        return value

    @field_validator('comorbidities')
    @classmethod
    def validate_comorbidities(cls, value):
        if value is not None and len(value) != 10:
            raise ValueError('comorbidities must be a list of exactly 10 values')
        if value is not None and any(item not in (0, 1) for item in value):
            raise ValueError('comorbidities must contain only 0 or 1 values')
        return value


def build_drug_batches(req: DDIRequest):
    """Build bounded molecular batches shared by prediction and explanation."""
    graph_a = smiles_to_graph(req.smiles_a)
    graph_b = smiles_to_graph(req.smiles_b)
    if graph_a is None or graph_b is None:
        raise HTTPException(
            status_code=422,
            detail='Invalid, unsupported, or single-atom SMILES string for one or both drugs.',
        )

    for name, graph in (('drug A', graph_a), ('drug B', graph_b)):
        atom_count = int(graph.num_nodes)
        bond_count = int(graph.edge_index.size(1) // 2)
        if atom_count > MAX_MOLECULE_ATOMS:
            raise HTTPException(
                status_code=422,
                detail=f'{name} has {atom_count} atoms; the limit is {MAX_MOLECULE_ATOMS}.',
            )
        if bond_count > MAX_MOLECULE_BONDS:
            raise HTTPException(
                status_code=422,
                detail=f'{name} has {bond_count} bonds; the limit is {MAX_MOLECULE_BONDS}.',
            )

    return Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b])


def toxicity_response(smiles: str, score: float):
    """State whether the molecular structure had a FAERS-derived training label."""
    training_label_available = is_toxicity_known(smiles)
    coverage_note = (
        'A matched FAERS-derived toxicity training label is available for this structure.'
        if training_label_available
        else 'No matched FAERS-derived toxicity training label is available for this structure.'
    )
    return {
        'score': score,
        'known': training_label_available,
        'training_label_available': training_label_available,
        'coverage_note': coverage_note,
    }


def risk_calibration_response() -> dict:
    """Describe whether the loaded checkpoint supplies a saved calibration map."""
    if CALIBRATION and CALIBRATION.get('status') == 'fitted':
        return {
            'status': 'internally_calibrated',
            'method': CALIBRATION.get('method'),
            'fitted_on': CALIBRATION.get('fitted_on'),
            'note': (
                'Calibration was fitted on the internal validation split only. It is not '
                'evidence of calibrated cold-start, external, or clinical performance.'
            ),
        }
    return {
        'status': 'uncalibrated',
        'method': None,
        'fitted_on': None,
        'note': 'This checkpoint is uncalibrated; no saved calibration map is available.',
    }


def explanation_response() -> dict:
    """Describe whether this architecture has a compatible explanation path."""
    if getattr(model, 'architecture_version', None) == MODEL_ARCHITECTURE_EDGE_AWARE:
        return {
            'available': False,
            'endpoint': None,
            'note': (
                'The edge-aware candidate has no validated API explanation method yet. '
                'Do not interpret an embedding attribution from the legacy method as an '
                'edge-aware pair-risk explanation.'
            ),
        }
    return {
        'available': True,
        'endpoint': '/explain (separate, slower endpoint)',
        'note': (
            'Available only as a legacy embedding attribution. It is not a validated '
            'pair-risk explanation.'
        ),
    }


def readiness_error() -> str | None:
    """Return a readiness failure when required toxicity-coverage data is unavailable."""
    if KNOWN_TOXICITY_SMILES:
        return None
    return TOXICITY_BRIDGE_ERROR or (
        'Toxicity bridge loaded without any usable canonical SMILES entries.'
    )


@app.post('/predict')
def predict_ddi(req: DDIRequest):
    batch_a, batch_b = build_drug_batches(req)

    # The patient encoder remains disabled: it has not been trained with linked
    # patient, exposure, and outcome data, so applying it would add random bias.
    patient_context_note = (
        'Patient context fields were accepted but NOT applied. The patient '
        'conditioning module is not trained on linked patient-outcome data.'
    )

    with torch.no_grad():
        risk, tox_a, tox_b = model(batch_a, batch_b, patient=None)

    raw_risk_score = float(torch.sigmoid(risk))
    risk_score = float(apply_calibrator([raw_risk_score], CALIBRATION)[0])
    calibration_response = risk_calibration_response()
    explanation = explanation_response()
    return {
        'disclaimer': 'Research prototype output. Not clinical advice. Not FDA/regulatory reviewed.',
        'interaction_risk_estimate': risk_score,
        'interaction_risk_score_raw': raw_risk_score,
        'interaction_risk_note': (
            'This is a research-model estimate, not a clinical probability. '
            + calibration_response['note']
        ),
        'interaction_label_note': (
            'The research task distinguishes reported TWOSIDES pairs from sampled '
            'unreported pairs. An unreported pair is not evidence that the pair is safe.'
        ),
        'score_calibration': calibration_response,
        'interaction_predicted': risk_score >= DECISION_THRESHOLD,
        'decision_threshold_used': DECISION_THRESHOLD,
        'patient_context_applied': False,
        'patient_context_note': patient_context_note,
        'drug_a_toxicity': toxicity_response(req.smiles_a, float(tox_a)),
        'drug_b_toxicity': toxicity_response(req.smiles_b, float(tox_b)),
        'explanation_available_at': explanation['endpoint'],
        'explanation_note': explanation['note'],
    }


@app.post('/explain')
def explain_ddi(req: DDIRequest):
    """Run the expensive embedding explanation with bounded local concurrency."""
    if not explanation_response()['available']:
        raise HTTPException(
            status_code=501,
            detail=(
                'The edge-aware candidate has no validated API explanation method yet. '
                'This endpoint remains available only for the legacy GAT architecture.'
            ),
        )
    batch_a, batch_b = build_drug_batches(req)
    if not EXPLANATION_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail='An explanation is already running. Please retry shortly.',
        )

    try:
        explanation = full_explanation_pipeline(
            model, batch_a, req.smiles_a, batch_b, req.smiles_b
        )
    finally:
        EXPLANATION_SEMAPHORE.release()

    return {
        'disclaimer': (
            'Explanation identifies atoms that influenced each molecule embedding and '
            'cross-checks them against a small functional-group heuristic. It is not a '
            'validated literature review or a pair-risk explanation.'
        ),
        'explanation': explanation,
    }


@app.get('/health')
def health():
    bridge_error = readiness_error()
    return {
        'status': 'ok',
        'ready': bridge_error is None,
        'model_loaded': True,
        'model_type': 'GNN' if not checkpoint.get('use_chemberta', False) else 'ChemBERTa',
        'model_architecture': checkpoint.get('architecture_version', 'legacy_gat_v1'),
        'score_calibration_status': risk_calibration_response()['status'],
        'explanation_status': 'available' if explanation_response()['available'] else 'not_available',
        'model_auroc': float(checkpoint.get('auroc')) if checkpoint.get('auroc') is not None else None,
        'model_epoch': int(checkpoint['epoch']) if checkpoint.get('epoch') is not None else None,
        'model_checkpoint_sha256': CHECKPOINT_SHA256,
        'toxicity_bridge_loaded': len(KNOWN_TOXICITY_SMILES) > 0,
        'toxicity_bridge_size': len(KNOWN_TOXICITY_SMILES),
        'toxicity_bridge_error': bridge_error,
        'toxicity_bridge_source_rows': (
            TOXICITY_BRIDGE_SUMMARY['source_rows']
            if TOXICITY_BRIDGE_SUMMARY is not None else None
        ),
        'toxicity_bridge_conflicting_structures_excluded': (
            TOXICITY_BRIDGE_SUMMARY['excluded_conflicting_structures']
            if TOXICITY_BRIDGE_SUMMARY is not None else None
        ),
        'decision_threshold': DECISION_THRESHOLD,
        'patient_context_enabled': False,
        'max_smiles_length': MAX_SMILES_LENGTH,
        'max_molecule_atoms': MAX_MOLECULE_ATOMS,
        'max_molecule_bonds': MAX_MOLECULE_BONDS,
        'max_concurrent_explanations': MAX_CONCURRENT_EXPLANATIONS,
    }


@app.get('/ready')
def ready():
    """Readiness endpoint used by deployment health checks."""
    bridge_error = readiness_error()
    if bridge_error is not None:
        raise HTTPException(status_code=503, detail=bridge_error)
    return {
        'status': 'ready',
        'model_checkpoint_sha256': CHECKPOINT_SHA256,
        'toxicity_bridge_size': len(KNOWN_TOXICITY_SMILES),
    }
