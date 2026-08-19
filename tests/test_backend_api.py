"""Integration and contract tests for the research API backend."""

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from backend import main


CLIENT = TestClient(main.app)
ASPIRIN = 'CC(=O)OC1=CC=CC=C1C(=O)O'
ACETAMINOPHEN = 'CC(=O)NC1=CC=C(C=C1)O'


def test_health_reports_checkpoint_and_runtime_limits():
    response = CLIENT.get('/health')

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['ready'] is True
    assert payload['model_loaded'] is True
    assert payload['patient_context_enabled'] is False
    assert len(payload['model_checkpoint_sha256']) == 64
    assert payload['max_molecule_atoms'] == main.MAX_MOLECULE_ATOMS
    assert payload['toxicity_bridge_error'] is None
    assert payload['toxicity_bridge_size'] == len(main.KNOWN_TOXICITY_SMILES)
    assert payload['toxicity_bridge_conflicting_structures_excluded'] == (
        main.TOXICITY_BRIDGE_SUMMARY['excluded_conflicting_structures']
    )


def test_ready_endpoint_requires_the_toxicity_bridge():
    response = CLIENT.get('/ready')

    assert response.status_code == 200
    assert response.json()['status'] == 'ready'


def test_ready_endpoint_reports_a_missing_toxicity_bridge(monkeypatch):
    monkeypatch.setattr(main, 'KNOWN_TOXICITY_SMILES', set())
    monkeypatch.setattr(main, 'TOXICITY_BRIDGE_ERROR', 'bridge fixture unavailable')

    response = CLIENT.get('/ready')

    assert response.status_code == 503
    assert response.json()['detail'] == 'bridge fixture unavailable'


def test_local_frontend_origin_is_allowed_by_cors():
    response = CLIENT.options(
        '/predict',
        headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type',
        },
    )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'


def test_unconfigured_origin_is_not_allowed_by_cors():
    response = CLIENT.options(
        '/predict',
        headers={
            'Origin': 'https://untrusted.example',
            'Access-Control-Request-Method': 'POST',
        },
    )

    assert response.status_code == 400
    assert 'access-control-allow-origin' not in response.headers


def test_prediction_reports_label_coverage_and_disabled_patient_context():
    response = CLIENT.post(
        '/predict',
        json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['patient_context_applied'] is False
    assert 'uncalibrated' in payload['interaction_risk_note']
    assert 'not evidence that the pair is safe' in payload['interaction_label_note']
    assert payload['score_calibration']['status'] == 'uncalibrated'
    assert payload['drug_a_toxicity']['known'] is True
    assert payload['drug_a_toxicity']['training_label_available'] is True
    assert 'FAERS-derived' in payload['drug_a_toxicity']['coverage_note']
    assert payload['explanation_available_at'] == '/explain (separate, slower endpoint)'


def test_invalid_or_single_atom_smiles_returns_422():
    response = CLIENT.post(
        '/predict',
        json={'smiles_a': 'C', 'smiles_b': ACETAMINOPHEN},
    )

    assert response.status_code == 422


def test_atom_limit_is_enforced():
    request = main.DDIRequest(smiles_a='C' * (main.MAX_MOLECULE_ATOMS + 1), smiles_b='CCO')

    with pytest.raises(HTTPException, match='atoms') as error:
        main.build_drug_batches(request)

    assert error.value.status_code == 422


def test_second_explanation_request_is_rejected_while_one_is_active():
    assert main.EXPLANATION_SEMAPHORE.acquire(blocking=False)
    try:
        response = CLIENT.post(
            '/explain',
            json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
        )
    finally:
        main.EXPLANATION_SEMAPHORE.release()

    assert response.status_code == 429


def test_edge_aware_candidate_does_not_advertise_legacy_explanations(monkeypatch):
    monkeypatch.setattr(main.model, 'architecture_version', main.MODEL_ARCHITECTURE_EDGE_AWARE)

    explanation = CLIENT.post(
        '/explain',
        json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
    )
    health = CLIENT.get('/health')

    assert explanation.status_code == 501
    assert health.json()['explanation_status'] == 'not_available'
