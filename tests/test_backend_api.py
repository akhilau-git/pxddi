"""Integration and contract tests for the research API backend."""

import asyncio
import re

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from src.data_prep.prepare_twosides import (
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_RICH,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
)
from src.data_prep.molecular_motifs import MOTIF_FEATURE_DIM
from src.models.ddi_model import PxDDIModel
from src.models.uncertainty import fit_split_conformal_binary

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
    assert payload['api_documentation_enabled'] is True
    assert re.fullmatch(r'[A-Za-z0-9._-]{8,128}', response.headers['x-request-id'])
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert 'model_auroc' not in payload
    assert payload['stored_validation_evidence']['status'] == 'available'
    assert payload['stored_validation_evidence']['auroc'] is not None
    assert 'not transductive test' in payload['stored_validation_evidence']['note']
    assert payload['conformal_uncertainty_status'] == 'not_available'
    assert payload['structural_applicability_domain_status'] == 'not_available'
    assert payload['structural_applicability_domain_error'] == (
        'checkpoint_has_no_structural_domain_reference_set'
    )
    assert payload['auxiliary_toxicity_head_status']['status'] == (
        'historical_contract_not_recorded'
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
            'Access-Control-Request-Headers': 'content-type,x-request-id',
        },
    )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'
    assert 'x-request-id' in response.headers['access-control-allow-headers'].lower()


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


def test_untrusted_host_header_is_rejected_including_before_routing():
    response = CLIENT.get('/health', headers={'host': 'untrusted.example'})

    assert response.status_code == 400


def test_trusted_host_configuration_rejects_wildcard(monkeypatch):
    monkeypatch.setenv('PXDDI_TRUSTED_HOSTS', '*.example.org')

    with pytest.raises(RuntimeError, match='explicit bare host names'):
        main.configured_trusted_hosts()


def test_boolean_environment_setting_is_explicit(monkeypatch):
    monkeypatch.setenv('PXDDI_ENABLE_DOCS', 'off')
    assert main.boolean_from_environment('PXDDI_ENABLE_DOCS', True) is False
    monkeypatch.setenv('PXDDI_ENABLE_DOCS', 'maybe')
    with pytest.raises(RuntimeError, match='true or false'):
        main.boolean_from_environment('PXDDI_ENABLE_DOCS', True)


def test_origin_configuration_rejects_wildcards_and_paths(monkeypatch):
    monkeypatch.setenv('PXDDI_ALLOWED_ORIGINS', 'https://example.org/app')

    with pytest.raises(RuntimeError, match='explicit HTTP'):
        main.configured_origins()


def test_valid_request_id_is_returned_for_correlation():
    response = CLIENT.get('/health', headers={'x-request-id': 'local-run-0001'})

    assert response.status_code == 200
    assert response.headers['x-request-id'] == 'local-run-0001'


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
    assert payload['prediction_uncertainty']['status'] == 'not_available'
    assert payload['structural_applicability_domain']['status'] == 'not_available'
    assert 'cannot assess structural out-of-domain status' in (
        payload['structural_applicability_domain']['note']
    )
    assert payload['drug_a_toxicity']['known'] is True
    assert payload['drug_a_toxicity']['training_label_available'] is True
    assert 0 <= payload['drug_a_toxicity']['score'] <= 1
    assert 0 <= payload['drug_b_toxicity']['score'] <= 1
    assert 'not a clinical toxicity probability' in payload['drug_a_toxicity']['model_score_note']
    assert 'FAERS-derived' in payload['drug_a_toxicity']['coverage_note']
    assert payload['explanation_available_at'] == '/explain (separate, slower endpoint)'


def test_prediction_rejects_unknown_or_coerced_request_fields_without_echoing_input():
    response = CLIENT.post(
        '/predict',
        json={
            'smiles_a': ASPIRIN,
            'smiles_b': ACETAMINOPHEN,
            'age_band': '3',
            'unexpected_option': 'must not be ignored',
        },
    )

    assert response.status_code == 422
    detail = response.json()['detail']
    assert any(error['loc'] == ['body', 'age_band'] for error in detail)
    assert any(error['loc'] == ['body', 'unexpected_option'] for error in detail)
    assert all('input' not in error for error in detail)
    assert 'must not be ignored' not in response.text


def test_oversized_request_is_rejected_before_model_processing(monkeypatch):
    monkeypatch.setattr(main, 'MAX_REQUEST_BYTES', 80)
    response = CLIENT.post(
        '/predict',
        json={'smiles_a': 'C' * 60, 'smiles_b': 'C' * 60},
    )

    assert response.status_code == 413
    assert 'exceeds the 80-byte API limit' in response.json()['detail']
    assert response.headers['cache-control'] == 'no-store'


def test_chunked_oversized_request_is_rejected_without_reaching_application(monkeypatch):
    monkeypatch.setattr(main, 'MAX_REQUEST_BYTES', 80)
    reached_application = False
    input_messages = iter([
        {'type': 'http.request', 'body': b'x' * 60, 'more_body': True},
        {'type': 'http.request', 'body': b'x' * 60, 'more_body': False},
    ])
    output_messages = []

    async def receive():
        return next(input_messages)

    async def send(message):
        output_messages.append(message)

    async def downstream(scope, receive, send):
        nonlocal reached_application
        reached_application = True

    scope = {
        'type': 'http',
        'asgi': {'version': '3.0'},
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': '/predict',
        'raw_path': b'/predict',
        'query_string': b'',
        'headers': [],
        'client': ('testclient', 50000),
        'server': ('testserver', 80),
    }

    asyncio.run(main.MaxRequestBodyMiddleware(downstream)(scope, receive, send))

    assert reached_application is False
    assert output_messages[0]['type'] == 'http.response.start'
    assert output_messages[0]['status'] == 413


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


def test_prediction_request_is_rejected_while_process_limit_is_active(monkeypatch):
    monkeypatch.setattr(main, 'PREDICTION_SEMAPHORE', main.threading.BoundedSemaphore(1))
    assert main.PREDICTION_SEMAPHORE.acquire(blocking=False)
    try:
        response = CLIENT.post(
            '/predict',
            json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
        )
    finally:
        main.PREDICTION_SEMAPHORE.release()

    assert response.status_code == 429
    assert 'busy' in response.json()['detail']


def test_explanation_also_respects_the_shared_inference_limit(monkeypatch):
    monkeypatch.setattr(main, 'PREDICTION_SEMAPHORE', main.threading.BoundedSemaphore(1))
    assert main.PREDICTION_SEMAPHORE.acquire(blocking=False)
    try:
        response = CLIENT.post(
            '/explain',
            json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
        )
    finally:
        main.PREDICTION_SEMAPHORE.release()

    assert response.status_code == 429
    assert 'inference service is busy' in response.json()['detail']


def test_edge_aware_candidate_does_not_advertise_legacy_explanations(monkeypatch):
    monkeypatch.setattr(main.model, 'architecture_version', main.MODEL_ARCHITECTURE_EDGE_AWARE)

    explanation = CLIENT.post(
        '/explain',
        json={'smiles_a': ASPIRIN, 'smiles_b': ACETAMINOPHEN},
    )
    health = CLIENT.get('/health')

    assert explanation.status_code == 501
    assert health.json()['explanation_status'] == 'not_available'


def test_backend_builds_and_scores_rich_graphs_for_an_edge_aware_candidate(monkeypatch):
    """A selected candidate must receive the graph schema it was trained on."""
    candidate = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version=main.MODEL_ARCHITECTURE_EDGE_AWARE,
        edge_feature_dim=NUM_BOND_FEATURES,
    ).eval()
    monkeypatch.setattr(main, 'model', candidate)
    monkeypatch.setattr(main, 'MODEL_FEATURE_SCHEMA', FEATURE_SCHEMA_RICH)

    graph_a, graph_b = main.build_drug_batches(
        main.DDIRequest(smiles_a=ASPIRIN, smiles_b=ACETAMINOPHEN)
    )
    risk, _, _ = candidate(graph_a, graph_b)

    assert graph_a.x.shape[1] == RICH_NUM_ATOM_FEATURES
    assert graph_b.edge_attr.shape[1] == NUM_BOND_FEATURES
    assert risk.shape == (1,)


def test_checkpoint_feature_schema_rejects_architecture_mismatches():
    assert main.checkpoint_feature_schema({
        'architecture_version': main.MODEL_ARCHITECTURE_EDGE_AWARE,
        'feature_schema': FEATURE_SCHEMA_RICH,
    }) == FEATURE_SCHEMA_RICH
    with pytest.raises(RuntimeError, match='edge-aware'):
        main.checkpoint_feature_schema({
            'architecture_version': main.MODEL_ARCHITECTURE_EDGE_AWARE,
            'feature_schema': FEATURE_SCHEMA_LEGACY,
        })
    assert main.checkpoint_feature_schema({
        'architecture_version': 'cross_attention_edge_aware_gat_v1',
        'feature_schema': FEATURE_SCHEMA_RICH,
    }) == FEATURE_SCHEMA_RICH


def test_checkpoint_structural_domain_reports_seen_and_distant_drugs(monkeypatch):
    domain, error = main.checkpoint_structural_domain({
        'applicability_domain': {
            'method': 'nearest_train_ecfp_tanimoto_v1',
            'radius': 2,
            'num_bits': 1024,
            'include_chirality': True,
            'minimum_tanimoto_similarity': 0.6,
            'reference_canonical_smiles': ['CCO', 'CCN'],
        },
    })
    assert error is None
    assert domain is not None
    monkeypatch.setattr(main, 'STRUCTURAL_DOMAIN', domain)
    monkeypatch.setattr(main, 'STRUCTURAL_DOMAIN_ERROR', None)

    response = main.structural_domain_response('CCO', 'c1ccccc1')

    assert response['status'] == 'available_structure_similarity_diagnostic'
    assert response['drug_a']['exactly_seen_in_training'] is True
    assert response['drug_b']['outside_structural_domain'] is True
    assert response['outside_structural_domain'] is True
    assert 'does not measure pair novelty' in response['note']


def test_checkpoint_conformal_state_is_applied_to_calibrated_score(monkeypatch):
    conformal = fit_split_conformal_binary(
        labels=[0, 0, 1, 1], probabilities=[0.1, 0.2, 0.8, 0.9], alpha=0.2,
        fitted_on='validation_conformal_partition',
    )
    monkeypatch.setitem(main.checkpoint, 'conformal', conformal)

    response = main.uncertainty_response(0.5)

    assert response['status'] == 'available_internal_validation_only'
    assert response['method'] == 'split_conformal_binary_v1'
    assert response['fitted_on'] == 'validation_conformal_partition'
    assert response['abstain'] is True


def test_backend_builds_motif_features_for_a_motif_candidate(monkeypatch):
    candidate = PxDDIModel(
        in_channels=RICH_NUM_ATOM_FEATURES,
        hidden_channels=8,
        architecture_version='motif_edge_aware_gat_v1',
        edge_feature_dim=NUM_BOND_FEATURES,
        motif_feature_dim=MOTIF_FEATURE_DIM,
        motif_hidden_channels=4,
    ).eval()
    monkeypatch.setattr(main, 'model', candidate)
    monkeypatch.setattr(main, 'MODEL_FEATURE_SCHEMA', FEATURE_SCHEMA_RICH)
    monkeypatch.setattr(main, 'MODEL_REQUIRES_MOTIF_FEATURES', True)

    graph_a, graph_b = main.build_drug_batches(
        main.DDIRequest(smiles_a=ASPIRIN, smiles_b=ACETAMINOPHEN)
    )
    risk, _, _ = candidate(graph_a, graph_b)

    assert graph_a.motif_features.shape == (1, MOTIF_FEATURE_DIM)
    assert graph_b.motif_features.shape == (1, MOTIF_FEATURE_DIM)
    assert risk.shape == (1,)
