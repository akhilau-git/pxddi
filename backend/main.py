"""FastAPI service for the research-only PxDDI model."""

from pathlib import Path
from typing import List, Optional
import hmac
import hashlib
import logging
import os
import re
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch
from torch_geometric.data import Batch


BACKEND_DIR = Path(__file__).resolve().parent
SRC_PATH = BACKEND_DIR.parent / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from models.ddi_model import (
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_CROSS_ATTENTION_EDGE_AWARE,
    MODEL_ARCHITECTURE_MOTIF_EDGE_AWARE,
    MODEL_ARCHITECTURE_GRAPH_FP_FUSION,
    MODEL_ARCHITECTURE_AUDITDDI_MEMORY,
    architecture_uses_edge_features,
    architecture_requires_motif_features,
    architecture_requires_fingerprint_features,
    model_from_checkpoint,
)
from models.neighbor_memory import AuditableNeighborMemory
from models.calibration import apply_calibrator
from models.applicability_domain import APPLICABILITY_DOMAIN_METHOD
from models.uncertainty import conformal_prediction_sets
from models.explainability import full_explanation_pipeline
from data_prep.prepare_twosides import (
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_RICH,
    smiles_to_graph,
)


LOGGER = logging.getLogger('pxddi.api')
REQUEST_ID_PATTERN = re.compile(r'^[A-Za-z0-9._-]{8,128}$')

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
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f'{name} must be a positive integer') from error
    if value <= 0:
        raise RuntimeError(f'{name} must be a positive integer')
    return value


def non_negative_integer_from_environment(name: str, default: int) -> int:
    """Read an optional non-negative integer deployment setting."""
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f'{name} must be a non-negative integer') from error
    if value < 0:
        raise RuntimeError(f'{name} must be a non-negative integer')
    return value


def boolean_from_environment(name: str, default: bool) -> bool:
    """Read an explicit boolean deployment setting without silent coercion."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise RuntimeError(f'{name} must be true or false')


def configured_origins():
    default_origins = 'http://localhost:3000,http://127.0.0.1:3000'
    raw_origins = os.environ.get('PXDDI_ALLOWED_ORIGINS', default_origins)
    origins = [origin.strip() for origin in raw_origins.split(',') if origin.strip()]
    if not origins:
        raise RuntimeError('PXDDI_ALLOWED_ORIGINS must contain at least one origin')
    for origin in origins:
        parsed = urlparse(origin)
        if (
            origin == '*'
            or parsed.scheme not in {'http', 'https'}
            or not parsed.netloc
            or parsed.path
            or parsed.params
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise RuntimeError(
                'PXDDI_ALLOWED_ORIGINS must contain explicit HTTP(S) origins without paths.'
            )
    return origins


def configured_trusted_hosts() -> list[str]:
    """Return an explicit Host-header allowlist for the HTTP service.

    This protects the local/container service against Host-header attacks. A
    public deployment must replace the development defaults with its actual
    API domain through ``PXDDI_TRUSTED_HOSTS``; TLS, authentication, and a
    global rate limit still belong at the deployment gateway.
    """
    default_hosts = 'localhost,127.0.0.1,testserver'
    raw_hosts = os.environ.get('PXDDI_TRUSTED_HOSTS', default_hosts)
    hosts = [host.strip() for host in raw_hosts.split(',') if host.strip()]
    if not hosts:
        raise RuntimeError('PXDDI_TRUSTED_HOSTS must contain at least one host.')
    if any('*' in host or '://' in host or '/' in host for host in hosts):
        raise RuntimeError(
            'PXDDI_TRUSTED_HOSTS must contain explicit bare host names, not URLs, paths, or *.'
        )
    return hosts


def configured_deployment_mode() -> str:
    """Return the explicitly constrained API operating mode."""
    mode = os.environ.get('PXDDI_DEPLOYMENT_MODE', 'development').strip().lower()
    if mode not in {'development', 'production'}:
        raise RuntimeError('PXDDI_DEPLOYMENT_MODE must be development or production.')
    return mode


def configured_api_key(deployment_mode: str) -> str | None:
    """Read an optional local API key, mandatory in production mode."""
    value = os.environ.get('PXDDI_API_KEY')
    if value is not None:
        value = value.strip()
    if deployment_mode == 'production' and not value:
        raise RuntimeError('PXDDI_API_KEY must be set in production mode.')
    if value and len(value) < 32:
        raise RuntimeError('PXDDI_API_KEY must contain at least 32 characters.')
    return value or None


def validate_production_configuration(
    deployment_mode: str,
    api_key: str | None,
    allowed_origins: list[str],
    trusted_hosts: list[str],
    enable_docs: bool,
    rate_limit_per_minute: int,
) -> None:
    """Fail closed when someone tries to expose the local service publicly."""
    if deployment_mode != 'production':
        return
    if not api_key:
        raise RuntimeError('PXDDI_API_KEY must be set in production mode.')
    if enable_docs:
        raise RuntimeError('PXDDI_ENABLE_DOCS must be false in production mode.')
    if any(urlparse(origin).scheme != 'https' for origin in allowed_origins):
        raise RuntimeError('PXDDI_ALLOWED_ORIGINS must use HTTPS in production mode.')
    if any(host in {'localhost', '127.0.0.1', 'testserver'} for host in trusted_hosts):
        raise RuntimeError('PXDDI_TRUSTED_HOSTS must contain public API hosts in production mode.')
    if rate_limit_per_minute <= 0:
        raise RuntimeError(
            'PXDDI_RATE_LIMIT_PER_MINUTE must be positive in production mode.'
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint_path(configured_path: str | None = None) -> Path:
    """Return an existing checkpoint path without allowing path traversal.

    ``PXDDI_CHECKPOINT_PATH`` accepts either a path relative to
    ``backend/checkpoints`` (for example ``candidates/model.pt``) or an
    absolute path inside that directory.  This lets a reviewed candidate be
    served explicitly, while preventing an environment setting from loading an
    arbitrary pickle-like file elsewhere on the host.
    """
    checkpoints_dir = BACKEND_DIR / 'checkpoints'
    raw_path = configured_path or os.environ.get(
        'PXDDI_CHECKPOINT_PATH', 'pxddi_model.pt'
    )
    requested_path = Path(raw_path)
    selected_path = requested_path if requested_path.is_absolute() else checkpoints_dir / requested_path
    selected_path = selected_path.resolve()
    try:
        selected_path.relative_to(checkpoints_dir.resolve())
    except ValueError as error:
        raise RuntimeError(
            'PXDDI_CHECKPOINT_PATH must point inside backend/checkpoints.'
        ) from error
    if not selected_path.is_file():
        raise RuntimeError(f'Configured checkpoint does not exist: {selected_path}')
    return selected_path


def checkpoint_feature_schema(checkpoint_metadata: dict) -> str:
    """Read and validate the graph schema required by a versioned checkpoint."""
    architecture = checkpoint_metadata.get('architecture_version', 'legacy_gat_v1')
    feature_schema = checkpoint_metadata.get('feature_schema', FEATURE_SCHEMA_LEGACY)
    if feature_schema not in {FEATURE_SCHEMA_LEGACY, FEATURE_SCHEMA_RICH}:
        raise RuntimeError(f'Unsupported checkpoint feature schema: {feature_schema!r}.')
    if architecture_uses_edge_features(architecture) and feature_schema != FEATURE_SCHEMA_RICH:
        raise RuntimeError(
            'An edge-aware checkpoint must declare the rich molecular feature schema.'
        )
    if architecture == 'legacy_gat_v1' and feature_schema != FEATURE_SCHEMA_LEGACY:
        raise RuntimeError(
            'A legacy GAT checkpoint must declare the legacy molecular feature schema.'
        )
    return feature_schema


MAX_SMILES_LENGTH = positive_integer_from_environment('PXDDI_MAX_SMILES_LENGTH', 1000)
MAX_MOLECULE_ATOMS = positive_integer_from_environment('PXDDI_MAX_MOLECULE_ATOMS', 200)
MAX_MOLECULE_BONDS = positive_integer_from_environment('PXDDI_MAX_MOLECULE_BONDS', 250)
MAX_REQUEST_BYTES = positive_integer_from_environment('PXDDI_MAX_REQUEST_BYTES', 16 * 1024)
MAX_CONCURRENT_PREDICTIONS = positive_integer_from_environment(
    'PXDDI_MAX_CONCURRENT_PREDICTIONS', 2
)
MAX_CONCURRENT_EXPLANATIONS = positive_integer_from_environment(
    'PXDDI_MAX_CONCURRENT_EXPLANATIONS', 1
)
ENABLE_API_DOCUMENTATION = boolean_from_environment('PXDDI_ENABLE_DOCS', True)
ALLOWED_ORIGINS = configured_origins()
TRUSTED_HOSTS = configured_trusted_hosts()
DEPLOYMENT_MODE = configured_deployment_mode()
API_KEY = configured_api_key(DEPLOYMENT_MODE)
RATE_LIMIT_PER_MINUTE = non_negative_integer_from_environment(
    'PXDDI_RATE_LIMIT_PER_MINUTE', 60 if DEPLOYMENT_MODE == 'production' else 0
)
validate_production_configuration(
    DEPLOYMENT_MODE,
    API_KEY,
    ALLOWED_ORIGINS,
    TRUSTED_HOSTS,
    ENABLE_API_DOCUMENTATION,
    RATE_LIMIT_PER_MINUTE,
)

app = FastAPI(
    title='PxDDI API',
    docs_url='/docs' if ENABLE_API_DOCUMENTATION else None,
    redoc_url=None,
    openapi_url='/openapi.json' if ENABLE_API_DOCUMENTATION else None,
)

# The default permits only the local frontend. Deployments must explicitly set
# PXDDI_ALLOWED_ORIGINS to their own comma-separated frontend origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type', 'X-Request-ID', 'X-API-Key'],
    expose_headers=['X-Request-ID'],
)


class MaxRequestBodyMiddleware:
    """Enforce a small buffered JSON body limit, including chunked requests."""

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reject(scope, receive, send, status_code: int, detail: str) -> None:
        await JSONResponse(status_code=status_code, content={'detail': detail})(
            scope, receive, send
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get('headers', [])}
        content_length = headers.get(b'content-length')
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                await self._reject(
                    scope, receive, send, 400, 'Content-Length must be a valid integer.'
                )
                return
            if declared_size < 0 or declared_size > MAX_REQUEST_BYTES:
                await self._reject(
                    scope,
                    receive,
                    send,
                    413,
                    f'Request body exceeds the {MAX_REQUEST_BYTES}-byte API limit.',
                )
                return

        messages = []
        received_bytes = 0
        while True:
            message = await receive()
            messages.append(message)
            if message['type'] == 'http.request':
                received_bytes += len(message.get('body', b''))
                if received_bytes > MAX_REQUEST_BYTES:
                    await self._reject(
                        scope,
                        receive,
                        send,
                        413,
                        f'Request body exceeds the {MAX_REQUEST_BYTES}-byte API limit.',
                    )
                    return
                if not message.get('more_body', False):
                    break
            elif message['type'] == 'http.disconnect':
                break

        async def replay_body():
            if messages:
                return messages.pop(0)
            return {'type': 'http.disconnect'}

        await self.app(scope, replay_body, send)


class PerClientFixedWindowRateLimiter:
    """Small in-process rate limiter for the protected inference endpoints.

    This limits an accidentally exposed single worker. A production gateway
    must still enforce a shared, durable rate limit across all workers and
    instances; this class intentionally makes no global-distribution claim.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}

    def retry_after_seconds(self, client_key: str) -> int | None:
        if self.requests_per_minute <= 0:
            return None
        now = time.monotonic()
        current_window = int(now // 60)
        with self._lock:
            # Bound the local bookkeeping even if an attacker rotates source
            # addresses. This is not a replacement for gateway protection.
            self._windows = {
                key: value
                for key, value in self._windows.items()
                if value[0] == current_window
            }
            window, count = self._windows.get(client_key, (current_window, 0))
            if window != current_window:
                count = 0
            if count >= self.requests_per_minute:
                return max(1, int(60 - (now % 60)))
            self._windows[client_key] = (current_window, count + 1)
        return None


PER_CLIENT_RATE_LIMITER = PerClientFixedWindowRateLimiter(RATE_LIMIT_PER_MINUTE)
PROTECTED_API_PATHS = frozenset({'/predict', '/explain'})


class RequestGuardMiddleware(BaseHTTPMiddleware):
    """Guard request availability and log non-sensitive request metadata.

    This deliberately never logs a request body, SMILES string, or secret. It
    applies optional local API-key authentication and a per-process rate limit
    to inference routes, but is not a substitute for a gateway rate limit,
    durable audit log, TLS termination, or clinical audit trail.
    """

    @staticmethod
    def _finalize_response(
        request: Request,
        response,
        request_id: str,
        started_at: float,
    ):
        """Attach stable response metadata and log no request-body content."""
        response.headers['X-Request-ID'] = request_id
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        LOGGER.info(
            'api_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f',
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started_at) * 1000,
        )
        return response

    async def dispatch(self, request: Request, call_next):
        supplied_request_id = request.headers.get('x-request-id', '')
        request_id = (
            supplied_request_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else uuid.uuid4().hex
        )
        started_at = time.perf_counter()
        if request.method == 'POST' and request.url.path in PROTECTED_API_PATHS:
            if API_KEY is not None:
                supplied_key = request.headers.get('x-api-key', '')
                if not hmac.compare_digest(supplied_key, API_KEY):
                    response = JSONResponse(
                        status_code=401,
                        content={'detail': 'Valid API credentials are required.'},
                    )
                    return self._finalize_response(request, response, request_id, started_at)
            client_key = request.client.host if request.client else 'unknown_client'
            retry_after_seconds = PER_CLIENT_RATE_LIMITER.retry_after_seconds(client_key)
            if retry_after_seconds is not None:
                response = JSONResponse(
                    status_code=429,
                    content={'detail': 'Rate limit exceeded. Please retry later.'},
                    headers={'Retry-After': str(retry_after_seconds)},
                )
                return self._finalize_response(request, response, request_id, started_at)
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                'api_request_failed request_id=%s method=%s path=%s',
                request_id,
                request.method,
                request.url.path,
            )
            raise

        return self._finalize_response(request, response, request_id, started_at)


# Middleware is added in reverse wrapping order. This gives the request path
# TrustedHost -> RequestGuard -> MaxRequestBody -> CORS -> application.
app.add_middleware(MaxRequestBodyMiddleware)
app.add_middleware(RequestGuardMiddleware)
# Add this after CORS so it is the outermost middleware, including for CORS
# preflight requests. Otherwise a malicious Host header could bypass the check.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

CHECKPOINT_PATH = resolve_checkpoint_path()
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
except Exception:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
model = model_from_checkpoint(checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
MODEL_FEATURE_SCHEMA = checkpoint_feature_schema(checkpoint)
MODEL_REQUIRES_MOTIF_FEATURES = architecture_requires_motif_features(
    checkpoint.get('architecture_version', 'legacy_gat_v1')
)
MODEL_REQUIRES_FP_FEATURES = architecture_requires_fingerprint_features(
    checkpoint.get('architecture_version', 'legacy_gat_v1')
)

NEIGHBOR_MEMORY = None
if checkpoint.get('use_neighbor_memory') and checkpoint.get('neighbor_memory_state'):
    try:
        NEIGHBOR_MEMORY = AuditableNeighborMemory.from_state(checkpoint['neighbor_memory_state'])
        LOGGER.info('AuditableNeighborMemory successfully restored into API.')
    except Exception as exc:
        LOGGER.warning('Could not restore AuditableNeighborMemory: %s', exc)


def compute_ecfp_tensor(smiles: str, radius: int = 2, n_bits: int = 1024) -> torch.Tensor:
    """Compute normalized 1024-bit Morgan fingerprint tensor for candidate models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros((1, n_bits), dtype=torch.float)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    import numpy as np
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr).unsqueeze(0)

DECISION_THRESHOLD = float(checkpoint.get('threshold', 0.5))
CALIBRATION = checkpoint.get('calibration')
CHECKPOINT_SHA256 = file_sha256(CHECKPOINT_PATH)
EXPLANATION_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_EXPLANATIONS)
PREDICTION_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)


def _optional_float(value) -> float | None:
    """Convert checkpoint scalar metadata without making health checks fragile."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if 0 <= numeric <= 1 else None


def stored_validation_evidence(checkpoint_metadata: dict) -> dict:
    """Describe stored validation metadata without presenting it as test evidence."""
    auroc = _optional_float(
        checkpoint_metadata.get('validation_auroc', checkpoint_metadata.get('auroc'))
    )
    epoch_value = checkpoint_metadata.get('epoch')
    epoch = int(epoch_value) if isinstance(epoch_value, (int, float)) else None
    selection_split = checkpoint_metadata.get(
        'model_selection_split',
        'internal validation; exact historical split manifest is unavailable',
    )
    return {
        'status': 'available' if auroc is not None else 'unavailable',
        'auroc': auroc,
        'epoch': epoch,
        'selection_metric': checkpoint_metadata.get('model_selection_metric', 'AUROC'),
        'selection_split': selection_split,
        'note': (
            'This is stored internal model-selection metadata. It is not '
            'transductive test, cold-start, external, or clinical performance.'
        ),
    }


def auxiliary_toxicity_head_status(checkpoint_metadata: dict) -> dict:
    """State whether auxiliary-head training uses the recorded logits contract.

    The interaction forward path remains loadable for legacy checkpoints.  But
    a checkpoint that predates the explicit logits-vs-probability record cannot
    be used as evidence for an auxiliary toxicity-training claim.
    """
    if checkpoint_metadata.get('toxicity_head_output') == 'logits_v1':
        return {
            'status': 'current_logits_training_contract_recorded',
            'note': (
                'The candidate records raw toxicity logits for the auxiliary '
                'BCE-with-logits training objective. Its toxicity output remains '
                'an auxiliary research score, not a clinical probability.'
            ),
        }
    return {
        'status': 'historical_contract_not_recorded',
        'note': (
            'This checkpoint predates the recorded auxiliary-toxicity logits '
            'training contract. It remains a loadable research DDI reference, but '
            'it must not support an auxiliary-toxicity performance claim; retrain '
            'a candidate with the current pipeline before making one.'
        ),
    }


class CheckpointStructuralDomain:
    """Optional checkpoint-backed nearest-training-structure diagnostic.

    Candidate checkpoints produced by a future audited run may carry the
    canonical training-structure reference list. The API can then flag a query
    whose drug structure is far from every training drug. The diagnostic is a
    review guardrail, never a reliability or safety guarantee.
    """

    REQUIRED_METHOD = APPLICABILITY_DOMAIN_METHOD

    def __init__(self, metadata: dict) -> None:
        if metadata.get('method') != self.REQUIRED_METHOD:
            raise ValueError('unsupported applicability-domain method')
        reference_smiles = metadata.get('reference_canonical_smiles')
        if not isinstance(reference_smiles, list) or not reference_smiles:
            raise ValueError('missing canonical training-structure references')
        if len(reference_smiles) > 50_000:
            raise ValueError('too many structural-domain reference structures')
        radius = int(metadata.get('radius', 2))
        num_bits = int(metadata.get('num_bits', 1024))
        minimum_similarity = float(metadata.get('minimum_tanimoto_similarity', 0.4))
        if radius <= 0 or num_bits < 128 or not 0 <= minimum_similarity <= 1:
            raise ValueError('invalid applicability-domain parameters')
        if metadata.get('include_chirality', True) is not True:
            raise ValueError('only chirality-aware ECFP checkpoint domain state is supported')

        canonical_references = []
        for smiles in reference_smiles:
            molecule = Chem.MolFromSmiles(str(smiles).strip())
            if molecule is None:
                raise ValueError('invalid canonical training structure in checkpoint metadata')
            canonical_references.append(
                Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            )
        self.method = self.REQUIRED_METHOD
        self.radius = radius
        self.num_bits = num_bits
        self.minimum_similarity = minimum_similarity
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=num_bits,
            includeChirality=True,
        )
        self.reference_smiles = frozenset(canonical_references)
        self.reference_fingerprints = tuple(
            self.generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            for smiles in sorted(self.reference_smiles)
        )

    def score_smiles(self, smiles: str) -> dict:
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None:
            raise ValueError('SMILES could not be parsed for structural-domain scoring')
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        fingerprint = self.generator.GetFingerprint(molecule)
        similarity = max(DataStructs.BulkTanimotoSimilarity(
            fingerprint, self.reference_fingerprints
        ))
        return {
            'canonical_smiles': canonical,
            'nearest_train_tanimoto': float(similarity),
            'exactly_seen_in_training': canonical in self.reference_smiles,
            'outside_structural_domain': bool(similarity < self.minimum_similarity),
        }


def checkpoint_structural_domain(
    checkpoint_metadata: dict,
) -> tuple[CheckpointStructuralDomain | None, str | None]:
    """Load optional domain state while keeping legacy checkpoints loadable."""
    metadata = checkpoint_metadata.get('applicability_domain')
    if metadata is None:
        return None, 'checkpoint_has_no_structural_domain_reference_set'
    if not isinstance(metadata, dict):
        return None, 'checkpoint_structural_domain_metadata_is_invalid'
    try:
        return CheckpointStructuralDomain(metadata), None
    except (TypeError, ValueError) as error:
        return None, f'checkpoint_structural_domain_unavailable: {error}'


STRUCTURAL_DOMAIN, STRUCTURAL_DOMAIN_ERROR = checkpoint_structural_domain(checkpoint)

print(
    'Loaded model. '
    f"AUROC={checkpoint.get('auroc')}, threshold={DECISION_THRESHOLD}, "
    f'checkpoint_sha256={CHECKPOINT_SHA256}'
)


class DDIRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    smiles_a: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    smiles_b: str = Field(min_length=1, max_length=MAX_SMILES_LENGTH)
    age_band: Optional[StrictInt] = None
    sex: Optional[StrictInt] = None
    comorbidities: Optional[List[StrictInt]] = None

    @field_validator('smiles_a', 'smiles_b', mode='before')
    @classmethod
    def strip_smiles(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator('age_band')
    @classmethod
    def validate_age_band(cls, value):
        if value is not None:
            raise ValueError('Patient context is not supported by this model.')
        return value

    @field_validator('sex')
    @classmethod
    def validate_sex(cls, value):
        if value is not None:
            raise ValueError('Patient context is not supported by this model.')
        return value

    @field_validator('comorbidities')
    @classmethod
    def validate_comorbidities(cls, value):
        if value is not None:
            raise ValueError('Patient context is not supported by this model.')
        return value


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, error: RequestValidationError
):
    """Return stable validation errors without echoing submitted structures."""
    safe_errors = [
        {
            'loc': list(item['loc']),
            'msg': item['msg'],
            'type': item['type'],
        }
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={'detail': safe_errors})


def build_drug_batches(req: DDIRequest):
    """Build bounded molecular batches shared by prediction and explanation."""
    graph_a = smiles_to_graph(
        req.smiles_a,
        feature_schema=MODEL_FEATURE_SCHEMA,
        include_motif_features=MODEL_REQUIRES_MOTIF_FEATURES,
    )
    graph_b = smiles_to_graph(
        req.smiles_b,
        feature_schema=MODEL_FEATURE_SCHEMA,
        include_motif_features=MODEL_REQUIRES_MOTIF_FEATURES,
    )
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

    if MODEL_REQUIRES_FP_FEATURES:
        graph_a.fingerprint_features = compute_ecfp_tensor(req.smiles_a)
        graph_b.fingerprint_features = compute_ecfp_tensor(req.smiles_b)

    return Batch.from_data_list([graph_a]), Batch.from_data_list([graph_b])


def toxicity_response(smiles: str, score: float):
    """Separate an auxiliary model score from FAERS-label coverage."""
    training_label_available = is_toxicity_known(smiles)
    coverage_note = (
        'A matched FAERS-derived toxicity training label is available for this structure.'
        if training_label_available
        else 'No matched FAERS-derived toxicity training label is available for this structure.'
    )
    return {
        'score': score,
        'model_score_note': (
            'This is an auxiliary research-model score, not a clinical toxicity '
            'probability. FAERS coverage describes supervision availability only.'
        ),
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


def uncertainty_response(probability: float) -> dict:
    """Return a saved conformal set only when the checkpoint carries its state."""
    conformal = checkpoint.get('conformal')
    if not isinstance(conformal, dict) or conformal.get('status') != 'fitted':
        return {
            'status': 'not_available',
            'method': None,
            'abstain': None,
            'prediction_set': None,
            'note': (
                'This checkpoint has no saved validation-only conformal state. '
                'Do not infer uncertainty from the score alone.'
            ),
        }
    try:
        prediction_sets = conformal_prediction_sets([probability], conformal)
    except (KeyError, TypeError, ValueError) as error:
        return {
            'status': 'not_available_invalid_checkpoint_state',
            'method': conformal.get('method'),
            'abstain': None,
            'prediction_set': None,
            'note': f'Checkpoint conformal state is incomplete: {error}',
        }
    return {
        'status': 'available_internal_validation_only',
        'method': conformal.get('method'),
        'alpha': conformal.get('alpha'),
        'fitted_on': conformal.get('fitted_on'),
        'prediction_set': str(prediction_sets['prediction_set'][0]),
        'abstain': bool(prediction_sets['abstain'][0]),
        'no_interaction_p_value': float(prediction_sets['no_interaction_p_value'][0]),
        'interaction_p_value': float(prediction_sets['interaction_p_value'][0]),
        'note': conformal.get(
            'interpretation_warning',
            'This is internal validation uncertainty only, not clinical confidence.',
        ),
    }


def checkpoint_uncertainty_status() -> str:
    """Expose availability in health without representing it as a prediction."""
    return uncertainty_response(0.5)['status']


def structural_domain_response(smiles_a: str, smiles_b: str) -> dict:
    """Return a checkpoint-backed OOD flag without treating it as safety evidence."""
    if STRUCTURAL_DOMAIN is None:
        return {
            'status': 'not_available',
            'method': None,
            'outside_structural_domain': None,
            'note': (
                'This checkpoint has no saved training-structure reference set, so '
                'the API cannot assess structural out-of-domain status. '
                f'Reason: {STRUCTURAL_DOMAIN_ERROR}.'
            ),
        }
    try:
        drug_a = STRUCTURAL_DOMAIN.score_smiles(smiles_a)
        drug_b = STRUCTURAL_DOMAIN.score_smiles(smiles_b)
    except ValueError as error:
        return {
            'status': 'not_available_invalid_query',
            'method': STRUCTURAL_DOMAIN.method,
            'outside_structural_domain': None,
            'note': f'Could not assess the supplied structures: {error}',
        }
    pair_minimum = min(
        drug_a['nearest_train_tanimoto'], drug_b['nearest_train_tanimoto']
    )
    outside = bool(pair_minimum < STRUCTURAL_DOMAIN.minimum_similarity)
    return {
        'status': 'available_structure_similarity_diagnostic',
        'method': STRUCTURAL_DOMAIN.method,
        'minimum_tanimoto_similarity': STRUCTURAL_DOMAIN.minimum_similarity,
        'drug_a': drug_a,
        'drug_b': drug_b,
        'pair_minimum_nearest_train_tanimoto': pair_minimum,
        'outside_structural_domain': outside,
        'note': (
            'This is a nearest-training-drug structural similarity diagnostic only. '
            'It does not measure pair novelty, prove prediction reliability, or make '
            'an unreported pair safe.'
        ),
    }


def explanation_response() -> dict:
    """Describe whether this architecture has a compatible explanation path."""
    if getattr(model, 'architecture_version', None) != 'legacy_gat_v1':
        return {
            'available': False,
            'endpoint': None,
            'note': (
                'This non-legacy candidate has no validated API explanation method yet. '
                'Do not interpret an embedding attribution from the legacy method as an '
                'edge-aware or motif-aware pair-risk explanation.'
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
    if not PREDICTION_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail='The prediction service is busy. Please retry shortly.',
        )
    try:
        batch_a, batch_b = build_drug_batches(req)

        # Patient context is rejected at request validation because this model
        # has not been trained with linked patient, exposure, and outcome data.
        patient_context_note = (
            'Patient-specific fields are not accepted. The patient-conditioning '
            'module is not trained on linked patient-outcome data.'
        )

        memory_features = None
        auditable_evidence = None
        if NEIGHBOR_MEMORY is not None:
            mem_res = NEIGHBOR_MEMORY.score_pair_memory(req.smiles_a, req.smiles_b)
            memory_features = torch.tensor([[
                mem_res['neighbor_density'],
                mem_res['max_support'],
                mem_res['structural_confidence'],
            ]], dtype=torch.float)
            auditable_evidence = {
                'status': 'available',
                'method': 'tanimoto_knn_training_memory_v1',
                'neighbor_interaction_density': round(float(mem_res['neighbor_density']), 4),
                'max_supported_interaction': round(float(mem_res['max_support']), 4),
                'structural_confidence': round(float(mem_res['structural_confidence']), 4),
                'audit_trail': mem_res['audit_trail'],
            }
        else:
            auditable_evidence = {
                'status': 'not_available',
                'method': None,
                'note': 'This checkpoint does not carry an active training neighbor memory.',
            }

        with torch.inference_mode():
            if getattr(model, 'use_neighbor_memory', False):
                risk, tox_a, tox_b = model(batch_a, batch_b, patient=None, memory_features=memory_features)
            else:
                risk, tox_a, tox_b = model(batch_a, batch_b, patient=None)

        raw_risk_score = float(torch.sigmoid(risk))
        risk_score = float(apply_calibrator([raw_risk_score], CALIBRATION)[0])
        calibration_response = risk_calibration_response()
        explanation = explanation_response()
        uncertainty = uncertainty_response(risk_score)
        structural_domain = structural_domain_response(req.smiles_a, req.smiles_b)
        return {
            'disclaimer': 'Research prototype output. Not clinical advice. Not FDA/regulatory reviewed.',
            'model_architecture': checkpoint.get('architecture_version', 'legacy_gat_v1'),
            'stored_validation_evidence': stored_validation_evidence(checkpoint),
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
            'prediction_uncertainty': uncertainty,
            'structural_applicability_domain': structural_domain,
            'auditable_evidence': auditable_evidence,
            'interaction_predicted': risk_score >= DECISION_THRESHOLD,
            'decision_threshold_used': DECISION_THRESHOLD,
            'patient_context_applied': False,
            'patient_context_note': patient_context_note,
            'drug_a_toxicity': toxicity_response(
                req.smiles_a, float(torch.sigmoid(tox_a))
            ),
            'drug_b_toxicity': toxicity_response(
                req.smiles_b, float(torch.sigmoid(tox_b))
            ),
            'explanation_available_at': explanation['endpoint'],
            'explanation_note': explanation['note'],
        }
    finally:
        PREDICTION_SEMAPHORE.release()


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
    # An explanation performs several model forwards. It must consume the same
    # finite inference capacity as a prediction, otherwise one explanation can
    # still exhaust memory while the prediction semaphore is full.
    if not PREDICTION_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail='The inference service is busy. Please retry shortly.',
        )
    if not EXPLANATION_SEMAPHORE.acquire(blocking=False):
        PREDICTION_SEMAPHORE.release()
        raise HTTPException(
            status_code=429,
            detail='An explanation is already running. Please retry shortly.',
        )

    try:
        batch_a, batch_b = build_drug_batches(req)
        explanation = full_explanation_pipeline(
            model, batch_a, req.smiles_a, batch_b, req.smiles_b
        )
    finally:
        EXPLANATION_SEMAPHORE.release()
        PREDICTION_SEMAPHORE.release()

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
        'model_feature_schema': MODEL_FEATURE_SCHEMA,
        'model_requires_motif_features': MODEL_REQUIRES_MOTIF_FEATURES,
        'score_calibration_status': risk_calibration_response()['status'],
        'conformal_uncertainty_status': checkpoint_uncertainty_status(),
        'structural_applicability_domain_status': (
            'available_structure_similarity_diagnostic'
            if STRUCTURAL_DOMAIN is not None else 'not_available'
        ),
        'structural_applicability_domain_error': STRUCTURAL_DOMAIN_ERROR,
        'explanation_status': 'available' if explanation_response()['available'] else 'not_available',
        'stored_validation_evidence': stored_validation_evidence(checkpoint),
        'auxiliary_toxicity_head_status': auxiliary_toxicity_head_status(checkpoint),
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
        'max_request_bytes': MAX_REQUEST_BYTES,
        'max_concurrent_predictions': MAX_CONCURRENT_PREDICTIONS,
        'max_concurrent_explanations': MAX_CONCURRENT_EXPLANATIONS,
        'api_documentation_enabled': ENABLE_API_DOCUMENTATION,
        'deployment_mode': DEPLOYMENT_MODE,
        'api_key_protection_enabled': API_KEY is not None,
        'per_client_rate_limit_per_minute': RATE_LIMIT_PER_MINUTE,
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
