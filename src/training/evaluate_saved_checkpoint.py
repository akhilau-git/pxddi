"""Dedicated evaluation script for one explicit, provenance-checked checkpoint.

This script SKIPS ALL TRAINING and directly resumes the post-training evaluation
pipeline (Platt calibration, threshold optimization, conformal prediction sets,
applicability domain audits, and S1/S2/Transductive test set scoring).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))

def _checkpoint_path_from_environment() -> Path:
    raw_path = os.environ.get('PXDDI_CHECKPOINT_PATH')
    if not raw_path:
        raise ValueError(
            'Set PXDDI_CHECKPOINT_PATH to the one checkpoint you intend to evaluate. '
            'This script does not search fallback locations.'
        )
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f'PXDDI_CHECKPOINT_PATH does not exist: {path}')
    return path


def _set_or_check_environment(name: str, expected: object) -> None:
    value = str(expected).lower() if isinstance(expected, bool) else str(expected)
    existing = os.environ.get(name)
    if existing is not None and existing != value:
        raise ValueError(
            f'{name}={existing!r} conflicts with the selected checkpoint value {value!r}.'
        )
    os.environ[name] = value


def configure_evaluation_from_checkpoint(checkpoint_path: Path) -> None:
    """Configure the pipeline from verified checkpoint provenance only."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('Selected checkpoint is not a metadata dictionary.')
    provenance = checkpoint.get('evaluation_provenance')
    required = {
        'model_architecture': 'PXDDI_MODEL_ARCHITECTURE',
        'hidden_channels': 'PXDDI_HIDDEN_CHANNELS',
        'data_cap': 'PXDDI_DATA_CAP',
        'model_seed': 'PXDDI_MODEL_SEED',
        'split_seed': 'PXDDI_SPLIT_SEED',
        'negative_sampling_strategy': 'PXDDI_NEGATIVE_SAMPLING_STRATEGY',
        'negative_sampling_protocol': 'PXDDI_NEGATIVE_SAMPLING_PROTOCOL',
        'use_toxicity_pair_features': 'PXDDI_USE_TOXICITY_PAIR_FEATURES',
        'toxicity_loss_weight': 'PXDDI_TOXICITY_LOSS_WEIGHT',
        'model_selection_validation_fraction': 'PXDDI_MODEL_SELECTION_VALIDATION_FRACTION',
    }
    if not isinstance(provenance, dict):
        raise ValueError(
            'Selected checkpoint lacks audited evaluation provenance and cannot be '
            'safely re-evaluated. Train it again with the current pipeline.'
        )
    missing = sorted(set(required).difference(provenance))
    if missing:
        raise ValueError(
            'Selected checkpoint has incomplete evaluation provenance; missing: '
            f'{missing}.'
        )
    for checkpoint_key, environment_name in required.items():
        _set_or_check_environment(environment_name, provenance[checkpoint_key])
    os.environ['PXDDI_EVALUATE_ONLY'] = '1'


CHECKPOINT_PATH = _checkpoint_path_from_environment()
configure_evaluation_from_checkpoint(CHECKPOINT_PATH)

from src.training.train_full_pipeline_v2 import main

if __name__ == "__main__":
    print("=" * 72)
    print("  PxDDI: Provenance-Checked Evaluation from Saved Checkpoint")
    print("  Training Status: ALREADY COMPLETED (0 training epochs will be run)")
    print("=" * 72)
    main()
