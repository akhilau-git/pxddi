"""Dedicated evaluation script for trained AuditDDI checkpoints.

This script SKIPS ALL TRAINING and directly resumes the post-training evaluation
pipeline (Platt calibration, threshold optimization, conformal prediction sets,
applicability domain audits, and S1/S2/Transductive test set scoring).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))

# Ensure PXDDI_EVALUATE_ONLY is active before importing the training pipeline
os.environ["PXDDI_EVALUATE_ONLY"] = "1"

# Default to AuditDDI memory fusion architecture if not specified
if "PXDDI_MODEL_ARCHITECTURE" not in os.environ:
    os.environ["PXDDI_MODEL_ARCHITECTURE"] = "auditddi_memory_fusion_v1"
if "PXDDI_EXPERIMENT_NAME" not in os.environ:
    os.environ["PXDDI_EXPERIMENT_NAME"] = "auditddi_memory_fusion_multitask"
if "PXDDI_STUDY_ID" not in os.environ:
    os.environ["PXDDI_STUDY_ID"] = "study_auditddi_screening"

from src.training.train_full_pipeline_v2 import main

if __name__ == "__main__":
    print("=" * 72)
    print("  AuditDDI: Resuming Evaluation from Saved Checkpoint")
    print("  Training Status: ALREADY COMPLETED (0 training epochs will be run)")
    print("=" * 72)
    main()
