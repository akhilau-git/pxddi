"""Tests for API request validation independent of model inference."""

import pytest
from pydantic import ValidationError

from backend.main import DDIRequest


def test_patient_context_fields_are_rejected_until_supported_by_training_data():
    with pytest.raises(ValidationError, match='not supported'):
        DDIRequest(
            smiles_a='CCO',
            smiles_b='CCN',
            comorbidities=[2] * 10,
        )


def test_smiles_length_is_limited():
    with pytest.raises(ValidationError):
        DDIRequest(smiles_a='C' * 1001, smiles_b='CCN')
