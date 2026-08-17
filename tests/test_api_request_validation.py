"""Tests for API request validation independent of model inference."""

import pytest
from pydantic import ValidationError

from backend.main import DDIRequest


def test_comorbidities_must_be_binary():
    with pytest.raises(ValidationError, match='only 0 or 1'):
        DDIRequest(
            smiles_a='CCO',
            smiles_b='CCN',
            comorbidities=[2] * 10,
        )


def test_smiles_length_is_limited():
    with pytest.raises(ValidationError):
        DDIRequest(smiles_a='C' * 1001, smiles_b='CCN')
