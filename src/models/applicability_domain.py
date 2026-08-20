"""Lightweight molecular applicability-domain flags for PxDDI evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


APPLICABILITY_DOMAIN_METHOD = 'nearest_train_ecfp_tanimoto_v1'


class MorganApplicabilityDomain:
    """Compare each queried drug with the unique drugs in the training split.

    This is a structural-neighbour flag, not a DDI-pair distribution detector,
    uncertainty guarantee, or clinical eligibility rule.  A pair is flagged if
    either of its two drugs lacks a sufficiently similar training molecule.
    """

    def __init__(
        self,
        radius: int = 2,
        num_bits: int = 1024,
        minimum_similarity: float = 0.4,
    ) -> None:
        if radius <= 0:
            raise ValueError('radius must be positive.')
        if num_bits < 128:
            raise ValueError('num_bits must be at least 128.')
        if not 0 <= minimum_similarity <= 1:
            raise ValueError('minimum_similarity must lie between zero and one.')
        self.radius = radius
        self.num_bits = num_bits
        self.minimum_similarity = minimum_similarity
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=num_bits, includeChirality=True
        )
        self._reference_fingerprints = []
        self._reference_smiles: set[str] = set()
        self._query_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _canonicalize(smiles: str) -> str:
        molecule = Chem.MolFromSmiles(str(smiles).strip())
        if molecule is None:
            raise ValueError(f'Cannot make an applicability-domain fingerprint from {smiles!r}.')
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    def _fingerprint(self, canonical_smiles: str):
        molecule = Chem.MolFromSmiles(canonical_smiles)
        if molecule is None:  # Defensive after _canonicalize.
            raise ValueError(f'Cannot parse canonical SMILES {canonical_smiles!r}.')
        return self.generator.GetFingerprint(molecule)

    def fit(self, train_smiles) -> dict[str, Any]:
        canonical_smiles = sorted({self._canonicalize(smiles) for smiles in train_smiles})
        if not canonical_smiles:
            raise ValueError('Applicability-domain reference requires at least one training SMILES.')
        self._reference_smiles = set(canonical_smiles)
        self._reference_fingerprints = [
            self._fingerprint(smiles) for smiles in canonical_smiles
        ]
        self._query_cache = {}
        return self.summary()

    def score_smiles(self, smiles: str) -> dict[str, Any]:
        if not self._reference_fingerprints:
            raise RuntimeError('Call fit() before scoring applicability domain.')
        canonical_smiles = self._canonicalize(smiles)
        if canonical_smiles in self._query_cache:
            return dict(self._query_cache[canonical_smiles])
        fingerprint = self._fingerprint(canonical_smiles)
        similarity = max(DataStructs.BulkTanimotoSimilarity(
            fingerprint, self._reference_fingerprints
        ))
        result = {
            'canonical_smiles': canonical_smiles,
            'nearest_train_tanimoto': float(similarity),
            'exactly_seen_in_training': canonical_smiles in self._reference_smiles,
            'outside_structural_domain': bool(similarity < self.minimum_similarity),
        }
        self._query_cache[canonical_smiles] = result
        return dict(result)

    def score_pairs(self, source_smiles, target_smiles) -> dict[str, np.ndarray]:
        if len(source_smiles) != len(target_smiles):
            raise ValueError('source_smiles and target_smiles must have equal lengths.')
        sources = [self.score_smiles(smiles) for smiles in source_smiles]
        targets = [self.score_smiles(smiles) for smiles in target_smiles]
        source_similarity = np.asarray(
            [entry['nearest_train_tanimoto'] for entry in sources], dtype=float
        )
        target_similarity = np.asarray(
            [entry['nearest_train_tanimoto'] for entry in targets], dtype=float
        )
        pair_minimum = np.minimum(source_similarity, target_similarity)
        return {
            'source_nearest_train_tanimoto': source_similarity,
            'target_nearest_train_tanimoto': target_similarity,
            'pair_minimum_nearest_train_tanimoto': pair_minimum,
            'source_exactly_seen_in_training': np.asarray(
                [entry['exactly_seen_in_training'] for entry in sources], dtype=bool
            ),
            'target_exactly_seen_in_training': np.asarray(
                [entry['exactly_seen_in_training'] for entry in targets], dtype=bool
            ),
            'structural_ood_flag': pair_minimum < self.minimum_similarity,
        }

    def summary(self) -> dict[str, Any]:
        return {
            'method': APPLICABILITY_DOMAIN_METHOD,
            'fingerprint': 'ECFP/Morgan',
            'radius': self.radius,
            'num_bits': self.num_bits,
            'include_chirality': True,
            'minimum_tanimoto_similarity': self.minimum_similarity,
            'reference_unique_training_drugs': len(self._reference_fingerprints),
            'interpretation_warning': (
                'This is nearest-training-drug structural similarity only. It does not '
                'measure DDI-pair novelty, prove prediction reliability, or establish '
                'clinical applicability.'
            ),
        }
