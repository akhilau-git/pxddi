"""Auditable Neighbor Interaction Memory for AuditDDI Cold-Start Prediction.

In inductive cold-start DDI prediction (especially S1, where neither drug has
been observed in training), parametric neural networks suffer from severe
out-of-distribution extrapolation failure, often collapsing to near-random guessing.

AuditDDI addresses this by coupling the molecular graph representation with an
Auditable Neighbor Interaction Memory. For any novel drug pair (A, B):
1. The memory retrieves the top-K structural analogs of Drug A and Drug B from the
   training graph based on Morgan fingerprint Tanimoto similarity.
2. It queries the empirical interaction sub-graph among those retrieved analogs.
3. It computes continuous evidence scores:
   - neighbor_interaction_density: similarity-weighted interaction rate across analog pairs
   - max_supported_interaction: highest similarity-weighted analog interaction
   - analog_structural_confidence: average proximity of the pair to training space
4. It emits an explicit, human-auditable evidence trail detailing the exact nearest
   analogs and their observed interactions.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


AUDITABLE_MEMORY_METHOD = 'tanimoto_knn_interaction_memory_v1'


class AuditableNeighborMemory:
    """Non-parametric structural memory that retrieves analog interaction evidence."""

    def __init__(
        self,
        k_neighbors: int = 5,
        radius: int = 2,
        num_bits: int = 1024,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError('k_neighbors must be at least 1.')
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.num_bits = num_bits
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=num_bits, includeChirality=True
        )
        self.training_smiles: list[str] = []
        self.training_fps: list[Any] = []
        self._smiles_to_idx: dict[str, int] = {}
        self._interaction_matrix: np.ndarray | None = None
        self._fitted = False
        self._top_k_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _get_drug_neighbors(self, canonical_smiles: str) -> tuple[np.ndarray, np.ndarray]:
        if canonical_smiles in self._top_k_cache:
            return self._top_k_cache[canonical_smiles]
        fp = self._fingerprint(canonical_smiles)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, self.training_fps))
        k = min(self.k_neighbors, len(self.training_smiles))
        top_k = np.argsort(sims)[-k:][::-1]
        weights = sims[top_k]
        self._top_k_cache[canonical_smiles] = (top_k, weights)
        return top_k, weights

    @staticmethod
    def _canonical(smiles: str) -> str:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            raise ValueError(f'Invalid SMILES for memory retrieval: {smiles!r}')
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _fingerprint(self, canonical_smiles: str):
        mol = Chem.MolFromSmiles(canonical_smiles)
        return self._generator.GetFingerprint(mol)

    def fit(
        self,
        train_smiles_a: list[str],
        train_smiles_b: list[str],
        train_labels: list[float] | np.ndarray,
    ) -> dict[str, Any]:
        """Index all unique training drugs and construct the empirical interaction matrix."""
        if len(train_smiles_a) != len(train_smiles_b) or len(train_smiles_a) != len(train_labels):
            raise ValueError('train_smiles_a, train_smiles_b, and train_labels must match in length.')

        self._top_k_cache.clear()
        unique_smiles_set: set[str] = set()
        clean_pairs: list[tuple[str, str, float]] = []
        for sa, sb, y in zip(train_smiles_a, train_smiles_b, train_labels):
            try:
                ca, cb = self._canonical(sa), self._canonical(sb)
                unique_smiles_set.add(ca)
                unique_smiles_set.add(cb)
                clean_pairs.append((ca, cb, float(y)))
            except Exception:
                continue

        self.training_smiles = sorted(unique_smiles_set)
        self._smiles_to_idx = {s: i for i, s in enumerate(self.training_smiles)}
        self.training_fps = [self._fingerprint(s) for s in self.training_smiles]

        n_drugs = len(self.training_smiles)
        self._interaction_matrix = np.zeros((n_drugs, n_drugs), dtype=np.float32)

        # Symmetrically populate the training interaction matrix
        for ca, cb, y in clean_pairs:
            idx_a = self._smiles_to_idx[ca]
            idx_b = self._smiles_to_idx[cb]
            self._interaction_matrix[idx_a, idx_b] = max(self._interaction_matrix[idx_a, idx_b], y)
            self._interaction_matrix[idx_b, idx_a] = max(self._interaction_matrix[idx_b, idx_a], y)

        self._fitted = True
        return {
            'method': AUDITABLE_MEMORY_METHOD,
            'unique_training_drugs': n_drugs,
            'k_neighbors': self.k_neighbors,
            'matrix_sparsity': float(np.mean(self._interaction_matrix > 0)),
        }

    def score_pair_memory(
        self,
        smiles_a: str,
        smiles_b: str,
        *,
        exclude_query_pair: bool = False,
    ) -> dict[str, Any]:
        """Retrieve nearest analogs and compute auditable interaction evidence.

        ``exclude_query_pair`` is required when deriving features for a row
        that was used to fit this memory.  Without it, a training pair can
        retrieve its own interaction-matrix entry through exact structural
        matches and thereby expose its label to the model.  Evaluation queries
        are not in the fitted matrix, so they retain the normal retrieval path.
        """
        if not self._fitted or self._interaction_matrix is None:
            raise RuntimeError('AuditableNeighborMemory must be fitted before scoring.')

        ca = self._canonical(smiles_a)
        cb = self._canonical(smiles_b)

        top_k_a, weights_a = self._get_drug_neighbors(ca)
        top_k_b, weights_b = self._get_drug_neighbors(cb)

        sub_matrix = self._interaction_matrix[np.ix_(top_k_a, top_k_b)]
        query_pair_excluded = False
        if exclude_query_pair:
            source_index = self._smiles_to_idx.get(ca)
            target_index = self._smiles_to_idx.get(cb)
            if source_index is not None and target_index is not None:
                source_positions = np.flatnonzero(top_k_a == source_index)
                target_positions = np.flatnonzero(top_k_b == target_index)
                if len(source_positions) and len(target_positions):
                    # Copy before zeroing so feature generation never mutates
                    # the fitted evidence matrix shared by later rows.
                    sub_matrix = sub_matrix.copy()
                    sub_matrix[np.ix_(source_positions, target_positions)] = 0.0
                    query_pair_excluded = True
        outer_weights = np.outer(weights_a, weights_b)
        sum_weights = float(np.sum(outer_weights))

        if sum_weights > 0:
            weighted_density = float(np.sum(outer_weights * sub_matrix) / sum_weights)
            max_support = float(np.max(outer_weights * sub_matrix))
        else:
            weighted_density = 0.5
            max_support = 0.0

        confidence = float(0.5 * (weights_a[0] + weights_b[0]))

        # Construct human-readable audit trail
        audit_trail = []
        for i, idx_a in enumerate(top_k_a[:3]):
            for j, idx_b in enumerate(top_k_b[:3]):
                audit_trail.append({
                    'analog_a': self.training_smiles[idx_a],
                    'analog_b': self.training_smiles[idx_b],
                    'similarity_a': round(float(weights_a[i]), 3),
                    'similarity_b': round(float(weights_b[j]), 3),
                    'observed_interaction': float(sub_matrix[i, j]),
                })

        return {
            'neighbor_density': weighted_density,
            'max_support': max_support,
            'structural_confidence': confidence,
            'query_pair_excluded': query_pair_excluded,
            'audit_trail': audit_trail,
        }

    def score_batch(
        self,
        smiles_list_a: list[str],
        smiles_list_b: list[str],
        *,
        exclude_query_pairs: bool = False,
    ) -> np.ndarray:
        """Vectorized extraction of memory features [density, max_support, confidence]."""
        n = len(smiles_list_a)
        features = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            try:
                res = self.score_pair_memory(
                    smiles_list_a[i],
                    smiles_list_b[i],
                    exclude_query_pair=exclude_query_pairs,
                )
                features[i, 0] = res['neighbor_density']
                features[i, 1] = res['max_support']
                features[i, 2] = res['structural_confidence']
            except Exception:
                features[i, 0] = 0.5
                features[i, 1] = 0.0
                features[i, 2] = 0.0
        return features

    def export_state(self) -> dict[str, Any]:
        """Export state bundle for reproducible checkpoint serialization."""
        if not self._fitted or self._interaction_matrix is None:
            raise RuntimeError('AuditableNeighborMemory must be fitted before exporting state.')
        import torch
        return {
            'method': AUDITABLE_MEMORY_METHOD,
            'k_neighbors': self.k_neighbors,
            'radius': self.radius,
            'num_bits': self.num_bits,
            'training_smiles': self.training_smiles,
            'interaction_matrix': torch.from_numpy(self._interaction_matrix),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore memory from an exported state bundle."""
        self.k_neighbors = int(state.get('k_neighbors', 5))
        self.radius = int(state.get('radius', 2))
        self.num_bits = int(state.get('num_bits', 1024))
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.num_bits, includeChirality=True
        )
        self.training_smiles = list(state['training_smiles'])
        self._smiles_to_idx = {s: i for i, s in enumerate(self.training_smiles)}
        self.training_fps = [self._fingerprint(s) for s in self.training_smiles]
        mat = state['interaction_matrix']
        import torch
        if isinstance(mat, torch.Tensor):
            self._interaction_matrix = mat.cpu().numpy()
        else:
            self._interaction_matrix = np.asarray(mat, dtype=np.float32)
        self._top_k_cache = {}
        self._fitted = True

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> AuditableNeighborMemory:
        """Factory constructor restoring memory directly from state dictionary."""
        instance = cls(
            k_neighbors=int(state.get('k_neighbors', 5)),
            radius=int(state.get('radius', 2)),
            num_bits=int(state.get('num_bits', 1024)),
        )
        instance.load_state(state)
        return instance
