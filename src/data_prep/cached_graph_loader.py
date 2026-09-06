"""High-performance cached graph and multi-modal mini-batch loader for AuditDDI.

Pre-computes and caches:
1. PyG molecular graphs (Rich atom/bond schema) in RAM.
2. 1024-bit Morgan ECFP fingerprints.
3. 50-dim multi-hot PharmGKB pharmacogenomic gene/enzyme vectors.
4. Clinical FAERS toxicity scores.

Eliminates repetitive RDKit parsing bottlenecks during PyTorch training loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader

from .prepare_twosides import FEATURE_SCHEMA_RICH, smiles_to_graph

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


class MolecularCache:
    """In-memory cache for all unique drugs in the master graph."""

    def __init__(self, gene_dim: int = 50) -> None:
        self.gene_dim = gene_dim
        self.graphs: dict[str, Data] = {}
        self.fingerprints: dict[str, torch.Tensor] = {}
        self.gene_vectors: dict[str, torch.Tensor] = {}
        self.gene_masks: dict[str, torch.Tensor] = {}
        self.toxicity_scalars: dict[str, torch.Tensor] = {}
        self.toxicity_masks: dict[str, torch.Tensor] = {}

    def register_drug(
        self,
        smiles: str,
        gene_vector: list[int] | list[float] | None = None,
        toxicity_score: float | None = None,
    ) -> bool:
        """Parse and cache a single drug's multi-modal representations."""
        if smiles in self.graphs:
            return True

        graph = smiles_to_graph(
            smiles,
            feature_schema=FEATURE_SCHEMA_RICH,
            include_fingerprint_features=True,
        )
        if graph is None:
            return False

        self.graphs[smiles] = graph

        # ECFP Fingerprint (1024-bit)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            bit_vec = _MORGAN_GEN.GetFingerprint(mol)
            np_vec = np.zeros(1024, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bit_vec, np_vec)
            self.fingerprints[smiles] = torch.from_numpy(np_vec)
        else:
            self.fingerprints[smiles] = torch.zeros(1024, dtype=torch.float32)

        # PharmGKB Gene Vector (Multi-Hot)
        if gene_vector and len(gene_vector) == self.gene_dim:
            self.gene_vectors[smiles] = torch.tensor(gene_vector, dtype=torch.float32)
            self.gene_masks[smiles] = torch.tensor(1.0, dtype=torch.float32)
        else:
            self.gene_vectors[smiles] = torch.zeros(self.gene_dim, dtype=torch.float32)
            self.gene_masks[smiles] = torch.tensor(0.0, dtype=torch.float32)

        # FAERS Clinical Toxicity Score
        if toxicity_score is not None and not pd.isna(toxicity_score):
            self.toxicity_scalars[smiles] = torch.tensor(float(toxicity_score), dtype=torch.float32)
            self.toxicity_masks[smiles] = torch.tensor(1.0, dtype=torch.float32)
        else:
            self.toxicity_scalars[smiles] = torch.tensor(0.0, dtype=torch.float32)
            self.toxicity_masks[smiles] = torch.tensor(0.0, dtype=torch.float32)

        return True

    def populate_from_master_nodes(self, master_nodes_path: str | Path) -> int:
        """Pre-populate the entire cache from master_drug_nodes.csv."""
        df_nodes = pd.read_csv(master_nodes_path)
        count = 0
        for _, row in df_nodes.iterrows():
            smi = str(row['drug_id']).strip()
            gvec_raw = row.get('gene_vector_json')
            gvec = None
            if pd.notna(gvec_raw):
                try:
                    gvec = json.loads(gvec_raw) if isinstance(gvec_raw, str) else list(gvec_raw)
                except Exception:
                    gvec = None

            tox = row.get('toxicity_score')
            tox_val = float(tox) if pd.notna(tox) else None

            if self.register_drug(smi, gene_vector=gvec, toxicity_score=tox_val):
                count += 1
        print(f'MolecularCache populated: {count} drugs cached with graphs, ECFP, genes, and toxicity.')
        return count


class CachedDDIPairDataset(Dataset):
    """Fast indexed dataset referencing the MolecularCache."""

    def __init__(
        self,
        edges_df: pd.DataFrame,
        molecular_cache: MolecularCache,
        source_col: str = 'drug_a_id',
        target_col: str = 'drug_b_id',
        label_col: str = 'label',
    ) -> None:
        super().__init__()
        self.cache = molecular_cache
        self.samples: list[tuple[str, str, float]] = []

        # Validate that edges have registered drugs
        for _, row in edges_df.iterrows():
            sa, sb = str(row[source_col]).strip(), str(row[target_col]).strip()
            if sa in self.cache.graphs and sb in self.cache.graphs:
                raw_label = row.get(label_col, 1.0)
                lbl = 1.0 if (raw_label is True or raw_label == 1.0 or str(raw_label).lower() in {'1', 'true'}) else 0.0
                self.samples.append((sa, sb, lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sa, sb, lbl = self.samples[idx]
        return {
            'graph_a': self.cache.graphs[sa],
            'graph_b': self.cache.graphs[sb],
            'fp_a': self.cache.fingerprints[sa],
            'fp_b': self.cache.fingerprints[sb],
            'gene_a': self.cache.gene_vectors[sa],
            'gene_b': self.cache.gene_vectors[sb],
            'gene_mask_a': self.cache.gene_masks[sa],
            'gene_mask_b': self.cache.gene_masks[sb],
            'tox_a': self.cache.toxicity_scalars[sa],
            'tox_b': self.cache.toxicity_scalars[sb],
            'tox_mask_a': self.cache.toxicity_masks[sa],
            'tox_mask_b': self.cache.toxicity_masks[sb],
            'label': torch.tensor(lbl, dtype=torch.float32),
        }


def multimodal_collate_fn(batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate mini-batch combining PyG Batches and stacked feature tensors."""
    graph_a_list = [item['graph_a'] for item in batch_items]
    graph_b_list = [item['graph_b'] for item in batch_items]

    return {
        'drug_a': Batch.from_data_list(graph_a_list),
        'drug_b': Batch.from_data_list(graph_b_list),
        'fp_a': torch.stack([item['fp_a'] for item in batch_items]),
        'fp_b': torch.stack([item['fp_b'] for item in batch_items]),
        'gene_a': torch.stack([item['gene_a'] for item in batch_items]),
        'gene_b': torch.stack([item['gene_b'] for item in batch_items]),
        'gene_mask_a': torch.stack([item['gene_mask_a'] for item in batch_items]),
        'gene_mask_b': torch.stack([item['gene_mask_b'] for item in batch_items]),
        'tox_a': torch.stack([item['tox_a'] for item in batch_items]),
        'tox_b': torch.stack([item['tox_b'] for item in batch_items]),
        'tox_mask_a': torch.stack([item['tox_mask_a'] for item in batch_items]),
        'tox_mask_b': torch.stack([item['tox_mask_b'] for item in batch_items]),
        'labels': torch.stack([item['label'] for item in batch_items]),
    }


def build_cached_multimodal_dataloader(
    edges_df: pd.DataFrame,
    molecular_cache: MolecularCache,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    source_col: str = 'drug_a_id',
    target_col: str = 'drug_b_id',
    label_col: str = 'label',
) -> DataLoader:
    """Build high-throughput DataLoader using RAM-cached molecular and multi-modal features."""
    dataset = CachedDDIPairDataset(
        edges_df=edges_df,
        molecular_cache=molecular_cache,
        source_col=source_col,
        target_col=target_col,
        label_col=label_col,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )
