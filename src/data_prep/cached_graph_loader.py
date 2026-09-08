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

    def __init__(self, gene_dim: int = 50, target_dim: int = 50, geo_dim: int = 2, pdb_dim: int = 50) -> None:
        self.gene_dim = gene_dim
        self.target_dim = target_dim
        self.geo_dim = geo_dim
        self.pdb_dim = pdb_dim
        self.graphs: dict[str, Data] = {}
        self.fingerprints: dict[str, torch.Tensor] = {}
        self.gene_vectors: dict[str, torch.Tensor] = {}
        self.gene_masks: dict[str, torch.Tensor] = {}
        self.toxicity_scalars: dict[str, torch.Tensor] = {}
        self.toxicity_masks: dict[str, torch.Tensor] = {}
        self.target_vectors: dict[str, torch.Tensor] = {}
        self.target_masks: dict[str, torch.Tensor] = {}
        self.geo_vectors: dict[str, torch.Tensor] = {}
        self.geo_masks: dict[str, torch.Tensor] = {}
        self.pdb_vectors: dict[str, torch.Tensor] = {}
        self.pdb_masks: dict[str, torch.Tensor] = {}

    def register_drug(
        self,
        smiles: str,
        gene_vector: list[int] | list[float] | None = None,
        toxicity_score: float | None = None,
        target_vector: list[int] | list[float] | None = None,
        geo_vector: list[float] | None = None,
        pdb_vector: list[int] | list[float] | None = None,
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
        if gene_vector and len(gene_vector) > 0:
            gv = np.array(gene_vector, dtype=np.float32)
            if len(gv) != self.gene_dim:
                padded = np.zeros(self.gene_dim, dtype=np.float32)
                l = min(len(gv), self.gene_dim)
                padded[:l] = gv[:l]
                gv = padded
            self.gene_vectors[smiles] = torch.tensor(gv, dtype=torch.float32)
            self.gene_masks[smiles] = torch.tensor(1.0 if np.any(gv > 0) else 0.0, dtype=torch.float32)
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

        # BindingDB Target Vector (Multi-Hot / Affinity)
        if target_vector and len(target_vector) > 0:
            tv = np.array(target_vector, dtype=np.float32)
            if len(tv) != self.target_dim:
                padded = np.zeros(self.target_dim, dtype=np.float32)
                l = min(len(tv), self.target_dim)
                padded[:l] = tv[:l]
                tv = padded
            self.target_vectors[smiles] = torch.tensor(tv, dtype=torch.float32)
            self.target_masks[smiles] = torch.tensor(1.0 if np.any(tv > 0) else 0.0, dtype=torch.float32)
        else:
            self.target_vectors[smiles] = torch.zeros(self.target_dim, dtype=torch.float32)
            self.target_masks[smiles] = torch.tensor(0.0, dtype=torch.float32)

        # GEO Disease Transcriptomic Signature Vector
        if geo_vector and len(geo_vector) > 0:
            geov = np.array(geo_vector, dtype=np.float32)
            if len(geov) != self.geo_dim:
                padded = np.zeros(self.geo_dim, dtype=np.float32)
                l = min(len(geov), self.geo_dim)
                padded[:l] = geov[:l]
                geov = padded
            self.geo_vectors[smiles] = torch.tensor(geov, dtype=torch.float32)
            self.geo_masks[smiles] = torch.tensor(1.0 if np.any(geov > 0) else 0.0, dtype=torch.float32)
        else:
            self.geo_vectors[smiles] = torch.zeros(self.geo_dim, dtype=torch.float32)
            self.geo_masks[smiles] = torch.tensor(0.0, dtype=torch.float32)

        # PDB 3D Macromolecular Target Vector
        if pdb_vector and len(pdb_vector) > 0:
            pdbv = np.array(pdb_vector, dtype=np.float32)
            if len(pdbv) != self.pdb_dim:
                padded = np.zeros(self.pdb_dim, dtype=np.float32)
                l = min(len(pdbv), self.pdb_dim)
                padded[:l] = pdbv[:l]
                pdbv = padded
            self.pdb_vectors[smiles] = torch.tensor(pdbv, dtype=torch.float32)
            self.pdb_masks[smiles] = torch.tensor(1.0 if np.any(pdbv > 0) else 0.0, dtype=torch.float32)
        else:
            self.pdb_vectors[smiles] = torch.zeros(self.pdb_dim, dtype=torch.float32)
            self.pdb_masks[smiles] = torch.tensor(0.0, dtype=torch.float32)

        return True

    def populate_from_master_nodes(self, master_nodes_path: str | Path) -> int:
        """Pre-populate the entire cache from master_drug_nodes.csv with auto-dimension detection."""
        df_nodes = pd.read_csv(master_nodes_path)
        count = 0
        node_id_col = 'drug_id' if 'drug_id' in df_nodes.columns else ('canonical_smiles' if 'canonical_smiles' in df_nodes.columns else df_nodes.columns[0])

        # Auto-detect dimensions from first non-empty serialized vector in dataset
        gene_col_cands = ['gene_vector_multihot', 'gene_vector', 'genes_multihot', 'pharmgkb_gene_vector']
        target_col_cands = ['bindingdb_target_vector', 'target_vector_multihot', 'target_vector', 'bindingdb_vector']
        geo_col_cands = ['geo_signature_vector', 'geo_vector', 'disease_signature_vector', 'geo_signatures_vector']
        pdb_col_cands = ['pdb_vector_multihot', 'pdb_vector', 'pdb_signature_vector', 'pdb_targets_vector']
        tox_col_cands = ['toxicity_score', 'clinical_toxicity', 'faers_toxicity_score', 'tox_score', 'faers_score']

        for c in gene_col_cands:
            if c in df_nodes.columns:
                non_nulls = df_nodes[c].dropna()
                for val in non_nulls:
                    try:
                        parsed = json.loads(val) if isinstance(val, str) else list(val)
                        if parsed and len(parsed) > 0:
                            self.gene_dim = len(parsed)
                            break
                    except Exception:
                        pass
                break

        for c in target_col_cands:
            if c in df_nodes.columns:
                non_nulls = df_nodes[c].dropna()
                for val in non_nulls:
                    try:
                        parsed = json.loads(val) if isinstance(val, str) else list(val)
                        if parsed and len(parsed) > 0:
                            self.target_dim = len(parsed)
                            break
                    except Exception:
                        pass
                break

        for c in geo_col_cands:
            if c in df_nodes.columns:
                non_nulls = df_nodes[c].dropna()
                for val in non_nulls:
                    try:
                        parsed = json.loads(val) if isinstance(val, str) else list(val)
                        if parsed and len(parsed) > 0:
                            self.geo_dim = len(parsed)
                            break
                    except Exception:
                        pass
                break

        for c in pdb_col_cands:
            if c in df_nodes.columns:
                non_nulls = df_nodes[c].dropna()
                for val in non_nulls:
                    try:
                        parsed = json.loads(val) if isinstance(val, str) else list(val)
                        if parsed and len(parsed) > 0:
                            self.pdb_dim = len(parsed)
                            break
                    except Exception:
                        pass
                break

        for _, row in df_nodes.iterrows():
            smi = str(row[node_id_col]).strip()

            # Extract gene vector
            gene_vec = None
            for c in gene_col_cands:
                if c in row and pd.notna(row[c]):
                    val = row[c]
                    try:
                        gene_vec = json.loads(val) if isinstance(val, str) else list(val)
                        break
                    except Exception:
                        pass

            # Extract toxicity score
            tox_score = None
            for c in tox_col_cands:
                if c in row and pd.notna(row[c]):
                    try:
                        tox_score = float(row[c])
                        break
                    except Exception:
                        pass

            # Extract BindingDB target vector
            target_vec = None
            for c in target_col_cands:
                if c in row and pd.notna(row[c]):
                    val = row[c]
                    try:
                        target_vec = json.loads(val) if isinstance(val, str) else list(val)
                        break
                    except Exception:
                        pass

            # Extract GEO transcriptomic vector
            geo_vec = None
            for c in geo_col_cands:
                if c in row and pd.notna(row[c]):
                    val = row[c]
                    try:
                        geo_vec = json.loads(val) if isinstance(val, str) else list(val)
                        break
                    except Exception:
                        pass

            # Extract PDB macromolecular target vector
            pdb_vec = None
            for c in pdb_col_cands:
                if c in row and pd.notna(row[c]):
                    val = row[c]
                    try:
                        pdb_vec = json.loads(val) if isinstance(val, str) else list(val)
                        break
                    except Exception:
                        pass

            if self.register_drug(
                smi,
                gene_vector=gene_vec,
                toxicity_score=tox_score,
                target_vector=target_vec,
                geo_vector=geo_vec,
                pdb_vector=pdb_vec,
            ):
                count += 1

        n_genes = sum(1 for m in self.gene_masks.values() if m.item() > 0)
        n_tox = sum(1 for m in self.toxicity_masks.values() if m.item() > 0)
        n_targets = sum(1 for m in self.target_masks.values() if m.item() > 0)
        n_geo = sum(1 for m in self.geo_masks.values() if m.item() > 0)
        n_pdb = sum(1 for m in self.pdb_masks.values() if m.item() > 0)

        print(
            f"MolecularCache populated: {count} drugs cached "
            f"[Graphs: {len(self.graphs)}, ECFP: {len(self.fingerprints)}, "
            f"PharmGKB: {n_genes} (dim={self.gene_dim}), FAERS: {n_tox}, "
            f"BindingDB: {n_targets} (dim={self.target_dim}), GEO: {n_geo} (dim={self.geo_dim}), "
            f"PDB: {n_pdb} (dim={self.pdb_dim})]"
        )
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
        neighbor_memory: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cache = molecular_cache
        self.samples: list[tuple[str, str, float]] = []

        actual_source = source_col
        if actual_source not in edges_df.columns:
            for cand in ['source', 'drug1_id', 'drug_a_id', 'drug_1', 'drug_a']:
                if cand in edges_df.columns:
                    actual_source = cand
                    break

        actual_target = target_col
        if actual_target not in edges_df.columns:
            for cand in ['target', 'drug2_id', 'drug_b_id', 'drug_2', 'drug_b']:
                if cand in edges_df.columns:
                    actual_target = cand
                    break

        # Validate that edges have registered drugs and strictly validate labels
        valid_positives = {True, 1, 1.0, '1', '1.0', 'true'}
        valid_negatives = {False, 0, 0.0, '0', '0.0', 'false'}
        for _, row in edges_df.iterrows():
            sa, sb = str(row[actual_source]).strip(), str(row[actual_target]).strip()
            if sa in self.cache.graphs and sb in self.cache.graphs:
                if label_col in row and pd.notna(row[label_col]):
                    raw_label = row[label_col]
                    clean_str = str(raw_label).strip().lower()
                    if raw_label in valid_positives or clean_str in {'1', '1.0', 'true'}:
                        lbl = 1.0
                    elif raw_label in valid_negatives or clean_str in {'0', '0.0', 'false'}:
                        lbl = 0.0
                    else:
                        raise ValueError(
                            f"Invalid or corrupted label '{raw_label}' encountered for edge ({sa}, {sb}). "
                            f"Expected binary value (1/0, True/False)."
                        )
                elif label_col in row and pd.isna(row[label_col]):
                    raise ValueError(
                        f"Missing/NaN label encountered for edge ({sa}, {sb}). Label column cannot be null."
                    )
                else:
                    lbl = 1.0
                self.samples.append((sa, sb, lbl))

        self.memory_features: list[torch.Tensor] = []
        if neighbor_memory is not None and hasattr(neighbor_memory, 'score_batch') and self.samples:
            try:
                da_list = [s[0] for s in self.samples]
                db_list = [s[1] for s in self.samples]
                exclude_query = bool(kwargs.get('exclude_query_pairs', kwargs.get('is_train', False)))
                m_arr = neighbor_memory.score_batch(da_list, db_list, exclude_query_pairs=exclude_query)
                self.memory_features = [torch.from_numpy(v).float() for v in m_arr]
            except Exception:
                self.memory_features = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sa, sb, lbl = self.samples[index]
        res = {
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
            'target_a': self.cache.target_vectors.get(sa, torch.zeros(self.cache.target_dim, dtype=torch.float32)),
            'target_b': self.cache.target_vectors.get(sb, torch.zeros(self.cache.target_dim, dtype=torch.float32)),
            'target_mask_a': self.cache.target_masks.get(sa, torch.tensor(0.0, dtype=torch.float32)),
            'target_mask_b': self.cache.target_masks.get(sb, torch.tensor(0.0, dtype=torch.float32)),
            'geo_a': self.cache.geo_vectors.get(sa, torch.zeros(self.cache.geo_dim, dtype=torch.float32)),
            'geo_b': self.cache.geo_vectors.get(sb, torch.zeros(self.cache.geo_dim, dtype=torch.float32)),
            'geo_mask_a': self.cache.geo_masks.get(sa, torch.tensor(0.0, dtype=torch.float32)),
            'geo_mask_b': self.cache.geo_masks.get(sb, torch.tensor(0.0, dtype=torch.float32)),
            'pdb_a': self.cache.pdb_vectors.get(sa, torch.zeros(self.cache.pdb_dim, dtype=torch.float32)),
            'pdb_b': self.cache.pdb_vectors.get(sb, torch.zeros(self.cache.pdb_dim, dtype=torch.float32)),
            'pdb_mask_a': self.cache.pdb_masks.get(sa, torch.tensor(0.0, dtype=torch.float32)),
            'pdb_mask_b': self.cache.pdb_masks.get(sb, torch.tensor(0.0, dtype=torch.float32)),
            'label': torch.tensor(lbl, dtype=torch.float32),
        }
        if self.memory_features and index < len(self.memory_features):
            res['memory_features'] = self.memory_features[index]
        else:
            res['memory_features'] = torch.zeros(3, dtype=torch.float32)
        return res


def multimodal_collate_fn(batch_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate mini-batch combining PyG Batches and stacked feature tensors."""
    graph_a_list = [item['graph_a'] for item in batch_items]
    graph_b_list = [item['graph_b'] for item in batch_items]

    batch = {
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
        'target_a': torch.stack([item['target_a'] for item in batch_items]),
        'target_b': torch.stack([item['target_b'] for item in batch_items]),
        'target_mask_a': torch.stack([item['target_mask_a'] for item in batch_items]),
        'target_mask_b': torch.stack([item['target_mask_b'] for item in batch_items]),
        'geo_a': torch.stack([item['geo_a'] for item in batch_items]),
        'geo_b': torch.stack([item['geo_b'] for item in batch_items]),
        'geo_mask_a': torch.stack([item['geo_mask_a'] for item in batch_items]),
        'geo_mask_b': torch.stack([item['geo_mask_b'] for item in batch_items]),
        'pdb_a': torch.stack([item['pdb_a'] for item in batch_items]),
        'pdb_b': torch.stack([item['pdb_b'] for item in batch_items]),
        'pdb_mask_a': torch.stack([item['pdb_mask_a'] for item in batch_items]),
        'pdb_mask_b': torch.stack([item['pdb_mask_b'] for item in batch_items]),
        'labels': torch.stack([item['label'] for item in batch_items]),
    }
    if 'memory_features' in batch_items[0]:
        batch['memory_features'] = torch.stack([item['memory_features'] for item in batch_items])
    return batch


def build_cached_multimodal_dataloader(
    edges_df: pd.DataFrame,
    molecular_cache: MolecularCache,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    source_col: str = 'drug_a_id',
    target_col: str = 'drug_b_id',
    label_col: str = 'label',
    neighbor_memory: Any = None,
    **kwargs: Any,
) -> DataLoader:
    """Build high-throughput DataLoader using RAM-cached molecular and multi-modal features."""
    is_train = bool(kwargs.pop('is_train', shuffle))
    exclude_query = bool(kwargs.pop('exclude_query_pairs', is_train))
    dataset = CachedDDIPairDataset(
        edges_df=edges_df,
        molecular_cache=molecular_cache,
        source_col=source_col,
        target_col=target_col,
        label_col=label_col,
        neighbor_memory=neighbor_memory,
        is_train=is_train,
        exclude_query_pairs=exclude_query,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )
