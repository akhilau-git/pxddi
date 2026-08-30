"""Pretrain PxDDI's edge-aware molecular encoder on leakage-filtered ChEMBL.

This is an experimental research workflow, never a deployment workflow.  It
uses ChEMBL structures without DDI labels, excludes every molecule outside the
PxDDI transductive-training partition, and writes a separate encoder-only
checkpoint for explicit later fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = PROJECT_ROOT / 'src'
for candidate in (PROJECT_ROOT, REPOSITORY_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from data_prep.chembl_pretraining import (
    build_pretraining_exclusion_set,
    select_chembl_pretraining_corpus,
)
from data_prep.prepare_twosides import (
    FEATURE_SCHEMA_RICH,
    NUM_BOND_FEATURES,
    RICH_NUM_ATOM_FEATURES,
    smiles_to_graph,
)
from models.encoder_pretraining import (
    PRETRAINING_ARTIFACT_TYPE,
    EdgeAwareContrastivePretrainer,
    augment_edge_aware_batch,
    bidirectional_nt_xent_loss,
)


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def _unit_interval(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0 <= value < 1:
        raise ValueError(f'{name} must be in [0, 1).')
    return value


DATA_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
RESULTS_BASE = Path(os.environ.get('PXDDI_RESULTS_BASE', '/content/drive/MyDrive/pxddi-results'))
CHEMREPS_PATH = Path(
    os.environ.get('PXDDI_CHEMBL_CHEMREPS_PATH', DATA_BASE / 'chembl' / 'chembl_37_chemreps.txt.gz')
)
TWOSIDES_EDGES = Path(
    os.environ.get('PXDDI_TWOSIDES_EDGES_PATH', DATA_BASE / 'twosides' / 'drug_drug_edges.csv')
)
SPLIT_SEED = _positive_int('PXDDI_SPLIT_SEED', 42)
PRETRAIN_SEED = _positive_int('PXDDI_PRETRAIN_SEED', 2026)
DATA_CAP = _positive_int('PXDDI_DATA_CAP', 200000)
MAXIMUM_MOLECULES = _positive_int('PXDDI_PRETRAIN_MAX_MOLECULES', 50000)
EPOCHS = _positive_int('PXDDI_PRETRAIN_EPOCHS', 20)
BATCH_SIZE = _positive_int('PXDDI_PRETRAIN_BATCH_SIZE', 256)
HIDDEN_CHANNELS = _positive_int('PXDDI_HIDDEN_CHANNELS', 128)
ATOM_MASK_RATE = _unit_interval('PXDDI_PRETRAIN_ATOM_MASK_RATE', 0.15)
BOND_MASK_RATE = _unit_interval('PXDDI_PRETRAIN_BOND_MASK_RATE', 0.15)
TEMPERATURE = float(os.environ.get('PXDDI_PRETRAIN_TEMPERATURE', '0.2'))
if TEMPERATURE <= 0:
    raise ValueError('PXDDI_PRETRAIN_TEMPERATURE must be positive.')
LEARNING_RATE = float(os.environ.get('PXDDI_PRETRAIN_LEARNING_RATE', '0.001'))
if LEARNING_RATE <= 0:
    raise ValueError('PXDDI_PRETRAIN_LEARNING_RATE must be positive.')
NEGATIVE_SAMPLING_STRATEGY = os.environ.get('PXDDI_NEGATIVE_SAMPLING_STRATEGY', 'degree_matched')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = Path(
    os.environ.get(
        'PXDDI_PRETRAIN_ARTIFACTS_DIR',
        RESULTS_BASE / 'pretraining' / f'chembl_edge_aware_{RUN_ID}',
    )
)
CHECKPOINT_PATH = Path(
    os.environ.get('PXDDI_PRETRAIN_CHECKPOINT_PATH', RUN_DIR / 'chembl_pretrained_encoder.pt')
)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot serialize {type(value).__name__}.')
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=default), encoding='utf-8')


def _safe_save(bundle: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(bundle, temporary_path)
        checked = torch.load(temporary_path, map_location='cpu', weights_only=False)
        if not isinstance(checked, dict) or set(checked) != set(bundle):
            raise ValueError('Pretraining checkpoint verification failed.')
        digest = _file_hash(temporary_path)
        os.replace(temporary_path, path)
        return digest
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ChEMBLGraphDataset(Dataset):
    """Lazy rich-graph construction for a selected, graph-compatible corpus."""

    def __init__(self, corpus: pd.DataFrame) -> None:
        self.smiles = corpus['canonical_smiles'].tolist()

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int):
        graph = smiles_to_graph(self.smiles[index], feature_schema=FEATURE_SCHEMA_RICH)
        if graph is None:
            raise RuntimeError(
                'A selected ChEMBL structure became graph-incompatible. '
                'Review the corpus selection artifact.'
            )
        return graph


def _plot_history(history: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history['epoch'], history['contrastive_loss'], color='#0072B2', linewidth=2)
    axis.set(
        title='ChEMBL self-supervised pretraining',
        xlabel='Epoch',
        ylabel='NT-Xent contrastive loss',
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    figure.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    if BATCH_SIZE < 2:
        raise ValueError('PXDDI_PRETRAIN_BATCH_SIZE must be at least two for contrastive loss.')
    _set_seed(PRETRAIN_SEED)
    if RUN_DIR.exists():
        raise FileExistsError(f'Pretraining artifact directory already exists: {RUN_DIR}')
    RUN_DIR.mkdir(parents=True)
    print(f'Pretraining on: {DEVICE}')

    excluded_smiles, split_audit = build_pretraining_exclusion_set(
        TWOSIDES_EDGES,
        data_cap=DATA_CAP,
        split_seed=SPLIT_SEED,
        negative_sampling_strategy=NEGATIVE_SAMPLING_STRATEGY,
    )
    corpus, corpus_summary = select_chembl_pretraining_corpus(
        CHEMREPS_PATH,
        excluded_smiles=excluded_smiles,
        maximum_molecules=MAXIMUM_MOLECULES,
        seed=PRETRAIN_SEED,
    )
    if len(corpus) < BATCH_SIZE:
        raise ValueError(
            f'Only {len(corpus)} ChEMBL molecules remain after strict exclusions; '
            f'at least one full batch of {BATCH_SIZE} is required.'
        )
    corpus_path = RUN_DIR / 'chembl_pretraining_corpus.csv'
    corpus.to_csv(corpus_path, index=False)
    corpus_summary['corpus_csv'] = str(corpus_path)
    corpus_summary['corpus_csv_sha256'] = _file_hash(corpus_path)
    _write_json(RUN_DIR / 'pretraining_split_audit.json', split_audit)
    _write_json(RUN_DIR / 'corpus_summary.json', corpus_summary)
    print(
        f"Selected {len(corpus)} ChEMBL molecules; excluded "
        f"{corpus_summary['rows_excluded_for_twosides_non_train_leakage']} "
        'rows overlapping non-train PxDDI structures.'
    )

    loader = DataLoader(ChEMBLGraphDataset(corpus), batch_size=BATCH_SIZE, shuffle=True)
    model = EdgeAwareContrastivePretrainer(
        RICH_NUM_ATOM_FEATURES, NUM_BOND_FEATURES, HIDDEN_CHANNELS
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    history_rows: list[dict[str, float | int]] = []
    skipped_singleton_batches = 0
    started_at = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for graph_batch in loader:
            if graph_batch.num_graphs < 2:
                skipped_singleton_batches += 1
                continue
            graph_batch = graph_batch.to(DEVICE)
            first_view = augment_edge_aware_batch(
                graph_batch,
                atom_feature_mask_rate=ATOM_MASK_RATE,
                bond_feature_mask_rate=BOND_MASK_RATE,
            )
            second_view = augment_edge_aware_batch(
                graph_batch,
                atom_feature_mask_rate=ATOM_MASK_RATE,
                bond_feature_mask_rate=BOND_MASK_RATE,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = bidirectional_nt_xent_loss(
                model(first_view), model(second_view), temperature=TEMPERATURE
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError('No ChEMBL batch with at least two molecules was available.')
        mean_loss = total_loss / batch_count
        history_rows.append({'epoch': epoch, 'contrastive_loss': mean_loss})
        print(f'Pretraining epoch {epoch}/{EPOCHS}: contrastive loss={mean_loss:.4f}')

    history = pd.DataFrame(history_rows)
    history_path = RUN_DIR / 'pretraining_history.csv'
    history.to_csv(history_path, index=False)
    _plot_history(history, RUN_DIR / 'pretraining_loss')
    elapsed = time.perf_counter() - started_at
    checkpoint = {
        'artifact_type': PRETRAINING_ARTIFACT_TYPE,
        'encoder_state_dict': model.encoder.state_dict(),
        'encoder_configuration': {
            'in_channels': RICH_NUM_ATOM_FEATURES,
            'edge_feature_dim': NUM_BOND_FEATURES,
            'hidden_channels': HIDDEN_CHANNELS,
            'feature_schema': FEATURE_SCHEMA_RICH,
        },
        'objective': {
            'name': 'bidirectional_nt_xent_v1',
            'temperature': TEMPERATURE,
            'atom_feature_mask_rate': ATOM_MASK_RATE,
            'bond_feature_mask_rate': BOND_MASK_RATE,
        },
        'pretraining_leakage_policy': 'exclude_all_non_train_twosides_structures_v1',
        'source_corpus': corpus_summary,
        'pretraining_split_audit': split_audit,
        'pretrain_seed': PRETRAIN_SEED,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    checkpoint_hash = _safe_save(checkpoint, CHECKPOINT_PATH)
    manifest = {
        'artifact_type': PRETRAINING_ARTIFACT_TYPE,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': str(DEVICE),
        'configuration': {
            'data_base': str(DATA_BASE),
            'results_base': str(RESULTS_BASE),
            'twosides_edges': str(TWOSIDES_EDGES),
            'chemreps_path': str(CHEMREPS_PATH),
            'split_seed': SPLIT_SEED,
            'pretrain_seed': PRETRAIN_SEED,
            'data_cap': DATA_CAP,
            'maximum_molecules': MAXIMUM_MOLECULES,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'hidden_channels': HIDDEN_CHANNELS,
            'atom_feature_mask_rate': ATOM_MASK_RATE,
            'bond_feature_mask_rate': BOND_MASK_RATE,
            'temperature': TEMPERATURE,
            'negative_sampling_strategy': NEGATIVE_SAMPLING_STRATEGY,
        },
        'pretraining_split_audit': split_audit,
        'source_corpus': corpus_summary,
        'history_csv': str(history_path),
        'history_sha256': _file_hash(history_path),
        'checkpoint': {
            'path': str(CHECKPOINT_PATH),
            'sha256': checkpoint_hash,
        },
        'training_wall_clock_seconds': elapsed,
        'skipped_singleton_contrastive_batches': skipped_singleton_batches,
        'promotion_policy': (
            'This encoder is a research warm start only. It does not replace or '
            'serve backend/checkpoints/pxddi_model.pt.'
        ),
    }
    _write_json(RUN_DIR / 'pretraining_manifest.json', manifest)
    print(f'Pretraining artifacts saved to: {RUN_DIR}')
    print(f'Encoder checkpoint: {CHECKPOINT_PATH}')


if __name__ == '__main__':
    main()
