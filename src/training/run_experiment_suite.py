"""Run controlled PxDDI baselines, ablations, and repeated-seed experiments.

This launcher never replaces ``checkpoints/pxddi_model.pt``. Every candidate
gets its own checkpoint and artifact directory. Promotion is a deliberate
post-review action after the study results are inspected.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train_full_pipeline_v2.py'
DRIVE_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
PRESET = os.environ.get('PXDDI_EXPERIMENT_PRESET', 'screening').strip().lower()

EXPERIMENTS = (
    {
        'name': 'legacy_gat_ddi_only',
        'architecture': 'legacy_gat_v1',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'legacy_gat_multitask',
        'architecture': 'legacy_gat_v1',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
    {
        'name': 'edge_aware_ddi_only',
        'architecture': 'edge_aware_gat_v2',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'edge_aware_multitask',
        'architecture': 'edge_aware_gat_v2',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(',') if part.strip()]
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError('PXDDI_EXPERIMENT_SEEDS must contain positive integers.')
    return seeds


def experiment_settings() -> tuple[list[int], int]:
    """Choose a quick screening study or a repeated-seed paper study."""
    if PRESET == 'screening':
        return _parse_seeds(os.environ.get('PXDDI_EXPERIMENT_SEEDS', '42')), int(
            os.environ.get('PXDDI_EXPERIMENT_EPOCHS', '120')
        )
    if PRESET == 'paper':
        return _parse_seeds(os.environ.get('PXDDI_EXPERIMENT_SEEDS', '11,23,37')), int(
            os.environ.get('PXDDI_EXPERIMENT_EPOCHS', '200')
        )
    raise ValueError("PXDDI_EXPERIMENT_PRESET must be 'screening' or 'paper'.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def bootstrap_mean_confidence_interval(
    values: list[float],
    seed: int = 0,
    resamples: int = 10000,
) -> dict[str, float | int] | None:
    """Return a nonparametric 95% interval across independent experiment runs."""
    if len(values) < 2:
        return None
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    sampled_means = generator.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return {
        'run_count': int(len(array)),
        'mean': float(array.mean()),
        'standard_deviation': float(array.std(ddof=1)),
        'ci_95_lower': float(np.percentile(sampled_means, 2.5)),
        'ci_95_upper': float(np.percentile(sampled_means, 97.5)),
    }


def find_completed_run(artifact_base: Path) -> Path:
    runs = sorted(
        (path for path in artifact_base.glob('run_*') if (path / 'run_manifest.json').is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f'No completed run manifest found under {artifact_base}.')
    return runs[0]


def collect_metric_rows(experiment_name: str, seed: int, run_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for split_name, metrics in manifest['results'].items():
        rows.append({
            'experiment': experiment_name,
            'seed': seed,
            'split': split_name,
            'run_directory': str(run_dir),
            'checkpoint_sha256': manifest['checkpoint']['sha256'],
            'model_architecture': manifest['configuration']['model_architecture'],
            'use_toxicity_pair_features': manifest['configuration']['use_toxicity_pair_features'],
            'toxicity_loss_weight': manifest['configuration']['toxicity_loss_weight'],
            **{
                key: metrics.get(key)
                for key in ('auroc', 'average_precision', 'f1', 'mcc', 'brier_score_raw', 'brier_score_calibrated')
            },
        })
    return rows


def save_comparison_plot(table: pd.DataFrame, destination: Path) -> None:
    """Save a grouped AUROC plot; single-run screening has no error bars."""
    from matplotlib import pyplot as plt

    grouped = table.groupby(['experiment', 'split'])['auroc'].agg(['mean', 'std']).reset_index()
    experiments = list(dict.fromkeys(grouped['experiment']))
    splits = ['Transductive', 'S1', 'S2']
    positions = np.arange(len(experiments))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 5.5))
    colors = {'Transductive': '#0072B2', 'S1': '#D55E00', 'S2': '#009E73'}
    for index, split in enumerate(splits):
        subset = grouped[grouped['split'] == split].set_index('experiment').reindex(experiments)
        axis.bar(
            positions + (index - 1) * width,
            subset['mean'],
            width,
            yerr=subset['std'].fillna(0.0),
            capsize=3,
            color=colors[split],
            label=split,
        )
    axis.set(
        title='PxDDI controlled experiment comparison',
        xlabel='Model configuration',
        ylabel='AUROC',
        xticks=positions,
        xticklabels=experiments,
        ylim=(0, 1),
    )
    axis.tick_params(axis='x', rotation=20)
    axis.legend()
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    figure.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    seeds, epochs = experiment_settings()
    study_id = datetime.now(timezone.utc).strftime('study_%Y%m%dT%H%M%SZ')
    study_dir = DRIVE_BASE / 'experiments' / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    write_json(study_dir / 'study_plan.json', {
        'study_id': study_id,
        'preset': PRESET,
        'seeds': seeds,
        'epochs_per_run': epochs,
        'experiments': EXPERIMENTS,
        'promotion_policy': (
            'No candidate is promoted to checkpoints/pxddi_model.pt by this suite. '
            'Review S1/S2, calibration, and repeated-seed results first.'
        ),
    })

    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for seed in seeds:
            run_root = study_dir / experiment['name'] / f'seed_{seed}'
            artifact_base = run_root / 'artifacts'
            checkpoint_path = run_root / 'checkpoints' / f"{experiment['name']}_seed_{seed}.pt"
            environment = os.environ.copy()
            environment.update({
                'PXDDI_SEED': str(seed),
                'PXDDI_EPOCHS': str(epochs),
                'PXDDI_MODEL_ARCHITECTURE': experiment['architecture'],
                'PXDDI_USE_TOXICITY_PAIR_FEATURES': str(experiment['use_toxicity_pair_features']).lower(),
                'PXDDI_TOXICITY_LOSS_WEIGHT': str(experiment['toxicity_loss_weight']),
                'PXDDI_ARTIFACTS_BASE': str(artifact_base),
                'PXDDI_CHECKPOINT_PATH': str(checkpoint_path),
                'PXDDI_PUBLISH_LATEST_RESULTS': 'false',
            })
            print(f"Running {experiment['name']} seed={seed}; checkpoint={checkpoint_path}")
            subprocess.run([sys.executable, str(TRAINING_SCRIPT)], check=True, env=environment)
            run_dir = find_completed_run(artifact_base)
            rows.extend(collect_metric_rows(experiment['name'], seed, run_dir))

    table = pd.DataFrame(rows)
    table.to_csv(study_dir / 'experiment_results.csv', index=False)
    summary: dict[str, Any] = {}
    for experiment_name, experiment_table in table.groupby('experiment'):
        summary[experiment_name] = {}
        for split_name, split_table in experiment_table.groupby('split'):
            summary[experiment_name][split_name] = {
                metric: bootstrap_mean_confidence_interval(split_table[metric].dropna().tolist())
                for metric in ('auroc', 'average_precision', 'f1', 'mcc', 'brier_score_calibrated')
            }
    write_json(study_dir / 'study_summary.json', summary)
    save_comparison_plot(table, study_dir / 'experiment_comparison')
    print(f'Experiment study saved to: {study_dir}')


if __name__ == '__main__':
    main()
