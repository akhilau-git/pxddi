"""Run controlled PxDDI baselines, ablations, and repeated-seed experiments.

This launcher never replaces ``checkpoints/pxddi_model.pt``. Every candidate
gets its own checkpoint and artifact directory. Promotion is a deliberate
post-review action after the study results are inspected.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_TRAINING_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train_full_pipeline_v2.py'
ECFP_BASELINE_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train_ecfp_logistic_baseline.py'
RUNNER_SCRIPTS = {
    'gnn': GNN_TRAINING_SCRIPT,
    'ecfp_sgd_logistic': ECFP_BASELINE_SCRIPT,
}
DRIVE_BASE = Path(os.environ.get('PXDDI_DATA_BASE', '/content/drive/MyDrive/pxddi-data'))
PRESET = os.environ.get('PXDDI_EXPERIMENT_PRESET', 'screening').strip().lower()
REFERENCE_EXPERIMENT = os.environ.get(
    'PXDDI_EXPERIMENT_REFERENCE', 'legacy_gat_multitask'
).strip()
SPLIT_NAMES = (
    'transductive_train',
    'validation',
    'posthoc_validation',
    'transductive_test',
    's1_test',
    's2_test',
)
METRIC_NAMES = (
    'auroc', 'average_precision', 'f1', 'mcc', 'balanced_accuracy',
    'brier_score_calibrated',
)

EXPERIMENTS = (
    {
        'name': 'ecfp_sgd_logistic',
        'runner': 'ecfp_sgd_logistic',
        'architecture': 'ecfp_sgd_logistic_v1',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
        'epochs': 30,
    },
    {
        'name': 'legacy_gat_ddi_only',
        'runner': 'gnn',
        'architecture': 'legacy_gat_v1',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'legacy_gat_multitask',
        'runner': 'gnn',
        'architecture': 'legacy_gat_v1',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
    {
        'name': 'edge_aware_ddi_only',
        'runner': 'gnn',
        'architecture': 'edge_aware_gat_v2',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'edge_aware_multitask',
        'runner': 'gnn',
        'architecture': 'edge_aware_gat_v2',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
    {
        'name': 'motif_edge_aware_ddi_only',
        'runner': 'gnn',
        'architecture': 'motif_edge_aware_gat_v1',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'motif_edge_aware_multitask',
        'runner': 'gnn',
        'architecture': 'motif_edge_aware_gat_v1',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
    {
        'name': 'cross_attention_edge_aware_ddi_only',
        'runner': 'gnn',
        'architecture': 'cross_attention_edge_aware_gat_v1',
        'use_toxicity_pair_features': False,
        'toxicity_loss_weight': 0.0,
    },
    {
        'name': 'cross_attention_edge_aware_multitask',
        'runner': 'gnn',
        'architecture': 'cross_attention_edge_aware_gat_v1',
        'use_toxicity_pair_features': True,
        'toxicity_loss_weight': 0.3,
    },
)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(',') if part.strip()]
    if not seeds or any(seed <= 0 for seed in seeds):
        raise ValueError('PXDDI_EXPERIMENT_SEEDS must contain positive integers.')
    return seeds


def resolve_experiments_base(
    configured_path: str | Path | None = None,
    data_base: Path = DRIVE_BASE,
) -> Path:
    """Select a writable output root separately from the read-only data root.

    Shared Google Drive folders are often mounted through a read-only shortcut.
    PxDDI can still read its source CSVs there, while ``PXDDI_EXPERIMENTS_BASE``
    directs every study artifact and candidate checkpoint to the user's own
    writable Drive location.
    """
    if configured_path is None:
        configured_path = os.environ.get('PXDDI_EXPERIMENTS_BASE')
    return Path(configured_path) if configured_path else data_base / 'experiments'


def experiment_settings() -> tuple[list[int], int]:
    """Choose a quick screening study or a repeated-seed paper study."""
    if PRESET == 'screening':
        return _parse_seeds(os.environ.get('PXDDI_EXPERIMENT_SEEDS', '42')), int(
            os.environ.get('PXDDI_EXPERIMENT_EPOCHS', '120')
        )
    if PRESET == 'paper':
        return _parse_seeds(os.environ.get('PXDDI_EXPERIMENT_SEEDS', '11,23,37,53,71')), int(
            os.environ.get('PXDDI_EXPERIMENT_EPOCHS', '200')
        )
    raise ValueError("PXDDI_EXPERIMENT_PRESET must be 'screening' or 'paper'.")


def selected_experiments(requested_names: str | None = None) -> tuple[dict[str, Any], ...]:
    """Choose an explicit set of configurations for a fair, bounded study.

    Screening runs every ablation once.  The paper preset repeats the two
    directly comparable multi-task models by default, avoiding an unnecessarily
    expensive 20-run Colab job after the ablation question has been answered.
    ``PXDDI_EXPERIMENT_NAMES`` can always request a deliberate alternative.
    """
    by_name = {experiment['name']: experiment for experiment in EXPERIMENTS}
    value = requested_names
    if value is None:
        value = os.environ.get('PXDDI_EXPERIMENT_NAMES')
    if value is None or not value.strip():
        names = (
            tuple(by_name)
            if PRESET == 'screening'
            else ('legacy_gat_multitask', 'edge_aware_multitask')
        )
    else:
        names = tuple(name.strip() for name in value.split(',') if name.strip())
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            f'PXDDI_EXPERIMENT_NAMES contains unknown experiments: {unknown}. '
            f'Available: {sorted(by_name)}.'
        )
    if not names:
        raise ValueError('At least one experiment must be selected.')
    return tuple(by_name[name] for name in names)


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
        'run_count': len(array),
        'mean': float(array.mean()),
        'standard_deviation': float(array.std(ddof=1)),
        'ci_95_lower': float(np.percentile(sampled_means, 2.5)),
        'ci_95_upper': float(np.percentile(sampled_means, 97.5)),
    }


def paired_bootstrap_difference_confidence_interval(
    reference_values: list[float],
    candidate_values: list[float],
    seed: int = 0,
    resamples: int = 10000,
) -> dict[str, float | int] | None:
    """Return a paired 95% CI for candidate minus reference across matched seeds."""
    if len(reference_values) != len(candidate_values):
        raise ValueError('Paired comparisons require the same number of matched seeds.')
    if len(reference_values) < 2:
        return None
    differences = np.asarray(candidate_values, dtype=float) - np.asarray(reference_values, dtype=float)
    generator = np.random.default_rng(seed)
    sampled_means = generator.choice(
        differences, size=(resamples, len(differences)), replace=True
    ).mean(axis=1)
    return {
        'matched_seed_count': len(differences),
        'mean_difference_candidate_minus_reference': float(differences.mean()),
        'standard_deviation': float(differences.std(ddof=1)),
        'ci_95_lower': float(np.percentile(sampled_means, 2.5)),
        'ci_95_upper': float(np.percentile(sampled_means, 97.5)),
    }


def paired_wilcoxon_signed_rank_test(
    reference_values: list[float],
    candidate_values: list[float],
) -> dict[str, Any]:
    """Run a conservative matched-seed test only when it has enough seeds.

    Five seeds is the minimum configured paper study.  With so few independent
    training seeds, the test is supplementary evidence alongside the paired
    effect-size interval, not a substitute for an external validation study.
    """
    if len(reference_values) != len(candidate_values):
        raise ValueError('Wilcoxon comparisons require matched seed counts.')
    if len(reference_values) < 5:
        return {
            'status': 'not_run_fewer_than_five_matched_seeds',
            'test': 'two_sided_wilcoxon_signed_rank',
            'matched_seed_count': len(reference_values),
            'p_value_raw': None,
        }
    differences = np.asarray(candidate_values, dtype=float) - np.asarray(reference_values, dtype=float)
    if np.allclose(differences, 0.0):
        return {
            'status': 'all_matched_differences_zero',
            'test': 'two_sided_wilcoxon_signed_rank',
            'matched_seed_count': len(differences),
            'statistic': 0.0,
            'p_value_raw': 1.0,
        }
    try:
        result = wilcoxon(candidate_values, reference_values, alternative='two-sided', method='auto')
    except ValueError as error:
        return {
            'status': 'not_run_invalid_matched_differences',
            'test': 'two_sided_wilcoxon_signed_rank',
            'matched_seed_count': len(differences),
            'p_value_raw': None,
            'reason': str(error),
        }
    return {
        'status': 'evaluated',
        'test': 'two_sided_wilcoxon_signed_rank',
        'matched_seed_count': len(differences),
        'statistic': float(result.statistic),
        'p_value_raw': float(result.pvalue),
        'interpretation_warning': (
            'This test compares only matched random seeds on one fixed benchmark. '
            'It does not establish external or clinical superiority.'
        ),
    }


def holm_adjust_p_values(comparison_rows: list[dict[str, Any]]) -> None:
    """Apply Holm correction across every valid study comparison in place."""
    eligible = [
        row for row in comparison_rows
        if row.get('statistical_test', {}).get('p_value_raw') is not None
    ]
    ordered = sorted(
        eligible,
        key=lambda row: row['statistical_test']['p_value_raw'],
    )
    adjusted_so_far = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        raw_p_value = float(row['statistical_test']['p_value_raw'])
        adjusted_so_far = max(adjusted_so_far, (total - index) * raw_p_value)
        row['statistical_test']['p_value_holm_adjusted'] = min(1.0, adjusted_so_far)
        row['statistical_test']['multiple_testing_correction'] = 'holm_family_wise_error_rate'


def split_manifest_signature(manifest: dict[str, Any]) -> str:
    """Hash only the split evidence needed to establish a fair comparison."""
    split_manifest = manifest.get('split_manifest')
    if not isinstance(split_manifest, dict):
        raise ValueError('Run manifest is missing its split_manifest.')
    missing = set(SPLIT_NAMES).difference(split_manifest)
    if missing:
        raise ValueError(f'Run manifest is missing split evidence for: {sorted(missing)}.')
    evidence = {
        name: {
            'sha256': split_manifest[name].get('sha256'),
            'rows': split_manifest[name].get('rows'),
            'label_counts': split_manifest[name].get('label_counts'),
        }
        for name in SPLIT_NAMES
    }
    if any(not details['sha256'] for details in evidence.values()):
        raise ValueError('Run manifest has a split with no SHA-256 hash.')
    encoded = json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


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
    split_signature = split_manifest_signature(manifest)
    input_hash = manifest.get('input_sha256', {}).get('twosides_edges')
    if not input_hash:
        raise ValueError('Run manifest is missing the TWOSIDES input SHA-256 hash.')
    rows = []
    for split_name, metrics in manifest['results'].items():
        rows.append({
            'experiment': experiment_name,
            'seed': seed,
            'split': split_name,
            'run_directory': str(run_dir),
            'checkpoint_sha256': manifest['checkpoint']['sha256'],
            'twosides_input_sha256': input_hash,
            'split_manifest_signature': split_signature,
            'negative_label_meaning': manifest['configuration']['negative_label_meaning'],
            'model_architecture': manifest['configuration']['model_architecture'],
            'use_toxicity_pair_features': manifest['configuration']['use_toxicity_pair_features'],
            'toxicity_loss_weight': manifest['configuration']['toxicity_loss_weight'],
            **{
                key: metrics.get(key)
                for key in (*METRIC_NAMES, 'brier_score_raw')
            },
        })
    return rows


def validate_study_comparability(table: pd.DataFrame) -> None:
    """Refuse cross-model comparisons unless every model used the same split per seed."""
    required = {
        'experiment', 'seed', 'split', 'twosides_input_sha256',
        'split_manifest_signature', 'negative_label_meaning',
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f'Experiment table is missing comparability columns: {sorted(missing)}.')
    duplicated = table.duplicated(subset=['experiment', 'seed', 'split'], keep=False)
    if duplicated.any():
        raise ValueError('Experiment table contains duplicate experiment/seed/split rows.')
    for seed, seed_rows in table.groupby('seed'):
        for column in (
            'twosides_input_sha256', 'split_manifest_signature', 'negative_label_meaning'
        ):
            values = seed_rows[column].dropna().unique()
            if len(values) != 1:
                raise ValueError(
                    f'Experiment comparisons for seed {seed} are invalid: {column} differs '
                    'between configurations.'
                )


def paired_comparison_summary(
    table: pd.DataFrame,
    reference_experiment: str,
) -> dict[str, Any]:
    """Summarize matched-seed differences without overstating single-run results."""
    available = set(table['experiment'])
    if reference_experiment not in available:
        raise ValueError(
            f'Reference experiment {reference_experiment!r} was not run. '
            f'Available: {sorted(available)}.'
        )
    comparison_rows: list[dict[str, Any]] = []
    reference = table[table['experiment'] == reference_experiment]
    for experiment in sorted(available.difference({reference_experiment})):
        candidate = table[table['experiment'] == experiment]
        for split in sorted(set(reference['split']).intersection(candidate['split'])):
            reference_split = reference[reference['split'] == split].set_index('seed')
            candidate_split = candidate[candidate['split'] == split].set_index('seed')
            shared_seeds = sorted(set(reference_split.index).intersection(candidate_split.index))
            for metric in METRIC_NAMES:
                if metric not in reference_split or metric not in candidate_split:
                    continue
                paired = pd.DataFrame({
                    'reference': pd.to_numeric(reference_split.loc[shared_seeds, metric], errors='coerce'),
                    'candidate': pd.to_numeric(candidate_split.loc[shared_seeds, metric], errors='coerce'),
                }).dropna()
                comparison_rows.append({
                    'reference_experiment': reference_experiment,
                    'candidate_experiment': experiment,
                    'split': split,
                    'metric': metric,
                    'matched_seeds': [int(seed) for seed in paired.index.tolist()],
                    'statistics': paired_bootstrap_difference_confidence_interval(
                        paired['reference'].tolist(), paired['candidate'].tolist()
                    ),
                    'statistical_test': paired_wilcoxon_signed_rank_test(
                        paired['reference'].tolist(), paired['candidate'].tolist()
                    ),
                })
    holm_adjust_p_values(comparison_rows)
    return {
        'reference_experiment': reference_experiment,
        'warning': (
            'A confidence interval is reported only with two or more matched seeds. '
            'Wilcoxon tests run only with at least five matched seeds, then Holm '
            'correction controls the study-wide family-wise error rate. Screening '
            'results remain directional evidence, not statistical proof.'
        ),
        'comparisons': comparison_rows,
    }


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
    experiments = selected_experiments()
    experiment_names = {experiment['name'] for experiment in experiments}
    reference_experiment = REFERENCE_EXPERIMENT
    if reference_experiment not in experiment_names:
        if len(experiments) == 1:
            reference_experiment = experiments[0]['name']
        else:
            raise ValueError(
                f'PXDDI_EXPERIMENT_REFERENCE={reference_experiment!r} is not selected. '
                f'Selected experiments: {sorted(experiment_names)}.'
            )
    study_id = datetime.now(timezone.utc).strftime('study_%Y%m%dT%H%M%SZ')
    experiments_base = resolve_experiments_base()
    study_dir = experiments_base / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    write_json(study_dir / 'study_plan.json', {
        'study_id': study_id,
        'preset': PRESET,
        'input_data_base': str(DRIVE_BASE),
        'experiments_base': str(experiments_base),
        'seeds': seeds,
        'epochs_per_run': epochs,
        'experiments': experiments,
        'reference_experiment': reference_experiment,
        'promotion_policy': (
            'No candidate is promoted to checkpoints/pxddi_model.pt by this suite. '
            'Review S1/S2, calibration, and repeated-seed results first.'
        ),
    })

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for seed in seeds:
            run_root = study_dir / experiment['name'] / f'seed_{seed}'
            artifact_base = run_root / 'artifacts'
            try:
                run_dir = find_completed_run(artifact_base)
                print(f"Skipping {experiment['name']} seed={seed}; already completed at {run_dir}")
                rows.extend(collect_metric_rows(experiment['name'], seed, run_dir))
                continue
            except FileNotFoundError:
                pass
            checkpoint_suffix = '.npz' if experiment['runner'] == 'ecfp_sgd_logistic' else '.pt'
            checkpoint_path = run_root / 'checkpoints' / f"{experiment['name']}_seed_{seed}{checkpoint_suffix}"
            environment = os.environ.copy()
            environment.update({
                'PXDDI_SEED': str(seed),
                'PXDDI_ARTIFACTS_BASE': str(artifact_base),
                'PXDDI_CHECKPOINT_PATH': str(checkpoint_path),
                'PXDDI_PUBLISH_LATEST_RESULTS': 'false',
                'PXDDI_NEGATIVE_SAMPLING_STRATEGY': experiment.get('negative_sampling_strategy', 'degree_matched'),
            })
            if experiment['runner'] == 'gnn':
                environment.update({
                    'PXDDI_EPOCHS': str(epochs),
                    'PXDDI_MODEL_ARCHITECTURE': experiment['architecture'],
                    'PXDDI_USE_TOXICITY_PAIR_FEATURES': str(experiment['use_toxicity_pair_features']).lower(),
                    'PXDDI_TOXICITY_LOSS_WEIGHT': str(experiment['toxicity_loss_weight']),
                })
            else:
                environment['PXDDI_ECFP_EPOCHS'] = str(experiment['epochs'])
            training_script = RUNNER_SCRIPTS[experiment['runner']]
            print(f"Running {experiment['name']} seed={seed}; checkpoint={checkpoint_path}")
            subprocess.run([sys.executable, str(training_script)], check=True, env=environment)
            run_dir = find_completed_run(artifact_base)
            rows.extend(collect_metric_rows(experiment['name'], seed, run_dir))

    table = pd.DataFrame(rows)
    validate_study_comparability(table)
    table.to_csv(study_dir / 'experiment_results.csv', index=False)
    summary: dict[str, Any] = {}
    for experiment_name, experiment_table in table.groupby('experiment'):
        experiment_name_str = str(experiment_name)
        summary[experiment_name_str] = {}
        for split_name, split_table in experiment_table.groupby('split'):
            split_name_str = str(split_name)
            summary[experiment_name_str][split_name_str] = {
                metric: bootstrap_mean_confidence_interval(split_table[metric].dropna().tolist())
                for metric in METRIC_NAMES
            }
    write_json(study_dir / 'study_summary.json', summary)
    write_json(
        study_dir / 'paired_comparisons.json',
        paired_comparison_summary(table, reference_experiment),
    )
    save_comparison_plot(table, study_dir / 'experiment_comparison')
    print(f'Experiment study saved to: {study_dir}')


if __name__ == '__main__':
    main()
