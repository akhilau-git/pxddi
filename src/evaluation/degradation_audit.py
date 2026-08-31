"""Performance degradation audit across structural distance bins.

This module computes how AUROC, Brier score, and abstention rate degrade
as drug pairs become structurally more distant from the training set.

A "degradation curve" shows:
  - x-axis: nearest-training-drug Tanimoto bin (0.0-0.2, 0.2-0.4, ...)
  - y-axis: AUROC with bootstrap 95% CI per bin
  - repeat for each evaluation split (Transductive, S1, S2)

No existing DDI paper produces this curve.  It is a direct empirical answer
to the question: "Exactly how structurally far does a drug pair need to be
before this model stops working?"

Usage::

    from src.evaluation.degradation_audit import DegradationAudit

    audit = DegradationAudit(n_bootstrap=1000, seed=42)
    result = audit.run(
        labels=labels,
        scores=predictions,
        pair_min_tanimoto=pair_min_tanimoto_array,
        split_name='S1',
    )
    audit.save(result, output_path / 'degradation_audit.json')
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]  # 1.001 so Tanimoto=1.0 falls in last bin
_BIN_LABELS = ['[0.0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]']
DEGRADATION_AUDIT_METHOD = 'tanimoto_bin_degradation_bootstrap_v1'


def _bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | str | int | None]:
    """Return mean AUROC and 95% CI via bootstrap resampling."""
    n = len(labels)
    if n == 0 or len(np.unique(labels)) < 2:
        return {'mean': None, 'ci_lower': None, 'ci_upper': None, 'n_samples': n}
    try:
        point_estimate = float(roc_auc_score(labels, scores))
    except Exception:
        return {'mean': None, 'ci_lower': None, 'ci_upper': None, 'n_samples': n}

    bootstrap_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        boot_scores = scores[idx]
        if len(np.unique(boot_labels)) < 2:
            continue
        try:
            bootstrap_aurocs.append(float(roc_auc_score(boot_labels, boot_scores)))
        except Exception:
            continue

    if len(bootstrap_aurocs) < 10:
        return {
            'mean': point_estimate,
            'ci_lower': None,
            'ci_upper': None,
            'n_samples': n,
            'n_bootstrap_valid': len(bootstrap_aurocs),
        }

    arr = np.asarray(bootstrap_aurocs)
    return {
        'mean': point_estimate,
        'ci_lower': float(np.percentile(arr, 2.5)),
        'ci_upper': float(np.percentile(arr, 97.5)),
        'ci_method': 'percentile_bootstrap_95',
        'n_samples': n,
        'n_bootstrap_valid': len(bootstrap_aurocs),
    }


def _bootstrap_brier(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | str | int | None]:
    """Return mean Brier score and 95% CI via bootstrap resampling."""
    n = len(labels)
    if n == 0:
        return {'mean': None, 'ci_lower': None, 'ci_upper': None, 'n_samples': 0}

    point_estimate = float(np.mean((scores - labels) ** 2))
    bootstrap_briers = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_brier = float(np.mean((scores[idx] - labels[idx]) ** 2))
        bootstrap_briers.append(boot_brier)

    arr = np.asarray(bootstrap_briers)
    return {
        'mean': point_estimate,
        'ci_lower': float(np.percentile(arr, 2.5)),
        'ci_upper': float(np.percentile(arr, 97.5)),
        'ci_method': 'percentile_bootstrap_95',
        'n_samples': n,
    }


class DegradationAudit:
    """Compute per-bin degradation statistics with bootstrap confidence intervals."""

    def __init__(self, n_bootstrap: int = 1000, seed: int = 42) -> None:
        if n_bootstrap < 10:
            raise ValueError('n_bootstrap must be at least 10.')
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def run(
        self,
        labels: list | np.ndarray,
        scores: list | np.ndarray,
        pair_min_tanimoto: list | np.ndarray,
        split_name: str,
    ) -> dict[str, Any]:
        """Run degradation audit for one split.

        Parameters
        ----------
        labels : binary 0/1 array
        scores : predicted probability array
        pair_min_tanimoto : min(drug_a_tanimoto, drug_b_tanimoto) per pair
        split_name : e.g. 'Transductive', 'S1', 'S2'
        """
        labels_arr = np.asarray(labels, dtype=float)
        scores_arr = np.asarray(scores, dtype=float)
        tanimoto_arr = np.asarray(pair_min_tanimoto, dtype=float)

        if not (len(labels_arr) == len(scores_arr) == len(tanimoto_arr)):
            raise ValueError('labels, scores, and pair_min_tanimoto must have equal length.')

        rng = np.random.default_rng(self.seed)
        bin_indices = np.digitize(tanimoto_arr, _BIN_EDGES[1:-1])  # 0-indexed bins

        bins = []
        for bin_idx, bin_label in enumerate(_BIN_LABELS):
            mask = bin_indices == bin_idx
            bin_labels = labels_arr[mask]
            bin_scores = scores_arr[mask]
            n = int(mask.sum())

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                auroc = _bootstrap_auroc(bin_labels, bin_scores, self.n_bootstrap, rng)
                brier = _bootstrap_brier(bin_labels, bin_scores, self.n_bootstrap, rng)

            bins.append({
                'tanimoto_bin': bin_label,
                'n_pairs': n,
                'positive_rate': float(bin_labels.mean()) if n > 0 else None,
                'auroc': auroc,
                'brier': brier,
            })

        # Overall (all bins combined)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            overall_auroc = _bootstrap_auroc(labels_arr, scores_arr, self.n_bootstrap, rng)
            overall_brier = _bootstrap_brier(labels_arr, scores_arr, self.n_bootstrap, rng)

        return {
            'method': DEGRADATION_AUDIT_METHOD,
            'split': split_name,
            'n_bootstrap': self.n_bootstrap,
            'seed': self.seed,
            'bin_edges': _BIN_EDGES,
            'bin_labels': _BIN_LABELS,
            'overall': {
                'n_pairs': len(labels_arr),
                'auroc': overall_auroc,
                'brier': overall_brier,
            },
            'bins': bins,
            'interpretation': (
                'AUROC and Brier score are reported per nearest-training-drug Tanimoto '
                'bin, with 95% percentile bootstrap confidence intervals. '
                'Bins with fewer than 30 pairs should be interpreted with caution. '
                'This measures structural-distance-dependent performance, not '
                'clinical applicability or out-of-distribution safety.'
            ),
        }

    def run_all_splits(
        self,
        split_results: dict[str, dict[str, list]],
        pair_ood_scores: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Run degradation audit for multiple splits at once.

        Parameters
        ----------
        split_results : dict mapping split_name ->
            {'labels': [...], 'scores': [...]}
        pair_ood_scores : dict mapping split_name ->
            pair_min_tanimoto array
        """
        results = {}
        for split_name, data in split_results.items():
            if split_name not in pair_ood_scores:
                continue
            results[split_name] = self.run(
                labels=data['labels'],
                scores=data['scores'],
                pair_min_tanimoto=pair_ood_scores[split_name],
                split_name=split_name,
            )
        return {
            'method': DEGRADATION_AUDIT_METHOD,
            'splits': results,
        }

    @staticmethod
    def save(result: dict[str, Any], path: str | Path) -> None:
        """Save degradation audit result as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=2, default=str)

    @staticmethod
    def to_table_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return flat rows suitable for a LaTeX/pandas table in the paper."""
        rows = []
        split = result.get('split', 'unknown')
        for bin_data in result.get('bins', []):
            auroc = bin_data['auroc']
            brier = bin_data['brier']
            rows.append({
                'split': split,
                'tanimoto_bin': bin_data['tanimoto_bin'],
                'n_pairs': bin_data['n_pairs'],
                'auroc_mean': auroc.get('mean'),
                'auroc_ci_lower': auroc.get('ci_lower'),
                'auroc_ci_upper': auroc.get('ci_upper'),
                'brier_mean': brier.get('mean'),
                'brier_ci_lower': brier.get('ci_lower'),
                'brier_ci_upper': brier.get('ci_upper'),
            })
        return rows
