"""Reliability diagrams for calibration visualization across evaluation splits.

A reliability diagram (calibration curve) bins predictions into 10 equal-width
probability buckets and plots:
  - x-axis: mean predicted probability per bin
  - y-axis: observed positive rate (fraction of actual positives) per bin

A perfectly calibrated model has all points on the diagonal.
Overconfident models curve below; underconfident models curve above.

This module produces:
  1. A per-split reliability diagram (Transductive, S2, S1)
  2. A combined three-panel figure for the paper

The increasing miscalibration from Transductive → S2 → S1 is itself a
publishable finding: "The model's own confidence degrades on cold-start pairs."

Usage::

    from src.evaluation.reliability_diagram import plot_reliability_diagram

    plot_reliability_diagram(
        split_results={
            'Transductive': {'labels': [...], 'scores': [...]},
            'S2': {'labels': [...], 'scores': [...]},
            'S1': {'labels': [...], 'scores': [...]},
        },
        output_path=Path('figures/reliability_diagram'),
        n_bins=10,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_SPLIT_COLORS = {
    'Transductive': '#0072B2',  # Blue
    'S2': '#009E73',            # Green
    'S1': '#D55E00',            # Orange-red
    'S2-dev': '#56B4E9',
    'S1-dev': '#E69F00',
}

RELIABILITY_DIAGRAM_METHOD = 'equal_width_10bin_reliability_diagram_v1'


def compute_calibration_bins(
    labels: list | np.ndarray,
    scores: list | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute per-bin calibration statistics."""
    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(scores_arr, bin_edges[1:-1])

    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        n = int(mask.sum())
        if n == 0:
            bins.append({
                'bin_lower': float(bin_edges[i]),
                'bin_upper': float(bin_edges[i + 1]),
                'n_samples': 0,
                'mean_predicted': None,
                'fraction_positive': None,
            })
        else:
            bins.append({
                'bin_lower': float(bin_edges[i]),
                'bin_upper': float(bin_edges[i + 1]),
                'n_samples': n,
                'mean_predicted': float(scores_arr[mask].mean()),
                'fraction_positive': float(labels_arr[mask].mean()),
            })

    # ECE = weighted mean absolute calibration error
    total = len(labels_arr)
    ece = sum(
        (b['n_samples'] / total) * abs(b['mean_predicted'] - b['fraction_positive'])
        for b in bins
        if b['n_samples'] > 0
        and b['mean_predicted'] is not None
        and b['fraction_positive'] is not None
    )

    return {
        'method': RELIABILITY_DIAGRAM_METHOD,
        'n_bins': n_bins,
        'n_total': total,
        'ece': float(ece),
        'bins': bins,
    }


def plot_reliability_diagram(
    split_results: dict[str, dict[str, list]],
    output_path: str | Path,
    n_bins: int = 10,
    dpi: int = 180,
) -> dict[str, Any]:
    """Plot and save a multi-panel reliability diagram for the paper.

    Parameters
    ----------
    split_results : dict mapping split_name -> {'labels': [...], 'scores': [...]}
    output_path : path WITHOUT extension (saves .png and .pdf)
    n_bins : number of equal-width probability bins
    dpi : output resolution

    Returns
    -------
    dict with calibration statistics per split (for JSON archiving)
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        raise ImportError('matplotlib is required for reliability diagrams.')

    split_names = [s for s in ['Transductive', 'S2', 'S1'] if s in split_results]
    if not split_names:
        split_names = list(split_results.keys())

    n_panels = len(split_names)
    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs = GridSpec(1, n_panels, figure=fig, wspace=0.35)

    calibration_stats = {}

    for panel_idx, split_name in enumerate(split_names):
        data = split_results[split_name]
        stats = compute_calibration_bins(data['labels'], data['scores'], n_bins)
        calibration_stats[split_name] = stats

        ax = fig.add_subplot(gs[0, panel_idx])
        color = _SPLIT_COLORS.get(split_name, '#333333')

        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Perfect calibration')

        # Calibration curve
        valid_bins = [
            b for b in stats['bins']
            if b['mean_predicted'] is not None and b['fraction_positive'] is not None
        ]
        if valid_bins:
            xs = [b['mean_predicted'] for b in valid_bins]
            ys = [b['fraction_positive'] for b in valid_bins]
            sizes = [max(10, b['n_samples'] / stats['n_total'] * 500) for b in valid_bins]
            ax.plot(xs, ys, 'o-', color=color, linewidth=2, markersize=5, label=split_name)
            ax.scatter(xs, ys, s=sizes, color=color, alpha=0.4, zorder=5)

        ax.set(
            title=f'{split_name}\nECE={stats["ece"]:.3f}  n={stats["n_total"]:,}',
            xlabel='Mean predicted probability',
            ylabel='Fraction of positives' if panel_idx == 0 else '',
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
            aspect='equal',
        )
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.2)

    fig.suptitle(
        'PxDDI Reliability Diagrams — Calibration across evaluation protocols',
        fontsize=12,
        y=1.02,
    )
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(dest.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    return {
        'method': RELIABILITY_DIAGRAM_METHOD,
        'n_bins': n_bins,
        'splits': calibration_stats,
        'output_png': str(dest.with_suffix('.png')),
        'output_pdf': str(dest.with_suffix('.pdf')),
        'interpretation': (
            'A perfectly calibrated model follows the diagonal. '
            'Increasing ECE from Transductive to S1 indicates the model '
            'is less reliably calibrated on cold-start drug pairs. '
            'This does not imply clinical safety or DDI ruling-out capability.'
        ),
    }
