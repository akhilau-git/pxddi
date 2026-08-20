"""Write a cross-seed stability report from completed candidate explanation JSONs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.explanation_stability import compare_explanation_artifacts


def _input_paths() -> list[Path]:
    raw = os.environ.get('PXDDI_EXPLANATION_ARTIFACTS', '')
    paths = [Path(part.strip()) for part in raw.split(',') if part.strip()]
    if len(paths) < 2:
        raise ValueError(
            'PXDDI_EXPLANATION_ARTIFACTS must contain at least two comma-separated '
            'candidate_occlusion_explanations.json paths.'
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Explanation artifacts not found: {missing}.')
    return paths


def main() -> None:
    paths = _input_paths()
    artifacts = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    report = compare_explanation_artifacts(artifacts)
    output_path = Path(os.environ.get(
        'PXDDI_EXPLANATION_STABILITY_OUTPUT',
        paths[0].parent / 'cross_seed_explanation_stability.json',
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        'input_artifact_paths': [str(path) for path in paths],
        **report,
    }, indent=2, sort_keys=True), encoding='utf-8')
    print(f'Cross-seed explanation stability report saved to: {output_path}')


if __name__ == '__main__':
    main()
