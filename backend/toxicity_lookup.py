"""
toxicity_lookup.py
Loads the real toxicity bridge (built during training) so the backend
can honestly report whether a drug's toxicity score is REAL data or
an unset default — never silently presenting 'unknown' as 'safe'.
"""
import sys
from pathlib import Path
import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
from data_prep.pubchem_bridge import canonicalize, resolve_toxicity_bridge

def load_known_toxicity_smiles(
    bridge_csv_path: Path,
) -> tuple[set[str], str | None, dict[str, int] | None]:
    """Load clean toxicity-label coverage and preserve health diagnostics.

    A structure with conflicting FAERS-derived source scores is intentionally
    excluded from the usable training-label set. The API must not describe it
    as a clean toxicity label merely because it appears somewhere in the raw
    bridge CSV.
    """
    try:
        bridge = pd.read_csv(bridge_csv_path)
        if 'canonical_smiles' not in bridge.columns:
            raise ValueError("toxicity bridge is missing the 'canonical_smiles' column")
        resolved, summary, _ = resolve_toxicity_bridge(bridge)
        return set(resolved['canonical_smiles'].astype(str)), None, summary
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        message = f'Could not load toxicity bridge: {error}'
        print(f'Warning: {message}')
        return set(), message, None

CHECKPOINTS_DIR = Path(__file__).resolve().parent / 'checkpoints'
(
    KNOWN_TOXICITY_SMILES,
    TOXICITY_BRIDGE_ERROR,
    TOXICITY_BRIDGE_SUMMARY,
) = load_known_toxicity_smiles(CHECKPOINTS_DIR / 'toxicity_smiles_bridge.csv')


def is_toxicity_known(smiles: str, known_canonical_smiles=None) -> bool:
    """Check bridge coverage using canonical, rather than raw, SMILES."""
    canonical_smiles = canonicalize(smiles)
    known_smiles = (
        KNOWN_TOXICITY_SMILES
        if known_canonical_smiles is None
        else known_canonical_smiles
    )
    return canonical_smiles is not None and canonical_smiles in known_smiles
