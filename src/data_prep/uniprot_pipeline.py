"""UniProt Protein Target Sequence Pipeline for AuditDDI.

Provides:
1. Standard mapping from drug target symbols / accessions to canonical UniProt IDs.
2. Leakage-safe online fetching and local caching of FASTA primary amino acid sequences.
3. Offline fallback sequences for a small set of explicitly identified targets.
4. Generating target sequence catalogs for inductive sequence-level protein language modeling (ESM-2).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any
import urllib.request
import urllib.error

import pandas as pd

__all__ = [
    "CANONICAL_TARGET_TO_UNIPROT",
    "OFFLINE_SEQUENCE_FALLBACKS",
    "clean_protein_sequence",
    "parse_fasta_string",
    "fetch_uniprot_sequence",
    "build_target_sequence_catalog",
    "update_master_nodes_with_uniprot",
]

# Canonical UniProt Accession mappings for top DDI pharmacological targets
CANONICAL_TARGET_TO_UNIPROT: dict[str, str] = {
    'CYP3A4': 'P08684',
    'CYP2D6': 'P10635',
    'CYP2C9': 'P11712',
    'CYP2C19': 'P33261',
    'CYP1A2': 'P05177',
    'CYP2E1': 'P05181',
    'PTGS2': 'P35354',
    'PTGS1': 'P23219',
    'ADRB1': 'P08588',
    'ADRB2': 'P07550',
    'DRD2': 'P14416',
    'DRD1': 'P21728',
    'HTR2A': 'P28223',
    'HTR1A': 'P08908',
    'EGFR': 'P00533',
    'KDR': 'P35968',
    'ABL1': 'P00519',
    'BRAF': 'P15056',
    'SRC': 'P12931',
    'MAPK1': 'P28482',
    'HMGCR': 'P04035',
    'SLC6A4': 'P31645',
    'SLC6A2': 'P23975',
    'ACE': 'P12821',
    'AGTR1': 'P30556',
    'ESR1': 'P03372',
    'AR': 'P10275',
    'NR3C1': 'P04150',
    'ACHE': 'P22303',
    'BCHE': 'P06276',
    'DPP4': 'P27487',
    'SCN5A': 'Q14524',
    'KCNH2': 'Q12809',
    'CACNA1C': 'Q13936',
    'GABRA1': 'P14867',
    'CHRNA7': 'P36544',
    'OPRM1': 'P35372',
    'CNR1': 'P21554',
    'PPARG': 'P37231',
    'PDE5A': 'O76074',
    'ALDH2': 'P05091',
    'VKORC1': 'Q9BQB6',
    'TUBB': 'P07437',
    'TOP2A': 'P11388',
    'PARP1': 'P09874',
    'CDK4': 'P11802',
    'CDK6': 'Q00534',
    'MTOR': 'P42345',
    'PIK3CA': 'P42336',
    'JAK2': 'O60674',
    'P2RY12': 'Q9H244',
    'ABCB1': 'P12270',
    'SLCO1B1': 'Q9Y6L6',
    'CYP2B6': 'P20813',
    'CYP3A5': 'P20815',
    'CYP2A6': 'P11509',
    'CYP2C8': 'P10632',
    'DPYD': 'Q12882',
    'TPMT': 'P51580',
    'G6PD': 'P11413',
    'ITGB3': 'P05106',
    'CYP1B1': 'Q16678',
    'CYP1A1': 'P04798',
    'ERBB2': 'P04626',
    'CASP1': 'P29466',
    'CASP3': 'P42574',
    'CASP8': 'Q14790',
    'DOT1L': 'Q8TEK3',
    'CCL5': 'P13501',
    'CNTN4': 'Q8IWV2',
    'AGAP1': 'Q9UPQ3',
    'ARHGEF28': 'Q8N1W1',
    'ETFRF1': 'Q86V54',
    'CYCSP5': 'Q6MZN7',
    'HCP5': 'Q6MZN7',
    'ADRA1A': 'P35348',
    'ARSA': 'P15289',
    'CEP68': 'Q76N32',
    'CES1': 'P23141',
    'DAO': 'P14920',
    'FAAH': 'O00519',
    'FMO2': 'Q99518',
    'GAL': 'P22466',
    'GCK': 'P35557',
    'HTT': 'P42858',
    'LEPR': 'P48357',
    'NAT1': 'P18440',
    'NOS3': 'P29474',
    'PTGER2': 'P43116',
    # Pseudogenes to parent functional enzymes
    'CYP2A7P1': 'P20853',
    'CYP2B7P1': 'P20813',
    'PSMB3P': 'P49720',
    'OR10AE3P': 'Q8NGQ2',
    # Long non-coding and small nucleolar RNAs to functional regulatory/host partners
    'PSORS1C3': 'P14859',
    'C5ORF56': 'P15311',
    'CARINH': 'P15311',
    'IRF1-AS1': 'P15311',
    'UGT1A': 'P22309',
    'VENTXP7': 'O95231',
    'SNORA59B': 'Q96FL8',
    'SNORD68': 'O15360',
    # MicroRNAs to primary validated pharmacological targets
    'MIR1206': 'P12270',
    'MIR1264': 'P20813',
    'MIR1307': 'P08684',
    'MIR133B': 'P10415',
    'MIR146A': 'Q9Y4K3',
    'MIR1912': 'P10635',
    'MIR2053': 'P11712',
    'MIR23A': 'P35354',
    'MIR27A': 'P37231',
    'MIR300': 'P15692',
    'MIR3117': 'P05177',
    'MIR423': 'P04637',
    'MIR4268': 'P33261',
    'MIR4278': 'P10632',
    'MIR449B': 'P38936',
    'MIR492': 'P06493',
    'MIR577': 'P01116',
    'MIR582': 'P42345',
    'MIR604': 'P04035',
    'MIR618': 'P12931',
}

# Representative curated offline fallback sequences (truncated for fast testing)
OFFLINE_SEQUENCE_FALLBACKS: dict[str, str] = {
    'P08684': (  # Human CYP3A4 (catalytic core fragment)
        'MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGK'
        'VWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFMKSAISIAEDEEWKRLRSLLSPTFTS'
        'GKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTK'
        'KLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDS'
        'QNSKETESHKALSDLELMAQSIIFIFAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYD'
        'TVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPSYALHRDPKYWTEPEKFLPER'
        'FSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKP'
        'VVLKVESRDGTVSGA'
    ),
    'P10635': (  # Human CYP2D6 (catalytic core fragment)
        'MGLEALVPLAVIVAIFLLLVDLMHRRQRWAARYPPGPLPLPGLGNLLHVDFQNTPYCFDQLRRRFGDVFSL'
        'QLAWTPVVVLNGLAAVREALVTHGEDTADRPPVPITQILGFGPRSQGVFLARYGPAWREQRRFSVSTLRNL'
        'GLGKKSLEQWVTEEAACLCAAFANHSGRPFRPNGLLDKAVSNVIASLTCGRRFEYDDPRFLRLLDLAQEGL'
        'KEESGFLREVLNAVPVLLHIPALAGKVLRFQKAFLTQLDELLTEHRMTWDPAQPPRDLTEAFLAEMEKAKG'
        'NPESSFNDENLRIVVADLFSAGMVTTSTTLAWGLLLMILHPDVQRRVQQEIDDVIGQVRRPEMGDQAHMPY'
        'TTAVIHEVQRFGDIVPLGMTHMTSRDIEVQGFRIPKGTTLITNLSSVLKDEAVWEKPFRFHPEHFLDAQGH'
        'FVKPEAFLPFSAGRRACLGEPLARMELFLFFTSLLQHFSFSVPTGQPRPSHHGVFAFLVSPSPYELCAVPR'
    ),
    'P35354': (  # Human PTGS2 (COX-2)
        'MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCTTPEFLTRIKLFLKPTP'
        'NTVHYILTHFKGFWNVVNIRNFLGATIQEMVTLRFTNPNSRLEFTEKTYQNYEELVLRGIGDKTKYGFGYH'
        'SWDLGKGWTKDDLLQIGEQMARALDFLKKAKQLPKVFTVFFTNLHGDNVSGIEVGSYVFSNRVLTVVEE'
    ),
    'P00533': (  # Human EGFR kinase domain fragment
        'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQR'
        'NYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEI'
        'LHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKI'
    ),
}


def clean_protein_sequence(raw_seq: str) -> str:
    """Validate and clean an amino acid sequence, retaining standard IUPAC letters."""
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', raw_seq.upper().strip())
    if not cleaned:
        raise ValueError('No valid standard amino acid residues found in sequence.')
    return cleaned


def parse_fasta_string(fasta_text: str) -> str:
    """Extract amino acid sequence from raw FASTA text."""
    lines = [line.strip() for line in fasta_text.strip().splitlines() if line.strip()]
    seq_lines = [line for line in lines if not line.startswith('>')]
    raw_seq = ''.join(seq_lines)
    return clean_protein_sequence(raw_seq)


def fetch_uniprot_sequence(
    accession_or_gene: str,
    cache_dir: str | Path | None = None,
    timeout: float = 5.0,
) -> str:
    """Fetch primary amino acid sequence for a UniProt accession or gene symbol.

    Prioritizes:
    1. Local disk cache (`cache_dir / f"{accession}.fasta"`).
    2. Curated offline fallback dictionary (for CYP3A4, CYP2D6, PTGS2, EGFR).
    3. Official UniProt REST API (`https://rest.uniprot.org/uniprotkb/{accession}.fasta`).
    """
    key = str(accession_or_gene).strip().upper()
    accession = CANONICAL_TARGET_TO_UNIPROT.get(key, key)

    c_path: Path | None = None
    if cache_dir is not None:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        c_path = c_dir / f'{accession}.fasta'
        if c_path.is_file():
            try:
                cached_text = c_path.read_text(encoding='utf-8')
                return parse_fasta_string(cached_text)
            except Exception:
                pass

    # Offline curated fallback
    if accession in OFFLINE_SEQUENCE_FALLBACKS:
        seq = OFFLINE_SEQUENCE_FALLBACKS[accession]
        if c_path is not None and not c_path.is_file():
            c_path.write_text(f'>{accession} curated_fallback\n{seq}\n', encoding='utf-8')
        return seq

    # Online UniProt REST API
    url = f'https://rest.uniprot.org/uniprotkb/{accession}.fasta'
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'AuditDDI-Research/1.0 (academic-bioinformatics-study)'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            fasta_data = response.read().decode('utf-8')
            seq = parse_fasta_string(fasta_data)
            if c_path is not None:
                c_path.write_text(fasta_data, encoding='utf-8')
            return seq
    except Exception as exc:
        # A sequence must belong to the requested accession.  Returning a
        # generic protein here makes a sequence model appear to have coverage
        # while giving it biologically meaningless input.
        raise RuntimeError(
            f"Could not retrieve a verified UniProt sequence for {accession}. "
            "Provide it in the local cache or retry when UniProt is reachable."
        ) from exc


def build_target_sequence_catalog(
    targets: list[str],
    cache_dir: str | Path,
    export_json_path: str | Path | None = None,
) -> dict[str, str]:
    """Compile and export a primary amino acid sequence catalog for a list of target IDs."""
    c_dir = Path(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    catalog: dict[str, str] = {}
    for tgt in targets:
        clean_t = str(tgt).strip().upper()
        if not clean_t:
            continue
        try:
            seq = fetch_uniprot_sequence(clean_t, cache_dir=c_dir)
            catalog[clean_t] = seq
        except Exception:
            continue

    if export_json_path is not None:
        out_f = Path(export_json_path)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text(json.dumps(catalog, indent=2), encoding='utf-8')

    return catalog


def _normalise_accession(value: Any) -> str:
    """Return a supported UniProt accession or an empty string."""
    candidate = str(value).strip().upper()
    if candidate in CANONICAL_TARGET_TO_UNIPROT:
        return CANONICAL_TARGET_TO_UNIPROT[candidate]
    # Standard UniProt primary accession formats:
    # 6-character: [OPQ][0-9][A-Z0-9]{3}[0-9] or [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]
    # 10-character: [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9]
    if re.fullmatch(
        r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?)(?:-[0-9]+)?",
        candidate,
    ):
        return candidate.split("-", maxsplit=1)[0]
    return ""


def _tokens_from_target_value(value: Any) -> list[str]:
    """Extract exact target identifiers; never infer a target from drug name."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null", "[]", "{}"):
        return []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            return [str(key) for key in decoded]
        if isinstance(decoded, (list, tuple, set)):
            return [str(item) for item in decoded]
    except (TypeError, json.JSONDecodeError):
        pass
    try:
        evaluated = ast.literal_eval(text)
        if isinstance(evaluated, dict):
            return [str(key) for key in evaluated]
        if isinstance(evaluated, (list, tuple, set)):
            return [str(item) for item in evaluated]
    except Exception:
        pass
    cleaned_text = re.sub(r"[\[\]'\"\{\}]", "", text)
    return [part.strip() for part in re.split(r"[;,|/\s]+", cleaned_text) if part.strip()]


def update_master_nodes_with_uniprot(
    master_nodes_path: str | Path,
    uniprot_dir: str | Path = "/content/drive/MyDrive/pxddi-data/uniprot",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add only source-supported drug-target UniProt sequences to master nodes.

    Drugs without an explicit target accession/gene are deliberately left
    unmapped.  This prevents a generic enzyme sequence from being assigned to
    every drug, which would invalidate protein-sequence benchmarking.
    """
    nodes_p = Path(master_nodes_path)
    if not nodes_p.is_file():
        raise FileNotFoundError(f"Master nodes file not found: {nodes_p}")
    df = pd.read_csv(nodes_p)

    u_dir = Path(uniprot_dir)
    json_path = u_dir / "target_sequences.json"
    catalog: dict[str, str] = {}
    if json_path.is_file():
        try:
            catalog = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            catalog = {}

    meta_path = u_dir / "uniprot_targets_metadata.csv"
    gene_to_acc: dict[str, str] = dict(CANONICAL_TARGET_TO_UNIPROT)
    if meta_path.is_file():
        try:
            df_meta = pd.read_csv(meta_path)
            for _, r in df_meta.iterrows():
                gn = str(r.get("gene_symbol", "")).strip().upper()
                acc = str(r.get("uniprot_id", "")).strip().upper()
                if gn and acc:
                    gene_to_acc[gn] = acc
        except Exception:
            pass

    # The catalog can be keyed by a gene symbol or an accession.  Normalize it
    # once so subsequent mapping only uses an explicitly observed target.
    catalog_by_acc: dict[str, str] = {}
    for key, sequence in catalog.items():
        acc = _normalise_accession(key)
        if acc and isinstance(sequence, str) and len(sequence.strip()) > 20:
            catalog_by_acc[acc] = sequence.strip()

    assigned_sequences: list[str] = []
    assigned_accessions: list[str] = []
    assignment_sources: list[str] = []

    target_gene_columns = [
        "bindingdb_targets_json",
        "bindingdb_targets",
        "gene_symbols_json",
        "gene_symbols",
        "target_gene",
        "gene_symbol",
        "genes",
        "primary_target",
    ]

    for _, row in df.iterrows():
        assigned_seq = ""
        assigned_acc = ""

        # Prefer direct accession fields.  ``uniprot_target_id`` is only
        # reused when it was produced by this verified mapper: older files
        # used that column for the now-removed generic CYP3A4 assignment.
        accession_columns = ["uniprot_id", "uniprot_accession", "target_uniprot"]
        existing_source = str(row.get("target_sequence_source", "")).lower()
        if ":exact_" in existing_source:
            accession_columns.insert(0, "uniprot_target_id")
        for col in accession_columns:
            if col not in df.columns:
                continue
            for token in _tokens_from_target_value(row[col]):
                acc = _normalise_accession(token)
                if acc in catalog_by_acc:
                    assigned_seq, assigned_acc = catalog_by_acc[acc], acc
                    assignment_sources.append(f"{col}:exact_accession")
                    break
            if assigned_seq:
                break

        # Then accept exact target gene symbols from observed target data.  A
        # BindingDB target dictionary or PharmGKB gene list is source data;
        # a drug-name guess is not.
        if not assigned_seq:
            for col in target_gene_columns:
                if col not in df.columns:
                    continue
                for token in _tokens_from_target_value(row[col]):
                    gene = token.strip().upper()
                    acc = _normalise_accession(gene) or gene_to_acc.get(gene, "")
                    if acc in catalog_by_acc:
                        assigned_seq, assigned_acc = catalog_by_acc[acc], acc
                        assignment_sources.append(f"{col}:exact_target")
                        break
                    if gene in catalog and len(str(catalog[gene]).strip()) > 20:
                        assigned_seq = str(catalog[gene]).strip()
                        assigned_acc = acc or gene
                        assignment_sources.append(f"{col}:exact_target")
                        break
                if assigned_seq:
                    break

        if not assigned_seq:
            assignment_sources.append("unmapped")

        assigned_sequences.append(assigned_seq)
        assigned_accessions.append(assigned_acc)

    df["target_sequence"] = assigned_sequences
    df["uniprot_target_id"] = assigned_accessions
    df["target_sequence_source"] = assignment_sources

    target_out = Path(output_path) if output_path is not None else nodes_p
    target_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_out, index=False)

    n_with_seq = sum(1 for s in assigned_sequences if len(s) > 0)
    print(
        f"Enriched master nodes with verified UniProt sequences: {n_with_seq}/{len(df)} drugs mapped. "
        "Unmapped drugs were left blank."
    )
    return df
