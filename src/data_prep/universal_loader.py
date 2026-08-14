"""
universal_loader.py — v2, more robust
Now searches subfolders recursively and matches extensions
case-insensitively (fixes common upload-structure mismatches).
"""

import glob, os
import pandas as pd
import xml.etree.ElementTree as ET
try:
    import fitz
except ImportError:
    fitz = None
    print("WARNING: run pip install pymupdf")

def load_tabular_file(filepath, chunksize=None):
    sep = '\t' if 'tsv' in filepath.lower() else ','
    compression = 'gzip' if filepath.lower().endswith('.gz') else None
    try:
        if chunksize:
            chunks = pd.read_csv(filepath, sep=sep, compression=compression,
                                  chunksize=chunksize, low_memory=False, on_bad_lines='skip')
            return pd.concat(chunks, ignore_index=True)
        return pd.read_csv(filepath, sep=sep, compression=compression,
                            low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"  [SKIPPED] {filepath}: {e}"); return None

def load_xml_file(filepath):
    try:
        root = ET.parse(filepath).getroot()
        return pd.DataFrame([{c.tag: c.text for c in r} for r in root])
    except Exception as e:
        print(f"  [SKIPPED] {filepath}: {e}"); return None

def load_pdf_text(filepath):
    if fitz is None: return None
    try:
        doc = fitz.open(filepath)
        text = "\n".join(p.get_text() for p in doc)
        return pd.DataFrame([{"source_file": filepath, "text": text}])
    except Exception as e:
        print(f"  [SKIPPED] {filepath}: {e}"); return None

def load_all_files_in_folder(folder_path, chunksize=None, row_cap_per_file=None, recursive=True):
    """
    v2: searches subfolders (recursive=True) and matches extensions
    case-insensitively — e.g. .CSV, .Csv, .csv all work now.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(
            f"'{folder_path}' is not a valid folder. "
            f"Run the os.walk diagnostic cell to see your real folder structure."
        )

    all_files = []
    walk_pattern = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
    for root, _, files in walk_pattern:
        for f in files:
            ext = f.lower()
            if ext.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz', '.xml', '.pdf')):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} files under {folder_path} (recursive={recursive}):")
    for f in all_files:
        print(f"   - {f}")

    if len(all_files) == 0:
        print("\n!! NO FILES FOUND. Possible causes:")
        print("   1. Folder is genuinely empty — check Drive in your browser")
        print("   2. Files have an extension not in [.csv .tsv .csv.gz .tsv.gz .xml .pdf]")
        print("   3. Wrong folder path — re-run the os.walk diagnostic to confirm")
        return None

    dfs = []
    for fp in all_files:
        print(f"Loading: {fp}")
        low = fp.lower()
        if low.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz')):
            df = load_tabular_file(fp, chunksize)
        elif low.endswith('.xml'):
            df = load_xml_file(fp)
        elif low.endswith('.pdf'):
            df = load_pdf_text(fp)
        else:
            df = None
        if df is not None:
            if row_cap_per_file: df = df.head(row_cap_per_file)
            df['__source_file'] = os.path.basename(fp)
            dfs.append(df)

    if not dfs:
        raise ValueError(f"Files were found but none could be parsed successfully.")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nTOTAL rows: {len(combined)} from {len(dfs)} files")
    return combined