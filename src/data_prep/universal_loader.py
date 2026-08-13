"""Loads EVERY file in a folder (csv/tsv/gz/xml/pdf) — fixes 'only 1 of 100 files' bug."""
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
    compression = 'gzip' if filepath.endswith('.gz') else None
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

def load_all_files_in_folder(folder_path, chunksize=None, row_cap_per_file=None):
    patterns = ['*.csv', '*.tsv', '*.csv.gz', '*.tsv.gz', '*.xml', '*.pdf']
    all_files = []
    for p in patterns: all_files.extend(glob.glob(os.path.join(folder_path, p)))
    print(f"Found {len(all_files)} files in {folder_path}")
    for f in all_files: print(f"   - {os.path.basename(f)}")
    dfs = []
    for fp in all_files:
        print(f"Loading: {os.path.basename(fp)}")
        if fp.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz')): df = load_tabular_file(fp, chunksize)
        elif fp.endswith('.xml'): df = load_xml_file(fp)
        elif fp.endswith('.pdf'): df = load_pdf_text(fp)
        else: df = None
        if df is not None:
            if row_cap_per_file: df = df.head(row_cap_per_file)
            df['__source_file'] = os.path.basename(fp)
            dfs.append(df)
    if not dfs: raise ValueError(f"No loadable files in {folder_path}")
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"TOTAL rows: {len(combined)} from {len(dfs)} files")
    return combined
