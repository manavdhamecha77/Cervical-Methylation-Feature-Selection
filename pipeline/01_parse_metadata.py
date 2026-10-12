"""
01_parse_metadata.py
--------------------
Parse the GSE99511 SOFT file to extract sample-level metadata (GSM ID,
diagnosis label) and build an Illumina-compatible sample sheet from the
raw IDAT filenames.

Inputs:
    data/GSE99511/source/GSE99511_family.soft.gz
    data/GSE99511/raw/  (IDAT files)

Outputs:
    data/GSE99511/metadata/metadata.csv           – GSM ID → diagnosis mapping
    data/GSE99511/metadata/samplesheet.csv        – methylprep sample sheet
"""

import csv
import gzip
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "GSE99511"
SOFT_FILE   = DATASET_DIR / "source" / "GSE99511_family.soft.gz"
METADATA_DIR = DATASET_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUT = METADATA_DIR / "metadata.csv"
RAW_DIR     = DATASET_DIR / "raw"
SAMPLESHEET_OUT = METADATA_DIR / "samplesheet.csv"


# ── 1. Parse SOFT file ──────────────────────────────────────────────────────

def extract_diagnosis(characteristics: list[str], title: str) -> str:
    text = " ".join(characteristics).lower() + " " + title.lower()
    if "normal" in text:
        return "Normal"
    if "cin3" in text or "cin 3" in text or "precancer" in text:
        return "CIN3"
    if "scc" in text or "squamous cell carcinoma" in text or "carcinoma" in text:
        return "SCC"
    return "Unknown"


print("Parsing SOFT file ...")
samples, current = [], None

with gzip.open(SOFT_FILE, "rt", encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("^SAMPLE ="):
            if current:
                samples.append(current)
            current = {"gsm_id": "", "title": "", "source_name": "", "characteristics": []}
        if current is None:
            continue
        if line.startswith("!Sample_geo_accession ="):
            current["gsm_id"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_title ="):
            current["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_source_name_ch1 ="):
            current["source_name"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1 ="):
            current["characteristics"].append(line.split("=", 1)[1].strip())

if current:
    samples.append(current)

rows = []
for s in samples:
    rows.append({
        "gsm_id":       s["gsm_id"],
        "sample_title": s["title"],
        "source_name":  s["source_name"],
        "disease":      extract_diagnosis(s["characteristics"], s["title"]),
        "characteristics": " | ".join(s["characteristics"]),
    })

with open(METADATA_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

counts = {}
for r in rows:
    counts[r["disease"]] = counts.get(r["disease"], 0) + 1
print(f"  Parsed {len(rows)} samples → {METADATA_OUT}")
print(f"  Distribution: {counts}")


# ── 2. Build sample sheet from IDAT filenames ───────────────────────────────

print("\nBuilding sample sheet from IDAT filenames ...")
PATTERN = re.compile(r"^(GSM\d+)_(\d+)_(R\d+C\d+)_Grn\.idat\.gz$")

sheet_rows = []
for grn in sorted(RAW_DIR.glob("*_Grn.idat.gz")):
    m = PATTERN.match(grn.name)
    if not m:
        print(f"  WARNING: cannot parse {grn.name}")
        continue
    gsm, sentrix_id, position = m.groups()
    red = grn.parent / grn.name.replace("_Grn.idat.gz", "_Red.idat.gz")
    if not red.exists():
        print(f"  WARNING: missing Red file for {gsm}")
        continue
    sheet_rows.append({"Sample_Name": gsm, "Sentrix_ID": sentrix_id,
                        "Sentrix_Position": position})

df = pd.DataFrame(sheet_rows)
df.to_csv(SAMPLESHEET_OUT, index=False)
print(f"  {len(df)} samples → {SAMPLESHEET_OUT}")
