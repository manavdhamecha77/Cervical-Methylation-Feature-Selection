"""
annotator.py
------------
CpG annotation and biological overlap analysis.

1. Load Illumina 450K manifest (GPL13534) if available
2. Annotate selected CpGs with gene name, chromosome, genomic region, CpG island relation
3. Check overlap with known cervical methylation markers
4. Check pairwise correlation among selected CpGs (redundancy for sensor panels)

NOTE: No annotations are fabricated. If the manifest is not found at the expected
path, the function returns empty annotation with a clear warning.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MANIFEST_PATHS = [
    os.path.join(ROOT, "data", "GSE99511", "raw",
                 "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"),
    os.path.join(ROOT, "data", "GSE99511", "intermediate",
                 "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"),
]

# Columns to extract from the manifest
MANIFEST_USECOLS = [
    "IlmnID", "CHR", "MAPINFO",
    "UCSC_RefGene_Name", "UCSC_RefGene_Group",
    "Relation_to_UCSC_CpG_Island", "UCSC_CpG_Islands_Name",
]

MANIFEST_RENAME = {
    "IlmnID":                      "CpG",
    "CHR":                         "Chromosome",
    "MAPINFO":                     "Position",
    "UCSC_RefGene_Name":           "Gene",
    "UCSC_RefGene_Group":          "Gene_Region",
    "Relation_to_UCSC_CpG_Island": "CpG_Island_Relation",
    "UCSC_CpG_Islands_Name":       "CpG_Island_Name",
}


def load_manifest(verbose: bool = True) -> pd.DataFrame:
    """
    Try to load the Illumina 450K manifest from known paths.
    Returns a DataFrame indexed by CpG ID, or an empty DataFrame with a warning.
    """
    for path in MANIFEST_PATHS:
        if not os.path.exists(path):
            continue
        if verbose:
            print(f"[Annotator] Loading manifest: {path}")
        try:
            annot = pd.read_csv(
                path,
                skiprows=7,
                low_memory=False,
                usecols=lambda c: c in MANIFEST_USECOLS,
            )
            annot = annot.rename(columns=MANIFEST_RENAME)
            # Keep only actual CpG probes
            annot = annot[annot["CpG"].str.startswith("cg", na=False)].copy()
            annot = annot.drop_duplicates(subset="CpG").set_index("CpG")
            if verbose:
                print(f"[Annotator] Manifest loaded: {len(annot):,} probes")
            return annot
        except Exception as e:
            if verbose:
                print(f"[Annotator] WARNING: Could not parse manifest at {path}: {e}")

    if verbose:
        print("[Annotator] WARNING: Manifest not found at expected paths.")
        print("  Expected: data/GSE99511/raw/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz")
        print("  File is included in your repo — it is likely already present.")
    return pd.DataFrame(columns=list(MANIFEST_RENAME.values()))


def annotate_cpgs(cpg_list: list, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Look up a list of CpG IDs in the manifest.
    Returns a DataFrame with one row per CpG, NA where not found.
    """
    if manifest.empty:
        df = pd.DataFrame({"CpG": cpg_list})
        for col in ["Chromosome", "Position", "Gene", "Gene_Region",
                    "CpG_Island_Relation", "CpG_Island_Name"]:
            df[col] = np.nan
        return df

    result = manifest.reindex(cpg_list).reset_index()
    result = result.rename(columns={"index": "CpG"})
    if "CpG" not in result.columns and manifest.index.name == "CpG":
        result.insert(0, "CpG", cpg_list)
    return result


def known_marker_overlap(
    cpg_list: list,
    manifest: pd.DataFrame,
    known_genes: list,
) -> pd.DataFrame:
    """
    Check how many selected CpGs map to known cervical cancer methylation marker genes.

    Returns a DataFrame with columns:
        CpG, Gene, Chromosome, Position, CpG_Island_Relation, known_marker_gene

    NOTE: This is for SANITY CHECKING only, not validation.
    The known markers are from clinical literature (e.g. FAM19A4, PAX1, ZNF582).
    Overlap does not validate a CpG; absence does not invalidate it.
    """
    annot_df = annotate_cpgs(cpg_list, manifest)

    if manifest.empty or "Gene" not in annot_df.columns:
        annot_df["known_marker_gene"] = False
        annot_df["matched_known_gene"] = ""
        return annot_df

    # Extract primary gene name (take first semicolon-separated entry)
    def _primary_gene(val):
        if pd.isna(val) or val == "":
            return ""
        return str(val).split(";")[0].strip().upper()

    known_upper = {g.upper() for g in known_genes}
    annot_df["primary_gene"]     = annot_df["Gene"].apply(_primary_gene)
    annot_df["known_marker_gene"] = annot_df["primary_gene"].isin(known_upper)
    annot_df["matched_known_gene"] = annot_df["primary_gene"].apply(
        lambda g: g if g in known_upper else ""
    )
    return annot_df


def correlation_among_selected(
    cpg_list: list,
    X_full: np.ndarray,
    cpg_names: list,
    imputed: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation among selected CpGs.
    Uses globally mean-imputed X (for descriptive purposes only — clearly flagged).

    Returns a (n_cpg x n_cpg) correlation DataFrame.
    Also returns summary statistics (max, mean absolute correlation).

    NOTE: This is computed on the full dataset as a descriptive/exploratory step,
    NOT used for any CV metric. The resulting correlation is for sensor panel design:
    highly correlated CpGs are redundant and should be pruned.
    """
    cpg_arr = np.array(cpg_names)
    available = [c for c in cpg_list if c in cpg_names]
    missing   = [c for c in cpg_list if c not in cpg_names]
    if missing:
        print(f"[Annotator] {len(missing)} CpGs not found in feature matrix, skipping them.")

    if not available:
        return pd.DataFrame(), {}

    indices = [np.where(cpg_arr == c)[0][0] for c in available]
    X_sub   = X_full[:, indices].astype(np.float64)

    if not imputed:
        # Mean-impute for descriptive correlation only
        col_means = np.nanmean(X_sub, axis=0)
        col_means = np.where(np.isnan(col_means), 0.5, col_means)
        nan_mask  = np.isnan(X_sub)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X_sub[rows, cols] = col_means[cols]

    # Pearson correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(X_sub.T)

    corr_df = pd.DataFrame(corr_matrix, index=available, columns=available)

    # Summary
    n = len(available)
    if n > 1:
        upper = corr_matrix[np.triu_indices(n, k=1)]
        abs_upper = np.abs(upper)
        summary = {
            "n_cpgs":          n,
            "max_abs_corr":    round(abs_upper.max(), 3),
            "mean_abs_corr":   round(abs_upper.mean(), 3),
            "median_abs_corr": round(np.median(abs_upper), 3),
            "frac_high_corr":  round((abs_upper > 0.7).mean(), 3),
            "note":            "Computed on globally imputed full dataset (descriptive only)",
        }
    else:
        summary = {"n_cpgs": n, "note": "Only 1 CpG — no pairwise correlation."}

    return corr_df, summary


def build_annotation_report(
    selector: str,
    k: int,
    cpg_list: list,
    manifest: pd.DataFrame,
    known_genes: list,
    X_full: np.ndarray,
    cpg_names: list,
) -> dict:
    """
    Full annotation report for a given (selector, k) consensus panel.
    Returns a dict with DataFrames and summary statistics.
    """
    # Annotate
    annot_df = known_marker_overlap(cpg_list, manifest, known_genes)

    # Correlation
    if len(cpg_list) >= 2:
        corr_df, corr_summary = correlation_among_selected(
            cpg_list, X_full, cpg_names, imputed=False
        )
    else:
        corr_df      = pd.DataFrame()
        corr_summary = {}

    # Known marker overlap summary
    n_known = int(annot_df["known_marker_gene"].sum()) if "known_marker_gene" in annot_df.columns else 0
    known_cpgs = (
        annot_df[annot_df["known_marker_gene"]]["CpG"].tolist()
        if "known_marker_gene" in annot_df.columns
        else []
    )

    return {
        "selector":        selector,
        "k":               k,
        "n_cpgs":          len(cpg_list),
        "annotation":      annot_df,
        "correlation":     corr_df,
        "corr_summary":    corr_summary,
        "n_known_markers": n_known,
        "known_cpgs":      known_cpgs,
        "note":            (
            "Known marker overlap is for biological sanity checking ONLY. "
            "Overlap does not validate and absence does not invalidate a CpG."
        ),
    }
