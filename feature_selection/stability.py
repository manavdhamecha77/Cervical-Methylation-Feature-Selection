"""
stability.py
------------
Feature selection stability analysis.

For each (selector, k) combination:
  - Jaccard index: mean pairwise Jaccard between all fold feature sets
  - Kuncheva index (KI): bias-corrected stability measure
  - Per-CpG selection frequency across all outer folds × repeats
  - Consensus panel: CpGs selected in the most folds

Kuncheva Index
--------------
KI(S) = (1/C(n,2)) * Σ_{i<j} (|Si ∩ Sj| / |Si ∪ Sj|  -  k/p)
         / (1 - k/p)

where k = panel size, p = total feature space size, C(n,2) = num fold pairs.
Range: [-1, 1]. Higher = more stable. 0 = random chance, 1 = perfectly stable.

Reference: Kuncheva (2007) Journal of Machine Learning Research.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter


def jaccard_pairwise(feature_sets: list) -> float:
    """
    Mean pairwise Jaccard similarity across all pairs of feature sets.
    Each element of feature_sets is a set (or list) of feature identifiers.
    Returns float in [0, 1]. Returns NaN if fewer than 2 sets.
    """
    sets = [set(fs) for fs in feature_sets]
    n = len(sets)
    if n < 2:
        return float("nan")

    scores = []
    for i, j in combinations(range(n), 2):
        inter = len(sets[i] & sets[j])
        union = len(sets[i] | sets[j])
        if union == 0:
            scores.append(0.0)
        else:
            scores.append(inter / union)
    return float(np.mean(scores)) if scores else float("nan")


def kuncheva_index(feature_sets: list, k: int, p: int) -> float:
    """
    Kuncheva Index (stability measure corrected for chance).

    Parameters
    ----------
    feature_sets : list of sets/lists of selected feature INDICES or NAMES
    k            : panel size (number of selected features per fold)
    p            : total feature space size (before selection)

    Returns float in [-1, 1], or NaN if calculation is undefined.
    """
    sets = [set(fs) for fs in feature_sets]
    n = len(sets)
    if n < 2:
        return float("nan")
    if p <= k:
        return float("nan")

    baseline = k / p
    if baseline == 1.0:
        return float("nan")

    scores = []
    for i, j in combinations(range(n), 2):
        inter = len(sets[i] & sets[j])
        # KI per pair = (|inter|/k - k/p) / (1 - k/p)
        ki_pair = (inter / k - baseline) / (1.0 - baseline)
        scores.append(ki_pair)

    return float(np.mean(scores)) if scores else float("nan")


def compute_stability(fold_df: pd.DataFrame,
                      total_cpgs: int) -> pd.DataFrame:
    """
    Compute stability metrics for every (selector, k, prefilter_n) combination.

    Parameters
    ----------
    fold_df    : DataFrame with columns [selector, k, prefilter_n, selected_cpgs]
                 where selected_cpgs is a comma-separated string of CpG IDs
    total_cpgs : total feature space size p (e.g. 405336 post-QC CpGs)

    Returns
    -------
    stability_df: one row per (selector, k, prefilter_n) with
                  jaccard_mean, kuncheva_index, n_folds
    """
    rows = []
    group_cols = ["selector", "k"]
    if "prefilter_n" in fold_df.columns:
        group_cols.append("prefilter_n")

    for keys, grp in fold_df.groupby(group_cols):
        if isinstance(keys, (str, int, float)):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        k   = int(rec["k"])

        # Parse selected CpG sets for each fold
        feat_sets = []
        for val in grp["selected_cpgs"].dropna():
            if isinstance(val, str) and val.strip():
                feat_sets.append(set(val.split(",")))
            elif isinstance(val, (list, set)):
                feat_sets.append(set(val))

        rec["n_folds"]       = len(feat_sets)
        rec["jaccard_mean"]  = round(jaccard_pairwise(feat_sets), 4) if feat_sets else np.nan
        rec["kuncheva_index"] = round(
            kuncheva_index(feat_sets, k, total_cpgs), 4
        ) if feat_sets else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


def compute_selection_frequency(fold_df: pd.DataFrame) -> dict:
    """
    For each (selector, k, prefilter_n), return a DataFrame:
      columns: CpG, frequency (0-1), count (int)
    sorted by frequency descending.

    Returns dict keyed by (selector, k) tuple.
    """
    result = {}
    group_cols = ["selector", "k"]
    if "prefilter_n" in fold_df.columns:
        group_cols.append("prefilter_n")

    for keys, grp in fold_df.groupby(group_cols):
        if isinstance(keys, (str, int, float)):
            keys = (keys,)
        key = tuple(keys)

        n_folds = len(grp)
        counter = Counter()
        for val in grp["selected_cpgs"].dropna():
            if isinstance(val, str) and val.strip():
                for cpg in val.split(","):
                    if cpg:
                        counter[cpg] += 1

        freq_df = pd.DataFrame([
            {"CpG": cpg, "count": cnt, "frequency": round(cnt / n_folds, 4)}
            for cpg, cnt in counter.items()
        ]).sort_values("frequency", ascending=False).reset_index(drop=True)

        result[key] = freq_df

    return result


def consensus_panel(frequency_dict: dict, selector: str,
                    k: int, top_n: int = None) -> list:
    """
    Return the consensus panel for a given (selector, k): CpGs sorted by
    selection frequency, optionally truncated to top_n.
    Tries both (selector, k) and (selector, k, prefilter_n) keys.
    """
    top_n = top_n or k
    # Find matching key
    matching = [key for key in frequency_dict if key[0] == selector and key[1] == k]
    if not matching:
        return []
    key = matching[0]
    freq_df = frequency_dict[key]
    return freq_df.head(top_n)["CpG"].tolist()


def build_frequency_heatmap_data(frequency_dict: dict,
                                 selectors: list,
                                 k: int,
                                 top_n: int = 50) -> pd.DataFrame:
    """
    Build a (CpG x selector) heatmap dataframe of selection frequencies.
    Rows: union of top-top_n CpGs across all selectors for given k.
    Columns: selector names.
    Values: frequency (0-1).
    """
    all_cpgs = set()
    for sel in selectors:
        cpgs = consensus_panel(frequency_dict, sel, k, top_n=top_n)
        all_cpgs.update(cpgs)

    heatmap = pd.DataFrame(index=sorted(all_cpgs))
    for sel in selectors:
        matching = [key for key in frequency_dict if key[0] == sel and key[1] == k]
        if not matching:
            heatmap[sel] = 0.0
            continue
        freq_df = frequency_dict[matching[0]].set_index("CpG")
        heatmap[sel] = freq_df["frequency"].reindex(heatmap.index, fill_value=0.0)

    # Sort rows by mean frequency across selectors
    heatmap["_mean"] = heatmap.mean(axis=1)
    heatmap = heatmap.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    return heatmap
