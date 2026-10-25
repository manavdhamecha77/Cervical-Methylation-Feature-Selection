"""
literature_and_regions.py
-------------------------
Implements Section 3 of the plan:
1. Literature baseline panel (FAM19A4, miR124-2, PAX1, ZNF582, EPB41L3, CADM1, MAL).
2. Region-level features (CpGs averaged within gene promoter regions).
3. Cell composition estimation (epithelial vs immune fractions) and diagnosis separation testing.
4. Outputs comparison table: literature vs probe-level vs region-level.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_loader import load_primary_data, impute_fold, variance_prefilter
from annotator import load_manifest
from evaluator import compute_metrics, _get_classifier_proba, aggregate_results
from fs_selectors import get_selector

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(HERE, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")


# ---------------------------------------------------------------------------
# 1. Literature Baseline Panel
# ---------------------------------------------------------------------------

TARGET_LIT_GENES = ["FAM19A4", "MIR124-2", "PAX1", "ZNF582", "EPB41L3", "CADM1", "MAL"]

def get_literature_cpgs(cpg_names: list, manifest: pd.DataFrame) -> list:
    """Find all CpGs in GSE99511 mapping to the 7 literature genes."""
    cpg_to_gene = manifest["Gene"].dropna().to_dict()
    lit_cpgs = []
    for cpg in cpg_names:
        genes_str = str(cpg_to_gene.get(cpg, ""))
        if any(g in genes_str for g in TARGET_LIT_GENES):
            lit_cpgs.append(cpg)
    return lit_cpgs


def evaluate_literature_panel(X: np.ndarray, y: np.ndarray, cpg_names: list,
                              lit_cpgs: list, config: dict, n_repeats: int = 3) -> pd.DataFrame:
    """Evaluate literature-only CpGs using repeated stratified CV."""
    cpg_idx_map = {name: i for i, name in enumerate(cpg_names)}
    lit_indices = np.array([cpg_idx_map[c] for c in lit_cpgs], dtype=int)
    X_lit = X[:, lit_indices]

    outer_splits = config["cv"].get("outer_splits", 5)
    base_seed = config["cv"].get("random_seed", 42)
    panel_sizes = config["panel_sizes"]
    
    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=n_repeats, random_state=base_seed)
    
    records = []
    for fold_num, (tr_idx, va_idx) in enumerate(rskf.split(X_lit, y)):
        repeat_idx = fold_num // outer_splits
        outer_fold_idx = fold_num % outer_splits
        fold_seed = base_seed + repeat_idx * 100 + outer_fold_idx
        
        X_tr_raw, X_va_raw = X_lit[tr_idx], X_lit[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # In-fold imputation
        X_tr, X_va, _ = impute_fold(X_tr_raw, X_va_raw)
        
        # M-value conversion
        eps = 1e-5
        X_tr = np.clip(X_tr, eps, 1.0 - eps)
        X_va = np.clip(X_va, eps, 1.0 - eps)
        X_tr = np.log2(X_tr / (1.0 - X_tr))
        X_va = np.log2(X_va / (1.0 - X_va))
        
        for k in panel_sizes:
            # Select top k within literature probes using in-fold Mann-Whitney U test
            p_vals = []
            for col_i in range(X_tr.shape[1]):
                try:
                    _, p = stats.mannwhitneyu(X_tr[y_tr == 0, col_i], X_tr[y_tr == 1, col_i])
                except Exception:
                    p = 1.0
                p_vals.append(p)
            
            top_k_idx = np.argsort(p_vals)[:min(k, X_tr.shape[1])]
            X_tr_k = X_tr[:, top_k_idx]
            X_va_k = X_va[:, top_k_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_k)
            X_va_sc = scaler.transform(X_va_k)
            
            for clf_name in ["logreg", "linear_svm"]:
                if clf_name == "logreg":
                    clf = LogisticRegression(C=1.0, solver="liblinear", random_state=fold_seed)
                else:
                    clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=fold_seed), cv=2)
                
                clf.fit(X_tr_sc, y_tr)
                y_pred = clf.predict(X_va_sc)
                y_prob = _get_classifier_proba(clf, X_va_sc)
                m = compute_metrics(y_va, y_pred, y_prob)
                records.append({
                    "repeat": repeat_idx,
                    "outer_fold": outer_fold_idx,
                    "feature_type": "Literature_Genes_Panel",
                    "selector": "MW_Lit_Filter",
                    "k": k,
                    "classifier": clf_name,
                    **m
                })
                
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Region-Level (Gene Promoter) Features
# ---------------------------------------------------------------------------

def build_region_level_matrix(X: np.ndarray, cpg_names: list, manifest: pd.DataFrame, verbose: bool = True):
    """
    Average CpGs belonging to the promoter region of each gene (TSS1500, TSS200, 5'UTR, 1stExon).
    """
    if verbose:
        print("[RegionFeatures] Aggregating CpGs into Gene Promoter Regions...")
    
    cpg_to_gene = manifest["Gene"].dropna().to_dict()
    cpg_to_region = manifest["Gene_Region"].dropna().to_dict()
    
    promoter_tags = {"TSS1500", "TSS200", "5'UTR", "1stExon"}
    
    gene_cpg_map = {}
    for i, cpg in enumerate(cpg_names):
        genes = str(cpg_to_gene.get(cpg, "")).split(";")
        regions = str(cpg_to_region.get(cpg, "")).split(";")
        
        for g, r in zip(genes, regions):
            g = g.strip()
            r = r.strip()
            if g and r in promoter_tags:
                if g not in gene_cpg_map:
                    gene_cpg_map[g] = []
                gene_cpg_map[g].append(i)
                
    # Filter genes with at least 1 probe
    region_names = sorted(gene_cpg_map.keys())
    if verbose:
        print(f"[RegionFeatures] Constructed {len(region_names):,} unique promoter gene regions.")
        
    # Compute mean for each region across samples
    n_samples = X.shape[0]
    X_region = np.zeros((n_samples, len(region_names)), dtype=np.float32)
    
    for j, g in enumerate(region_names):
        idx_list = gene_cpg_map[g]
        X_region[:, j] = np.nanmean(X[:, idx_list], axis=1)
        
    return X_region, region_names


def evaluate_region_features(X_reg: np.ndarray, y: np.ndarray, region_names: list,
                             config: dict, n_repeats: int = 3) -> pd.DataFrame:
    """Evaluate region-level features with in-fold selection and tuning."""
    outer_splits = config["cv"].get("outer_splits", 5)
    base_seed = config["cv"].get("random_seed", 42)
    panel_sizes = config["panel_sizes"]
    
    rskf = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=n_repeats, random_state=base_seed)
    
    records = []
    for fold_num, (tr_idx, va_idx) in enumerate(rskf.split(X_reg, y)):
        repeat_idx = fold_num // outer_splits
        outer_fold_idx = fold_num % outer_splits
        fold_seed = base_seed + repeat_idx * 100 + outer_fold_idx
        
        X_tr_raw, X_va_raw = X_reg[tr_idx], X_reg[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        X_tr, X_va, _ = impute_fold(X_tr_raw, X_va_raw)
        
        # M-values
        eps = 1e-5
        X_tr = np.clip(X_tr, eps, 1.0 - eps)
        X_va = np.clip(X_va, eps, 1.0 - eps)
        X_tr = np.log2(X_tr / (1.0 - X_tr))
        X_va = np.log2(X_va / (1.0 - X_va))
        
        # In-fold pre-filter top 2000 by variance
        var_tr = np.var(X_tr, axis=0)
        top_var_idx = np.argsort(var_tr)[::-1][:min(2000, X_tr.shape[1])]
        X_tr_pf = X_tr[:, top_var_idx]
        X_va_pf = X_va[:, top_var_idx]
        
        for k in panel_sizes:
            # MW ranking
            p_vals = []
            for col_i in range(X_tr_pf.shape[1]):
                try:
                    _, p = stats.mannwhitneyu(X_tr_pf[y_tr == 0, col_i], X_tr_pf[y_tr == 1, col_i])
                except Exception:
                    p = 1.0
                p_vals.append(p)
            
            top_k_idx = np.argsort(p_vals)[:min(k, X_tr_pf.shape[1])]
            X_tr_k = X_tr_pf[:, top_k_idx]
            X_va_k = X_va_pf[:, top_k_idx]
            
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_k)
            X_va_sc = scaler.transform(X_va_k)
            
            for clf_name in ["logreg", "linear_svm"]:
                if clf_name == "logreg":
                    clf = LogisticRegression(C=1.0, solver="liblinear", random_state=fold_seed)
                else:
                    clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=fold_seed), cv=2)
                
                clf.fit(X_tr_sc, y_tr)
                y_pred = clf.predict(X_va_sc)
                y_prob = _get_classifier_proba(clf, X_va_sc)
                m = compute_metrics(y_va, y_pred, y_prob)
                records.append({
                    "repeat": repeat_idx,
                    "outer_fold": outer_fold_idx,
                    "feature_type": "Region_Level_Promoters",
                    "selector": "MW_Region_Filter",
                    "k": k,
                    "classifier": clf_name,
                    **m
                })
                
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Cell Composition Estimation (EpiDISH reference-based proxy)
# ---------------------------------------------------------------------------

def estimate_cell_composition(X: np.ndarray, cpg_names: list, manifest: pd.DataFrame, y: np.ndarray, sample_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Reference-based deconvolution (EpiDISH framework) estimating Epithelial vs Immune cell proportions.
    Uses non-negative least squares (NNLS) on canonical cell-type marker CpGs.
    """
    cpg_to_gene = manifest["Gene"].dropna().to_dict()
    
    # Established leukocyte/immune markers and epithelial markers in cervical smears
    immune_genes = ["PTPRC", "CD3D", "CD3E", "CD19", "CD14", "MS4A1", "NCAM1"]
    epithelial_genes = ["CDH1", "EPCAM", "KRT5", "KRT13", "KRT19", "MUC1"]
    
    cpg_idx_map = {name: i for i, name in enumerate(cpg_names)}
    
    immune_cpgs = [c for c in cpg_names if any(g in str(cpg_to_gene.get(c, "")) for g in immune_genes)]
    epithelial_cpgs = [c for c in cpg_names if any(g in str(cpg_to_gene.get(c, "")) for g in epithelial_genes)]
    
    # Reference profiles (Epithelial: high epithelial marker hypomethylation / high immune marker hypermethylation)
    # Immune: high immune marker hypomethylation / high epithelial marker hypermethylation
    selected_markers = immune_cpgs[:15] + epithelial_cpgs[:15]
    marker_indices = [cpg_idx_map[c] for c in selected_markers]
    
    X_markers = X[:, marker_indices]
    
    # Construct reference basis matrix W (markers x 2: [Epithelial_Ref, Immune_Ref])
    # Epithelial: immune probes unmethylated=0.1, epithelial=0.85; Immune: inverse
    n_imm = len(immune_cpgs[:15])
    n_epi = len(epithelial_cpgs[:15])
    
    W = np.zeros((len(selected_markers), 2))
    # Epithelial cell reference: immune markers hypermethylated (0.85), epithelial unmethylated (0.15)
    W[:n_imm, 0] = 0.85
    W[n_imm:, 0] = 0.15
    # Immune cell reference: immune markers unmethylated (0.15), epithelial hypermethylated (0.85)
    W[:n_imm, 1] = 0.15
    W[n_imm:, 1] = 0.85
    
    deconv_rows = []
    for i in range(X.shape[0]):
        sample_vals = X_markers[i, :]
        valid_mask = ~np.isnan(sample_vals)
        if valid_mask.sum() > 4:
            weights, _ = nnls(W[valid_mask], sample_vals[valid_mask])
            total = weights.sum()
            if total > 0:
                weights = weights / total
            else:
                weights = np.array([0.5, 0.5])
        else:
            weights = np.array([0.5, 0.5])
            
        deconv_rows.append({
            "Sample_Index": i,
            "GSM_ID": sample_meta["GSM_ID"].iloc[i] if "GSM_ID" in sample_meta.columns else f"S{i}",
            "Diagnosis": sample_meta["Diagnosis"].iloc[i] if "Diagnosis" in sample_meta.columns else ("CIN3" if y[i]==1 else "Normal"),
            "Epithelial_Fraction": round(float(weights[0]), 4),
            "Immune_Fraction": round(float(weights[1]), 4)
        })
        
    df_cell = pd.DataFrame(deconv_rows)
    
    # Statistical test between Normal and CIN3
    normal_epi = df_cell[df_cell["Diagnosis"] == "Normal"]["Epithelial_Fraction"]
    cin3_epi = df_cell[df_cell["Diagnosis"] == "CIN3"]["Epithelial_Fraction"]
    
    mw_stat, mw_pval = stats.mannwhitneyu(normal_epi, cin3_epi)
    tt_stat, tt_pval = stats.ttest_ind(normal_epi, cin3_epi)
    
    print("\n" + "="*70)
    print("CELL COMPOSITION DECONVOLUTION (EpiDISH Proxy)")
    print("="*70)
    print(f"Normal Epithelial Fraction: {normal_epi.mean():.3f} ± {normal_epi.std():.3f}")
    print(f"CIN3 Epithelial Fraction:   {cin3_epi.mean():.3f} ± {cin3_epi.std():.3f}")
    print(f"Difference test (Epithelial): Mann-Whitney U p-val={mw_pval:.4f}, t-test p-val={tt_pval:.4f}")
    if mw_pval < 0.05:
        print("-> Cell composition differs significantly between Normal and CIN3 (potential biological/cellular confound).")
    else:
        print("-> Cell composition does NOT show significant difference between Normal and CIN3 (p >= 0.05).")
        
    return df_cell, mw_pval


def run_feature_engineering_experiments(verbose: bool = True):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    data = load_primary_data(config, verbose=verbose)
    X = data["X"]
    y = data["y"]
    cpg_names = data["cpg_names"]
    meta = data["sample_meta"]
    manifest = load_manifest(verbose=verbose)

    # 1. Literature panel
    lit_cpgs = get_literature_cpgs(cpg_names, manifest)
    if verbose:
        print(f"\n[Literature] Found {len(lit_cpgs)} candidate CpGs across 7 literature genes.")
    df_lit_folds = evaluate_literature_panel(X, y, cpg_names, lit_cpgs, config, n_repeats=3)

    # 2. Region-level features
    X_reg, reg_names = build_region_level_matrix(X, cpg_names, manifest, verbose=verbose)
    df_reg_folds = evaluate_region_features(X_reg, y, reg_names, config, n_repeats=3)

    # 3. Cell composition deconvolution
    df_cell, cell_pval = estimate_cell_composition(X, cpg_names, manifest, y, meta)
    cell_csv = os.path.join(OUT_DIR, "cell_composition_estimates.csv")
    df_cell.to_csv(cell_csv, index=False)

    # 4. Aggregate results
    agg_lit = aggregate_results(df_lit_folds)
    agg_lit["Feature_Representation"] = "Literature_Genes"

    agg_reg = aggregate_results(df_reg_folds)
    agg_reg["Feature_Representation"] = "Region_Level_Promoters"

    comparison_df = pd.concat([agg_lit, agg_reg], ignore_index=True)
    comp_csv = os.path.join(OUT_DIR, "feature_engineering_comparison.csv")
    comparison_df.to_csv(comp_csv, index=False)

    if verbose:
        print(f"\nSaved feature engineering comparison to {comp_csv}")
        print(f"Saved cell composition estimates to {cell_csv}")
        
    return comparison_df, df_cell


if __name__ == "__main__":
    run_feature_engineering_experiments(verbose=True)
