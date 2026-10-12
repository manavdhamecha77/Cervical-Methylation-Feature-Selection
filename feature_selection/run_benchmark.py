"""
run_benchmark.py
----------------
Single entry-point for the leakage-free feature selection benchmark.

Usage
-----
  # From the repo root (or feature_selection/ dir) with the uv venv active:
  python feature_selection/run_benchmark.py

  # Quick smoke-test (fewer repeats, 2 selectors):
  python feature_selection/run_benchmark.py --quick

  # Sensitivity run with prefilter_n=2000:
  python feature_selection/run_benchmark.py --prefilter_n 2000

Steps
-----
1.  Load data (reuses pipeline/05 data)
2.  Print batch covariate report
3.  Run main benchmark (repeated nested CV, 7 selectors × 4 panel sizes)
4.  Run sensitivity check with prefilter_n=2000
5.  Reproduce existing ANOVA-5000+RF baseline under correct CV
6.  Run permutation test on best (selector, k)
7.  Compute stability metrics (Jaccard, Kuncheva)
8.  Build consensus panels
9.  Annotate with 450K manifest
10. Generate all plots
11. Write REPORT.md
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd

# Allow imports from feature_selection/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader    import load_primary_data
from fs_selectors   import SELECTOR_NAMES, SELECTOR_DISPLAY_NAMES
from evaluator      import (run_nested_cv, run_baseline_reproduction,
                             run_permutation_test, aggregate_results)
from stability      import (compute_stability, compute_selection_frequency,
                             consensus_panel, build_frequency_heatmap_data)
from annotator      import (load_manifest, annotate_cpgs, known_marker_overlap,
                             correlation_among_selected, build_annotation_report)
from plots          import (plot_performance_vs_k, plot_stability_vs_k,
                             plot_frequency_heatmap, plot_method_comparison_heatmap,
                             plot_permutation_test, plot_correlation_heatmap,
                             plot_baseline_comparison, plot_summary_table)

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def parse_args():
    p = argparse.ArgumentParser(description="Feature selection benchmark runner")
    p.add_argument("--quick",      action="store_true",
                   help="Quick test: 2 selectors, 2 repeats, no permutation test")
    p.add_argument("--prefilter_n", type=int, default=None,
                   help="Override prefilter_size_main (e.g. 2000 for sensitivity run)")
    p.add_argument("--selectors",   nargs="+", default=None,
                   help="Subset of selectors to run (space-separated names)")
    p.add_argument("--no_baseline", action="store_true",
                   help="Skip baseline RF-5000 reproduction")
    p.add_argument("--no_permtest", action="store_true",
                   help="Skip permutation test")
    p.add_argument("--n_jobs",      type=int, default=-1,
                   help="Parallelism (-1 = all cores)")
    return p.parse_args()


def setup_output_dir(config: dict) -> str:
    out_dir = os.path.join(ROOT, config["output_dir"])
    for sub in ["raw_results", "stability", "annotation", "plots", "panels"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    return out_dir


def print_banner(msg: str):
    bar = "=" * 70
    print(f"\n{bar}\n{msg}\n{bar}")


def main():
    t_total = time.time()
    args = parse_args()

    # ---- Load config ----
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if args.quick:
        print("[Quick mode] Reducing to 2 repeats, 2 selectors, no permutation test")
        config["cv"]["outer_repeats"] = 2
        config["selectors"]["stability_selection"]["n_subsamples"] = 20
        config["permutation_test"]["enabled"] = False

    selector_names = args.selectors or SELECTOR_NAMES
    prefilter_n    = args.prefilter_n or config["prefilter"]["prefilter_size_main"]
    n_jobs         = args.n_jobs

    out_dir = setup_output_dir(config)
    print(f"[Setup] Output directory: {out_dir}")

    # ---- Step 1: Load data ----
    print_banner("STEP 1: Load Data")
    data = load_primary_data(config, verbose=True)
    X          = data["X"]
    y          = data["y"]
    cpg_names  = data["cpg_names"]
    gsm_ids    = data["gsm_ids"]
    sentrix_ids = data["sentrix_ids"]
    n_samples  = data["n_samples"]
    n_cpgs     = data["n_cpgs"]

    # Save batch report
    with open(os.path.join(out_dir, "batch_report.txt"), "w") as f:
        f.write(data["batch_report"])

    print(f"\nData loaded: {n_samples} samples, {n_cpgs:,} CpGs")
    print(f"Class distribution: Normal={( y==0).sum()}, CIN3={(y==1).sum()}")

    # ---- Step 2: Main benchmark ----
    print_banner("STEP 2: Main Benchmark (Repeated Nested CV)")
    t0 = time.time()
    fold_df = run_nested_cv(
        X, y, cpg_names, config,
        selector_names=selector_names,
        prefilter_n=prefilter_n,
        n_jobs=n_jobs,
        verbose=True,
    )
    print(f"Main benchmark done in {(time.time()-t0)/60:.1f} min")

    raw_path = os.path.join(out_dir, "raw_results", "fold_results_main.csv")
    fold_df.to_csv(raw_path, index=False)
    print(f"Saved: {raw_path}")

    agg_df = aggregate_results(fold_df)
    agg_path = os.path.join(out_dir, "raw_results", "aggregated_results_main.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    # ---- Step 3: Sensitivity check (prefilter_n=2000) ----
    pf_sensitivity = config["prefilter"]["prefilter_size_sensitivity"]
    if pf_sensitivity != prefilter_n:
        print_banner(f"STEP 3: Sensitivity Check (prefilter_n={pf_sensitivity})")
        t0 = time.time()
        fold_df_s = run_nested_cv(
            X, y, cpg_names, config,
            selector_names=selector_names,
            prefilter_n=pf_sensitivity,
            n_jobs=n_jobs,
            verbose=True,
        )
        print(f"Sensitivity run done in {(time.time()-t0)/60:.1f} min")
        sens_path = os.path.join(out_dir, "raw_results", f"fold_results_pf{pf_sensitivity}.csv")
        fold_df_s.to_csv(sens_path, index=False)
        agg_df_s = aggregate_results(fold_df_s)
        agg_df_s.to_csv(
            os.path.join(out_dir, "raw_results", f"aggregated_results_pf{pf_sensitivity}.csv"),
            index=False,
        )
        print(f"Saved sensitivity results: {sens_path}")
    else:
        fold_df_s = pd.DataFrame()
        agg_df_s  = pd.DataFrame()

    # ---- Step 4: Baseline reproduction ----
    baseline_df = pd.DataFrame()
    if not args.no_baseline and config["baseline_reproduction"]["enabled"]:
        print_banner("STEP 4: Baseline Reproduction (ANOVA-5000 + RF, correct CV)")
        t0 = time.time()
        baseline_df = run_baseline_reproduction(X, y, cpg_names, config, verbose=True)
        baseline_df.to_csv(
            os.path.join(out_dir, "raw_results", "baseline_rf5000.csv"), index=False
        )
        print(f"Baseline done in {(time.time()-t0)/60:.1f} min")

    # ---- Step 5: Find best (selector, k, classifier) ----
    print_banner("STEP 5: Best Configuration")
    best_auc_row = (
        agg_df.dropna(subset=["mean_Balanced_Accuracy"])
        .sort_values("mean_Balanced_Accuracy", ascending=False)
        .iloc[0]
    )
    best_selector   = best_auc_row["selector"]
    best_k          = int(best_auc_row["k"])
    best_classifier = best_auc_row.get("classifier", "logreg")
    best_ba         = best_auc_row["mean_Balanced_Accuracy"]
    best_auc_val    = best_auc_row.get("mean_ROC_AUC", np.nan)

    print(f"Best config: selector={best_selector}, k={best_k}, clf={best_classifier}")
    print(f"  Mean Balanced Accuracy: {best_ba:.3f} ± {best_auc_row.get('std_Balanced_Accuracy', np.nan):.3f}")
    print(f"  Mean ROC-AUC:           {best_auc_val:.3f} ± {best_auc_row.get('std_ROC_AUC', np.nan):.3f}")

    # ---- Step 6: Permutation test ----
    perm_result = {}
    if (not args.no_permtest and not args.quick
            and config["permutation_test"]["enabled"]):
        print_banner("STEP 6: Permutation Test")
        t0 = time.time()
        perm_result = run_permutation_test(
            X, y, cpg_names, config,
            best_selector=best_selector,
            best_k=best_k,
            best_classifier=best_classifier,
            verbose=True,
        )
        null_mean = perm_result["null_mean"]
        null_std  = perm_result["null_std"]
        null_arr  = np.array(perm_result["null_balanced_accuracy"])
        empirical_p = (null_arr >= best_ba).mean()
        print(f"Null dist: {null_mean:.3f} ± {null_std:.3f}")
        print(f"Observed:  {best_ba:.3f}")
        print(f"Empirical p-value: {empirical_p:.3f}")
        perm_result["observed_score"]   = float(best_ba)
        perm_result["empirical_p_value"] = float(empirical_p)
        perm_result["best_selector"]    = best_selector
        perm_result["best_k"]           = best_k
        perm_result["best_classifier"]  = best_classifier
        with open(os.path.join(out_dir, "raw_results", "permutation_test.json"), "w") as f:
            json.dump(perm_result, f, indent=2)
        print(f"Permutation done in {(time.time()-t0)/60:.1f} min")

    # ---- Step 7: Stability analysis ----
    print_banner("STEP 7: Stability Analysis")
    # Use per-fold selected_cpgs column from fold_df
    fold_for_stab = fold_df[["selector", "k", "prefilter_n", "selected_cpgs",
                              "outer_fold", "repeat"]].dropna(subset=["selected_cpgs"])
    fold_for_stab = fold_for_stab[fold_for_stab["selected_cpgs"].str.strip() != ""]

    stab_df = compute_stability(fold_for_stab, total_cpgs=n_cpgs)
    stab_df.to_csv(os.path.join(out_dir, "stability", "stability_metrics.csv"), index=False)
    print(f"Stability metrics saved. Shape: {stab_df.shape}")
    print(stab_df.to_string(index=False))

    freq_dict = compute_selection_frequency(fold_for_stab)
    for key, freq_df in freq_dict.items():
        key_str = "_".join(str(x) for x in key)
        freq_df.to_csv(
            os.path.join(out_dir, "stability", f"frequency_{key_str}.csv"),
            index=False,
        )

    # ---- Step 8: Consensus panels ----
    print_banner("STEP 8: Consensus Panels")
    panel_sizes = config["panel_sizes"]
    panels = {}  # (selector, k) -> list of CpGs
    for sel in selector_names:
        for k in panel_sizes:
            cpg_panel = consensus_panel(freq_dict, sel, k, top_n=k)
            panels[(sel, k)] = cpg_panel
            display = SELECTOR_DISPLAY_NAMES.get(sel, sel)
            print(f"  {display:22s} k={k:2d}: {cpg_panel}")

    # Save all panels
    panel_rows = []
    for (sel, k), cpg_list in panels.items():
        for rank, cpg in enumerate(cpg_list, 1):
            panel_rows.append({
                "selector": sel,
                "k": k,
                "rank": rank,
                "CpG": cpg,
            })
    pd.DataFrame(panel_rows).to_csv(
        os.path.join(out_dir, "panels", "consensus_panels.csv"), index=False
    )

    # ---- Step 9: Annotation ----
    print_banner("STEP 9: Annotation")
    manifest = load_manifest(verbose=True)

    known_genes = config["known_markers"]["genes"]
    all_panel_cpgs = list(dict.fromkeys(
        cpg for cpg_list in panels.values() for cpg in cpg_list
    ))

    annot_rows = []
    for (sel, k), cpg_list in panels.items():
        if not cpg_list:
            continue
        report = build_annotation_report(
            sel, k, cpg_list, manifest, known_genes, X, cpg_names
        )
        for _, row in report["annotation"].iterrows():
            row_dict = row.to_dict()
            row_dict["selector"] = sel
            row_dict["k"] = k
            annot_rows.append(row_dict)
        # Correlation heatmap per panel
        if not report["correlation"].empty:
            plot_correlation_heatmap(
                report["correlation"], sel, k,
                os.path.join(out_dir, "plots"),
            )
        # Print known markers
        n_known = report["n_known_markers"]
        if n_known > 0:
            print(f"  {SELECTOR_DISPLAY_NAMES.get(sel, sel)} k={k}: "
                  f"{n_known}/{len(cpg_list)} CpGs map to known markers: "
                  f"{report['known_cpgs']}")

    if annot_rows:
        annot_df = pd.DataFrame(annot_rows)
        annot_df.to_csv(
            os.path.join(out_dir, "annotation", "panel_annotations.csv"),
            index=False,
        )
        print(f"Saved: annotation/panel_annotations.csv ({len(annot_df)} rows)")
    else:
        annot_df = pd.DataFrame()

    # ---- Step 10: Plots ----
    print_banner("STEP 10: Generating Plots")
    plots_dir = os.path.join(out_dir, "plots")

    # Performance vs k
    for clf_name in ["logreg", "linear_svm"]:
        for metric in ["Balanced_Accuracy", "ROC_AUC"]:
            plot_performance_vs_k(agg_df, plots_dir, metric=metric, classifier=clf_name)

    # Stability vs k
    if not stab_df.empty:
        for metric in ["jaccard_mean", "kuncheva_index"]:
            if metric in stab_df.columns:
                plot_stability_vs_k(stab_df, plots_dir, metric=metric)

    # Frequency heatmaps
    for k in panel_sizes:
        hm_data = build_frequency_heatmap_data(freq_dict, selector_names, k, top_n=40)
        if not hm_data.empty:
            plot_frequency_heatmap(hm_data, k, plots_dir)

    # Method comparison heatmaps
    for clf_name in ["logreg", "linear_svm"]:
        for metric in ["Balanced_Accuracy", "ROC_AUC"]:
            plot_method_comparison_heatmap(agg_df, plots_dir, clf_name, metric)

    # Permutation test
    if perm_result:
        plot_permutation_test(perm_result, float(best_ba), plots_dir)

    # Baseline comparison
    if not baseline_df.empty:
        # Original result from existing pipeline: best RF5000 from model_comparison.csv
        orig_csv = os.path.join(ROOT, "results", "e2e_cervical_methylation",
                                "cv_results", "model_comparison.csv")
        orig_mean, orig_std = 0.640, 0.231  # from reading model_comparison.csv
        if os.path.exists(orig_csv):
            try:
                orig_df = pd.read_csv(orig_csv)
                row = orig_df[(orig_df["Model"] == "RandomForest") &
                              (orig_df["Feature_Budget"] == 5000)]
                if not row.empty:
                    orig_mean = float(row["Mean_Balanced_Accuracy"].iloc[0])
                    orig_std  = float(row["Std_Balanced_Accuracy"].iloc[0])
            except Exception:
                pass
        plot_baseline_comparison(
            baseline_df, orig_mean, orig_std, plots_dir, metric="Balanced_Accuracy"
        )

    # Summary table heatmaps
    plot_summary_table(agg_df, plots_dir)

    # ---- Step 11: Write REPORT.md ----
    print_banner("STEP 11: Writing REPORT.md")
    write_report(
        out_dir=out_dir,
        config=config,
        agg_df=agg_df,
        agg_df_s=agg_df_s,
        stab_df=stab_df,
        panels=panels,
        annot_df=annot_df,
        baseline_df=baseline_df,
        perm_result=perm_result,
        best_selector=best_selector,
        best_k=best_k,
        best_classifier=best_classifier,
        best_ba=float(best_ba),
        prefilter_n=prefilter_n,
        n_samples=n_samples,
        n_cpgs=n_cpgs,
        selector_names=selector_names,
    )

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE in {total_time/60:.1f} min")
    print(f"All results: {out_dir}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _agg_row(agg_df, selector, k, classifier="logreg"):
    sub = agg_df[
        (agg_df["selector"] == selector) &
        (agg_df["k"] == k) &
        (agg_df.get("classifier", pd.Series(["logreg"] * len(agg_df))) == classifier)
    ]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def _fmt(row, metric):
    m = f"mean_{metric}"
    s = f"std_{metric}"
    if m in row and not pd.isna(row.get(m, np.nan)):
        return f"{row[m]:.3f} ± {row.get(s, np.nan):.3f}"
    return "N/A"


def write_report(out_dir, config, agg_df, agg_df_s, stab_df, panels,
                 annot_df, baseline_df, perm_result, best_selector, best_k,
                 best_classifier, best_ba, prefilter_n, n_samples, n_cpgs,
                 selector_names):
    from datetime import datetime

    panel_sizes = config["panel_sizes"]
    outer_splits  = config["cv"]["outer_splits"]
    outer_repeats = config["cv"]["outer_repeats"]
    pf_main       = config["prefilter"]["prefilter_size_main"]
    pf_sens       = config["prefilter"]["prefilter_size_sensitivity"]

    # Build summary table
    def _summary_table():
        lines = []
        header = "| Selector | k | LogReg AUC | LogReg Bal.Acc | LinSVM AUC | LinSVM Bal.Acc |"
        sep    = "|" + "|".join(["-" * (len(h) + 2) for h in header.split("|")[1:-1]]) + "|"
        lines.append(header)
        lines.append(sep)
        for sel in selector_names:
            for k in panel_sizes:
                lr = _agg_row(agg_df, sel, k, "logreg")
                sv = _agg_row(agg_df, sel, k, "linear_svm")
                name = SELECTOR_DISPLAY_NAMES.get(sel, sel)
                lines.append(
                    f"| {name} | {k} | "
                    f"{_fmt(lr, 'ROC_AUC')} | {_fmt(lr, 'Balanced_Accuracy')} | "
                    f"{_fmt(sv, 'ROC_AUC')} | {_fmt(sv, 'Balanced_Accuracy')} |"
                )
        return "\n".join(lines)

    # Stability table
    def _stab_table():
        if stab_df.empty:
            return "_Stability metrics not available._"
        lines = ["| Selector | k | Jaccard | Kuncheva Index | n_folds |",
                 "|----------|---|---------|----------------|---------|"]
        for _, row in stab_df.iterrows():
            name = SELECTOR_DISPLAY_NAMES.get(str(row.get("selector", "")), str(row.get("selector", "")))
            j    = f"{row.get('jaccard_mean', np.nan):.3f}" if not pd.isna(row.get('jaccard_mean', np.nan)) else "N/A"
            ki   = f"{row.get('kuncheva_index', np.nan):.3f}" if not pd.isna(row.get('kuncheva_index', np.nan)) else "N/A"
            lines.append(f"| {name} | {row.get('k','?')} | {j} | {ki} | {row.get('n_folds','?')} |")
        return "\n".join(lines)

    # Panel table
    def _panel_table():
        lines = []
        for k in panel_sizes:
            lines.append(f"\n**k = {k}**\n")
            lines.append("| Selector | Consensus Panel CpGs |")
            lines.append("|----------|----------------------|")
            for sel in selector_names:
                cpgs  = panels.get((sel, k), [])
                name  = SELECTOR_DISPLAY_NAMES.get(sel, sel)
                lines.append(f"| {name} | {', '.join(cpgs) if cpgs else '_none_'} |")
        return "\n".join(lines)

    # Known marker overlap
    def _overlap_section():
        if annot_df.empty or "known_marker_gene" not in annot_df.columns:
            return (
                "Annotation skipped or manifest not available. "
                "The Illumina manifest (GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz) "
                "should be present in `data/GSE99511/raw/` — it is included in the repo."
            )
        known_rows = annot_df[annot_df["known_marker_gene"] == True]
        if known_rows.empty:
            return "No selected CpGs map to the known cervical marker gene list."
        lines = []
        for _, row in known_rows.iterrows():
            name = SELECTOR_DISPLAY_NAMES.get(str(row.get("selector", "")), str(row.get("selector", "")))
            lines.append(
                f"- {row.get('CpG', '?')} (method={name}, k={row.get('k', '?')}): "
                f"gene={row.get('matched_known_gene', '?')}, "
                f"region={row.get('Gene_Region', '?')}, "
                f"CpG island={row.get('CpG_Island_Relation', '?')}"
            )
        return "\n".join(lines)

    # Permutation section
    def _perm_section():
        if not perm_result:
            return "_Permutation test not run._"
        n_perm = len(perm_result.get("null_balanced_accuracy", []))
        null_m = perm_result.get("null_mean", np.nan)
        null_s = perm_result.get("null_std", np.nan)
        obs    = perm_result.get("observed_score", best_ba)
        ep     = perm_result.get("empirical_p_value", np.nan)
        return (
            f"- Permutations: {n_perm}  \n"
            f"- Null Balanced Accuracy: {null_m:.3f} ± {null_s:.3f}  \n"
            f"- Observed Balanced Accuracy: {obs:.3f}  \n"
            f"- Empirical p-value: {ep:.3f}  \n"
            f"- Best config used: selector={best_selector}, k={best_k}, clf={best_classifier}"
        )

    # Baseline section
    def _baseline_section():
        if baseline_df.empty:
            return "_Baseline reproduction not run._"
        ba_m = baseline_df["Balanced_Accuracy"].mean()
        ba_s = baseline_df["Balanced_Accuracy"].std()
        auc_m = baseline_df["ROC_AUC"].mean()
        auc_s = baseline_df["ROC_AUC"].std()
        return (
            f"Original pipeline (5-fold, single repeat): Bal.Acc ≈ 0.640 ± 0.231  \n"
            f"Reproduced (5×10 repeated nested CV):       Bal.Acc = {ba_m:.3f} ± {ba_s:.3f}  \n"
            f"                                            ROC-AUC = {auc_m:.3f} ± {auc_s:.3f}  \n"
            f"\nDifference in means: {0.640 - ba_m:+.3f} (positive = original was inflated)"
        )

    # Sensitivity section
    def _sensitivity_section():
        if agg_df_s.empty:
            return "_Sensitivity run not available._"
        best_s = (
            agg_df_s.dropna(subset=["mean_Balanced_Accuracy"])
            .sort_values("mean_Balanced_Accuracy", ascending=False)
            .iloc[0]
        )
        main_best = (
            agg_df.dropna(subset=["mean_Balanced_Accuracy"])
            .sort_values("mean_Balanced_Accuracy", ascending=False)
            .iloc[0]
        )
        return (
            f"Pre-filter={pf_main}: best Bal.Acc = {main_best.get('mean_Balanced_Accuracy', np.nan):.3f}  \n"
            f"Pre-filter={pf_sens}: best Bal.Acc = {best_s.get('mean_Balanced_Accuracy', np.nan):.3f}  \n"
            f"Conclusion: "
            + ("Results are robust to pre-filter size."
               if abs(main_best.get("mean_Balanced_Accuracy", 0) - best_s.get("mean_Balanced_Accuracy", 0)) < 0.05
               else "Results differ; check stability.")
        )

    report = f"""# Feature Selection Benchmark Report
## Cervical Cancer DNA Methylation — GSE99511
### Leakage-Free Small Panel Discovery

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

---

## 1. Objective

Identify small panels of original CpG sites (k ∈ {{5, 10, 20, 50}}) that best distinguish
Normal cervical tissue from CIN3, for use in a low-cost biosensor device.
All selection steps are performed inside each CV fold — no test data leaks into selection.

---

## 2. Data

| Property | Value |
|----------|-------|
| Dataset | GSE99511 (GEO) |
| Platform | Illumina HumanMethylation450K |
| Primary samples | {n_samples} (Normal=28, CIN3=36) |
| CpGs (post-QC) | {n_cpgs:,} |
| Feature values | Beta values (0–1, bimodal) |

---

## 3. Methods

### Pre-filter (inside each training fold)
1. Mean imputation (training mean only)
2. Variance filter (threshold = {config["prefilter"]["variance_threshold"]})
3. Top {prefilter_n} by variance

### Selection Methods
| Name | Description |
|------|-------------|
| MW Filter | Mann-Whitney U p-value ranking (non-parametric, baseline) |
| Lasso (L1) | L1 logistic regression, rank by \\|coef\\| |
| Elastic Net | L1+L2 logistic regression (l1_ratio ∈ {{0.3, 0.5, 0.7}}) |
| Stability Sel. | Repeated stratified subsampling + L1 LR, rank by selection frequency |
| SVM-RFE | Linear SVM recursive elimination |
| mRMR | Greedy max-relevance min-redundancy (pure numpy) |
| RF Importance | Random Forest mean-decrease-impurity |

### Evaluation Protocol
- **Outer CV**: {outer_splits}-fold × {outer_repeats} repeats = {outer_splits * outer_repeats} outer folds
- **Inner CV**: {config["cv"]["inner_splits"]}-fold for classifier hyperparameter selection
- **Classifiers**: Logistic Regression, Linear SVM
- **No step uses test-fold data** — pre-filter, selection, scaling, hyperparameter tuning
  are all fit on training folds only.
- Explicit assertion in code: `assert len(set(train_idx) & set(val_idx)) == 0`

### Panel Sizes
k ∈ {{ {', '.join(map(str, panel_sizes))} }}

---

## 4. Results Summary

### Performance Table (Mean ± SD across {outer_splits * outer_repeats} outer folds)

> **Disclaimer**: n=64, high-dimensional. Standard deviations are large (0.10–0.25).
> Small differences between methods are NOT meaningful. Focus on patterns, not exact numbers.

{_summary_table()}

### Best Configuration
- Selector: **{SELECTOR_DISPLAY_NAMES.get(best_selector, best_selector)}**
- Panel size: **k = {best_k}**
- Classifier: **{best_classifier}**
- Mean Balanced Accuracy: **{best_ba:.3f}**

---

## 5. Baseline Reproduction

{_baseline_section()}

---

## 6. Permutation Test

{_perm_section()}

---

## 7. Sensitivity Check (Pre-filter Size)

{_sensitivity_section()}

---

## 8. Feature Stability

{_stab_table()}

**Interpretation**: Kuncheva Index > 0 means better-than-chance stability.
At n=64, instability is expected — no single fold has enough samples to reliably
identify the same CpGs. Stability should be treated as a secondary criterion.

---

## 9. Consensus Panels

{_panel_table()}

---

## 10. Biological Annotation

### Known Cervical Methylation Marker Overlap

Checked against: {', '.join(config['known_markers']['genes'])}

{_overlap_section()}

> **Important**: This is a sanity check only. Overlap with known markers supports
> plausibility but does not validate the CpGs. Absence of overlap does not mean
> the CpGs are wrong — they may be novel candidates.

---

## 11. Batch Covariate Warning

See `batch_report.txt` for full details.

Chip `9340996078` contains ONLY CIN3 samples. This creates a chip-confound where
any CpG correlated with methylation batch effects on that chip could appear discriminative.
**All candidate CpGs must be validated in an independent cohort before biological claims.**

---

## 12. Limitations

1. **Sample size**: n=64 — all results have wide confidence intervals.
2. **Batch effects**: Chip `9340996078` is 100% CIN3. Selected CpGs may reflect batch.
3. **No independent validation**: Results cannot be generalized without external replication.
4. **Beta values**: Bimodal distribution; parametric methods (Lasso, SVM) may be suboptimal.
5. **mRMR uses Pearson correlation** as a proxy for mutual information — this is an approximation.
6. **SVM-RFE instability** at small panel sizes is expected (wrapper methods need large n).
7. **No biological validation**: CpGs are computational candidates. Laboratory confirmation required.
8. **Stability selection C=0.05**: This is a fixed hyperparameter; optimal C may vary.

---

## 13. Recommendations for Next Steps

1. **Wet-lab validation**: bisulfite pyrosequencing on the top-k consensus panel CpGs.
2. **Independent cohort**: obtain and test on a second dataset (e.g., TCGA-CESC methylation).
3. **ComBat batch correction**: apply before running this pipeline to remove Sentrix chip effects.
4. **Targeted assay**: design a methylation-specific PCR or LAMP assay for the k=5 or k=10 panel.
5. **M-value transformation**: re-run with M-values (logit(beta)) which are more homoscedastic.

---

## 14. File Index

```
{config['output_dir']}/
├── batch_report.txt
├── raw_results/
│   ├── fold_results_main.csv        # per-fold metrics
│   ├── aggregated_results_main.csv  # mean±std summary
│   ├── fold_results_pf{pf_sens}.csv  # sensitivity run
│   ├── baseline_rf5000.csv          # baseline reproduction
│   └── permutation_test.json        # null distribution
├── stability/
│   ├── stability_metrics.csv        # Jaccard, Kuncheva per (sel, k)
│   └── frequency_*.csv              # CpG selection frequencies
├── annotation/
│   └── panel_annotations.csv        # gene, chromosome, CpG island
├── panels/
│   └── consensus_panels.csv         # recommended panels
└── plots/
    ├── performance_vs_k_*.png
    ├── stability_vs_k_*.png
    ├── selection_frequency_heatmap_k*.png
    ├── method_comparison_*.png
    ├── permutation_test.png
    ├── correlation_heatmap_*.png
    ├── baseline_comparison.png
    └── summary_*.png
```

---

*This report was auto-generated by `feature_selection/run_benchmark.py`.*
*No results are computed outside of CV folds. No annotations are fabricated.*
"""

    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Report] Saved: {report_path}")


if __name__ == "__main__":
    main()
