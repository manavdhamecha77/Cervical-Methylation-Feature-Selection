"""Run leakage-safe small-CpG-panel CV on GSE99511 and GSE46306.

Runs every configured selector at k = 5, 10, 20, 50 with logistic regression
and linear SVM. Writes fold-level and aggregated metrics, stability summaries,
and a self-contained report at html/result.html.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_loader import load_primary_data
from evaluator import run_nested_cv, aggregate_results
from fs_selectors import SELECTOR_NAMES, SELECTOR_DISPLAY_NAMES
from stability import compute_stability, compute_selection_frequency, consensus_panel
from validate_gse46306 import load_external

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(HERE, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark", "all_datasets")
HTML_DIR = os.path.join(ROOT, "html")


def _load_cohorts(config):
    primary = load_primary_data(config, verbose=True)
    matrix, y, meta = load_external(config)
    external = {
        "X": matrix.to_numpy(dtype=np.float32),
        "y": y,
        "cpg_names": list(matrix.columns),
        "gsm_ids": meta["sample_id"].astype(str).to_numpy(),
        "sentrix_ids": np.array(["not reported"] * len(y)),
        "diag_labels": meta["Diagnosis"].to_numpy(),
        "sample_meta": meta,
        "n_samples": len(y),
        "n_cpgs": matrix.shape[1],
    }
    return [("GSE99511", primary), ("GSE46306", external)]


def _write_html(summaries, cohort_info, cross_metrics=None):
    def esc(v):
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    metric_cols = [
        ("mean_Accuracy", "Accuracy"), ("mean_Balanced_Accuracy", "Balanced accuracy"),
        ("mean_ROC_AUC", "ROC-AUC"), ("mean_Sensitivity", "Sensitivity"),
        ("mean_Specificity", "Specificity"), ("mean_Precision", "Precision"),
        ("mean_F1", "F1"),
    ]
    html = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Small CpG panel results</title>
<style>body{font:15px/1.5 system-ui,Segoe UI,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#17212b}h1,h2{color:#153a58}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px}th,td{border:1px solid #ccd5dd;padding:6px 8px;text-align:left;white-space:nowrap}th{background:#eaf1f6;position:sticky;top:0}.scroll{overflow:auto;max-height:75vh}.note{background:#fff6da;padding:12px 16px;border-left:4px solid #d69a00}.good{background:#eef7ef;padding:12px 16px;border-left:4px solid #39834a}.small{color:#4d5b66;font-size:13px}code{background:#eef1f3;padding:2px 4px}</style></head><body>
<h1>Small CpG panel benchmark</h1><p>Generated from stratified 5-fold cross-validation. The target is Normal versus CIN3 detection. Each approach selects original CpG probes; no PCA or latent feature reduction is used. Feature selection and scaling are fit inside each training fold.</p>
<div class="note"><b>Interpretation:</b> the retained-feature percentage is k divided by the full cohort probe count. This is not a direct measurement of biological information loss. The observed change in cross-validation metrics across panel sizes is the practical task-specific measure of how much predictive signal the panel retained. Scores are exploratory and do not establish clinical reliability.</div>
<h2>Methods</h2><p>Within each training fold: mean imputation, variance filtering to at most 1,000 probes, selector fit, standardization, and classifier fit. Test-fold samples are held out from every fitting step. All seven selectors are evaluated at 5, 10, 20, and 50 CpGs with logistic regression and linear SVM. Stratified CV uses one 5-fold pass per dataset (5 outer-fold evaluations); classifiers use fixed C=1, without per-fold hyperparameter tuning. For tractability, stability selection uses 10 stratified subsamples, RF importance uses 100 trees, and SVM-RFE removes 50% per step. The variance prefilter is a screening step and may discard useful low-variance CpGs. Metrics are mean ± standard deviation across the five outer folds; uncertainty is high with only one CV repeat.</p>
<h2>Datasets</h2><ul>"""]
    for name, info in cohort_info.items():
        html.append(f"<li><b>{esc(name)}</b>: {info['n']} samples ({esc(info['classes'])}), {info['p']:,} CpGs before the fold-level variance prefilter.</li>")
    html.append("</ul><p>GSE46306 values were inverted from GEO's arcsine-square-root beta transform using sin(x)². Its Normal controls are HPV-negative and CIN3 cases HPV-positive, so its within-cohort scores are confounded by HPV status and are exploratory. GSE99511 contains 28 Normal and 36 CIN3 samples for the primary comparison.</p>")

    for dataset, df in summaries.items():
        html.append(f"<h2>{esc(dataset)} within-cohort results</h2><p>Feature retention is relative to {cohort_info[dataset]['p']:,} available CpGs.</p><div class='scroll'><table><thead><tr><th>Selector</th><th>Classifier</th><th>CpGs k</th><th>Feature retention</th><th>Accuracy mean±SD</th><th>Balanced accuracy mean±SD</th><th>ROC-AUC mean±SD</th><th>Sensitivity mean±SD</th><th>Specificity mean±SD</th><th>Precision mean±SD</th><th>F1 mean±SD</th><th>OOF TP/TN/FP/FN</th><th>Jaccard stability</th></tr></thead><tbody>")
        stab_path = os.path.join(OUT_DIR, dataset, "stability.csv")
        stab = pd.read_csv(stab_path) if os.path.exists(stab_path) else pd.DataFrame()
        for row in df.sort_values(["k", "mean_Balanced_Accuracy"], ascending=[True, False]).to_dict("records"):
            label = SELECTOR_DISPLAY_NAMES.get(row["selector"], row["selector"])
            match = stab[(stab.selector == row["selector"]) & (stab.k == row["k"])] if not stab.empty else pd.DataFrame()
            jacc = f"{match.iloc[0]['jaccard_mean']:.3f}" if not match.empty else "n/a"
            ret = int(row["k"]) / cohort_info[dataset]["p"] * 100
            vals = []
            for col, _ in metric_cols:
                stdcol = col.replace("mean_", "std_")
                vals.append(f"{row[col]:.3f} ± {row[stdcol]:.3f}" if pd.notna(row.get(col)) else "n/a")
            counts = "/".join(str(int(row.get(f"sum_{m}", 0))) for m in ["TP", "TN", "FP", "FN"])
            html.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [label, row["classifier"], row["k"], f"{ret:.4f}%", *vals, counts, jacc]) + "</tr>")
        html.append("</tbody></table></div>")

        best = df.sort_values(["mean_Balanced_Accuracy", "mean_ROC_AUC"], ascending=False).iloc[0]
        html.append(f"<p class='good'><b>Top {esc(dataset)} configuration by balanced accuracy:</b> {esc(SELECTOR_DISPLAY_NAMES.get(best['selector'], best['selector']))}, k={int(best['k'])}, {esc(best['classifier'])}; balanced accuracy {best['mean_Balanced_Accuracy']:.3f} ± {best['std_Balanced_Accuracy']:.3f}, accuracy {best['mean_Accuracy']:.3f} ± {best['std_Accuracy']:.3f}, sensitivity {best['mean_Sensitivity']:.3f} ± {best['std_Sensitivity']:.3f}, specificity {best['mean_Specificity']:.3f} ± {best['std_Specificity']:.3f}, AUC {best['mean_ROC_AUC']:.3f} ± {best['std_ROC_AUC']:.3f}. This is the best observed configuration within this benchmark, not a final validated model.</p>")

    if cross_metrics is not None and not cross_metrics.empty:
        html.append("<h2>Separate cross-cohort check: train GSE99511, test GSE46306</h2><p>This earlier holdout analysis tested only Mann–Whitney and RF-importance panels (k=5, 10, 20, 50), with selection/model fitting on GSE99511 and scoring on GSE46306. It is distinct from the within-cohort CV above. Scores are exploratory because of HPV-status confounding and preprocessing differences.</p><div class='scroll'><table><thead><tr><th>Selector</th><th>Classifier</th><th>k</th><th>Accuracy</th><th>Balanced accuracy</th><th>ROC-AUC</th><th>Sensitivity</th><th>Specificity</th><th>F1</th></tr></thead><tbody>")
        for row in cross_metrics.sort_values("Balanced_Accuracy", ascending=False).to_dict("records"):
            html.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in [SELECTOR_DISPLAY_NAMES.get(row["selector"], row["selector"]), row["classifier"], row["panel_size"], *[f"{row[m]:.3f}" for m in ["Accuracy", "Balanced_Accuracy", "ROC_AUC", "Sensitivity", "Specificity", "F1"]]]) + "</tr>")
        html.append("</tbody></table></div>")
        best_external = cross_metrics.sort_values("Balanced_Accuracy", ascending=False).iloc[0]
        html.append(f"<p><b>Top score in this limited holdout:</b> {esc(best_external['selector'])}, k={int(best_external['panel_size'])}, {esc(best_external['classifier'])}; balanced accuracy {best_external['Balanced_Accuracy']:.3f}, sensitivity {best_external['Sensitivity']:.3f}, specificity {best_external['Specificity']:.3f}. Its specificity is low, and it does not demonstrate reliable generalization.</p>")

    html.append("""<h2>Conclusion</h2><p>These runs show that small CpG panels can be evaluated, but do not establish a final or clinically reliable classifier. Use balanced accuracy, sensitivity, specificity, AUC, precision, and marker stability together; raw accuracy alone can hide missed CIN3 or excess false positives. GSE46306's higher within-cohort scores are confounded because HPV-negative controls are compared with HPV-positive CIN3 cases. The separate cross-cohort check also has low specificity. A 50-marker panel retains only about 0.012% of the GSE99511 CpG input and 0.011% of the GSE46306 CpG input, but this ratio is not a biological information-loss estimate. The observed metric changes across k are the empirical estimate of task-specific predictive signal retained. Validate a locked panel and threshold on a larger independent cohort with HPV status balanced across diagnosis before choosing a kit.</p>
<h2>Files</h2><p>Fold metrics, aggregate metrics, and selected consensus panels are stored beside this report under <code>results/feature_selection_benchmark/all_datasets/</code>.</p><p class="small">Definitions: sensitivity = CIN3 recall; specificity = Normal recall; balanced accuracy = average of sensitivity and specificity; F1 = positive-class harmonic mean of precision and sensitivity; ROC-AUC uses classifier scores. Feature retention = selected CpGs / probes in the analyzed input. Jaccard stability summarizes overlap of selected CpGs between outer folds and is exploratory.</p></body></html>""")
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(os.path.join(HTML_DIR, "result.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    # One five-fold pass, applied to both cohorts for a tractable, matched
    # comparison across all selectors and panel sizes.
    config["cv"]["outer_repeats"] = 1
    config["prefilter"]["prefilter_size_main"] = 1000
    config["cv"]["fixed_classifier_params"] = True
    config["selectors"]["stability_selection"]["n_subsamples"] = 10
    config["selectors"]["rf_importance"]["n_estimators"] = 100
    config["selectors"]["svm_rfe"]["step_fraction"] = 0.5
    config["classifiers"]["logreg"]["C_grid"] = [0.1, 1.0]
    config["classifiers"]["linear_svm"]["C_grid"] = [0.1, 1.0]
    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = {}
    cohort_info = {}
    for name, data in _load_cohorts(config):
        print(f"\n=== {name}: {data['n_samples']} samples, {data['n_cpgs']:,} CpGs ===", flush=True)
        dataset_dir = os.path.join(OUT_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)
        cohort_info[name] = {
            "n": data["n_samples"], "p": data["n_cpgs"],
            "classes": ", ".join(f"{label}={int((data['y'] == idx).sum())}" for idx, label in [(0, "Normal"), (1, "CIN3")]),
        }
        t0 = time.time()
        folds = run_nested_cv(
            data["X"], data["y"], data["cpg_names"], config,
            selector_names=SELECTOR_NAMES,
            prefilter_n=config["prefilter"]["prefilter_size_main"],
            n_jobs=2, verbose=True,
        )
        folds.to_csv(os.path.join(dataset_dir, "fold_metrics.csv"), index=False)
        summary = aggregate_results(folds)
        summary["dataset"] = name
        summary["panel_retained_pct"] = 100 * summary["k"] / data["n_cpgs"]
        summary["probe_count"] = data["n_cpgs"]
        summary.to_csv(os.path.join(dataset_dir, "summary_metrics.csv"), index=False)
        # The selector output is repeated for each classifier in fold_metrics;
        # deduplicate those identical panels so stability counts each fold once.
        panel_folds = folds.drop_duplicates(
            subset=["repeat", "outer_fold", "selector", "k", "prefilter_n"]
        )
        stability = compute_stability(panel_folds, data["n_cpgs"])
        stability.to_csv(os.path.join(dataset_dir, "stability.csv"), index=False)
        freq = compute_selection_frequency(panel_folds)
        panel_rows = []
        for selector in SELECTOR_NAMES:
            for k in config["panel_sizes"]:
                panel = consensus_panel(freq, selector, k, top_n=k)
                for rank, cpg in enumerate(panel, 1):
                    panel_rows.append({"selector": selector, "k": k, "rank": rank, "CpG": cpg})
        pd.DataFrame(panel_rows).to_csv(os.path.join(dataset_dir, "consensus_panels.csv"), index=False)
        summaries[name] = summary
        print(f"{name} done in {(time.time()-t0)/60:.1f} minutes", flush=True)

    # The existing external holdout covers the two historically strongest
    # selector families; record it separately and avoid mixing it with CV.
    cross_path = os.path.join(ROOT, "results", "feature_selection_benchmark", "external_validation_GSE46306", "external_metrics.csv")
    cross = pd.read_csv(cross_path) if os.path.exists(cross_path) else None
    _write_html(summaries, cohort_info, cross_metrics=cross)
    print(f"Wrote {os.path.join(HTML_DIR, 'result.html')}", flush=True)


if __name__ == "__main__":
    main()
