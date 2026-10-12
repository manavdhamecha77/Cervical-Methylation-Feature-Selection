"""Build a self-contained HTML report from the saved simulation summary."""
from html import escape
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = pd.read_csv(HERE / "simulation_runs.csv")
GROUPS = ["dataset", "k", "selector", "classifier", "scenario"]
SUMMARY = RUNS.groupby(GROUPS)[
    ["Accuracy", "Balanced_Accuracy", "ROC_AUC", "Sensitivity",
     "Specificity", "Precision", "F1", "TP", "TN", "FP", "FN"]
].agg(["mean", "std"]).reset_index()
SUMMARY.columns = [
    "_".join(str(part) for part in col if str(part))
    if isinstance(col, tuple) else str(col) for col in SUMMARY.columns
]
SUMMARY.to_csv(HERE / "simulation_summary.csv", index=False)
CHOICES = pd.read_csv(HERE / "model_choices.csv")
METRICS = ["Balanced_Accuracy", "ROC_AUC", "Sensitivity", "Specificity",
           "Precision", "F1", "Accuracy"]
DISPLAY = {"Balanced_Accuracy": "Balanced accuracy", "ROC_AUC": "ROC-AUC",
           "Sensitivity": "Sensitivity", "Specificity": "Specificity",
           "Precision": "Precision", "F1": "F1", "Accuracy": "Accuracy"}
SCENARIO_LABELS = {
    "clean": "Clean input (no simulated error)",
    "noise_sd_0.01": "Random noise, SD 0.01 beta",
    "noise_sd_0.03": "Random noise, SD 0.03 beta",
    "noise_sd_0.05": "Random noise, SD 0.05 beta",
    "offset_pm_0.02": "Sample offset, ±0.02 beta",
    "offset_pm_0.05": "Sample offset, ±0.05 beta",
    "gain_0.9_1.1": "Gain error, 0.9–1.1×",
    "gain_0.8_1.2": "Gain error, 0.8–1.2×",
    "drift_pm_0.02": "Ordered drift, ±0.02 beta",
    "drift_pm_0.05": "Ordered drift, ±0.05 beta",
    "missing_05pct": "Missing readings, 5%",
    "missing_10pct": "Missing readings, 10%",
    "missing_20pct": "Missing readings, 20%",
    "combined_moderate": "Combined moderate assumptions",
    "combined_severe": "Combined severe assumptions",
}

def metric(row, name):
    mean = row[f"{name}_mean"]
    std = row.get(f"{name}_std", float("nan"))
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"

html = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hypothetical Sensor Error Simulation</title>
<style>
body{font:15px/1.55 system-ui,Segoe UI,sans-serif;max-width:1400px;margin:28px auto;padding:0 20px;color:#17212b}
h1,h2,h3{color:#153a58}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 26px}
th,td{border:1px solid #ccd5dd;padding:7px 9px;text-align:left;white-space:nowrap}
th{background:#eaf1f6;position:sticky;top:0}.scroll{overflow:auto;max-height:70vh}
.note{background:#fff4d6;padding:14px 18px;border-left:4px solid #d69a00;margin:16px 0}
.info{background:#edf5fb;padding:14px 18px;border-left:4px solid #377fa8;margin:16px 0}
.small{color:#4d5b66;font-size:13px}code{background:#eef1f3;padding:2px 4px}
</style></head><body>
<h1>Hypothetical Sensor Error Simulation</h1>
<p>Scenario-based robustness analysis for small methylation CpG panels using GSE99511 and GSE46306.</p>
<div class="note"><b>Interpretation boundary:</b> no real sensor measurements were available.
Every error magnitude here is an assumed stress-test setting, not a published or measured
property of the proposed hardware. These results cannot establish real device, assay, or
clinical performance.</div>
<h2>What was simulated</h2>
<p>Five-fold stratified cross-validation was run separately within each dataset. For every
fold, feature filtering, marker selection, scaling, and model fitting used only the training
samples. Measurement perturbations were applied only to held-out samples in beta-value space;
values were clipped to [0, 1]. The same synthetic perturbation realization was applied to all
markers in each held-out sample for sample-wide calibration effects. Results pool out-of-fold
predictions. Non-clean scenarios use 50 Monte Carlo draws; shown values are mean ± SD across
draws. A clean run uses one deterministic evaluation.</p>
<div class="info"><b>Model choice:</b> for each dataset and panel size (5, 10, 20, 50), the
selector/classifier with the highest balanced accuracy in the existing one-pass five-fold
benchmark was chosen. This winner selection uses the same small datasets and is exploratory;
the apparent best scores can be optimistic. The simulation compares robustness for these
screened candidates and does not independently validate them.</div>
<h2>Assumed stress scenarios</h2>
<ul><li>Independent per-marker random noise: Gaussian SD 0.01, 0.03, or 0.05 beta.</li>
<li>Per-sample calibration offset: uniform ±0.02 or ±0.05 beta, shared across that sample’s markers.</li>
<li>Per-sample gain error around beta=0.5: 0.9–1.1× or 0.8–1.2×.</li>
<li>Ordered drift: offset changes linearly from −0.02 to +0.02 or −0.05 to +0.05 beta across a synthetic acquisition order unrelated to diagnosis.</li>
<li>Missing marker readings: 5%, 10%, or 20%; missing values are replaced by that marker’s training-fold median.</li>
<li>Combined moderate: noise SD 0.03, offset ±0.03, gain ±10%, drift ±0.03, 10% missing.</li>
<li>Combined severe: noise SD 0.05, offset ±0.05, gain ±20%, drift ±0.05, 20% missing.</li></ul>
<h2>Results</h2>"""]

for dataset in ["GSE99511", "GSE46306"]:
    html.append(f"<h3>{escape(dataset)}</h3>")
    choices = CHOICES[CHOICES["dataset"] == dataset].sort_values("k")
    for choice in choices.to_dict("records"):
        html.append(f"<p><b>Panel k={int(choice['k'])}:</b> {escape(choice['selector'])} + {escape(choice['classifier'])}. "
                    f"Selected from the existing benchmark (balanced accuracy {choice['benchmark_balanced_accuracy']:.3f}; "
                    f"ROC-AUC {choice['benchmark_roc_auc']:.3f}).</p>")
        part = SUMMARY[(SUMMARY["dataset"] == dataset) & (SUMMARY["k"] == choice["k"])].copy()
        part["order"] = part["scenario"].map({key: i for i, key in enumerate(SCENARIO_LABELS)})
        part = part.sort_values("order")
        html.append('<div class="scroll"><table><thead><tr><th>Scenario</th>' +
                    ''.join(f'<th>{DISPLAY[m]} mean ± SD</th>' for m in METRICS) +
                    '<th>Mean TP/TN/FP/FN</th></tr></thead><tbody>')
        for row in part.to_dict("records"):
            label = SCENARIO_LABELS.get(row["scenario"], row["scenario"])
            counts = "/".join(str(round(row[f"{m}_mean"])) for m in ["TP", "TN", "FP", "FN"])
            html.append('<tr><td>' + escape(label) + '</td>' +
                        ''.join(f'<td>{metric(row, m)}</td>' for m in METRICS) +
                        f'<td>{counts}</td></tr>')
        html.append('</tbody></table></div>')

html.append("""<h2>How to read this</h2>
<p>Look for sensitivity and false negatives as well as balanced accuracy and specificity.
Small cohorts make each result unstable, and cross-validation folds share the same source
cohort. The GSE46306 comparison is especially limited: its Normal controls are HPV-negative
and its CIN3 cases are HPV-positive, so diagnosis and HPV status are confounded. GSE99511 is
the main development cohort.</p>
<h2>Conclusion</h2>
<p>This analysis provides a reproducible way to ask whether candidate panels are fragile
under explicit hypothetical readout errors. It can help prioritize panels for future
bench testing, but it cannot tell us the actual noise, drift, calibration error, missing
rate, or clinical accuracy of a sensor that has not been measured. A next validation step
is to measure repeated reference samples on the intended assay/device, estimate its error
distributions, replace these assumed settings with measured values, and evaluate a locked
panel on an independent cohort.</p>
<h2>Reproducibility files</h2>
<p>Script: <code>sensor-sim/run_sensor_simulation.py</code>. Assumptions and run instructions:
<code>sensor-sim/README.md</code>. Per-run rows: <code>sensor-sim/simulation_runs.csv</code>;
summary table: <code>sensor-sim/simulation_summary.csv</code>; screened model choices:
<code>sensor-sim/model_choices.csv</code>.</p>
<p class="small">Metric definitions: sensitivity = CIN3 recall; specificity = Normal recall;
balanced accuracy = average of sensitivity and specificity. TP/TN/FP/FN are pooled
out-of-fold counts for each Monte Carlo draw, averaged in the displayed table.</p>
</body></html>""")
HTML_DIR = ROOT / "html"
HTML_DIR.mkdir(exist_ok=True)
(HTML_DIR / "sensor_simulation_results.html").write_text("\n".join(html), encoding="utf-8")
print("Wrote", HTML_DIR / "sensor_simulation_results.html")
