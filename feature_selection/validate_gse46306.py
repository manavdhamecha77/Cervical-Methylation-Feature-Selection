"""External cohort check: train on GSE99511, evaluate on GSE46306.

This is a held-out cohort check for the existing Normal-versus-CIN3 task.
It is not a clean validation of HPV-positive triage because GSE46306's normal
controls are HPV-negative and its CIN3 cases are HPV-positive.
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_loader import load_primary_data, impute_fold, variance_prefilter
from evaluator import _get_classifier_proba, _tune_classifier, compute_metrics
from fs_selectors import get_selector

warnings.filterwarnings("ignore")

CONFIG_PATH = os.path.join(HERE, "config.json")
GSE46306_DIR = os.path.join(ROOT, "data", "GSE46306", "GSE46306_clean")
EXTERNAL_MATRIX = os.path.join(GSE46306_DIR, "GSE46306_beta_matrix.csv.gz")
EXTERNAL_META = os.path.join(GSE46306_DIR, "GSE46306_sample_metadata.csv")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark",
                       "external_validation_GSE46306")


def load_external(config):
    meta = pd.read_csv(EXTERNAL_META, dtype={"sample_id": str, "group_label": str})
    if meta["sample_id"].duplicated().any():
        raise ValueError("GSE46306 metadata contains duplicate sample IDs")

    # The local metadata uses g2 for cancer; the GEO record calls that group g3.
    label_map = {"g0": "Normal", "g1": "CIN3", "g2": "Cancer"}
    unknown = sorted(set(meta["group_label"]) - set(label_map))
    if unknown:
        raise ValueError(f"Unexpected GSE46306 group codes: {unknown}")
    meta["Diagnosis"] = meta["group_label"].map(label_map)

    # Keep only the Normal/CIN3 comparison, matching the development task.
    meta = meta[meta["Diagnosis"].isin(["Normal", "CIN3"])].copy()
    expected_counts = {"Normal": 20, "CIN3": 18}
    observed_counts = meta["Diagnosis"].value_counts().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected external class counts: {observed_counts}")

    matrix = pd.read_csv(EXTERNAL_MATRIX, index_col=0, low_memory=False)
    matrix.columns = matrix.columns.astype(str)
    missing_samples = sorted(set(meta["sample_id"]) - set(matrix.columns))
    if missing_samples:
        raise ValueError(f"Samples absent from beta matrix: {missing_samples[:5]}")

    # GEO documents these as arcsine-square-root transformed beta values:
    # transformed = asin(sqrt(beta)); invert to return to the beta scale.
    values = matrix.loc[:, meta["sample_id"].tolist()].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("GSE46306 matrix has missing or non-numeric values")
    if values.min().min() < 0 or values.max().max() > (np.pi / 2 + 1e-6):
        raise ValueError("GSE46306 values are outside the documented transform range")
    beta = np.square(np.sin(values.to_numpy(dtype=np.float64)))

    probe_ids = matrix.index.astype(str)
    cpg_mask = np.char.startswith(probe_ids.to_numpy(dtype=str), "cg")
    cpg_names = probe_ids[cpg_mask].tolist()
    beta = beta[cpg_mask, :]
    external = pd.DataFrame(beta.T, index=meta["sample_id"].values,
                            columns=cpg_names)
    y = (meta.set_index("sample_id").loc[external.index, "Diagnosis"] == "CIN3").astype(int).to_numpy()
    return external, y, meta.reset_index(drop=True)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading GSE99511 development cohort...")
    train = load_primary_data(config, verbose=True)
    X_train_full = train["X"]
    y_train = train["y"]
    train_cpgs = train["cpg_names"]

    print("Loading and harmonizing GSE46306 external cohort...")
    external, y_external, external_meta = load_external(config)
    common_cpgs = [name for name in train_cpgs if name in external.columns]
    if len(common_cpgs) < 50:
        raise ValueError(f"Only {len(common_cpgs)} shared CpGs; cannot evaluate panels")
    train_index_by_cpg = {name: i for i, name in enumerate(train_cpgs)}
    train_indices = np.array([train_index_by_cpg[name] for name in common_cpgs], dtype=int)
    X_train = X_train_full[:, train_indices]
    X_external = external.loc[:, common_cpgs].to_numpy(dtype=np.float32)

    # Imputation, variance filtering, selection, and scaling are learned only
    # from GSE99511. GSE46306 labels are used only for final score calculation.
    X_train, X_external, _ = impute_fold(X_train, X_external)
    X_train_pf, X_external_pf, prefiltered_cpgs, _ = variance_prefilter(
        X_train, X_external, common_cpgs,
        threshold=config["prefilter"]["variance_threshold"],
        top_n=config["prefilter"]["prefilter_size_main"],
    )

    # These selectors were the strongest small-panel methods in the saved
    # internal benchmark. All four panel sizes and both classifiers are
    # evaluated; external performance is not used to select a winner.
    selectors = ["rf_importance", "mw_filter"]
    panel_sizes = config["panel_sizes"]
    classifiers = ["logreg", "linear_svm"]
    summary_rows = []
    prediction_rows = []
    panel_rows = []

    for selector_name in selectors:
        for k in panel_sizes:
            selector = get_selector(selector_name, k, config,
                                     random_state=config["cv"]["random_seed"])
            selector.fit(X_train_pf, y_train)
            selected_idx = selector.selected_indices_
            selected_cpgs = [prefiltered_cpgs[i] for i in selected_idx]
            Xtr = X_train_pf[:, selected_idx]
            Xte = X_external_pf[:, selected_idx]
            scaler = StandardScaler().fit(Xtr)
            Xtr_scaled = scaler.transform(Xtr)
            Xte_scaled = scaler.transform(Xte)

            for classifier_name in classifiers:
                classifier = _tune_classifier(
                    classifier_name, Xtr_scaled, y_train, config,
                    config["cv"]["random_seed"],
                )
                predicted = classifier.predict(Xte_scaled)
                score = _get_classifier_proba(classifier, Xte_scaled)
                metrics = compute_metrics(y_external, predicted, score)
                model_id = f"{selector_name}_k{k}_{classifier_name}"
                summary_rows.append({
                    "model_id": model_id,
                    "selector": selector_name,
                    "panel_size": k,
                    "classifier": classifier_name,
                    "development_samples": len(y_train),
                    "external_samples": len(y_external),
                    "external_normal": int((y_external == 0).sum()),
                    "external_cin3": int((y_external == 1).sum()),
                    **metrics,
                })
                for i, sample_id in enumerate(external_meta["sample_id"]):
                    prediction_rows.append({
                        "model_id": model_id,
                        "sample_id": sample_id,
                        "group_code": external_meta.iloc[i]["group_label"],
                        "diagnosis": external_meta.iloc[i]["Diagnosis"],
                        "true_cin3": int(y_external[i]),
                        "predicted_cin3": int(predicted[i]),
                        "score": float(score[i]),
                    })
                for rank, cpg in enumerate(selected_cpgs, start=1):
                    panel_rows.append({
                        "model_id": model_id,
                        "selector": selector_name,
                        "panel_size": k,
                        "rank": rank,
                        "CpG": cpg,
                    })

                # Save candidate scoring artifact. It expects ordinary beta
                # values for the listed CpGs, not GSE46306 transformed values.
                joblib.dump({
                    "model_id": model_id,
                    "selector_name": selector_name,
                    "panel_size": k,
                    "classifier_name": classifier_name,
                    "selected_cpgs": selected_cpgs,
                    "scaler": scaler,
                    "classifier": classifier,
                    "input_scale": "beta values in [0, 1]",
                }, os.path.join(OUT_DIR, f"{model_id}.joblib"))
                print(f"{model_id}: BA={metrics['Balanced_Accuracy']:.3f}, "
                      f"AUC={metrics['ROC_AUC']:.3f}, "
                      f"sensitivity={metrics['Sensitivity']:.3f}, "
                      f"specificity={metrics['Specificity']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "external_metrics.csv"), index=False)
    pd.DataFrame(prediction_rows).to_csv(os.path.join(OUT_DIR, "external_predictions.csv"), index=False)
    pd.DataFrame(panel_rows).to_csv(os.path.join(OUT_DIR, "selected_panels.csv"), index=False)
    external_meta.to_csv(os.path.join(OUT_DIR, "external_sample_metadata.csv"), index=False)

    top_rows = summary_df.sort_values(
        ["Balanced_Accuracy", "ROC_AUC"], ascending=False
    ).head(5)
    metrics_table = (
        "| Model | Balanced accuracy | ROC-AUC | Sensitivity | Specificity |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row.model_id} | {row.Balanced_Accuracy:.3f} | {row.ROC_AUC:.3f} | "
            f"{row.Sensitivity:.3f} | {row.Specificity:.3f} |"
            for row in top_rows.itertuples(index=False)
        )
    )
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# GSE46306 external cohort check

GSE99511 was used for feature prefiltering, feature selection, scaling, and
classifier tuning. GSE46306 labels were used only for final metrics.

The matrix values were converted from GEO's documented arcsine-square-root
transformation back to beta values with `sin(value) ** 2`. The main comparison
uses 20 Normal and 18 CIN3 samples; six cancer samples are excluded.

## Highest scoring evaluated configurations

These are descriptive external scores; they were not used to select or tune
the panel. Full metrics for all 16 evaluated configurations are in
`external_metrics.csv`.

{metrics_table}

This is a small, cautious external check. GSE46306 normal controls are HPV
negative and CIN3 cases are HPV positive, so class and HPV status are
confounded. Its preprocessing also differs from GSE99511. Do not interpret
these scores as validation for HPV-positive triage or as clinical performance.

The `.joblib` files are candidate models trained on all GSE99511 Normal/CIN3
samples. They are research artifacts, not final clinical models. They expect
beta values in the listed CpG order.
""")

    print(f"\nSaved external metrics, predictions, panels, and candidate models to: {OUT_DIR}")


if __name__ == "__main__":
    main()
