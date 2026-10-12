"""
plots.py
--------
All publication-quality visualizations for the feature selection benchmark.

Plots produced
--------------
1. performance_vs_k.png       — AUC and Balanced Accuracy vs k, per selector
2. stability_vs_k.png         — Jaccard and Kuncheva index vs k, per selector
3. selection_frequency_heatmap_k{k}.png — CpG selection frequency heatmap per k
4. method_comparison_heatmap.png — summary table as heatmap (AUC, Bal.Acc)
5. permutation_test.png       — null distribution vs observed score
6. correlation_heatmap_{sel}_k{k}.png — pairwise CpG correlation for consensus panel
7. baseline_comparison.png    — original vs corrected RF-5000 result
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from fs_selectors import SELECTOR_DISPLAY_NAMES

warnings.filterwarnings("ignore")

PALETTE = {
    "mw_filter":           "#1976D2",
    "lasso":               "#E53935",
    "elastic_net":         "#FB8C00",
    "stability_selection": "#8E24AA",
    "svm_rfe":             "#00897B",
    "mrmr":                "#C62828",
    "rf_importance":       "#2E7D32",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size":  10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ---- 1. Performance vs k ----

def plot_performance_vs_k(agg_df: pd.DataFrame, out_dir: str,
                          metric: str = "Balanced_Accuracy",
                          classifier: str = "logreg"):
    """
    Line plot: metric vs k, one line per selector, with ±1 std error band.
    """
    sub = agg_df[agg_df["classifier"] == classifier].copy() if "classifier" in agg_df.columns else agg_df.copy()

    selectors = sub["selector"].unique()
    k_vals    = sorted(sub["k"].unique())
    m_mean    = f"mean_{metric}"
    m_std     = f"std_{metric}"

    if m_mean not in sub.columns:
        print(f"  [plot_performance_vs_k] Column {m_mean} not found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for sel in selectors:
        df_sel = sub[sub["selector"] == sel].sort_values("k")
        label  = SELECTOR_DISPLAY_NAMES.get(sel, sel)
        color  = PALETTE.get(sel, None)
        ax.plot(df_sel["k"], df_sel[m_mean],
                marker="o", label=label, color=color, linewidth=1.8)
        ax.fill_between(
            df_sel["k"],
            df_sel[m_mean] - df_sel[m_std].fillna(0),
            df_sel[m_mean] + df_sel[m_std].fillna(0),
            alpha=0.12, color=color,
        )

    ax.axhline(y=0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Chance (0.5)")
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Panel Size k (CpGs)")
    ax.set_ylabel(f"Mean {metric.replace('_', ' ')} (± 1 SD)")
    ax.set_title(
        f"{metric.replace('_', ' ')} vs Panel Size\n"
        f"Classifier: {classifier}  |  Repeated nested CV (5×10)\n"
        "WARNING: n=64, interpret std ranges as wide confidence intervals",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.2, 1.0)
    fname = os.path.join(out_dir, f"performance_vs_k_{classifier}_{metric}.png")
    _save(fig, fname)


# ---- 2. Stability vs k ----

def plot_stability_vs_k(stab_df: pd.DataFrame, out_dir: str,
                        metric: str = "jaccard_mean"):
    """
    Line plot: stability metric vs k, one line per selector.
    """
    if metric not in stab_df.columns:
        print(f"  [plot_stability_vs_k] Column {metric} not found — skipping.")
        return

    selectors = stab_df["selector"].unique()
    k_vals    = sorted(stab_df["k"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    for sel in selectors:
        df_sel = stab_df[stab_df["selector"] == sel].sort_values("k")
        label  = SELECTOR_DISPLAY_NAMES.get(sel, sel)
        color  = PALETTE.get(sel, None)
        ax.plot(df_sel["k"], df_sel[metric],
                marker="s", label=label, color=color, linewidth=1.8)

    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_xlabel("Panel Size k (CpGs)")
    label_map = {
        "jaccard_mean":   "Mean Pairwise Jaccard Index",
        "kuncheva_index": "Kuncheva Stability Index",
    }
    ax.set_ylabel(label_map.get(metric, metric))
    ax.set_title(
        f"Feature Selection Stability vs Panel Size\n"
        f"({label_map.get(metric, metric)}, across 50 outer folds)",
        fontsize=10,
    )
    ax.axhline(y=0.0, color="grey", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fname = os.path.join(out_dir, f"stability_vs_k_{metric}.png")
    _save(fig, fname)


# ---- 3. Selection frequency heatmap ----

def plot_frequency_heatmap(heatmap_df: pd.DataFrame, k: int,
                           out_dir: str, top_n: int = 40):
    """
    Heatmap: CpGs (rows) × selectors (columns), color = selection frequency.
    """
    if heatmap_df.empty:
        return

    # Cap rows at top_n
    sub = heatmap_df.head(top_n)
    if sub.empty:
        return

    # Rename columns to display names
    rename = {s: SELECTOR_DISPLAY_NAMES.get(s, s) for s in sub.columns}
    sub = sub.rename(columns=rename)

    fig_h = max(6, len(sub) * 0.3)
    fig, ax = plt.subplots(figsize=(max(8, len(sub.columns) * 1.5), fig_h))
    sns.heatmap(
        sub,
        ax=ax,
        cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        annot=(len(sub) <= 25),
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Selection Frequency"},
    )
    ax.set_xlabel("Selection Method", fontsize=10)
    ax.set_ylabel("CpG ID", fontsize=10)
    ax.set_title(
        f"CpG Selection Frequency Across Outer Folds\n"
        f"Panel size k={k}  |  Top {len(sub)} CpGs  [Exploratory]",
        fontsize=10,
    )
    fname = os.path.join(out_dir, f"selection_frequency_heatmap_k{k}.png")
    _save(fig, fname)


# ---- 4. Method comparison summary heatmap ----

def plot_method_comparison_heatmap(agg_df: pd.DataFrame, out_dir: str,
                                   classifier: str = "logreg",
                                   metric: str = "ROC_AUC"):
    """
    Heatmap: selectors (rows) × k values (columns), values = mean metric.
    """
    sub = agg_df[agg_df["classifier"] == classifier].copy() if "classifier" in agg_df.columns else agg_df.copy()
    m_col = f"mean_{metric}"
    if m_col not in sub.columns:
        return

    pivot = sub.pivot_table(index="selector", columns="k", values=m_col)
    if pivot.empty:
        return

    # Rename selector rows
    pivot.index = [SELECTOR_DISPLAY_NAMES.get(s, s) for s in pivot.index]

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.7)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        vmin=0.5, vmax=0.9,
        linewidths=0.5,
        cbar_kws={"label": f"Mean {metric.replace('_', ' ')}"},
    )
    ax.set_xlabel("Panel Size k")
    ax.set_ylabel("Selection Method")
    ax.set_title(
        f"Mean {metric.replace('_', ' ')} by Method × Panel Size\n"
        f"Classifier: {classifier}  |  n=64, interpret with caution",
        fontsize=10,
    )
    fname = os.path.join(out_dir, f"method_comparison_{classifier}_{metric}.png")
    _save(fig, fname)


# ---- 5. Permutation test ----

def plot_permutation_test(perm_result: dict, observed_score: float,
                          out_dir: str, metric: str = "Balanced_Accuracy"):
    """
    Histogram of null distribution with observed score marked.
    """
    null_scores = perm_result.get("null_balanced_accuracy", [])
    if not null_scores:
        return

    null = np.array(null_scores)
    p_val = (null >= observed_score).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=20, color="#90CAF9", edgecolor="white", alpha=0.8,
            label=f"Null distribution (n={len(null)})")
    ax.axvline(x=observed_score, color="#D32F2F", linewidth=2.5,
               label=f"Observed: {observed_score:.3f}")
    ax.axvline(x=0.5, color="grey", linewidth=1, linestyle="--", alpha=0.6,
               label="Chance (0.5)")
    ax.set_xlabel(f"Mean {metric.replace('_', ' ')} (permuted labels)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation Test — Label-Shuffle Null Distribution\n"
        f"Empirical p-value: {p_val:.3f}  (proportion of null ≥ observed)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fname = os.path.join(out_dir, "permutation_test.png")
    _save(fig, fname)


# ---- 6. Correlation heatmap of consensus panel ----

def plot_correlation_heatmap(corr_df: pd.DataFrame, selector: str,
                             k: int, out_dir: str):
    """
    Heatmap of pairwise Pearson correlations among consensus panel CpGs.
    High correlation indicates redundancy (bad for a sensor panel).
    """
    if corr_df is None or corr_df.empty:
        return

    n = len(corr_df)
    fig_size = max(5, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    mask = np.zeros_like(corr_df.values, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = False  # show full matrix

    sns.heatmap(
        corr_df,
        ax=ax,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        annot=(n <= 20),
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
        square=True,
    )
    display_name = SELECTOR_DISPLAY_NAMES.get(selector, selector)
    ax.set_title(
        f"CpG Pairwise Correlation — Consensus Panel\n"
        f"Method: {display_name}  |  k={k}\n"
        "[Descriptive — computed on globally imputed full dataset]",
        fontsize=9,
    )
    fname = os.path.join(out_dir, f"correlation_heatmap_{selector}_k{k}.png")
    _save(fig, fname)


# ---- 7. Baseline comparison ----

def plot_baseline_comparison(
    baseline_df: pd.DataFrame,
    original_mean: float,
    original_std: float,
    out_dir: str,
    metric: str = "Balanced_Accuracy",
):
    """
    Bar chart comparing:
      - Original pipeline 5-fold CV (non-repeated) result
      - Reproduced result under correct repeated nested CV
    """
    if baseline_df.empty:
        return

    corrected_mean = baseline_df[metric].mean()
    corrected_std  = baseline_df[metric].std()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Original\n(5-fold, single run)", "Corrected\n(5×10 repeated\nnested CV)"],
        [original_mean, corrected_mean],
        yerr=[original_std, corrected_std],
        capsize=8, color=["#FF7043", "#42A5F5"],
        edgecolor="black", linewidth=0.8, width=0.5,
    )
    for bar, val in zip(bars, [original_mean, corrected_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(
        "ANOVA top-5000 + RandomForest\nOriginal vs Corrected CV Protocol",
        fontsize=10,
    )
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()
    fname = os.path.join(out_dir, "baseline_comparison.png")
    _save(fig, fname)


# ---- 8. Summary table heatmap (both classifiers) ----

def plot_summary_table(agg_df: pd.DataFrame, out_dir: str):
    """
    Combined summary table (all selectors × k × classifier) as a figure.
    """
    metrics = ["mean_Balanced_Accuracy", "mean_ROC_AUC"]
    available_metrics = [m for m in metrics if m in agg_df.columns]
    if not available_metrics:
        return

    for clf_name in agg_df["classifier"].unique() if "classifier" in agg_df.columns else ["all"]:
        if "classifier" in agg_df.columns:
            sub = agg_df[agg_df["classifier"] == clf_name]
        else:
            sub = agg_df.copy()

        for metric in available_metrics:
            pivot = sub.pivot_table(index="selector", columns="k", values=metric)
            if pivot.empty:
                continue
            pivot.index = [SELECTOR_DISPLAY_NAMES.get(s, s) for s in pivot.index]
            fig, ax = plt.subplots(
                figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.7))
            )
            sns.heatmap(
                pivot, ax=ax, cmap="RdYlGn",
                annot=True, fmt=".3f",
                vmin=0.45, vmax=0.90,
                linewidths=0.5,
                cbar_kws={"label": metric.replace("mean_", "").replace("_", " ")},
            )
            ax.set_xlabel("Panel Size k (CpGs)")
            ax.set_ylabel("Selection Method")
            label = metric.replace("mean_", "").replace("_", " ")
            ax.set_title(
                f"Mean {label}\nClassifier: {clf_name}  |  "
                "n=64 — bars are wide confidence intervals",
                fontsize=9,
            )
            fname = os.path.join(out_dir, f"summary_{clf_name}_{metric}.png")
            _save(fig, fname)
