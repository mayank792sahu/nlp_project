"""Builds the cross-model comparison artifacts: confusion-matrix plots, the
comparison table, and the comparison bar chart, saved under results/."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from finsent.config import CONFIG, resolve_path
from finsent.models.base import LABELS
from finsent.palette import (
    BASELINE,
    GRIDLINE,
    MODEL_COLORS,
    SEQUENTIAL_BLUE,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_BLUE_CMAP = LinearSegmentedColormap.from_list("finsent_blue", SEQUENTIAL_BLUE)


def _apply_chart_chrome(fig, ax) -> None:
    """Matches the project's palette: surface/page tokens, ink-colored text,
    hairline recessive gridlines, no heavy chart-junk borders."""
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)


def _results_dir(key: str):
    d = resolve_path(CONFIG["results"][key])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_all_metrics(results: dict) -> None:
    out_path = _results_dir("metrics_dir") / "all_models.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def plot_confusion_matrix(cm: list, model_name: str, labels=LABELS) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=_BLUE_CMAP, xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False, annot_kws={"color": TEXT_PRIMARY, "fontsize": 11},
        linewidths=2, linecolor=SURFACE,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(model_name, fontweight="bold")
    _apply_chart_chrome(fig, ax)
    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(_results_dir("figures_dir") / f"{safe_name}_confusion.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "neutral_f1": metrics["per_class"]["neutral"]["f1-score"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(_results_dir("tables_dir") / "comparison.csv", index=False)
    return df


def plot_comparison_bar(df: pd.DataFrame) -> None:
    # Two metrics compared per model -> metric identity gets the categorical colors here
    # (distinct from MODEL_COLORS, which identifies models elsewhere in the dashboard).
    metric_colors = {"macro_f1": MODEL_COLORS["VADER"], "neutral_f1": MODEL_COLORS["Loughran-McDonald"]}
    metric_labels = {"macro_f1": "Macro-F1", "neutral_f1": "Neutral-class F1"}

    melted = df.melt(id_vars="model", value_vars=["macro_f1", "neutral_f1"], var_name="metric", value_name="score")
    melted["metric_label"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(
        data=melted, x="model", y="score", hue="metric_label", ax=ax,
        palette=[metric_colors["macro_f1"], metric_colors["neutral_f1"]],
        width=0.6, edgecolor="none",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel(None)
    ax.set_title("Model Comparison — Macro-F1 vs. Neutral-Class F1", fontweight="bold")
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, loc="upper left", bbox_to_anchor=(0, 1.15), ncol=2)
    _apply_chart_chrome(fig, ax)
    fig.tight_layout()
    fig.savefig(_results_dir("figures_dir") / "comparison.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def generate_full_report(results: dict) -> pd.DataFrame:
    """Runs the full reporting pipeline: saves raw metrics JSON, per-model confusion
    matrices, the comparison table, and the comparison bar chart."""
    save_all_metrics(results)
    for model_name, metrics in results.items():
        plot_confusion_matrix(metrics["confusion_matrix"], model_name)
    df = build_comparison_table(results)
    plot_comparison_bar(df)
    return df
