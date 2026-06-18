from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure9_polish"
PLOT_OUT = OUT / "polished"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for path in (PLOT_OUT, SOURCE_OUT, QA_OUT):
    path.mkdir(parents=True, exist_ok=True)

CANON = (
    ROOT
    / "Publication/paper/submission_package_communications_biology_20260609/07_source_data_and_code_availability/github_release/source_data/main_figures/canonical_source_tables"
)
LEGACY_FIG9 = (
    ROOT
    / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_9_practical_method_selection_source_data.csv"
)

FIGURE_BASENAME = "Figure_9_practical_method_selection_guide_polished"

FAMILY_RENAME = {
    "linear/probabilistic": "linear",
    "deep generative/autoencoder": "deep",
    "graph/diffusion": "graph",
    "metric/structure-aware": "metric",
}
FAMILY_ORDER = ["linear", "deep", "graph", "metric"]
FAMILY_COLORS = {
    "linear": "#3D6FB6",
    "deep": "#BF6F6B",
    "graph": "#C79B38",
    "metric": "#3F9C9A",
}
CONTROL_COLOR = "#8A8A8A"

GOOD = "#9DCCD9"
MEDIUM = "#F1D78D"
POOR = "#DF5B4F"
HEADER = "#C8C8C8"
SUBHEADER = "#DEDEDE"
ROW_GREY = "#F1F1F1"
GRID = "#FFFFFF"
TEXT = "#1F1F1F"

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_cmap",
    ["#E8EFE7", "#86BFC4", "#2E6E8E", "#253266"],
)
SENS_CMAP = LinearSegmentedColormap.from_list(
    "sensitivity_cmap",
    ["#E9F2EF", "#E3C56D", "#C7655E"],
)

RECOMMENDATION_GROUPS = [
    ("Rare types", ["TriMap", "SQuaD-MDS", "UMAP", "SPDR"]),
    ("Trajectory", ["TriMap", "PCA", "SQuaD-MDS", "PHATE", "scGBM"]),
    ("Clustering", ["TriMap", "t-SNE", "GLMPCA"]),
    ("Large scale", ["PCA", "TriMap", "UMAP", "PHATE"]),
    ("Noise", ["TriMap", "SPDR", "t-SNE", "UMAP"]),
    ("Balanced", ["TriMap", "UMAP", "t-SNE", "PHATE", "SPDR"]),
]
RECOMMENDED_METHODS = []
for _, methods in RECOMMENDATION_GROUPS:
    for method in methods:
        if method not in RECOMMENDED_METHODS:
            RECOMMENDED_METHODS.append(method)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.4,
        "axes.titlesize": 6.8,
        "axes.labelsize": 5.8,
        "xtick.labelsize": 4.9,
        "ytick.labelsize": 4.9,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)


def display_method(name: str) -> str:
    return {
        "t-SNE": "t-SNE",
        "SQuaD-MDS": "SQuaD-MDS",
        "SSNMDI": "SSNMDI",
        "tGPLVM": "tGPLVM",
        "scvis": "scvis",
        "scScope": "scScope",
        "scVI": "scVI",
    }.get(str(name), str(name))


def panel_header(ax: plt.Axes, label: str, title: str, dx: float = -0.06, dy: float = 1.035, title_dx: float = 0.075) -> None:
    ax.text(dx, dy, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0, fontweight="bold", color="#171717")
    ax.text(dx + title_dx, dy, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, color="#171717")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def time_label(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds >= 7200:
        return f">{seconds / 3600:.0f}h"
    return f"{seconds / 3600:.1f}h"


def memory_label(gb: float) -> str:
    if pd.isna(gb):
        return ""
    gb = float(gb)
    if gb < 1:
        return f"{gb * 1024:.0f}M"
    if gb < 10:
        return f"{gb:.1f}G"
    return f"{gb:.0f}G"


def score_symbol(score: float) -> str:
    if pd.isna(score):
        return ""
    if score >= 0.70:
        return "+"
    if score >= 0.50:
        return "±"
    return "-"


def score_class(score: float) -> str:
    if pd.isna(score):
        return "missing"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "medium"
    return "poor"


def runtime_class(seconds: float) -> str:
    if pd.isna(seconds):
        return "missing"
    if seconds <= 60:
        return "good"
    if seconds <= 3600:
        return "medium"
    return "poor"


def memory_class(gb: float) -> str:
    if pd.isna(gb):
        return "missing"
    if gb <= 1.0:
        return "good"
    if gb <= 16.0:
        return "medium"
    return "poor"


def class_color(cls: str) -> str:
    return {"good": GOOD, "medium": MEDIUM, "poor": POOR, "missing": "#F6F6F6"}.get(cls, "#F6F6F6")


def load_manifest() -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, int]]:
    manifest = pd.read_csv(CANON / "canonical_method_manifest.csv")
    full = manifest[
        manifest["benchmark_scope"].eq("full_26_method_benchmark")
        & (~manifest["is_variant"].astype(bool))
    ].sort_values("method_order")
    method_order = full["method_id"].tolist()
    family_map = full.set_index("method_id")["method_family"].map(FAMILY_RENAME).to_dict()
    order_map = full.set_index("method_id")["method_order"].to_dict()
    return full, method_order, family_map, order_map


def load_core_data(method_order: list[str], family_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completed_score_path = ROOT / "Publication/paper/revision_figures/figure3_polish/source_data/Figure_3_completed_score_matrix.csv"
    if completed_score_path.exists():
        scores = pd.read_csv(completed_score_path)
    else:
        scores = pd.read_csv(CANON / "original_score_matrix.csv")
    scores = scores[scores["method_id"].isin(method_order)].copy()
    scores["family_short"] = scores["method_id"].map(family_map)

    eff = pd.read_csv(CANON / "original_efficiency_scaling_long.csv")
    eff = eff[eff["method_id"].isin(method_order)].copy()
    eff["family_short"] = eff["method_id"].map(family_map)

    completed_stability_path = (
        ROOT / "Publication/paper/revision_figures/figure3_polish/source_data/Figure_3_completed_stability_scores_long.csv"
    )
    original_stability = pd.read_csv(CANON / "original_stability_score_long.csv")
    if completed_stability_path.exists():
        completed_stability = pd.read_csv(completed_stability_path).rename(columns={"Method": "method_id"})
        completed_stability["parent_method"] = completed_stability["method_id"]
        completed_keys = set(zip(completed_stability["method_id"], completed_stability["perturbation_axis"]))
        original_fallback = original_stability[
            ~original_stability.apply(lambda row: (row["method_id"], row["perturbation_axis"]) in completed_keys, axis=1)
        ].copy()
        stability = pd.concat(
            [
                completed_stability[["method_id", "parent_method", "perturbation_axis", "score"]],
                original_fallback[["method_id", "parent_method", "perturbation_axis", "score"]],
            ],
            ignore_index=True,
        )
    else:
        stability = original_stability
    stability = stability[stability["method_id"].isin(method_order)].copy()
    stability["family_short"] = stability["method_id"].map(family_map)

    fig9 = pd.read_csv(LEGACY_FIG9)
    return scores, eff, stability, fig9


def build_decision_table(scores: pd.DataFrame, eff: pd.DataFrame, stability: pd.DataFrame, family_map: dict[str, str]) -> pd.DataFrame:
    score_lookup = scores.set_index("method_id")
    eff_lookup = eff[eff["n_cells"].isin([5000, 20000, 50000])].set_index(["method_id", "n_cells"])

    stab_pivot = stability[
        stability["perturbation_axis"].isin(["dropout", "batch_number", "batch_strength"])
    ].pivot_table(index="method_id", columns="perturbation_axis", values="score", aggfunc="mean")
    if "batch_number" in stab_pivot.columns and "batch_strength" in stab_pivot.columns:
        stab_pivot["batch"] = stab_pivot[["batch_number", "batch_strength"]].mean(axis=1)
    elif "batch_number" in stab_pivot.columns:
        stab_pivot["batch"] = stab_pivot["batch_number"]
    else:
        stab_pivot["batch"] = np.nan

    rows = []
    row_index = 0
    for objective, methods in RECOMMENDATION_GROUPS:
        for method in methods:
            row = {
                "objective": objective,
                "method_id": method,
                "row_index": row_index,
                "family_short": family_map.get(method, "unknown"),
            }
            for metric in ["local", "global", "kmeans"]:
                value = float(score_lookup.loc[method, metric]) if method in score_lookup.index else np.nan
                row[metric] = value
                row[f"{metric}_symbol"] = score_symbol(value)
                row[f"{metric}_class"] = score_class(value)
            for n_cells in [5000, 20000, 50000]:
                if (method, n_cells) in eff_lookup.index:
                    runtime = float(eff_lookup.loc[(method, n_cells), "runtime_seconds"])
                    memory = float(eff_lookup.loc[(method, n_cells), "peak_memory_gb"])
                else:
                    runtime = np.nan
                    memory = np.nan
                row[f"runtime_{n_cells}"] = runtime
                row[f"runtime_{n_cells}_label"] = time_label(runtime)
                row[f"runtime_{n_cells}_class"] = runtime_class(runtime)
                row[f"memory_{n_cells}"] = memory
                row[f"memory_{n_cells}_label"] = memory_label(memory)
                row[f"memory_{n_cells}_class"] = memory_class(memory)
            for metric in ["dropout", "batch"]:
                value = float(stab_pivot.loc[method, metric]) if method in stab_pivot.index and metric in stab_pivot.columns else np.nan
                row[metric] = value
                row[f"{metric}_symbol"] = score_symbol(value)
                row[f"{metric}_class"] = score_class(value)
            rows.append(row)
            row_index += 1
    return pd.DataFrame(rows)


def build_recommendation_counts(
    decision: pd.DataFrame,
    method_order: list[str],
    family_map: dict[str, str],
    order_map: dict[str, int],
) -> pd.DataFrame:
    observed = (
        decision.groupby("method_id", as_index=False)
        .agg(appearances=("objective", "nunique"))
    )
    counts = pd.DataFrame({"method_id": method_order}).merge(observed, on="method_id", how="left")
    counts["appearances"] = counts["appearances"].fillna(0).astype(int)
    counts["family_short"] = counts["method_id"].map(family_map)
    counts["method_order"] = counts["method_id"].map(order_map)
    counts = counts.sort_values(["appearances", "method_order"], ascending=[False, True])
    return counts


def build_score_heatmap(scores: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    cols = ["local", "global", "kmeans", "runtime_score", "memory_score", "stability_median", "overall_mean"]
    heat = scores[scores["method_id"].isin(method_order)][["method_id", "family_short"] + cols].copy()
    heat["method_id"] = pd.Categorical(heat["method_id"], categories=method_order, ordered=True)
    heat = heat.sort_values("method_id")
    return heat


def build_50k_efficiency(eff: pd.DataFrame, family_map: dict[str, str], method_order: list[str]) -> pd.DataFrame:
    selected = eff[eff["n_cells"].eq(50000) & eff["method_id"].isin(method_order)].copy()
    selected["family_short"] = selected["method_id"].map(family_map)
    selected["method_id"] = pd.Categorical(selected["method_id"], categories=method_order, ordered=True)
    selected = selected.sort_values("method_id")
    return selected[["method_id", "family_short", "n_cells", "runtime_seconds", "peak_memory_gb"]]


def build_stability_all(stability: pd.DataFrame, family_map: dict[str, str], method_order: list[str]) -> pd.DataFrame:
    keep = stability[
        stability["method_id"].isin(method_order)
        & stability["perturbation_axis"].isin(["dropout", "batch_number", "batch_strength"])
    ].copy()
    keep["display_axis"] = keep["perturbation_axis"].replace({"batch_number": "batch", "batch_strength": "batch"})
    grouped = keep.groupby(["method_id", "display_axis"], as_index=False)["score"].mean()
    grouped["family_short"] = grouped["method_id"].map(family_map)
    grouped["method_id"] = pd.Categorical(grouped["method_id"], categories=method_order, ordered=True)
    grouped = grouped.sort_values("method_id")
    return grouped


def build_revision_control_matrices(family_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = ROOT / "Publication/paper/revision_figures/figure6_polish/source_data/Figure_6_targeted_sensitivity_controls_full_source_data.csv"
    fig6 = pd.read_csv(source_path)
    layer_order = ["dimension", "workflow", "hvg", "scvi"]
    layer_display = {
        "dimension": "Latent dim.",
        "workflow": "Workflow",
        "hvg": "Input genes",
        "scvi": "scVI ref.",
    }
    methods = sorted(fig6["method_id"].dropna().astype(str).unique())
    method_order = ["GLMPCA", "PCA", "pCMF", "scGBM", "SAUCIE", "UMAP", "t-SNE", "PHATE", "PaCMAP", "scVI"]
    methods = [m for m in method_order if m in methods] + [m for m in methods if m not in method_order]

    coverage_rows = []
    for method in methods:
        for layer in layer_order:
            sub = fig6[fig6["method_id"].astype(str).eq(method) & fig6["source"].eq(layer)]
            coverage_rows.append(
                {
                    "method_id": method,
                    "control_layer": layer,
                    "control_label": layer_display[layer],
                    "n_runs": int(len(sub)),
                    "n_datasets": int(sub["dataset_id"].nunique()) if len(sub) else 0,
                    "family_short": family_map.get(method, "control"),
                }
            )
    coverage = pd.DataFrame(coverage_rows)

    sens_rows = []

    dim = fig6[fig6["source"].eq("dimension")].copy()
    dim_med = dim.groupby(["method_id", "dataset_id", "dimension"], as_index=False)["ari"].median()
    for method, g_method in dim_med.groupby("method_id"):
        vals = []
        for _, g in g_method.groupby("dataset_id"):
            piv = g.set_index("dimension")["ari"]
            if 2 in piv.index and 50 in piv.index:
                vals.append(abs(float(piv.loc[50] - piv.loc[2])))
            else:
                vals.append(float(piv.max() - piv.min()))
        if vals:
            sens_rows.append(
                {
                    "method_id": str(method),
                    "control_layer": "dimension",
                    "control_label": layer_display["dimension"],
                    "sensitivity": float(np.median(vals)),
                    "effect_definition": "median absolute ARI change from 2D to 50D, fallback to within-dataset range",
                    "family_short": family_map.get(str(method), "control"),
                }
            )

    workflow = fig6[fig6["source"].eq("workflow")].copy()
    wf_med = workflow.groupby(["method_id", "dataset_id", "workflow"], as_index=False)["ari"].median()
    for method, g_method in wf_med.groupby("method_id"):
        vals = []
        for _, g in g_method.groupby("dataset_id"):
            piv = g.set_index("workflow")["ari"]
            if "PCA50 to 2D" in piv.index and "Direct 2D" in piv.index:
                vals.append(abs(float(piv.loc["PCA50 to 2D"] - piv.loc["Direct 2D"])))
        if vals:
            sens_rows.append(
                {
                    "method_id": str(method),
                    "control_layer": "workflow",
                    "control_label": layer_display["workflow"],
                    "sensitivity": float(np.median(vals)),
                    "effect_definition": "median absolute ARI change between PCA50-to-2D and direct-2D workflows",
                    "family_short": family_map.get(str(method), "control"),
                }
            )

    hvg = fig6[fig6["source"].eq("hvg")].copy()
    hvg_med = hvg.groupby(["method_id", "dataset_id", "hvg_requested"], as_index=False)["ari"].median()
    for method, g_method in hvg_med.groupby("method_id"):
        vals = []
        for _, g in g_method.groupby("dataset_id"):
            vals.append(float(g["ari"].quantile(0.75) - g["ari"].quantile(0.25)))
        if vals:
            sens_rows.append(
                {
                    "method_id": str(method),
                    "control_layer": "hvg",
                    "control_label": layer_display["hvg"],
                    "sensitivity": float(np.median(vals)),
                    "effect_definition": "median within-dataset ARI IQR across requested HVG cutoffs",
                    "family_short": family_map.get(str(method), "control"),
                }
            )

    scvi = fig6[fig6["source"].eq("scvi")].copy()
    scvi_med = scvi.groupby(["method_id", "dataset_id", "dimension"], as_index=False)["ari"].median()
    for method, g_method in scvi_med.groupby("method_id"):
        vals = []
        for _, g in g_method.groupby("dataset_id"):
            vals.append(float(g["ari"].max() - g["ari"].min()))
        if vals:
            sens_rows.append(
                {
                    "method_id": str(method),
                    "control_layer": "scvi",
                    "control_label": layer_display["scvi"],
                    "sensitivity": float(np.median(vals)),
                    "effect_definition": "median within-dataset ARI range across scVI latent dimensions",
                    "family_short": "control",
                }
            )

    sensitivity = pd.DataFrame(sens_rows)
    sensitivity["method_id"] = pd.Categorical(sensitivity["method_id"], categories=methods, ordered=True)
    sensitivity["control_layer"] = pd.Categorical(sensitivity["control_layer"], categories=layer_order, ordered=True)
    sensitivity = sensitivity.sort_values(["method_id", "control_layer"])
    coverage["method_id"] = pd.Categorical(coverage["method_id"], categories=methods, ordered=True)
    coverage["control_layer"] = pd.Categorical(coverage["control_layer"], categories=layer_order, ordered=True)
    coverage = coverage.sort_values(["method_id", "control_layer"])
    return coverage, sensitivity


def draw_decision_table(ax: plt.Axes, decision: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_header(ax, "a", "Task-specific method-selection guide", dx=-0.012, dy=1.015, title_dx=0.028)

    cols = [
        ("method", "Method"),
        ("local", "Local"),
        ("global", "Global"),
        ("kmeans", "Cluster"),
        ("runtime_5000", "5k"),
        ("runtime_20000", "20k"),
        ("runtime_50000", "50k"),
        ("memory_5000", "5k"),
        ("memory_20000", "20k"),
        ("memory_50000", "50k"),
        ("dropout", "Dropout"),
        ("batch", "Batch"),
    ]
    widths = np.array([0.132, 0.070, 0.070, 0.074, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.071, 0.071])
    widths = widths / widths.sum() * 0.875
    x0 = 0.116
    edges = np.r_[x0, x0 + np.cumsum(widths)]

    header_top = 0.955
    header_mid = 0.905
    header_bottom = 0.858
    y_bottom = 0.022
    n_rows = len(decision)
    row_h = (header_bottom - y_bottom) / n_rows

    group_defs = [
        ("Structure and clustering", 1, 3),
        ("Runtime", 4, 6),
        ("Memory", 7, 9),
        ("Stability", 10, 11),
    ]
    ax.add_patch(Rectangle((edges[0], header_bottom), edges[1] - edges[0], header_mid - header_bottom, facecolor=SUBHEADER, edgecolor=GRID, lw=0.45))
    ax.text((edges[0] + edges[1]) / 2, (header_bottom + header_mid) / 2, "Method", ha="center", va="center", fontsize=5.5)
    for title, start, end in group_defs:
        ax.add_patch(Rectangle((edges[start], header_mid), edges[end + 1] - edges[start], header_top - header_mid, facecolor=HEADER, edgecolor=GRID, lw=0.45))
        ax.text((edges[start] + edges[end + 1]) / 2, (header_mid + header_top) / 2, title, ha="center", va="center", fontsize=5.7)
    for i, (_, label) in enumerate(cols):
        if i == 0:
            continue
        ax.add_patch(Rectangle((edges[i], header_bottom), edges[i + 1] - edges[i], header_mid - header_bottom, facecolor=SUBHEADER, edgecolor=GRID, lw=0.45))
        ax.text((edges[i] + edges[i + 1]) / 2, (header_bottom + header_mid) / 2, label, ha="center", va="center", fontsize=5.2)

    objective_ranges = []
    start = 0
    for objective, group in decision.groupby("objective", sort=False):
        count = len(group)
        objective_ranges.append((objective, start, start + count - 1))
        start += count

    for ridx, row in decision.iterrows():
        y1 = header_bottom - ridx * row_h
        y0 = y1 - row_h
        ax.add_patch(Rectangle((edges[0], y0), edges[1] - edges[0], row_h, facecolor=ROW_GREY, edgecolor=GRID, lw=0.35))
        ax.text((edges[0] + edges[1]) / 2, (y0 + y1) / 2, display_method(row["method_id"]), ha="center", va="center", fontsize=5.0)

        for cidx, (key, _) in enumerate(cols[1:], start=1):
            if key in ["local", "global", "kmeans", "dropout", "batch"]:
                text = row[f"{key}_symbol"]
                face = class_color(row[f"{key}_class"])
            elif key.startswith("runtime"):
                text = row[f"{key}_label"]
                face = class_color(row[f"{key}_class"])
            elif key.startswith("memory"):
                text = row[f"{key}_label"]
                face = class_color(row[f"{key}_class"])
            else:
                text = ""
                face = "#F6F6F6"
            ax.add_patch(Rectangle((edges[cidx], y0), edges[cidx + 1] - edges[cidx], row_h, facecolor=face, edgecolor=GRID, lw=0.35))
            ax.text((edges[cidx] + edges[cidx + 1]) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=4.8, color="#111111")

    for objective, start, end in objective_ranges:
        y_top = header_bottom - start * row_h
        y_low = header_bottom - (end + 1) * row_h
        x_bracket = 0.103
        ax.plot([x_bracket, x_bracket], [y_low + 0.002, y_top - 0.002], color="#1F1F1F", lw=0.7, clip_on=False)
        ax.plot([x_bracket, x_bracket + 0.018], [(y_low + y_top) / 2, (y_low + y_top) / 2], color="#1F1F1F", lw=0.7, clip_on=False)
        ax.text(0.090, (y_low + y_top) / 2, objective, ha="right", va="center", fontsize=5.5)

    legend_y = -0.012
    legend_items = [("Good/Fast", GOOD), ("Medium", MEDIUM), ("Poor/Slow", POOR)]
    x = 0.405
    for label, color in legend_items:
        ax.add_patch(Rectangle((x, legend_y), 0.026, 0.018, facecolor=color, edgecolor="none", clip_on=False))
        ax.text(x + 0.030, legend_y + 0.008, label, va="center", ha="left", fontsize=5.2)
        x += 0.145


def draw_counts(ax: plt.Axes, counts: pd.DataFrame) -> None:
    panel_header(ax, "b", "Repeated recommendations", dx=-0.16, dy=1.028, title_dx=0.13)
    plot = counts[counts["appearances"].gt(0)].sort_values(["appearances", "method_order"], ascending=[True, False])
    y = np.arange(len(plot))
    colors = [FAMILY_COLORS[f] for f in plot["family_short"]]
    ax.barh(y, plot["appearances"], color=colors, height=0.58, edgecolor="white", lw=0.30)
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in plot["method_id"]], fontsize=5.0)
    ax.set_xlabel("objectives")
    ax.set_xlim(0, max(6, plot["appearances"].max() + 0.6))
    ax.xaxis.grid(True, color="#E8E8E8", lw=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)


def draw_score_heatmap(ax: plt.Axes, heat: pd.DataFrame) -> None:
    panel_header(ax, "c", "Benchmark scores", dx=-0.08, dy=1.028, title_dx=0.080)
    cols = ["local", "global", "kmeans", "runtime_score", "memory_score", "stability_median", "overall_mean"]
    labels = ["Local", "Global", "Cluster", "Runtime", "Memory", "Stability", "Overall"]
    data = heat[cols].to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=SCORE_CMAP, vmin=0.35, vmax=1.0, interpolation="nearest")
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels([display_method(m) for m in heat["method_id"]], fontsize=3.8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=4.4)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.015, 0.18, 0.026, 0.64])
    cb = plt.colorbar(im, cax=cax, ticks=[0.4, 0.7, 1.0])
    cb.ax.tick_params(labelsize=4.4, length=1.5, pad=1)
    cb.outline.set_linewidth(0.35)


def scatter_all_methods(
    ax: plt.Axes,
    scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    label: str,
    title: str,
    x_label: str,
    y_label: str,
    labels: list[str],
    label_offsets: dict[str, tuple[int, int]] | None = None,
) -> None:
    panel_header(ax, label, title, dx=-0.14, dy=1.032, title_dx=0.12)
    for fam in FAMILY_ORDER:
        sub = scores[scores["family_short"].eq(fam)]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=24 + 85 * sub["overall_mean"].clip(0, 1),
            color=FAMILY_COLORS[fam],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    offsets = {
        "SPDR": (7, 8),
        "TriMap": (7, -10),
        "UMAP": (7, 7),
        "scvis": (7, 8),
        "SQuaD-MDS": (7, 8),
        "PCA": (7, 7),
        "PHATE": (7, -10),
        "scGBM": (7, 8),
        "t-SNE": (7, -10),
        "GLMPCA": (7, 8),
    }
    if label_offsets:
        offsets.update(label_offsets)
    for _, row in scores[scores["method_id"].isin(labels)].iterrows():
        dx, dy = offsets.get(row["method_id"], (6, 6))
        ax.annotate(
            display_method(row["method_id"]),
            (row[x_col], row[y_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=4.6,
            color="#2A2A2A",
            arrowprops=dict(arrowstyle="-", color="#B9B9B9", lw=0.35, shrinkA=1.5, shrinkB=1.5),
            clip_on=False,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(scores[x_col].min() - 0.04, min(1.08, scores[x_col].max() + 0.08))
    ax.set_ylim(scores[y_col].min() - 0.04, min(1.06, scores[y_col].max() + 0.06))
    ax.grid(True, color="#E8E8E8", lw=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)


def draw_50k_footprint(ax: plt.Axes, footprint: pd.DataFrame) -> None:
    panel_header(ax, "g", "50k-cell footprint", dx=-0.14, dy=1.032, title_dx=0.12)
    for fam in FAMILY_ORDER:
        sub = footprint[footprint["family_short"].eq(fam)]
        ax.scatter(
            sub["runtime_seconds"],
            sub["peak_memory_gb"],
            s=42,
            color=FAMILY_COLORS[fam],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.88,
            zorder=3,
        )
    label_methods = {"PCA", "TriMap", "UMAP", "PHATE", "SPDR", "EDGE", "VASC"}
    offsets = {
        "PCA": (7, 8),
        "TriMap": (7, 9),
        "UMAP": (7, 8),
        "PHATE": (7, -10),
        "SPDR": (7, 8),
        "EDGE": (7, 8),
        "VASC": (7, -10),
        "SQuaD-MDS": (7, -10),
        "t-SNE": (7, 7),
        "GLMPCA": (7, 8),
        "scGBM": (7, -10),
    }
    for _, row in footprint[footprint["method_id"].isin(label_methods)].iterrows():
        dx, dy = offsets.get(row["method_id"], (6, 6))
        ax.annotate(
            display_method(row["method_id"]),
            (row["runtime_seconds"], row["peak_memory_gb"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=4.4,
            color="#2A2A2A",
            arrowprops=dict(arrowstyle="-", color="#B9B9B9", lw=0.35, shrinkA=1.5, shrinkB=1.5),
            clip_on=False,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime (s)")
    ax.set_ylabel("memory (GB)")
    ax.grid(True, color="#E8E8E8", lw=0.45, which="major")
    ax.set_axisbelow(True)
    clean_axis(ax)


def draw_stability(ax: plt.Axes, stability: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    panel_header(ax, "h", "Stability scores", dx=-0.14, dy=1.032, title_dx=0.12)
    pivot = stability.pivot_table(
        index=["method_id", "family_short"],
        columns="display_axis",
        values="score",
        observed=False,
    ).reset_index()
    pivot["mean_score"] = pivot[["dropout", "batch"]].mean(axis=1)
    pivot["method_id"] = pd.Categorical(pivot["method_id"].astype(str), categories=method_order, ordered=True)
    pivot = pivot.sort_values("method_id")
    data = pivot[["dropout", "batch"]].to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap=SCORE_CMAP, vmin=0.10, vmax=0.86, interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Dropout", "Batch"], fontsize=4.4)
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels([display_method(m) for m in pivot["method_id"]], fontsize=3.8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.015, 0.17, 0.026, 0.66])
    cb = plt.colorbar(im, cax=cax, ticks=[0.15, 0.50, 0.85])
    cb.ax.tick_params(labelsize=4.2, length=1.5, pad=1)
    cb.outline.set_linewidth(0.35)
    return pivot[["method_id", "family_short", "dropout", "batch", "mean_score"]]


def draw_control_layer_effects(ax: plt.Axes, sensitivity: pd.DataFrame) -> pd.DataFrame:
    panel_header(ax, "i", "Control-layer ARI sensitivity", dx=-0.14, dy=1.032, title_dx=0.12)
    plot = sensitivity.dropna(subset=["sensitivity"]).copy()
    layer_order = ["dimension", "workflow", "hvg", "scvi"]
    layer_labels = {
        "dimension": "Latent\ndim.",
        "workflow": "Workflow",
        "hvg": "Input\ngenes",
        "scvi": "scVI\nref.",
    }
    plot["control_layer"] = pd.Categorical(plot["control_layer"].astype(str), categories=layer_order, ordered=True)
    summary = (
        plot.groupby("control_layer", observed=False)
        .agg(
            median_sensitivity=("sensitivity", "median"),
            q25=("sensitivity", lambda x: x.quantile(0.25)),
            q75=("sensitivity", lambda x: x.quantile(0.75)),
            max_sensitivity=("sensitivity", "max"),
            n_methods=("method_id", "nunique"),
        )
        .reset_index()
    )
    summary["control_layer"] = pd.Categorical(summary["control_layer"].astype(str), categories=layer_order, ordered=True)
    summary = summary.sort_values("control_layer")
    y = np.arange(len(summary))
    ax.hlines(
        y,
        summary["q25"],
        summary["q75"],
        color="#B8B8B8",
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        summary["median_sensitivity"],
        y,
        s=30,
        color="#3F9C9A",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.scatter(
        summary["max_sensitivity"],
        y,
        s=18,
        color="#C79B38",
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    for yi, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            row["max_sensitivity"] + 0.018,
            yi,
            f"n={int(row['n_methods'])}",
            ha="left",
            va="center",
            fontsize=4.15,
            color="#4A4A4A",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([layer_labels.get(str(layer), str(layer)) for layer in summary["control_layer"]], fontsize=4.45)
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(0.82)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.55, summary["max_sensitivity"].max() * 1.23))
    ax.set_xlabel("absolute ARI sensitivity")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#B8B8B8", lw=1.2, label="IQR"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#3F9C9A", markeredgecolor="white", markersize=4.0, label="median"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#C79B38", markeredgecolor="white", markersize=3.6, label="max"),
        ],
        loc="lower right",
        fontsize=3.6,
        handlelength=1.0,
        borderaxespad=0.2,
        columnspacing=0.5,
    )
    clean_axis(ax)
    return summary


def draw_revision_sensitivity(ax: plt.Axes, sensitivity: pd.DataFrame) -> None:
    panel_header(ax, "j", "Sensitivity landscape", dx=-0.14, dy=1.032, title_dx=0.12)
    plot = sensitivity.dropna(subset=["sensitivity"]).copy()
    layer_order = ["dimension", "workflow", "hvg", "scvi"]
    layer_labels = {
        "dimension": "Latent dim.",
        "workflow": "Workflow",
        "hvg": "Input genes",
        "scvi": "scVI ref.",
    }
    y_lookup = {layer: idx for idx, layer in enumerate(layer_order)}
    plot["y"] = plot["control_layer"].astype(str).map(y_lookup)
    rng = np.random.default_rng(20260605)
    plot["yj"] = plot["y"] + rng.normal(0, 0.055, len(plot))

    for family in FAMILY_ORDER + ["control"]:
        sub = plot[plot["family_short"].astype(str).eq(family)]
        if sub.empty:
            continue
        ax.scatter(
            sub["sensitivity"],
            sub["yj"],
            s=34,
            color=FAMILY_COLORS.get(family, CONTROL_COLOR),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )

    label_rows = plot[
        (
            plot["method_id"].astype(str).eq("scVI")
            & plot["control_layer"].astype(str).eq("dimension")
        )
        | (
            plot["method_id"].astype(str).eq("UMAP")
            & plot["control_layer"].astype(str).eq("workflow")
        )
        | (
            plot["method_id"].astype(str).eq("t-SNE")
            & plot["control_layer"].astype(str).eq("hvg")
        )
    ]
    offsets = {
        "scVI": (7, 8),
        "UMAP": (7, 8),
        "t-SNE": (7, -8),
        "scGBM": (7, 8),
        "GLMPCA": (7, -8),
        "PCA": (7, 8),
    }
    for _, row in label_rows.iterrows():
        dx, dy = offsets.get(str(row["method_id"]), (6, 6))
        ax.annotate(
            "scVI*" if str(row["method_id"]) == "scVI" else display_method(row["method_id"]),
            (row["sensitivity"], row["yj"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.4,
            color="#2B2B2B",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.35, shrinkA=1.5, shrinkB=1.5),
            clip_on=False,
        )

    ax.set_yticks(np.arange(len(layer_order)))
    ax.set_yticklabels([layer_labels[layer] for layer in layer_order], fontsize=4.6)
    ax.tick_params(axis="y", pad=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("ARI sensitivity")
    ax.set_xlim(-0.02, max(0.76, plot["sensitivity"].max() * 1.15))
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)


def add_family_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", lw=1.2, markersize=4, label=f)
        for f in FAMILY_ORDER
    ]
    handles.append(Line2D([0], [0], color=CONTROL_COLOR, marker="o", lw=1.2, markersize=4, label="targeted control"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.012),
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
        fontsize=5.6,
        title="Method family",
        title_fontsize=5.8,
    )


def write_outputs(
    panel_tables: dict[str, pd.DataFrame],
    base: Path,
    method_order: list[str],
) -> None:
    for name, table in panel_tables.items():
        table.to_csv(SOURCE_OUT / f"Figure_9_panel_{name}_plotted_values.csv", index=False)

    with Image.open(base.with_suffix(".png")) as img:
        arr = np.asarray(img.convert("RGB"))
        pixel_std = float(arr.std())
        width, height = img.size

    decision = panel_tables["a"]
    recommended = sorted(decision["method_id"].unique())
    formal_set = set(method_order)
    qa = {
        "figure": base.with_suffix(".png").name,
        "panel_count": 10,
        "recommendation_objectives": int(decision["objective"].nunique()),
        "decision_table_rows": int(len(decision)),
        "unique_recommended_methods": int(decision["method_id"].nunique()),
        "all_recommendations_in_full_26": bool(set(recommended).issubset(formal_set)),
        "contains_squad_hybrid_variant": bool("SQuaD-MDS hybrid" in set(recommended)),
        "panel_b_methods": int(panel_tables["b"].query("appearances > 0")["method_id"].nunique()),
        "panel_c_methods": int(panel_tables["c"]["method_id"].nunique()),
        "tradeoff_methods_plotted": int(panel_tables["d"]["method_id"].nunique()),
        "panel_g_50k_completed_methods": int(panel_tables["g"]["method_id"].nunique()),
        "panel_h_methods": int(panel_tables["h"]["method_id"].nunique()),
        "control_effect_layers": int(panel_tables["i"]["control_layer"].nunique()),
        "max_control_layer_sensitivity": float(panel_tables["i"]["max_sensitivity"].max()),
        "revision_sensitivity_methods": int(panel_tables["j"]["method_id"].nunique()),
        "revision_sensitivity_layers": int(panel_tables["j"]["control_layer"].nunique()),
        "hvg_control_includes_scvi": bool(
            "scVI"
            in set(
                panel_tables["j"][
                    panel_tables["j"]["control_layer"].astype(str).isin(["hvg", "scvi"])
                ]["method_id"].astype(str)
            )
        ),
        "png_width_px": width,
        "png_height_px": height,
        "png_pixel_std": pixel_std,
        "nonblank_png": bool(pixel_std > 5),
    }
    pd.DataFrame([qa]).to_csv(QA_OUT / "Figure_9_visual_qa_summary.csv", index=False)

    checklist = f"""# Figure 9 Practical Method-Selection Guide QA

Generated by: `Publication/paper/revision_figures/figure9_polish/make_figure9_practical_selection_top_tier.py`

## Figure Contract

- Core claim: the benchmark supports task-specific, evidence-weighted method selection rather than a single universal winner.
- Original manuscript coverage: panel a preserves the original Figure 8 decision-guide logic while redrawing it from canonical source data.
- Full benchmark scope: recommendation panels use the formal 26-method benchmark; SQuaD-MDS hybrid is not used as a recommendation.
- Revision-control scope: targeted-control panels summarize scVI reference, latent-dimension, workflow, and input-gene controls rather than full 26-method sweeps; scVI is not counted as a full-benchmark method.
- 50k-cell efficiency scope: panel g plots only methods with valid 50k-cell efficiency records and does not impute missing 50k runs.

## QA

- Panels: {qa['panel_count']}.
- Recommendation objectives: {qa['recommendation_objectives']}; decision-table rows: {qa['decision_table_rows']}; unique recommended methods: {qa['unique_recommended_methods']}.
- All recommended methods are within the formal 26-method benchmark: {qa['all_recommendations_in_full_26']}.
- Contains SQuaD-MDS hybrid variant in recommendations: {qa['contains_squad_hybrid_variant']}.
- Panel b recommended methods shown: {qa['panel_b_methods']}; panel c score-matrix methods: {qa['panel_c_methods']}; trade-off panels plot methods: {qa['tradeoff_methods_plotted']}; panel h stability methods: {qa['panel_h_methods']}.
- Panel g 50k completed methods: {qa['panel_g_50k_completed_methods']}; control-effect layers: {qa['control_effect_layers']}; maximum control-layer sensitivity: {qa['max_control_layer_sensitivity']:.3f}; sensitivity methods: {qa['revision_sensitivity_methods']}; sensitivity layers: {qa['revision_sensitivity_layers']}.
- Targeted controls include scVI where appropriate: {qa['hvg_control_includes_scvi']}.
- PNG dimensions: {qa['png_width_px']} x {qa['png_height_px']} px; nonblank: {qa['nonblank_png']}.

## Panel Map

- a, task-specific decision table retaining the original Figure 8 structure.
- b, frequency with which recommended formal methods appear across analysis objectives; methods not recommended in panel a are omitted from this display.
- c, score heatmap for all 26 formal methods across local, global, clustering, efficiency, stability, and overall domains.
- d, local topology score versus runtime score for all 26 methods.
- e, global topology score versus memory score for all 26 methods.
- f, clustering score versus stability for all 26 methods.
- g, 50k-cell runtime-memory footprint for methods with valid 50k-cell records.
- h, dropout and batch stability score matrix for all 26 formal methods.
- i, control-layer ARI sensitivity summary showing median, interquartile range, maximum and method count for each targeted-control layer.
- j, observed sensitivity landscape for all targeted-control layers; scVI is a control only.
"""
    (QA_OUT / "Figure_9_polished_visual_qa_checklist.md").write_text(checklist, encoding="utf-8")


def main() -> None:
    _, method_order, family_map, order_map = load_manifest()
    scores, eff, stability, legacy_fig9 = load_core_data(method_order, family_map)

    decision = build_decision_table(scores, eff, stability, family_map)
    counts = build_recommendation_counts(decision, method_order, family_map, order_map)
    heat = build_score_heatmap(scores, method_order)
    footprint = build_50k_efficiency(eff, family_map, method_order)
    stability_selected = build_stability_all(stability, family_map, method_order)
    revision_coverage, revision_sensitivity = build_revision_control_matrices(family_map)

    fig = plt.figure(figsize=(9.68, 12.45))
    gs = fig.add_gridspec(
        4,
        8,
        height_ratios=[3.30, 2.10, 1.92, 2.10],
        hspace=0.42,
        wspace=0.92,
        left=0.064,
        right=0.982,
        bottom=0.074,
        top=0.976,
    )

    panel_tables: dict[str, pd.DataFrame] = {}
    ax_a = fig.add_subplot(gs[0, :])
    draw_decision_table(ax_a, decision)
    panel_tables["a"] = decision

    ax_b = fig.add_subplot(gs[1, 0:2])
    draw_counts(ax_b, counts)
    panel_tables["b"] = counts

    ax_c = fig.add_subplot(gs[1, 2:5])
    draw_score_heatmap(ax_c, heat)
    panel_tables["c"] = heat

    ax_d = fig.add_subplot(gs[1, 5:8])
    scatter_all_methods(
        ax_d,
        scores,
        "runtime_score",
        "local",
        "d",
        "Local score versus runtime",
        "runtime score",
        "local score",
        ["SPDR", "TriMap", "UMAP", "scvis"],
        {"SPDR": (9, 12), "UMAP": (8, 12), "TriMap": (8, -13), "scvis": (8, 11)},
    )
    panel_tables["d"] = scores[["method_id", "family_short", "runtime_score", "local", "overall_mean"]]

    ax_e = fig.add_subplot(gs[2, 0:2])
    scatter_all_methods(
        ax_e,
        scores,
        "memory_score",
        "global",
        "e",
        "Global score versus memory",
        "memory score",
        "global score",
        ["SQuaD-MDS", "PCA", "PHATE", "TriMap"],
        {"SQuaD-MDS": (-18, 6), "PCA": (9, -10), "PHATE": (9, -16), "TriMap": (-10, 12)},
    )
    panel_tables["e"] = scores[["method_id", "family_short", "memory_score", "global", "overall_mean"]]

    ax_f = fig.add_subplot(gs[2, 2:5])
    scatter_all_methods(
        ax_f,
        scores,
        "stability_median",
        "kmeans",
        "f",
        "Clustering score versus stability",
        "stability score",
        "k-means score",
        ["TriMap", "scGBM", "SPDR"],
        {"scGBM": (9, 13), "SPDR": (9, 10), "TriMap": (9, -13)},
    )
    panel_tables["f"] = scores[["method_id", "family_short", "stability_median", "kmeans", "overall_mean"]]

    ax_g = fig.add_subplot(gs[2, 5:8])
    draw_50k_footprint(ax_g, footprint)
    panel_tables["g"] = footprint

    ax_h = fig.add_subplot(gs[3, 0:3])
    panel_tables["h"] = draw_stability(ax_h, stability_selected, method_order)

    ax_i = fig.add_subplot(gs[3, 3:5])
    panel_tables["i"] = draw_control_layer_effects(ax_i, revision_sensitivity)

    ax_j = fig.add_subplot(gs[3, 5:8])
    draw_revision_sensitivity(ax_j, revision_sensitivity)
    panel_tables["j"] = revision_sensitivity

    add_family_legend(fig)

    base = PLOT_OUT / FIGURE_BASENAME
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_outputs(panel_tables, base, method_order)


if __name__ == "__main__":
    main()
