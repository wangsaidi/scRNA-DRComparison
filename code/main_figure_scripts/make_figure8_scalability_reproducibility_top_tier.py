from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure8_polish"
PLOT_OUT = OUT / "polished"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for path in (PLOT_OUT, SOURCE_OUT, QA_OUT):
    path.mkdir(parents=True, exist_ok=True)

SOURCE = ROOT / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_8_scalability_reproducibility_source_data.csv"
FINAL_CANON = (
    ROOT
    / "Publication/paper/submission_package_communications_biology_20260609/07_source_data_and_code_availability/github_release/source_data/main_figures/canonical_source_tables"
)
METHODS = FINAL_CANON / "canonical_method_manifest.csv"

FIGURE_BASENAME = "Figure_8_scalability_reproducibility_polished"
CELL_ORDER = [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 73233]
COMMON_CELL = 50000
HIGH_SCALE_CELL = max(CELL_ORDER)

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
DIRECT_LABEL_METHODS = ["SAUCIE", "PCA", "UMAP", "TriMap", "SIMLR", "SCDRHA", "EDGE", "SPDR", "VASC"]
STATUS_COLORS = {
    "installed_verified": "#9EBBD5",
    "Source import verified": "#CFCFCF",
}
ENV_COLOR = "#C79B38"

COMPLETION_CMAP = LinearSegmentedColormap.from_list("completion", ["#F1F1F1", "#3F9C9A"])
SCORE_CMAP = LinearSegmentedColormap.from_list("efficiency_score", ["#F5F1EA", "#D8E5E2", "#7EB8B2", "#226A68"])
BURDEN_CMAP = LinearSegmentedColormap.from_list("burden", ["#F6F6F6", "#D9D1C1", "#B87F56", "#743A2F"])


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.8,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.1,
        "ytick.labelsize": 5.1,
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
    }.get(str(name), str(name))


def cell_label(value: float | int) -> str:
    value = int(value)
    if value == 73233:
        return "73k"
    if value >= 1000:
        return f"{int(round(value / 1000))}k"
    return str(value)


def time_label(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def memory_label(gb: float) -> str:
    if pd.isna(gb):
        return ""
    if gb < 1:
        return f"{gb * 1024:.0f}M"
    return f"{gb:.1f}G"


def panel_header(ax: plt.Axes, label: str, title: str, dx: float = -0.10, dy: float = 1.045, title_dx: float = 0.095) -> None:
    ax.text(dx, dy, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0, fontweight="bold", color="#171717")
    ax.text(dx + title_dx, dy, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color="#171717")


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    raw = pd.read_csv(SOURCE)
    manifest = pd.read_csv(METHODS)
    full = manifest[
        manifest["benchmark_scope"].eq("full_26_method_benchmark")
        & (~manifest["is_variant"].astype(bool))
    ].sort_values("method_order")
    method_order = full["method_id"].tolist()
    family_map = full.set_index("method_id")["method_family"].map(FAMILY_RENAME).to_dict()
    order_map = full.set_index("method_id")["method_order"].to_dict()

    eff = raw[raw["source"].eq("efficiency")].copy()
    eff = eff[eff["method_id"].isin(method_order)].copy()
    eff["method_id"] = pd.Categorical(eff["method_id"], categories=method_order, ordered=True)
    eff["family_short"] = eff["method_id"].astype(str).map(family_map)
    eff["method_order"] = eff["method_id"].astype(str).map(order_map)
    eff["n_cells"] = eff["n_cells"].astype(int)

    install = raw[raw["source"].eq("install_manifest")].copy()
    install["method_id"] = install["method"].fillna(install["method_id"])
    install = install[install["method_id"].isin(method_order)].copy()
    install["method_id"] = pd.Categorical(install["method_id"], categories=method_order, ordered=True)
    install["family_short"] = install["method_id"].astype(str).map(family_map)

    commits = raw[raw["source"].eq("source_commits")].copy()
    return eff, install, commits, method_order


def draw_scaling(ax: plt.Axes, eff: pd.DataFrame, metric: str, label: str, title: str, y_label: str) -> pd.DataFrame:
    selected = set(DIRECT_LABEL_METHODS)
    label_offsets = {
        "runtime_seconds": {
            "SAUCIE": (7, -6),
            "PCA": (7, 0),
            "UMAP": (7, 7),
            "TriMap": (7, -8),
            "SIMLR": (7, 9),
            "SCDRHA": (7, -8),
            "EDGE": (7, -12),
            "SPDR": (7, 0),
            "VASC": (7, 12),
        },
        "peak_memory_gb": {
            "SAUCIE": (7, -22),
            "PCA": (7, -12),
            "UMAP": (7, 14),
            "TriMap": (7, -2),
            "SIMLR": (7, -2),
            "SCDRHA": (7, 9),
            "EDGE": (7, -16),
            "SPDR": (7, 25),
            "VASC": (7, 5),
        },
    }
    records = []
    terminal_rows = []
    for method, group in eff.groupby("method_id", observed=True):
        method = str(method)
        group = group.sort_values("n_cells")
        fam = str(group["family_short"].iloc[0])
        alpha = 0.84 if method in selected else 0.32
        lw = 1.45 if method in selected else 0.75
        ax.plot(
            group["n_cells"],
            group[metric],
            color=FAMILY_COLORS.get(fam, "#777777"),
            linewidth=lw,
            alpha=alpha,
            marker="o" if method in selected else None,
            markersize=2.1 if method in selected else 0,
            markeredgecolor="white",
            markeredgewidth=0.25,
        )
        if method in selected:
            terminal_rows.append(group.tail(1).assign(method_id=method))
        records.append(group[["method_id", "family_short", "n_cells", metric]].copy())

    for terminal in terminal_rows:
        row = terminal.iloc[0]
        method = str(row["method_id"])
        fam = str(row["family_short"])
        dx, dy = label_offsets.get(metric, {}).get(method, (7, 0))
        ax.annotate(
            display_method(method),
            (row["n_cells"], row[metric]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=4.8,
            color=FAMILY_COLORS.get(fam, "#333333"),
            arrowprops=dict(arrowstyle="-", color="#B6B6B6", lw=0.32, shrinkA=1.5, shrinkB=1.5),
            clip_on=False,
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 140000)
    ax.set_xticks([100, 1000, 10000, 73233])
    ax.set_xticklabels(["100", "1k", "10k", "73k"])
    ax.set_xlabel("cells")
    ax.set_ylabel(y_label)
    ax.grid(True, color="#E8E8E8", linewidth=0.45, which="major")
    ax.set_axisbelow(True)
    panel_header(ax, label, title, dx=-0.12, dy=1.045, title_dx=0.11)
    clean_axis(ax)
    return pd.concat(records, ignore_index=True)


def draw_completion_matrix(fig: plt.Figure, ax: plt.Axes, eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    completed = (
        eff.assign(observed=1)
        .pivot_table(index="method_id", columns="n_cells", values="observed", aggfunc="max", observed=False)
        .reindex(index=method_order, columns=CELL_ORDER)
        .fillna(0)
    )
    im = ax.imshow(completed.to_numpy(), aspect="auto", cmap=COMPLETION_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels([display_method(m) for m in method_order], fontsize=4.8)
    ax.set_xticks(np.arange(len(CELL_ORDER)))
    ax.set_xticklabels([cell_label(c) for c in CELL_ORDER], rotation=45, ha="right")
    ax.tick_params(length=0)
    for x in np.arange(0.5, len(CELL_ORDER), 1):
        ax.axvline(x, color="white", linewidth=0.35)
    for y in np.arange(0.5, len(method_order), 1):
        ax.axhline(y, color="white", linewidth=0.35)
    panel_header(ax, "c", "Run-completion audit", dx=-0.08, dy=1.035, title_dx=0.075)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.012, 0.23, 0.018, 0.54])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 1])
    cb.ax.set_yticklabels(["no", "yes"])
    cb.ax.tick_params(labelsize=4.7, length=1.5, pad=1)
    cb.outline.set_linewidth(0.35)
    cax.set_title("run", fontsize=4.9, pad=2)
    return completed.reset_index()


def draw_endpoint_scatter(ax: plt.Axes, eff: pd.DataFrame) -> pd.DataFrame:
    endpoint = eff.sort_values("n_cells").groupby("method_id", observed=True).tail(1).copy()
    endpoint["size"] = np.interp(endpoint["n_cells"], [min(CELL_ORDER), max(CELL_ORDER)], [18, 58])
    for fam in FAMILY_ORDER:
        subset = endpoint[endpoint["family_short"].eq(fam)]
        ax.scatter(
            subset["runtime_seconds"],
            subset["peak_memory_gb"],
            s=subset["size"],
            color=FAMILY_COLORS[fam],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )
    labels = DIRECT_LABEL_METHODS
    offsets = {
        "PCA": (9, 9),
        "TriMap": (10, 8),
        "UMAP": (11, 15),
        "SAUCIE": (10, 10),
        "SIMLR": (9, -13),
        "EDGE": (10, 10),
        "SCDRHA": (11, -12),
        "VASC": (10, 8),
        "SPDR": (10, -12),
    }
    for _, row in endpoint[endpoint["method_id"].astype(str).isin(labels)].iterrows():
        dx, dy = offsets.get(str(row["method_id"]), (6, 6))
        ax.annotate(
            display_method(row["method_id"]),
            (row["runtime_seconds"], row["peak_memory_gb"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.9,
            color="#2B2B2B",
            arrowprops=dict(arrowstyle="-", color="#B8B8B8", lw=0.38, shrinkA=1.5, shrinkB=1.5),
            clip_on=False,
        )
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#E4E4E4",
            markeredgecolor="#777777",
            markeredgewidth=0.35,
            markersize=np.sqrt(np.interp(cells, [min(CELL_ORDER), max(CELL_ORDER)], [18, 58])),
            label=cell_label(cells),
        )
        for cells in [5000, 30000, 73233]
    ]
    ax.legend(
        handles=size_handles,
        title="endpoint",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.45,
        labelspacing=0.28,
        fontsize=4.8,
        title_fontsize=5.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.0, endpoint["runtime_seconds"].max() * 2.2)
    ax.set_ylim(0.75, endpoint["peak_memory_gb"].max() * 1.75)
    ax.set_xlabel("runtime at largest completed scale (s)")
    ax.set_ylabel("peak memory (GB)")
    ax.grid(True, color="#E8E8E8", linewidth=0.45, which="major")
    ax.set_axisbelow(True)
    panel_header(ax, "d", "Largest available endpoint", dx=-0.12, dy=1.045, title_dx=0.11)
    clean_axis(ax)
    return endpoint[["method_id", "family_short", "n_cells", "runtime_seconds", "peak_memory_gb"]]


def draw_incomplete_runs(ax: plt.Axes, eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    completed = (
        eff.assign(observed=1)
        .pivot_table(index="method_id", columns="n_cells", values="observed", aggfunc="max", observed=False)
        .reindex(index=method_order, columns=CELL_ORDER)
        .fillna(0)
    )
    family_lookup = eff.drop_duplicates("method_id").set_index("method_id")["family_short"].to_dict()
    summary = completed.copy()
    summary["completed_levels"] = summary[CELL_ORDER].sum(axis=1).astype(int)
    summary["missing_levels"] = len(CELL_ORDER) - summary["completed_levels"]
    summary["largest_completed_cells"] = [
        max([cell for cell in CELL_ORDER if row[cell] == 1], default=np.nan)
        for _, row in summary.iterrows()
    ]
    summary = summary.reset_index()
    summary["family_short"] = summary["method_id"].astype(str).map(family_lookup)
    plot = summary[summary["missing_levels"].gt(0)].copy()
    plot["method_id"] = pd.Categorical(plot["method_id"], categories=method_order, ordered=True)
    plot = plot.sort_values("method_id")

    y = np.arange(len(plot))
    colors = [FAMILY_COLORS.get(f, "#777777") for f in plot["family_short"]]
    ax.barh(y, plot["missing_levels"], color=colors, height=0.62, edgecolor="white", linewidth=0.35)
    for yi, (_, row) in enumerate(plot.iterrows()):
        ax.text(
            row["missing_levels"] + 0.09,
            yi,
            cell_label(row["largest_completed_cells"]),
            va="center",
            ha="left",
            fontsize=4.8,
            color="#333333",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in plot["method_id"]], fontsize=4.9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(5.6, plot["missing_levels"].max() + 0.9))
    ax.set_xlabel("missing scale levels")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "e", "Incomplete runs by method", dx=-0.15, dy=1.055, title_dx=0.13)
    clean_axis(ax)
    return summary[
        ["method_id", "family_short", "completed_levels", "missing_levels", "largest_completed_cells"]
    ]


def compute_completion_audit(eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    completed = (
        eff.assign(observed=1)
        .pivot_table(index="method_id", columns="n_cells", values="observed", aggfunc="max", observed=False)
        .reindex(index=method_order, columns=CELL_ORDER)
        .fillna(0)
    )
    family_lookup = eff.drop_duplicates("method_id").set_index("method_id")["family_short"].to_dict()
    audit = completed.copy()
    audit["completed_levels"] = audit[CELL_ORDER].sum(axis=1).astype(int)
    audit["missing_levels"] = len(CELL_ORDER) - audit["completed_levels"]
    audit["completion_fraction"] = audit["completed_levels"] / len(CELL_ORDER)
    audit["largest_completed_cells"] = [
        max([cell for cell in CELL_ORDER if row[cell] == 1], default=np.nan)
        for _, row in audit.iterrows()
    ]
    audit["missing_cell_levels"] = [
        ";".join([cell_label(cell) for cell in CELL_ORDER if row[cell] == 0])
        for _, row in audit.iterrows()
    ]
    audit = audit.reset_index()
    audit["family_short"] = audit["method_id"].astype(str).map(family_lookup)
    audit["method_id"] = pd.Categorical(audit["method_id"], categories=method_order, ordered=True)
    return audit.sort_values("method_id")


def compute_metric_score_by_scale(eff: pd.DataFrame, method_order: list[str], metric: str) -> pd.DataFrame:
    rows = []
    family_lookup = eff.drop_duplicates("method_id").set_index("method_id")["family_short"].to_dict()
    for cell in CELL_ORDER:
        sub = eff[eff["n_cells"].eq(cell)].copy()
        values = sub.set_index("method_id")[metric]
        fams = sub.set_index("method_id")["family_short"]
        ranks = values.rank(ascending=True, method="average")
        n_observed = int(values.notna().sum())
        for method in method_order:
            raw_value = values.get(method, np.nan)
            rank_value = ranks.get(method, np.nan)
            if pd.isna(raw_value):
                score = np.nan
            elif n_observed <= 1:
                score = 1.0
            else:
                score = 1.0 - (rank_value - 1.0) / (n_observed - 1.0)
            rows.append(
                {
                    "method_id": method,
                    "family_short": fams.get(method, family_lookup.get(method)),
                    "n_cells": cell,
                    metric: raw_value,
                    "within_scale_rank": rank_value,
                    "direction_score": score,
                    "completed": int(not pd.isna(raw_value)),
                }
            )
    return pd.DataFrame(rows)


def compute_scaling_exponents(eff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in eff.groupby("method_id", observed=True):
        group = group.sort_values("n_cells")
        fam = str(group["family_short"].iloc[0])
        for metric, display in [("runtime_seconds", "runtime"), ("peak_memory_gb", "memory")]:
            valid = group[(group[metric] > 0) & (group["n_cells"] > 0)]
            if len(valid) < 3:
                slope = np.nan
            else:
                slope = float(np.polyfit(np.log10(valid["n_cells"]), np.log10(valid[metric]), 1)[0])
            rows.append({"method_id": str(method), "family_short": fam, "metric": display, "scaling_exponent": slope})
    return pd.DataFrame(rows)


def compute_completion_adjusted_summary(eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    records = []
    for cell in CELL_ORDER:
        sub = eff[eff["n_cells"].eq(cell)].copy()
        if sub.empty:
            continue
        rt_min, rt_max = sub["runtime_seconds"].min(), sub["runtime_seconds"].max()
        mem_min, mem_max = sub["peak_memory_gb"].min(), sub["peak_memory_gb"].max()
        sub["runtime_direction_score"] = 1.0 if rt_max == rt_min else (rt_max - sub["runtime_seconds"]) / (rt_max - rt_min)
        sub["memory_direction_score"] = 1.0 if mem_max == mem_min else (mem_max - sub["peak_memory_gb"]) / (mem_max - mem_min)
        sub["cell_efficiency_score"] = sub[["runtime_direction_score", "memory_direction_score"]].mean(axis=1)
        records.append(sub[["method_id", "family_short", "n_cells", "runtime_direction_score", "memory_direction_score", "cell_efficiency_score"]])
    scored = pd.concat(records, ignore_index=True)
    grid = pd.MultiIndex.from_product([method_order, CELL_ORDER], names=["method_id", "n_cells"]).to_frame(index=False)
    scored = grid.merge(scored, on=["method_id", "n_cells"], how="left")
    scored["completed"] = scored["cell_efficiency_score"].notna().astype(int)
    scored["cell_efficiency_score"] = scored["cell_efficiency_score"].fillna(0.0)
    family_lookup = eff.drop_duplicates("method_id").set_index("method_id")["family_short"].to_dict()
    scored["family_short"] = scored["family_short"].fillna(scored["method_id"].map(family_lookup))
    summary = (
        scored.groupby(["method_id", "family_short"], observed=True)
        .agg(
            completion_adjusted_score=("cell_efficiency_score", "mean"),
            completed_levels=("completed", "sum"),
            mean_runtime_score=("runtime_direction_score", "mean"),
            mean_memory_score=("memory_direction_score", "mean"),
        )
        .reset_index()
    )
    summary["missing_levels"] = len(CELL_ORDER) - summary["completed_levels"]
    summary["method_id"] = pd.Categorical(summary["method_id"], categories=method_order, ordered=True)
    return summary


def normalize_higher_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values
    span = valid.max() - valid.min()
    if span == 0:
        return values.where(values.isna(), 1.0)
    return (values - valid.min()) / span


def normalize_lower_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values
    span = valid.max() - valid.min()
    if span == 0:
        return values.where(values.isna(), 1.0)
    return (valid.max() - values) / span


def draw_efficiency_score_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    eff: pd.DataFrame,
    method_order: list[str],
    metric: str,
    label: str,
    title: str,
    cbar_label: str,
    show_ylabels: bool = True,
    show_ylabel: bool = True,
) -> pd.DataFrame:
    table = compute_metric_score_by_scale(eff, method_order, metric)
    matrix = (
        table.pivot(index="method_id", columns="n_cells", values="direction_score")
        .reindex(index=method_order, columns=CELL_ORDER)
        .to_numpy()
    )
    masked = np.ma.masked_invalid(matrix)
    cmap = SCORE_CMAP.copy()
    cmap.set_bad("#EFEFEF")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(CELL_ORDER)))
    ax.set_xticklabels([cell_label(c) for c in CELL_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    if show_ylabels:
        ax.set_yticklabels([display_method(m) for m in method_order], fontsize=4.55)
        if show_ylabel:
            ax.set_ylabel("methods")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("cells")
    ax.tick_params(length=0)
    for x in np.arange(0.5, len(CELL_ORDER), 1):
        ax.axvline(x, color="white", linewidth=0.35)
    for y in np.arange(0.5, len(method_order), 1):
        ax.axhline(y, color="white", linewidth=0.35)
    cax = ax.inset_axes([1.015, 0.22, 0.018, 0.56])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=4.5, length=1.4, pad=1)
    cb.outline.set_linewidth(0.35)
    cax.set_title(cbar_label, fontsize=4.5, pad=2)
    panel_header(ax, label, title, dx=-0.13 if show_ylabels else -0.08, dy=1.035, title_dx=0.11 if show_ylabels else 0.075)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return table


def draw_scaling_exponents(ax: plt.Axes, eff: pd.DataFrame) -> pd.DataFrame:
    exponents = compute_scaling_exponents(eff)
    x_lookup = {"runtime": 0, "memory": 1}
    rng = np.random.default_rng(20260605)
    for metric in ["runtime", "memory"]:
        vals = exponents[exponents["metric"].eq(metric)].copy()
        x0 = x_lookup[metric]
        for fam in FAMILY_ORDER:
            sub = vals[vals["family_short"].eq(fam)].dropna(subset=["scaling_exponent"])
            jitter = rng.normal(0, 0.045, len(sub))
            ax.scatter(
                np.full(len(sub), x0) + jitter,
                sub["scaling_exponent"],
                s=16,
                color=FAMILY_COLORS[fam],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.86,
            )
        median = vals["scaling_exponent"].median(skipna=True)
        ax.plot([x0 - 0.23, x0 + 0.23], [median, median], color="#222222", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["runtime", "memory"])
    ax.set_ylabel("log-log slope")
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "e", "Scaling exponents", dx=-0.15, dy=1.055, title_dx=0.13)
    clean_axis(ax)
    return exponents


def draw_common_scale_lollipop(ax: plt.Axes, eff: pd.DataFrame, method_order: list[str], metric: str, label: str, title: str, x_label: str) -> pd.DataFrame:
    subset = eff[eff["n_cells"].eq(COMMON_CELL)].copy()
    table = pd.DataFrame({"method_id": method_order}).merge(
        subset[["method_id", "family_short", metric]],
        on="method_id",
        how="left",
    )
    table["method_id"] = pd.Categorical(table["method_id"], categories=method_order, ordered=True)
    y = np.arange(len(method_order))
    present = table[metric].notna()
    xmin = max(table[metric].min(skipna=True) * 0.75, 0.01)
    xmax = table[metric].max(skipna=True) * 1.35
    for idx, row in table.iterrows():
        if pd.isna(row[metric]):
            ax.scatter(xmin, idx, marker="x", s=12, color="#B5B5B5", linewidth=0.8, zorder=3)
            continue
        ax.plot([xmin, row[metric]], [idx, idx], color="#D7D7D7", linewidth=0.75, zorder=1)
        ax.scatter(row[metric], idx, s=16, color=FAMILY_COLORS.get(row["family_short"], "#777777"), edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in method_order], fontsize=4.6)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.grid(True, axis="x", color="#E8E8E8", linewidth=0.45, which="major")
    ax.set_axisbelow(True)
    panel_header(ax, label, title, dx=-0.12, dy=1.045, title_dx=0.11)
    clean_axis(ax)
    table["common_cell"] = COMMON_CELL
    return table


def draw_completion_adjusted_efficiency(ax: plt.Axes, eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    summary = compute_completion_adjusted_summary(eff, method_order)
    summary = summary.sort_values(["completion_adjusted_score", "completed_levels"], ascending=[False, False])

    for family in FAMILY_ORDER:
        sub = summary[summary["family_short"].astype(str).eq(family)]
        if sub.empty:
            continue
        ax.scatter(
            sub["completion_adjusted_score"],
            sub["completed_levels"],
            s=36 + 10 * (len(CELL_ORDER) - sub["missing_levels"]),
            color=FAMILY_COLORS.get(family, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
    label_methods = {"PCA"} | set(summary[summary["missing_levels"].gt(0)]["method_id"].astype(str))
    offsets = {
        "PCA": (-24, 8),
        "SIMLR": (5, 8),
        "SCDRHA": (-34, 18),
        "SSNMDI": (-36, -18),
        "scGAE": (18, 18),
        "DREAM": (18, -18),
    }
    for _, row in summary.iterrows():
        method = str(row["method_id"])
        if method not in label_methods:
            continue
        dx, dy = offsets.get(method, (5, 5))
        ax.annotate(
            display_method(method),
            (row["completion_adjusted_score"], row["completed_levels"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.35,
            color=FAMILY_COLORS.get(row["family_short"], "#4A4A4A"),
            arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.35, shrinkA=1, shrinkB=1),
        )
    ax.set_xlim(0.20, 1.04)
    ax.set_ylim(4.4, 10.6)
    ax.set_yticks([5, 8, 10])
    ax.set_xlabel("completion-adjusted efficiency")
    ax.set_ylabel("completed scale levels")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "f", "Completion-adjusted efficiency", dx=-0.08, dy=1.045, title_dx=0.075)
    clean_axis(ax)
    return summary


def draw_completion_fraction(ax: plt.Axes, eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    completed = (
        eff.assign(observed=1)
        .pivot_table(index="method_id", columns="n_cells", values="observed", aggfunc="max", observed=False)
        .reindex(index=method_order, columns=CELL_ORDER)
        .fillna(0)
    )
    family_lookup = eff.drop_duplicates("method_id").set_index("method_id")["family_short"].to_dict()
    summary = completed.copy()
    summary["completed_levels"] = summary[CELL_ORDER].sum(axis=1).astype(int)
    summary["completion_fraction"] = summary["completed_levels"] / len(CELL_ORDER)
    summary["largest_completed_cells"] = [
        max([cell for cell in CELL_ORDER if row[cell] == 1], default=np.nan)
        for _, row in summary.iterrows()
    ]
    summary = summary.reset_index()
    summary["family_short"] = summary["method_id"].astype(str).map(family_lookup)
    summary["method_id"] = pd.Categorical(summary["method_id"], categories=method_order, ordered=True)
    summary = summary.sort_values("method_id")
    y = np.arange(len(summary))
    colors = [FAMILY_COLORS.get(f, "#777777") for f in summary["family_short"]]
    ax.hlines(y, 0, summary["completion_fraction"], color=colors, lw=2.0, alpha=0.74)
    ax.scatter(summary["completion_fraction"], y, s=18, color=colors, edgecolor="white", linewidth=0.35, zorder=3)
    for yi, (_, row) in enumerate(summary.iterrows()):
        if row["completed_levels"] < len(CELL_ORDER):
            ax.text(
                row["completion_fraction"] + 0.018,
                yi,
                cell_label(row["largest_completed_cells"]),
                va="center",
                ha="left",
                fontsize=4.2,
                color="#555555",
            )
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in summary["method_id"]], fontsize=4.45)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("fraction of cell-scale levels completed")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "i", "Completion fraction", dx=-0.12, dy=1.045, title_dx=0.11)
    clean_axis(ax)
    return summary[["method_id", "family_short", "completed_levels", "completion_fraction", "largest_completed_cells"]]


def draw_family_scalability_burden(ax: plt.Axes, eff: pd.DataFrame) -> pd.DataFrame:
    exponents = compute_scaling_exponents(eff)
    burden = (
        exponents.pivot_table(index=["method_id", "family_short"], columns="metric", values="scaling_exponent", observed=True)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    burden["joint_scaling_burden"] = burden[["runtime", "memory"]].mean(axis=1)
    family_positions = {fam: idx for idx, fam in enumerate(FAMILY_ORDER)}
    rng = np.random.default_rng(20260616)
    for family in FAMILY_ORDER:
        sub = burden[burden["family_short"].eq(family)].dropna(subset=["joint_scaling_burden"])
        if sub.empty:
            continue
        x0 = family_positions[family]
        jitter = rng.normal(0, 0.045, len(sub))
        q1, q2, q3 = sub["joint_scaling_burden"].quantile([0.25, 0.50, 0.75])
        ax.plot([x0, x0], [q1, q3], color="#333333", linewidth=1.0, alpha=0.78, zorder=2)
        ax.plot([x0 - 0.18, x0 + 0.18], [q2, q2], color="#111111", linewidth=1.0, zorder=3)
        ax.scatter(
            np.full(len(sub), x0) + jitter,
            sub["joint_scaling_burden"],
            s=18,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=4,
        )
    ax.set_xticks(list(family_positions.values()))
    ax.set_xticklabels(FAMILY_ORDER, rotation=22, ha="right")
    ax.set_ylabel("mean log-log slope")
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "h", "Family scalability burden", dx=-0.16, dy=1.06, title_dx=0.14)
    clean_axis(ax)
    return burden[["method_id", "family_short", "runtime", "memory", "joint_scaling_burden"]]


def draw_high_scale_pareto_frontier(ax: plt.Axes, eff: pd.DataFrame) -> pd.DataFrame:
    common = eff[eff["n_cells"].eq(HIGH_SCALE_CELL)].copy()
    common = common.dropna(subset=["runtime_seconds", "peak_memory_gb"]).sort_values("runtime_seconds")
    pareto_flags = []
    for _, row in common.iterrows():
        dominates = common[
            (common["runtime_seconds"] <= row["runtime_seconds"])
            & (common["peak_memory_gb"] <= row["peak_memory_gb"])
            & (
                (common["runtime_seconds"] < row["runtime_seconds"])
                | (common["peak_memory_gb"] < row["peak_memory_gb"])
            )
        ]
        pareto_flags.append(dominates.empty)
    common["pareto_efficient"] = pareto_flags
    for fam in FAMILY_ORDER:
        subset = common[common["family_short"].eq(fam)]
        non_pareto = subset[~subset["pareto_efficient"]]
        pareto = subset[subset["pareto_efficient"]]
        ax.scatter(
            non_pareto["runtime_seconds"],
            non_pareto["peak_memory_gb"],
            s=22,
            color=FAMILY_COLORS[fam],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.70,
            zorder=2,
        )
        ax.scatter(
            pareto["runtime_seconds"],
            pareto["peak_memory_gb"],
            s=44,
            color=FAMILY_COLORS[fam],
            edgecolor="#111111",
            linewidth=0.55,
            alpha=0.96,
            zorder=4,
        )
    frontier = common[common["pareto_efficient"]].sort_values("runtime_seconds")
    if len(frontier) >= 2:
        ax.plot(frontier["runtime_seconds"], frontier["peak_memory_gb"], color="#222222", lw=0.75, ls=(0, (3, 2)), zorder=3)
    label_rows = pd.concat(
        [
            frontier,
            common.sort_values("runtime_seconds").head(2),
            common.sort_values("peak_memory_gb").head(2),
        ],
        ignore_index=True,
    ).drop_duplicates("method_id")
    for _, row in label_rows.iterrows():
        ax.text(
            row["runtime_seconds"] * 1.08,
            row["peak_memory_gb"],
            display_method(row["method_id"]),
            fontsize=4.5,
            va="center",
            ha="left",
            color="#2B2B2B",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"runtime at {cell_label(HIGH_SCALE_CELL)} cells (s)")
    ax.set_ylabel(f"peak memory at {cell_label(HIGH_SCALE_CELL)} cells (GB)")
    ax.grid(True, color="#E8E8E8", linewidth=0.45, which="major")
    ax.set_axisbelow(True)
    panel_header(ax, "i", f"{cell_label(HIGH_SCALE_CELL)}-cell resource frontier", dx=-0.12, dy=1.045, title_dx=0.11)
    clean_axis(ax)
    return common[["method_id", "family_short", "n_cells", "runtime_seconds", "peak_memory_gb", "pareto_efficient"]]


def compute_practical_profile(eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    summary = compute_completion_adjusted_summary(eff, method_order)
    exponents = (
        compute_scaling_exponents(eff)
        .pivot_table(index=["method_id", "family_short"], columns="metric", values="scaling_exponent", observed=True)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    endpoint = (
        eff.sort_values("n_cells")
        .groupby("method_id", observed=True)
        .tail(1)[["method_id", "n_cells", "runtime_seconds", "peak_memory_gb"]]
        .rename(
            columns={
                "n_cells": "largest_completed_cells",
                "runtime_seconds": "endpoint_runtime_seconds",
                "peak_memory_gb": "endpoint_peak_memory_gb",
            }
        )
    )
    high = eff[eff["n_cells"].eq(HIGH_SCALE_CELL)][["method_id", "runtime_seconds", "peak_memory_gb"]].copy()
    if high.empty:
        high["high_scale_resource_score"] = np.nan
    else:
        high["runtime_high_scale_score"] = normalize_lower_better(high["runtime_seconds"])
        high["memory_high_scale_score"] = normalize_lower_better(high["peak_memory_gb"])
        high["high_scale_resource_score"] = high[["runtime_high_scale_score", "memory_high_scale_score"]].mean(axis=1)
    profile = (
        pd.DataFrame({"method_id": method_order})
        .merge(summary[["method_id", "family_short", "completion_adjusted_score", "completed_levels", "missing_levels"]], on="method_id", how="left")
        .merge(exponents[["method_id", "runtime", "memory"]], on="method_id", how="left")
        .merge(endpoint, on="method_id", how="left")
        .merge(high[["method_id", "high_scale_resource_score"]], on="method_id", how="left")
    )
    profile["runtime_slope_score"] = normalize_lower_better(profile["runtime"])
    profile["memory_slope_score"] = normalize_lower_better(profile["memory"])
    profile["endpoint_reach_score"] = profile["largest_completed_cells"] / HIGH_SCALE_CELL
    profile["high_scale_resource_score"] = profile["high_scale_resource_score"].fillna(0.0)
    profile["integrated_practical_score"] = profile[
        [
            "completion_adjusted_score",
            "runtime_slope_score",
            "memory_slope_score",
            "endpoint_reach_score",
            "high_scale_resource_score",
        ]
    ].mean(axis=1)
    profile["method_id"] = pd.Categorical(profile["method_id"], categories=method_order, ordered=True)
    return profile.sort_values("method_id")


def draw_practical_profile_heatmap(fig: plt.Figure, ax: plt.Axes, eff: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    profile = compute_practical_profile(eff, method_order)
    components = [
        ("completion_adjusted_score", "completion-\nadjusted"),
        ("runtime_slope_score", "runtime\nslope"),
        ("memory_slope_score", "memory\nslope"),
        ("endpoint_reach_score", "endpoint\nreach"),
        ("high_scale_resource_score", f"{cell_label(HIGH_SCALE_CELL)}\nresource"),
        ("integrated_practical_score", "integrated\nscore"),
    ]
    matrix = profile[[col for col, _ in components]].to_numpy()
    im = ax.imshow(matrix, aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(components)))
    ax.set_xticklabels([label for _, label in components], rotation=0, ha="center")
    ax.set_yticks(np.arange(len(profile)))
    ax.set_yticklabels([display_method(m) for m in profile["method_id"].astype(str)], fontsize=4.55)
    ax.tick_params(length=0)
    for x in np.arange(0.5, len(components), 1):
        ax.axvline(x, color="white", linewidth=0.45)
    for y in np.arange(0.5, len(profile), 1):
        ax.axhline(y, color="white", linewidth=0.35)
    cax = ax.inset_axes([1.008, 0.24, 0.014, 0.52])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=4.5, length=1.4, pad=1)
    cb.outline.set_linewidth(0.35)
    cax.set_title("score", fontsize=4.5, pad=2)
    panel_header(ax, "j", "Practical scalability profile", dx=-0.045, dy=1.035, title_dx=0.04)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return profile


def add_family_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", linewidth=1.4, markersize=4, label=f)
        for f in FAMILY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.010),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.2,
        fontsize=5.8,
        title="Method family",
        title_fontsize=6.0,
    )


def write_outputs(panel_tables: dict[str, pd.DataFrame], eff: pd.DataFrame, install: pd.DataFrame, commits: pd.DataFrame, base: Path) -> None:
    for name, table in panel_tables.items():
        table.to_csv(SOURCE_OUT / f"Figure_8_panel_{name}_plotted_values.csv", index=False)
    eff.to_csv(SOURCE_OUT / "Figure_8_efficiency_scaling_full26_source_data.csv", index=False)
    install.to_csv(SOURCE_OUT / "Figure_8_install_manifest_full26_source_data.csv", index=False)
    commits.to_csv(SOURCE_OUT / "Figure_8_source_commit_records.csv", index=False)

    if hasattr(eff["method_id"], "cat"):
        method_order = list(eff["method_id"].cat.categories)
    else:
        method_order = sorted(eff["method_id"].astype(str).unique())
    completion_audit = compute_completion_audit(eff, method_order)
    completion_audit.to_csv(
        SOURCE_OUT / "Figure_8_completion_missingness_audit_source_data.csv",
        index=False,
    )

    with Image.open(base.with_suffix(".png")) as img:
        arr = np.asarray(img.convert("RGB"))
        pixel_std = float(arr.std())
        width, height = img.size

    available_rows = int(eff[["method_id", "n_cells"]].drop_duplicates().shape[0])
    expected_rows = 26 * len(CELL_ORDER)
    max_endpoint = eff.groupby("method_id", observed=True)["n_cells"].max()
    incomplete_methods = int(completion_audit["missing_levels"].gt(0).sum())
    qa = {
        "figure": base.with_suffix(".png").name,
        "panel_count": int(len(panel_tables)),
        "full_26_methods": int(eff["method_id"].nunique()),
        "cell_scale_points": len(CELL_ORDER),
        "available_efficiency_records": available_rows,
        "missing_efficiency_records": expected_rows - available_rows,
        "incomplete_methods": incomplete_methods,
        "methods_at_50000_cells": int(eff[eff["n_cells"].eq(COMMON_CELL)]["method_id"].nunique()),
        "methods_at_73233_cells": int(eff[eff["n_cells"].eq(73233)]["method_id"].nunique()),
        "min_largest_endpoint": int(max_endpoint.min()),
        "max_largest_endpoint": int(max_endpoint.max()),
        "install_manifest_methods": int(install["method_id"].nunique()),
        "environment_count": int(install["environment"].nunique()),
        "source_commit_records": int(commits["commit"].nunique()) if "commit" in commits.columns else 0,
        "complete_all_scales_methods": int(completion_audit["completed_levels"].eq(len(CELL_ORDER)).sum()),
        "pareto_methods_at_high_scale": int(panel_tables["i"]["pareto_efficient"].sum()),
        "practical_profile_methods": int(panel_tables["j"]["method_id"].nunique()),
        "png_width_px": width,
        "png_height_px": height,
        "png_pixel_std": pixel_std,
        "nonblank_png": bool(pixel_std > 5),
    }
    pd.DataFrame([qa]).to_csv(QA_OUT / "Figure_8_visual_qa_summary.csv", index=False)

    checklist = f"""# Figure 8 Scalability and Reproducibility QA

Generated by: `Publication/paper/revision_figures/figure8_polish/make_figure8_scalability_reproducibility_top_tier.py`

## Figure Contract

- Core claim: scalability is method-specific; runtime and memory traces must be interpreted together with cross-scale efficiency stability, endpoint completion, scaling burden and high-scale resource trade-offs.
- Original manuscript coverage: this figure expands the original computational-efficiency Figure 6 by retaining runtime and memory scaling while replacing coverage-style main panels with source-data-backed scalability analyses.
- Unit of analysis: method x downsampled 10x-73k cell-count run for efficiency panels; panel f averages direction-consistent runtime and memory scores across scale levels while assigning zero to missing scale-level runs.
- Missingness: absent completion cells indicate no valid efficiency record in the source data and are not imputed.
- Layout decision: completion and provenance/environment records are retained in source data and supplementary audit material; the main figure keeps only quantitative analyses derived from runtime and peak-memory source data.
- Labeling decision: all 26 methods are plotted in the scaling and endpoint panels; direct labels are restricted to representative or endpoint-informative methods to avoid obscuring the full benchmark landscape.

## QA

- Full benchmark methods plotted: {qa['full_26_methods']}.
- Cell-scale points: {qa['cell_scale_points']}; available efficiency records: {qa['available_efficiency_records']}; missing efficiency records: {qa['missing_efficiency_records']}; methods with incomplete scale coverage: {qa['incomplete_methods']}.
- Methods available at 50k cells: {qa['methods_at_50000_cells']}; at 73,233 cells: {qa['methods_at_73233_cells']}.
- Largest endpoint range: {qa['min_largest_endpoint']:,}-{qa['max_largest_endpoint']:,} cells.
- Install manifest methods: {qa['install_manifest_methods']}; environments: {qa['environment_count']}; source commit records: {qa['source_commit_records']}.
- Methods completing all scale levels: {qa['complete_all_scales_methods']}; {cell_label(HIGH_SCALE_CELL)}-cell Pareto-efficient methods: {qa['pareto_methods_at_high_scale']}; practical-profile methods: {qa['practical_profile_methods']}.
- PNG dimensions: {qa['png_width_px']} x {qa['png_height_px']} px; nonblank: {qa['nonblank_png']}.

## Panel Map

- a, runtime scaling over 10 downsampled cell-count levels; representative methods are directly labeled at their terminal observed scale.
- b, peak-memory scaling over the same cell-count levels; representative methods are directly labeled at their terminal observed scale.
- c, runtime rank-stability heatmap across tested cell-count levels; color encodes within-scale direction-consistent runtime rank percentile, with grey cells marking missing efficiency records.
- d, runtime-memory endpoint for the largest completed scale of each method, with point size encoding endpoint cell count.
- e, method-level log-log scaling exponents for runtime and memory.
- f, completion-adjusted efficiency score across all tested cell-count levels; labels mark missing scale levels.
- g, memory rank-stability heatmap across tested cell-count levels; color encodes within-scale direction-consistent peak-memory rank percentile, with grey cells marking missing efficiency records.
- h, family-level joint scaling burden computed as the mean of runtime and memory log-log slopes.
- i, runtime-memory Pareto frontier at the {cell_label(HIGH_SCALE_CELL)}-cell high-scale endpoint.
- j, integrated practical scalability profile combining completion-adjusted efficiency, runtime-slope score, memory-slope score, endpoint reach and {cell_label(HIGH_SCALE_CELL)}-cell resource score.
"""
    (QA_OUT / "Figure_8_polished_visual_qa_checklist.md").write_text(checklist, encoding="utf-8")


def main() -> None:
    eff, install, commits, method_order = load_data()

    fig = plt.figure(figsize=(9.68, 10.20))
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.48, 2.92, 1.62, 2.34],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.58,
        wspace=0.64,
        left=0.064,
        right=0.982,
        bottom=0.074,
        top=0.976,
    )

    panel_tables: dict[str, pd.DataFrame] = {}
    panel_tables["a"] = draw_scaling(fig.add_subplot(gs[0, 0:3]), eff, "runtime_seconds", "a", "Runtime scaling", "runtime (s)")
    panel_tables["b"] = draw_scaling(fig.add_subplot(gs[0, 3:6]), eff, "peak_memory_gb", "b", "Peak-memory scaling", "memory (GB)")
    panel_tables["c"] = draw_efficiency_score_heatmap(
        fig,
        fig.add_subplot(gs[1, 0:3]),
        eff,
        method_order,
        "runtime_seconds",
        "c",
        "Runtime rank stability",
        "score",
        show_ylabels=True,
    )
    panel_tables["g"] = draw_efficiency_score_heatmap(
        fig,
        fig.add_subplot(gs[1, 3:6]),
        eff,
        method_order,
        "peak_memory_gb",
        "g",
        "Memory rank stability",
        "score",
        show_ylabels=True,
        show_ylabel=False,
    )
    panel_tables["d"] = draw_endpoint_scatter(fig.add_subplot(gs[2, 0:2]), eff)
    panel_tables["e"] = draw_scaling_exponents(fig.add_subplot(gs[2, 2:4]), eff)
    panel_tables["h"] = draw_family_scalability_burden(fig.add_subplot(gs[2, 4:6]), eff)
    panel_tables["f"] = draw_completion_adjusted_efficiency(fig.add_subplot(gs[3, 0:2]), eff, method_order)
    panel_tables["i"] = draw_high_scale_pareto_frontier(fig.add_subplot(gs[3, 2:4]), eff)
    panel_tables["j"] = draw_practical_profile_heatmap(fig, fig.add_subplot(gs[3, 4:6]), eff, method_order)

    add_family_legend(fig)

    base = PLOT_OUT / FIGURE_BASENAME
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    write_outputs(panel_tables, eff, install, commits, base)


if __name__ == "__main__":
    main()
