from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure5_polish"
PLOT_OUT = OUT / "polished_real_only"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
CANON = ROOT / "Publication/paper/revision_figures/canonical_source_tables"

for path in (PLOT_OUT, SOURCE_OUT, QA_OUT):
    path.mkdir(parents=True, exist_ok=True)

CLUSTER_SOURCE = (
    ROOT
    / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_5_clustering_concordance_source_data.csv"
)
METHOD_SOURCE = (
    ROOT
    / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_1_method_taxonomy_source_data.csv"
)
META_SOURCE = ROOT / "datasets/real/benchmarker/Zhengmix4eq/cell_metadata.csv"
EMBEDDING_SOURCES = {
    "SPDR": ROOT / "results/benchmarker/Zhengmix4eq/SPDR_2.csv",
    "SSNMDI": ROOT / "results/benchmarker/Zhengmix4eq/SSNMDI_2.csv",
}


FAMILY_COLORS = {
    "linear": "#3D6FB6",
    "deep": "#BF6F6B",
    "graph": "#C79B38",
    "metric": "#3F9C9A",
    "unknown": "#7A7A7A",
}

FAMILY_RENAME = {
    "linear/probabilistic": "linear",
    "deep generative/autoencoder": "deep",
    "graph/diffusion": "graph",
    "metric/structure-aware": "metric",
}

LABEL_COLORS = {
    "b.cells": "#3D6FB6",
    "cd14.monocytes": "#3F9C9A",
    "naive.cytotoxic": "#C79B38",
    "regulatory.t": "#BF6F6B",
}

LABEL_DISPLAY = {
    "b.cells": "B cells",
    "cd14.monocytes": "CD14 monocytes",
    "naive.cytotoxic": "Naive cytotoxic",
    "regulatory.t": "Regulatory T",
}

ALG_ORDER = ["kmeans", "spectral", "louvain"]
ALG_DISPLAY = {"kmeans": "K-means", "spectral": "Spectral", "louvain": "Louvain"}
METRIC_ORDER = ["ARI", "NMI", "COMP", "HOMO", "SIL"]
METRIC_DISPLAY = {
    "ARI": "ARI",
    "NMI": "NMI",
    "COMP": "Completeness",
    "HOMO": "Homogeneity",
    "SIL": "Silhouette",
}
FAMILY_ORDER = ["linear", "deep", "graph", "metric"]


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.1,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.2,
        "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6,
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
    }
)

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "annotation_concordance", ["#F5F2EA", "#A9C9C7", "#3F9C9A", "#214D6C"]
)
HEX_CMAP = LinearSegmentedColormap.from_list(
    "agreement_density", ["#E3ECF3", "#83A9CC", "#2E6FA8", "#12395E"]
)


def display_method(name: str) -> str:
    return {
        "t-SNE": "t-SNE",
        "SQuaD-MDS": "SQuaD-MDS",
        "SSNMDI": "SSNMDI",
        "tGPLVM": "tGPLVM",
        "scvis": "scvis",
        "scScope": "scScope",
        "scGBM": "scGBM",
        "scGAE": "scGAE",
        "pCMF": "pCMF",
        "IVIS": "IVIS",
    }.get(name, name)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.1, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color="#111111",
    )


def load_method_families() -> dict[str, str]:
    taxonomy = pd.read_csv(METHOD_SOURCE)
    taxonomy = taxonomy[taxonomy["benchmark_scope"].eq("Full benchmark")].copy()
    taxonomy["family_short"] = taxonomy["method_family"].map(FAMILY_RENAME).fillna("unknown")
    return taxonomy.drop_duplicates("parent_method").set_index("parent_method")[
        "family_short"
    ].to_dict()


def canonical_full_methods() -> list[str]:
    methods = pd.read_csv(CANON / "canonical_method_manifest.csv")
    full = methods[methods["benchmark_scope"].eq("full_26_method_benchmark")].copy()
    return full.sort_values("method_order")["method_id"].tolist()


def load_clustering_data(family_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(CLUSTER_SOURCE)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    real_raw = raw[raw["dataset_category"].ne("simulate")].dropna(subset=["value"]).copy()
    real_raw["family"] = real_raw["parent_method"].map(family_map).fillna("unknown")

    parent = (
        real_raw.groupby(
            [
                "dataset_category",
                "dataset_id",
                "parent_method",
                "family",
                "clustering_algorithm",
                "metric",
            ],
            as_index=False,
        )["value"]
        .median()
        .copy()
    )
    return real_raw, parent


def load_embedding_example() -> pd.DataFrame:
    meta = pd.read_csv(META_SOURCE)
    meta = meta.rename(columns={"cell_type": "annotation_label"})
    pieces = []
    for method, path in EMBEDDING_SOURCES.items():
        emb = pd.read_csv(path)
        coord_cols = [c for c in emb.columns if c in {"0", "1"}]
        if len(coord_cols) != 2:
            raise ValueError(f"Could not find two coordinate columns in {path}")
        panel = pd.DataFrame(
            {
                "sample": np.arange(len(emb)),
                "x": pd.to_numeric(emb[coord_cols[0]], errors="coerce"),
                "y": pd.to_numeric(emb[coord_cols[1]], errors="coerce"),
                "method": method,
            }
        )
        panel = panel.merge(meta[["sample", "annotation_label"]], on="sample", how="left")
        pieces.append(panel)
    example = pd.concat(pieces, ignore_index=True)
    if example["annotation_label"].isna().any():
        raise ValueError("Missing annotation labels in Zhengmix4eq embedding example")
    return example


def audit_missing_parent_blocks(real_raw: pd.DataFrame, parent: pd.DataFrame) -> pd.DataFrame:
    dataset_table = (
        real_raw[["dataset_category", "dataset_id"]]
        .drop_duplicates()
        .sort_values(["dataset_category", "dataset_id"])
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "dataset_index"})
    )
    expected = pd.MultiIndex.from_product(
        [
            dataset_table["dataset_index"],
            sorted(real_raw["parent_method"].unique()),
            sorted(real_raw["clustering_algorithm"].unique()),
            sorted(real_raw["metric"].unique()),
        ],
        names=["dataset_index", "parent_method", "clustering_algorithm", "metric"],
    ).to_frame(index=False)
    expected = expected.merge(dataset_table, on="dataset_index", how="left").drop(
        columns=["dataset_index"]
    )
    observed = parent[
        ["dataset_category", "dataset_id", "parent_method", "clustering_algorithm", "metric"]
    ].copy()
    observed["present"] = True
    missing = expected.merge(
        observed,
        on=["dataset_category", "dataset_id", "parent_method", "clustering_algorithm", "metric"],
        how="left",
    )
    missing = missing[missing["present"].isna()].drop(columns=["present"])
    if missing.empty:
        return pd.DataFrame(
            columns=[
                "dataset_category",
                "dataset_id",
                "parent_method",
                "clustering_algorithm",
                "missing_metric_count",
            ]
        )
    return (
        missing.groupby(["dataset_category", "dataset_id", "parent_method", "clustering_algorithm"])
        .size()
        .reset_index(name="missing_metric_count")
        .sort_values(["dataset_category", "dataset_id", "parent_method", "clustering_algorithm"])
    )


def setup_axis(ax: plt.Axes) -> None:
    ax.tick_params(width=0.45, length=2.2, pad=1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#222222")


def draw_embedding_panel(fig: plt.Figure, gs_cell, example: pd.DataFrame) -> None:
    holder = fig.add_subplot(gs_cell)
    holder.axis("off")
    add_panel_label(holder, "a", x=-0.035, y=1.08)
    holder.text(
        0.0,
        1.08,
        "Zhengmix4eq annotation example",
        transform=holder.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
    )

    inner = gs_cell.subgridspec(2, 2, height_ratios=[1.0, 0.2], hspace=0.02, wspace=0.04)
    axes = [fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])]
    methods = ["SPDR", "SSNMDI"]
    for ax, method in zip(axes, methods):
        data = example[example["method"].eq(method)]
        for label, sub in data.groupby("annotation_label", sort=True):
            ax.scatter(
                sub["x"],
                sub["y"],
                s=2.2,
                c=LABEL_COLORS.get(label, "#7A7A7A"),
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(method, fontsize=6.6, fontweight="bold", pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_ax = fig.add_subplot(inner[1, :])
    legend_ax.axis("off")
    x_positions = [0.02, 0.28, 0.58, 0.82]
    labels = ["b.cells", "cd14.monocytes", "naive.cytotoxic", "regulatory.t"]
    for x, label in zip(x_positions, labels):
        legend_ax.scatter(
            [x],
            [0.45],
            s=14,
            c=LABEL_COLORS[label],
            edgecolor="white",
            linewidth=0.35,
            transform=legend_ax.transAxes,
            clip_on=False,
        )
        legend_ax.text(
            x + 0.025,
            0.45,
            LABEL_DISPLAY[label],
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.95,
            color="#333333",
        )


def draw_ari_heatmap(
    ax: plt.Axes,
    parent: pd.DataFrame,
    family_map: dict[str, str],
    canonical_methods: list[str],
) -> pd.DataFrame:
    ari = parent[parent["metric"].eq("ARI")].copy()
    heat = (
        ari.groupby(["parent_method", "clustering_algorithm"])["value"]
        .median()
        .unstack()
        .reindex(columns=ALG_ORDER)
    )
    order = [method for method in canonical_methods if method in heat.index]
    heat = heat.loc[order]

    im = ax.imshow(heat.values, aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=1)
    ax.set_title("ARI across clustering algorithms", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "b", x=-0.23, y=1.03)
    ax.set_xticks(np.arange(len(ALG_ORDER)))
    ax.set_xticklabels([ALG_DISPLAY[a] for a in ALG_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([display_method(x) for x in order], fontsize=4.7)
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for y, method in enumerate(order):
        fam = family_map.get(method, "unknown")
        ax.add_patch(
            Rectangle(
                (-0.73, y - 0.42),
                0.13,
                0.84,
                facecolor=FAMILY_COLORS.get(fam, "#7A7A7A"),
                edgecolor="none",
                clip_on=False,
            )
        )
    ax.set_xlim(-0.78, len(ALG_ORDER) - 0.5)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=1.8, width=0.45)
    cbar.set_label("median ARI", fontsize=5.2, labelpad=1.5)
    cbar.outline.set_linewidth(0.45)

    out = heat.reset_index().melt(
        id_vars="parent_method", var_name="clustering_algorithm", value_name="median_ari"
    )
    out["family"] = out["parent_method"].map(family_map).fillna("unknown")
    out["panel"] = "b"
    return out


def draw_metric_matrix(ax: plt.Axes, parent: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        parent.groupby(["metric", "clustering_algorithm"])["value"]
        .median()
        .unstack()
        .reindex(index=METRIC_ORDER, columns=ALG_ORDER)
    )
    im = ax.imshow(matrix.values, cmap=SCORE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Metric-algorithm medians", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "c", x=-0.18, y=1.04)
    ax.set_xticks(np.arange(len(ALG_ORDER)))
    ax.set_xticklabels([ALG_DISPLAY[a] for a in ALG_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(METRIC_ORDER)))
    ax.set_yticklabels([METRIC_DISPLAY[m] for m in METRIC_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            color = "white" if value >= 0.5 else "#252525"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color=color)
    return matrix.reset_index().melt(
        id_vars="metric", var_name="clustering_algorithm", value_name="median_value"
    ).assign(panel="c")


def draw_family_distribution(ax: plt.Axes, parent: pd.DataFrame) -> pd.DataFrame:
    ari = parent[parent["metric"].eq("ARI")].copy()
    ari["family"] = pd.Categorical(ari["family"], categories=FAMILY_ORDER, ordered=True)
    rng = np.random.default_rng(20260604)
    groups = [ari[ari["family"].eq(f)]["value"].dropna().values for f in FAMILY_ORDER]
    positions = np.arange(len(FAMILY_ORDER))
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 0.7},
        whiskerprops={"color": "#555555", "linewidth": 0.55},
        capprops={"color": "#555555", "linewidth": 0.55},
        boxprops={"linewidth": 0.55, "edgecolor": "#333333"},
    )
    for patch, family in zip(bp["boxes"], FAMILY_ORDER):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.62)

    for x, family in zip(positions, FAMILY_ORDER):
        vals = ari[ari["family"].eq(family)]["value"].dropna().values
        if len(vals) > 650:
            vals = rng.choice(vals, size=650, replace=False)
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=3.2,
            color="#5F6D78",
            alpha=0.16,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title("Family-level ARI distribution", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "d", x=-0.16, y=1.04)
    ax.set_xticks(positions)
    ax.set_xticklabels(FAMILY_ORDER, rotation=0, ha="center")
    ax.set_ylabel("ARI")
    ax.set_ylim(-0.05, 1.03)
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    return ari.assign(panel="d")[
        ["panel", "dataset_category", "dataset_id", "parent_method", "family", "clustering_algorithm", "value"]
    ].rename(columns={"value": "ari"})


def draw_ari_nmi_agreement(ax: plt.Axes, parent: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_category", "dataset_id", "parent_method", "family", "clustering_algorithm"]
    ari = parent[parent["metric"].eq("ARI")][keys + ["value"]].rename(columns={"value": "ARI"})
    nmi = parent[parent["metric"].eq("NMI")][keys + ["value"]].rename(columns={"value": "NMI"})
    merged = ari.merge(nmi, on=keys, how="inner")
    hb = ax.hexbin(
        merged["ARI"],
        merged["NMI"],
        gridsize=34,
        extent=(0, 1, 0, 1),
        mincnt=1,
        cmap=HEX_CMAP,
        linewidths=0,
        alpha=0.98,
    )
    counts_raw = hb.get_array()
    counts = counts_raw.compressed() if hasattr(counts_raw, "compressed") else np.asarray(counts_raw)
    if len(counts) > 0:
        hb.set_clim(1, float(np.quantile(counts, 0.92)))
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linewidth=0.55, linestyle=(0, (3, 2)))
    coef = np.polyfit(merged["ARI"], merged["NMI"], deg=1)
    xs = np.linspace(0, 1, 100)
    ys = coef[0] * xs + coef[1]
    mask = (ys >= 0) & (ys <= 1)
    ax.plot(xs[mask], ys[mask], color="#1F4E79", linewidth=0.75)
    r = float(np.corrcoef(merged["ARI"], merged["NMI"])[0, 1])
    ax.text(
        0.03,
        0.95,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color="#1F4E79",
        fontweight="bold",
    )
    ax.set_title("ARI-NMI agreement", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "e", x=-0.055, y=1.04)
    ax.set_xlabel("ARI")
    ax.set_ylabel("NMI")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    cbar = ax.figure.colorbar(hb, ax=ax, fraction=0.035, pad=0.012)
    cbar.ax.tick_params(labelsize=4.8, length=1.8, width=0.45)
    cbar.set_label("records/bin", fontsize=5.2, labelpad=1.5)
    cbar.outline.set_linewidth(0.45)
    merged["panel"] = "e"
    merged["pearson_r"] = r
    return merged


def draw_top_three_frequency(
    ax: plt.Axes, parent: pd.DataFrame, family_map: dict[str, str]
) -> pd.DataFrame:
    ari = parent[parent["metric"].eq("ARI")].copy()
    task_cols = ["dataset_category", "dataset_id", "clustering_algorithm"]
    ranked = []
    for _, block in ari.groupby(task_cols, sort=False):
        top = block.sort_values("value", ascending=False).head(3)
        ranked.append(top)
    top = pd.concat(ranked, ignore_index=True)
    counts = top["parent_method"].value_counts().rename_axis("parent_method").reset_index(name="appearances")
    n_slots = len(ranked) * 3
    counts["fraction"] = counts["appearances"] / n_slots
    counts["family"] = counts["parent_method"].map(family_map).fillna("unknown")
    counts = counts.sort_values(["appearances", "parent_method"], ascending=[False, True]).head(12)
    plot = counts.iloc[::-1]
    colors = [FAMILY_COLORS.get(f, "#7A7A7A") for f in plot["family"]]
    ax.barh(
        [display_method(x) for x in plot["parent_method"]],
        plot["appearances"],
        color=colors,
        edgecolor="white",
        linewidth=0.45,
        height=0.68,
    )
    ax.set_title("Top-three ARI frequency", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "f", x=-0.22, y=1.04)
    ax.set_xlabel(f"appearances out of {n_slots} slots")
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    ax.tick_params(axis="y", length=0)
    counts["panel"] = "f"
    counts["n_top_three_slots"] = n_slots
    return counts


def draw_dataset_rank_stability(
    ax: plt.Axes,
    parent: pd.DataFrame,
    canonical_methods: list[str],
    family_map: dict[str, str],
) -> pd.DataFrame:
    ari = parent[parent["metric"].eq("ARI")].copy()
    task_cols = ["dataset_category", "dataset_id", "clustering_algorithm"]
    task_coverage = ari.groupby(task_cols, observed=False)["parent_method"].nunique().reset_index(name="methods")
    complete_tasks = task_coverage.loc[task_coverage["methods"].ge(20), task_cols]
    ari = ari.merge(complete_tasks, on=task_cols, how="inner")
    ari["rank"] = ari.groupby(task_cols, observed=False)["value"].rank(method="average", ascending=False)
    summary = (
        ari.groupby("parent_method", observed=False)
        .agg(
            median_rank=("rank", "median"),
            q1_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
            q3_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
            median_ari=("value", "median"),
            top3_fraction=("rank", lambda x: float(np.mean(x <= 3))),
            dataset_algorithm_tasks=("rank", "size"),
        )
        .reindex(canonical_methods)
        .reset_index()
    )
    y = np.arange(len(summary))
    colors = [FAMILY_COLORS.get(family_map.get(method, "unknown"), "#7A7A7A") for method in summary["parent_method"]]
    for yi, (_, row), color in zip(y, summary.iterrows(), colors):
        ax.hlines(yi, row["q1_rank"], row["q3_rank"], color=color, lw=2.0, alpha=0.76)
        ax.scatter(
            row["median_rank"],
            yi,
            s=14 + 52 * row["top3_fraction"],
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_title("Dataset-level ARI rank stability", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "g", x=-0.055, y=1.04)
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in summary["parent_method"]])
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(canonical_methods) + 0.5)
    ax.set_xlabel("within-task rank (lower better)")
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    ax.tick_params(axis="y", labelsize=4.45, length=0, pad=1.1)
    ax.tick_params(axis="x", labelsize=5.0, length=2.2, pad=1.5)
    summary["panel"] = "g"
    summary["eligible_task_count"] = int(complete_tasks.shape[0])
    return summary


def draw_algorithm_top_frequency(
    ax: plt.Axes, parent: pd.DataFrame, aggregate_counts: pd.DataFrame
) -> pd.DataFrame:
    ari = parent[parent["metric"].eq("ARI")].copy()
    task_cols = ["dataset_category", "dataset_id", "clustering_algorithm"]
    pieces = []
    for _, block in ari.groupby(task_cols, sort=False):
        pieces.append(block.sort_values("value", ascending=False).head(3))
    top = pd.concat(pieces, ignore_index=True)
    counts = (
        top.groupby(["parent_method", "clustering_algorithm"], observed=False)
        .size()
        .reset_index(name="appearances")
    )
    top_methods = aggregate_counts.sort_values("appearances", ascending=False)["parent_method"].head(10).tolist()
    matrix = (
        counts[counts["parent_method"].isin(top_methods)]
        .pivot_table(index="parent_method", columns="clustering_algorithm", values="appearances", fill_value=0)
        .reindex(index=top_methods, columns=ALG_ORDER)
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=SCORE_CMAP)
    ax.set_title("Top-three frequency by algorithm", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "h", x=-0.22, y=1.04)
    ax.set_xticks(np.arange(len(ALG_ORDER)))
    ax.set_xticklabels([ALG_DISPLAY[a] for a in ALG_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([display_method(m) for m in matrix.index], fontsize=4.7)
    ax.tick_params(length=0, pad=1.1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            color = "white" if value >= np.nanmax(matrix.to_numpy()) * 0.55 else "#252525"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=4.8, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=1.8, width=0.45)
    cbar.set_label("appearances", fontsize=5.2, labelpad=1.5)
    cbar.outline.set_linewidth(0.45)
    out = matrix.reset_index().melt(
        id_vars="parent_method", var_name="clustering_algorithm", value_name="appearances"
    )
    out["panel"] = "h"
    return out


def save_outputs(fig: plt.Figure, basename: str) -> None:
    fig.savefig(PLOT_OUT / f"{basename}.svg", bbox_inches="tight")
    fig.savefig(PLOT_OUT / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(PLOT_OUT / f"{basename}.png", dpi=450, bbox_inches="tight")
    fig.savefig(
        PLOT_OUT / f"{basename}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def build_figure() -> None:
    family_map = load_method_families()
    canonical_methods = canonical_full_methods()
    real_raw, parent = load_clustering_data(family_map)
    example = load_embedding_example()
    missing_blocks = audit_missing_parent_blocks(real_raw, parent)

    real_raw.to_csv(SOURCE_OUT / "Figure_5_annotation_clustering_real_only_source_data.csv", index=False)
    example.to_csv(SOURCE_OUT / "Figure_5_Zhengmix4eq_embedding_example_source_data.csv", index=False)
    missing_blocks.to_csv(
        SOURCE_OUT / "Figure_5_annotation_clustering_real_only_missing_blocks.csv", index=False
    )

    fig = plt.figure(figsize=(7.2, 9.35))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        width_ratios=[1.05, 1.0, 1.25],
        height_ratios=[0.94, 0.82, 0.92, 0.78],
        wspace=0.62,
        hspace=0.36,
    )

    draw_embedding_panel(fig, gs[0, 0:2], example)
    ax_b = fig.add_subplot(gs[0:2, 2])
    panel_b = draw_ari_heatmap(ax_b, parent, family_map, canonical_methods)
    ax_c = fig.add_subplot(gs[1, 0])
    panel_c = draw_metric_matrix(ax_c, parent)
    ax_d = fig.add_subplot(gs[1, 1])
    panel_d = draw_family_distribution(ax_d, parent)
    ax_e = fig.add_subplot(gs[2, 0:2])
    panel_e = draw_ari_nmi_agreement(ax_e, parent)
    ax_f = fig.add_subplot(gs[2, 2])
    panel_f = draw_top_three_frequency(ax_f, parent, family_map)
    ax_g = fig.add_subplot(gs[3, 0:2])
    panel_g = draw_dataset_rank_stability(ax_g, parent, canonical_methods, family_map)
    ax_h = fig.add_subplot(gs[3, 2])
    panel_h = draw_algorithm_top_frequency(ax_h, parent, panel_f)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=5.0,
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor="none",
            label=f,
        )
        for f in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.010),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.35,
        fontsize=5.6,
        title="Method family",
        title_fontsize=5.8,
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.975, bottom=0.074)

    basename = "Figure_5_annotation_clustering_concordance_real_only"
    save_outputs(fig, basename)
    plt.close(fig)

    panel_data = []
    panel_data.append(panel_b)
    panel_data.append(panel_c)
    panel_data.append(panel_d)
    panel_data.append(panel_e)
    panel_data.append(panel_f)
    panel_data.append(panel_g)
    panel_data.append(panel_h)
    pd.concat(panel_data, ignore_index=True, sort=False).to_csv(
        SOURCE_OUT / f"{basename}_panel_data.csv", index=False
    )

    qa = {
        "panel_count": 8,
        "raw_rows": len(pd.read_csv(CLUSTER_SOURCE)),
        "real_raw_rows": len(real_raw),
        "real_parent_collapsed_rows": len(parent),
        "real_parent_expected_rows": real_raw["dataset_id"].nunique()
        * parent["parent_method"].nunique()
        * parent["clustering_algorithm"].nunique()
        * parent["metric"].nunique(),
        "real_parent_missing_rows": int(missing_blocks["missing_metric_count"].sum())
        if not missing_blocks.empty
        else 0,
        "real_parent_missing_combinations": len(missing_blocks),
        "real_dataset_count": real_raw["dataset_id"].nunique(),
        "dataset_algorithm_task_count": int(panel_g["eligible_task_count"].iloc[0]),
        "parent_method_count": parent["parent_method"].nunique(),
        "canonical_full_methods": len(canonical_methods),
        "ari_heatmap_methods": panel_b["parent_method"].nunique(),
        "rank_stability_methods": panel_g["parent_method"].nunique(),
        "method_id_count_raw_real": real_raw["method_id"].nunique(),
        "algorithm_count": parent["clustering_algorithm"].nunique(),
        "metric_count": parent["metric"].nunique(),
        "embedding_example_cells": example["sample"].nunique(),
        "embedding_example_labels": example["annotation_label"].nunique(),
        "median_ari_kmeans": parent[(parent["metric"].eq("ARI")) & (parent["clustering_algorithm"].eq("kmeans"))][
            "value"
        ].median(),
        "median_ari_spectral": parent[
            (parent["metric"].eq("ARI")) & (parent["clustering_algorithm"].eq("spectral"))
        ]["value"].median(),
        "median_ari_louvain": parent[(parent["metric"].eq("ARI")) & (parent["clustering_algorithm"].eq("louvain"))][
            "value"
        ].median(),
        "ari_nmi_pearson_r": panel_e["pearson_r"].iloc[0],
        "top_three_slots": panel_f["n_top_three_slots"].iloc[0],
        "top_three_leader": panel_f.sort_values("appearances", ascending=False)["parent_method"].iloc[0],
        "algorithm_top_frequency_methods": panel_h["parent_method"].nunique(),
    }
    pd.DataFrame([qa]).to_csv(SOURCE_OUT / f"{basename}_qa_summary.csv", index=False)

    checklist = f"""# Figure 5 Legend And Checklist

Generated by: `Publication/paper/revision_figures/figure5_polish/make_figure5_annotation_concordance_real_only.py`

## Figure Role

Figure 5 preserves the original Zhengmix4eq embedding example while reframing the
quantitative clustering benchmark as annotation-derived label concordance rather than
independent ground-truth recovery.

## Panel Logic

- a, Representative Zhengmix4eq embeddings for SPDR and SSNMDI, colored by
  annotation-derived labels. This preserves the original Figure 5a visual comparison.
- b, Median ARI for 26 parent methods across K-means, spectral clustering, and Louvain
  on available real datasets with clustering results. This upgrades the original
  k-means-only bar summary to all three clustering algorithms.
- c, Median values of ARI, NMI, completeness, homogeneity, and silhouette across the
  same algorithms. This defines the metric landscape without relying on a single score.
- d, Distribution of ARI values by method family. This supports category-level
  interpretation requested by the reviewers.
- e, ARI-NMI agreement across method-dataset-algorithm records. This checks whether
  the main conclusion is metric-specific.
- f, Frequency with which each parent method appears among the top three ARI values
  across dataset-algorithm tasks. This emphasizes repeated high performance rather than
  a single averaged winner.
- g, Dataset-level ARI rank stability across eligible dataset-algorithm tasks.
  This tests whether repeatedly high ARI values remain stable beyond a single
  averaged ranking.
- h, Algorithm-specific top-three ARI frequency among repeatedly high-performing
  methods. This shows whether panel f is stable across K-means, spectral clustering,
  and Louvain.

## Data QA

- Real datasets represented in clustering source data: {qa['real_dataset_count']}.
- Eligible dataset-algorithm tasks used for panel g: {qa['dataset_algorithm_task_count']}.
- Parent methods represented after variant collapse: {qa['parent_method_count']}.
- Canonical full-benchmark methods expected: {qa['canonical_full_methods']}.
- Methods shown in panel b: {qa['ari_heatmap_methods']}.
- Methods shown in panel g: {qa['rank_stability_methods']}.
- Clustering algorithms: {qa['algorithm_count']}.
- Clustering metrics: {qa['metric_count']}.
- Parent-method matrix rows after variant collapse: {qa['real_parent_collapsed_rows']}
  observed of {qa['real_parent_expected_rows']} expected rows.
- Missing parent-method metric rows: {qa['real_parent_missing_rows']} across
  {qa['real_parent_missing_combinations']} dataset-method-algorithm combinations.
- Zhengmix4eq cells in panel a: {qa['embedding_example_cells']}.
- Zhengmix4eq annotation labels in panel a: {qa['embedding_example_labels']}.
- Median ARI by algorithm: K-means {qa['median_ari_kmeans']:.3f}, spectral
  {qa['median_ari_spectral']:.3f}, Louvain {qa['median_ari_louvain']:.3f}.
- ARI-NMI Pearson correlation: {qa['ari_nmi_pearson_r']:.3f}.
- Top-three slots counted in panel f: {qa['top_three_slots']}.
- Methods shown in panel h: {qa['algorithm_top_frequency_methods']}.

## Manuscript Wording Guardrails

- Use "annotation-derived labels", "reference annotations", or "label concordance".
- Do not use "true labels", "ground truth cell types", or "cell type identification
  success" for these real-dataset clustering panels.
- State that ARI values are descriptive concordance metrics on two-dimensional
  embeddings, not proof of biological correctness.
- State that clustering algorithms are treated as sensitivity factors, not the main
  target of the benchmark.

## Output Files

- `polished_real_only/{basename}.svg`
- `polished_real_only/{basename}.pdf`
- `polished_real_only/{basename}.png`
- `polished_real_only/{basename}.tiff`
- `source_data/Figure_5_annotation_clustering_real_only_source_data.csv`
- `source_data/Figure_5_annotation_clustering_real_only_missing_blocks.csv`
- `source_data/Figure_5_Zhengmix4eq_embedding_example_source_data.csv`
- `source_data/{basename}_panel_data.csv`
- `source_data/{basename}_qa_summary.csv`
"""
    (QA_OUT / f"{basename}_legend_and_checklist.md").write_text(checklist, encoding="utf-8")


if __name__ == "__main__":
    build_figure()
