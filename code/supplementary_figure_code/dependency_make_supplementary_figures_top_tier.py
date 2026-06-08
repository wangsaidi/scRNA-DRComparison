from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[4]
SOURCE_IN = ROOT / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data"
CANON = ROOT / "Publication/paper/revision_figures/canonical_source_tables"
OUT = ROOT / "Publication/paper/revision_figures/supplementary_polish"
FIG_OUT = OUT / "supplementary_figures"
SRC_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for directory in [FIG_OUT, SRC_OUT, QA_OUT]:
    directory.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6.2,
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "savefig.facecolor": "white",
    }
)


PALETTE = {
    "blue": "#3D6FB6",
    "teal": "#3F9C9A",
    "green": "#75A35A",
    "gold": "#C99C35",
    "rose": "#BF6F6B",
    "violet": "#8B6BB0",
    "slate": "#5E6978",
    "gray": "#A7A7A7",
    "light": "#E9E9E9",
    "very_light": "#F7F7F7",
    "dark": "#2E2E2E",
}

FAMILY_COLORS = {
    "linear/probabilistic": PALETTE["blue"],
    "deep generative/autoencoder": PALETTE["rose"],
    "graph/diffusion": PALETTE["gold"],
    "metric/structure-aware": PALETTE["teal"],
    "targeted/control": "#7C7C7C",
    "other": PALETTE["gray"],
}

FAMILY_LABELS = {
    "linear/probabilistic": "linear",
    "deep generative/autoencoder": "deep",
    "graph/diffusion": "graph",
    "metric/structure-aware": "metric",
    "targeted/control": "targeted",
    "other": "other",
}

SCORE_CMAP = LinearSegmentedColormap.from_list("paper_score", ["#F5F5F2", "#B9D9D5", "#3F9C9A", "#16395C"])
GOLD_CMAP = LinearSegmentedColormap.from_list("paper_gold", ["#F7F4EA", "#E2C878", "#C99C35", "#6F5521"])
ROSE_CMAP = LinearSegmentedColormap.from_list("paper_rose", ["#F8F2F1", "#DFA6A3", "#BF6F6B", "#743D3B"])
BINARY_CMAP = LinearSegmentedColormap.from_list("binary_member", ["#F7F7F7", "#3F9C9A"])

METHOD_LABEL_REPLACEMENTS = {
    "Parametric UMAP 50": "Parametric UMAP\n50",
    "Parametric UMAP 200": "Parametric UMAP\n200",
    "SQuaD-MDS hybrid": "SQuaD-MDS\nhybrid",
}

METRIC_LABELS = {
    "ARI": "ARI",
    "NMI": "NMI",
    "HOMO": "homogeneity",
    "COMP": "completeness",
    "SIL": "silhouette",
    "ari": "ARI",
    "nmi": "NMI",
    "trustworthiness_k30": "trustworthiness",
    "runtime_seconds": "runtime (s)",
    "peak_memory_gb": "memory (GB)",
    "local": "local",
    "global": "global",
    "kmeans": "k-means",
    "louvain": "Louvain",
    "spectral": "spectral",
    "runtime_score": "runtime",
    "memory_score": "memory",
    "stability_median": "stability",
    "overall_mean": "overall",
}

PANEL_DECISIONS = []
MANIFEST_ROWS = []
LEGENDS = []


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE_IN / name)


def canonical_methods() -> list[str]:
    manifest = pd.read_csv(CANON / "canonical_method_manifest.csv")
    return (
        manifest[manifest["benchmark_scope"].eq("full_26_method_benchmark")]
        .sort_values("method_order")["method_id"]
        .astype(str)
        .tolist()
    )


CANONICAL = canonical_methods()
ORDER_MAP = {method: i for i, method in enumerate(CANONICAL)}


def method_manifest() -> pd.DataFrame:
    df = read_csv("Supplementary_Figure_S1_method_catalog_source_data.csv")
    df["implementation_language"] = df["implementation_language"].replace({"python": "Python"})
    return df


METHODS = method_manifest()
FAMILY_MAP = METHODS.drop_duplicates("parent_method").set_index("parent_method")["method_family"].to_dict()
FAMILY_MAP.update(METHODS.drop_duplicates("method_id").set_index("method_id")["method_family"].to_dict())
FAMILY_MAP["scVI"] = "targeted/control"


def display_method(method: object) -> str:
    text = str(method)
    return METHOD_LABEL_REPLACEMENTS.get(text, text)


def family_color(method: object) -> str:
    return FAMILY_COLORS.get(FAMILY_MAP.get(str(method), "other"), FAMILY_COLORS["other"])


def family_short(family: object) -> str:
    return FAMILY_LABELS.get(str(family), str(family))


def public_source(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    replacements = {
        "full_26_method_benchmark": "Full benchmark",
        "original_result_variant": "Result variant",
        "targeted_revision_control_only": "Targeted reference",
        "WP1_scVI_local": "scVI reference",
        "WP2_dimension_sensitivity": "latent-dimension sensitivity",
        "WP3_visualization_workflow": "visualization-workflow comparison",
        "WP4_input_gene_sensitivity": "input-gene sensitivity",
        "direct_2d": "Direct 2D",
        "pca50_to_2d": "PCA50 to 2D",
    }
    for col in out.select_dtypes(include="object").columns:
        s = out[col].astype("string")
        for old, new in replacements.items():
            s = s.str.replace(old, new, regex=False)
        out[col] = s
    return out


def save_panel_data(df: pd.DataFrame, name: str) -> str:
    path = SRC_OUT / name
    public_source(df).to_csv(path, index=False)
    return str(path.relative_to(OUT))


def save_figure(fig: plt.Figure, base: Path, source_file: str, dpi: int = 600) -> None:
    for ext in ["svg", "pdf", "png", "tiff"]:
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        MANIFEST_ROWS.append(
            {
                "figure": base.name,
                "format": ext,
                "path": str(out.relative_to(OUT)),
                "source_data": source_file,
            }
        )
    plt.close(fig)


def add_legend(fig_id: str, title: str, legend: str) -> None:
    LEGENDS.append({"figure": fig_id, "title": title, "legend": legend})


def decision(fig_id: str, status: str, reason: str, main_link: str) -> None:
    PANEL_DECISIONS.append({"figure": fig_id, "status": status, "reason": reason, "main_figure_link": main_link})


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.025) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold", color=PALETTE["dark"])


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def set_tick_size(ax: plt.Axes, size: float = 5.2) -> None:
    ax.tick_params(axis="both", labelsize=size, length=2.0, pad=1.5)


def add_family_strip(ax: plt.Axes, methods: list[str], orientation: str = "y") -> None:
    if orientation == "y":
        inset = ax.inset_axes([-0.035, 0.0, 0.012, 1.0], transform=ax.transAxes)
        colors = np.array([[matplotlib.colors.to_rgb(family_color(m))] for m in methods])
        inset.imshow(colors, aspect="auto")
        inset.set_xticks([])
        inset.set_yticks([])
    else:
        inset = ax.inset_axes([0.0, 1.01, 1.0, 0.016], transform=ax.transAxes)
        colors = np.array([[matplotlib.colors.to_rgb(family_color(m)) for m in methods]])
        inset.imshow(colors, aspect="auto")
        inset.set_xticks([])
        inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)


def heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap= SCORE_CMAP,
    vmin: float | None = 0,
    vmax: float | None = 1,
    cbar_label: str = "score",
    xrot: int = 40,
    annotate: bool = False,
    cbar: bool = True,
) -> None:
    matrix = data.astype(float).to_numpy()
    cmap_obj = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#F0F0F0")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([METRIC_LABELS.get(str(c), str(c)) for c in data.columns], rotation=xrot, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([display_method(x) for x in data.index])
    set_tick_size(ax, 4.7 if data.shape[0] > 22 else 5.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if annotate and data.shape[0] <= 12 and data.shape[1] <= 8:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    text_color = "white" if value > ((vmax or 1) * 0.58) else PALETTE["dark"]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.5, color=text_color)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
        cb.ax.tick_params(labelsize=4.8, length=1.8)
        cb.outline.set_linewidth(0.4)
        cb.set_label(cbar_label, fontsize=5.1)


def binary_matrix(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    xrot: int = 35,
    ysize: float | None = None,
) -> None:
    ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap=BINARY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=xrot, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([display_method(x) for x in data.index])
    set_tick_size(ax, ysize or (4.6 if data.shape[0] > 22 else 5.2))
    for x in np.arange(0.5, data.shape[1], 1):
        ax.axvline(x, color="white", lw=0.5)
    for y in np.arange(0.5, data.shape[0], 1):
        ax.axhline(y, color="white", lw=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def ranked_barh(
    ax: plt.Axes,
    values: pd.Series,
    title: str,
    xlabel: str,
    colors: list[str] | dict[object, str] | None = None,
    top: int | None = None,
) -> pd.DataFrame:
    vals = values.dropna().sort_values(ascending=True)
    if top is not None:
        vals = vals.tail(top)
    if colors is None:
        colors = [family_color(x) for x in vals.index]
    elif isinstance(colors, dict):
        colors = [colors.get(x, PALETTE["gray"]) for x in vals.index]
    ax.barh([display_method(x) for x in vals.index], vals.values, color=colors, edgecolor="white", lw=0.35)
    ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=PALETTE["light"], lw=0.45)
    ax.set_axisbelow(True)
    set_tick_size(ax, 5.0)
    return pd.DataFrame({"item": vals.index, "value": vals.values})


def line_scaling(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> pd.DataFrame:
    plot = df.dropna(subset=["parent_method", "n_cells", metric]).copy()
    plot["parent_method"] = plot["parent_method"].astype(str)
    plot = plot[plot["parent_method"].isin(CANONICAL)]
    for method, group in plot.groupby("parent_method"):
        group = group.sort_values("n_cells")
        ax.plot(
            group["n_cells"],
            group[metric],
            color=family_color(method),
            alpha=0.45,
            lw=0.8,
            marker="o",
            ms=2.0,
            mec="white",
            mew=0.25,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax.grid(True, color=PALETTE["light"], lw=0.45, which="major")
    ax.set_axisbelow(True)
    set_tick_size(ax)
    return plot


def collapse_parent(df: pd.DataFrame, method_col: str = "parent_method") -> pd.DataFrame:
    out = df.copy()
    if method_col not in out.columns and "method_id" in out.columns:
        out[method_col] = out["method_id"]
    out[method_col] = out[method_col].astype(str)
    out = out[out[method_col].isin(CANONICAL)].copy()
    out["method_order"] = out[method_col].map(ORDER_MAP)
    return out.sort_values("method_order")


def figure_s1() -> None:
    df = METHODS.copy()
    full = df[df["benchmark_scope"].eq("Full benchmark")].copy()
    full["method_id"] = pd.Categorical(full["method_id"], categories=CANONICAL, ordered=True)
    full = full.sort_values("method_id")

    family_cols = ["linear", "deep", "graph", "metric"]
    lang_cols = ["R", "Python", "MATLAB"]
    matrix = pd.DataFrame(0, index=full["method_id"].astype(str), columns=family_cols + lang_cols + ["source", "year"])
    family_to_col = {
        "linear/probabilistic": "linear",
        "deep generative/autoencoder": "deep",
        "graph/diffusion": "graph",
        "metric/structure-aware": "metric",
    }
    for _, row in full.iterrows():
        method = str(row["method_id"])
        matrix.loc[method, family_to_col.get(row["method_family"], "linear")] = 1
        lang = str(row["implementation_language"]).replace("python", "Python")
        if lang in matrix.columns:
            matrix.loc[method, lang] = 1
        matrix.loc[method, "source"] = int(pd.notna(row.get("source_or_url")) and str(row.get("source_or_url")).strip() != "")
        matrix.loc[method, "year"] = int(pd.notna(row.get("publication_year")))

    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1.0], height_ratios=[1.35, 1.0], hspace=0.13, wspace=0.12)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a", x=-0.11)
    panel_label(ax_b, "b", x=-0.12)
    panel_label(ax_c, "c", x=-0.12)

    binary_matrix(ax_a, matrix, "Canonical 26-method catalog", xrot=45, ysize=4.5)
    ax_a.axvline(3.5, color=PALETTE["dark"], lw=0.5)
    ax_a.axvline(6.5, color=PALETTE["dark"], lw=0.5)

    family_counts = full["method_family"].value_counts().reindex(FAMILY_COLORS.keys()).dropna()
    family_counts.index = [family_short(x) for x in family_counts.index]
    family_color_map = {
        family_short(k): FAMILY_COLORS[k]
        for k in full["method_family"].value_counts().reindex(FAMILY_COLORS.keys()).dropna().index
    }
    ranked_barh(ax_b, family_counts, "Family accounting", "methods", colors=family_color_map)
    for i, value in enumerate(family_counts.sort_values().values):
        ax_b.text(value + 0.15, i, str(int(value)), va="center", fontsize=5.1)
    ax_b.set_xlim(0, max(family_counts) + 1.2)

    not_full = df[~df["benchmark_scope"].eq("Full benchmark")].copy()
    y = np.arange(len(not_full))
    role_colors = {"Result variants": PALETTE["gold"], "Targeted analysis": PALETTE["gray"], "Targeted reference": PALETTE["gray"]}
    ax_c.barh(y, np.ones(len(not_full)), color=[role_colors.get(x, PALETTE["gray"]) for x in not_full["benchmark_scope"]], edgecolor="white", lw=0.45)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([display_method(x) for x in not_full["method_id"]], fontsize=5.0)
    ax_c.set_xlim(0, 1.1)
    ax_c.set_xlabel("listed")
    ax_c.set_title("Variants and targeted analyses", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    for i, row in enumerate(not_full.itertuples(index=False)):
        ax_c.text(1.02, i, str(row.benchmark_scope), va="center", fontsize=4.8, color=PALETTE["dark"])
    ax_c.set_xticks([0, 1])

    source = save_panel_data(df, "Supplementary_Figure_S1_method_catalog_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S1_method_catalog_polished", source)
    decision("S1", "redesigned", "Converted from sparse binary heatmaps into a canonical method catalog and inclusion audit.", "Figure 1")
    add_legend("S1", "Method catalog and benchmark inclusion audit.", "Supplementary Fig. S1 documents the formal 26-method benchmark scope, method families, implementation languages, source/year traceability, and non-full-benchmark variants or targeted reference analyses. Rows in panel a follow the canonical 26-method order used in the main figures.")


def figure_s2() -> None:
    df = read_csv("Supplementary_Figure_S2_full_dataset_atlas_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    counts = df["dataset_type"].value_counts().reindex(["real", "simulated"])
    axes[0].bar(counts.index, counts.values, color=[PALETTE["blue"], PALETTE["gold"]], width=0.64, edgecolor="white", lw=0.5)
    axes[0].set_title("100-dataset manuscript atlas", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[0].set_ylabel("datasets")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1.5, str(int(v)), ha="center", fontsize=6)
    axes[0].set_ylim(0, 58)
    set_tick_size(axes[0])

    real = df[df["dataset_type"].eq("real")]
    real_counts = real.groupby("dataset_group")["dataset_label"].nunique().sort_values()
    ranked_barh(axes[1], real_counts, "Real dataset groups", "datasets", colors=[PALETTE["blue"]] * len(real_counts), top=18)

    sim = df[df["dataset_type"].eq("simulated")]
    sim_counts = sim.groupby("param_group")["dataset_label"].nunique().sort_values()
    ranked_barh(axes[2], sim_counts, "Simulated perturbation axes", "configurations", colors=[PALETTE["gold"]] * len(sim_counts))

    cell_scale = df.dropna(subset=["cells", "dataset_type"]).copy()
    sns.boxplot(data=cell_scale, x="dataset_type", y="cells", ax=axes[3], color="#E5E9ED", linewidth=0.55, fliersize=0.8)
    palette = {"real": PALETTE["blue"], "simulated": PALETTE["gold"]}
    sns.stripplot(data=cell_scale, x="dataset_type", y="cells", ax=axes[3], palette=palette, size=2.2, alpha=0.70, jitter=0.22, hue="dataset_type", legend=False)
    axes[3].set_yscale("log")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("cells")
    axes[3].set_title("Cell-count range by atlas layer", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[3].grid(axis="y", color=PALETTE["light"], lw=0.45, which="major")
    set_tick_size(axes[3])
    source = save_panel_data(df, "Supplementary_Figure_S2_full_dataset_atlas_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S2_full_100_dataset_atlas_polished", source)
    decision("S2", "polished", "Retained atlas accounting but strengthened real/simulated scale and perturbation-axis evidence.", "Figure 2")
    add_legend("S2", "Full 100-dataset benchmark atlas.", "Supplementary Fig. S2 provides dataset-level accounting for the 50 real and 50 simulated datasets, separating manuscript atlas counts from scale and perturbation-axis summaries.")


def figure_s3() -> None:
    df = read_csv("Supplementary_Figure_S3_real_dataset_landscape_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    cats = sorted(df["category"].dropna().astype(str).unique())
    cat_colors = dict(zip(cats, sns.color_palette("Set2", n_colors=len(cats))))
    for cat, group in df.groupby("category"):
        axes[0].scatter(group["cells"], group["genes"], s=25, color=cat_colors[str(cat)], alpha=0.80, edgecolor="white", lw=0.3, label=str(cat))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel("genes")
    axes[0].set_title("Cell-gene scale by real-data category", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[0].grid(True, color=PALETTE["light"], lw=0.45)
    axes[0].legend(fontsize=4.6, ncol=2, loc="lower right")
    set_tick_size(axes[0])

    sns.boxplot(data=df, x="category", y="sparsity_pct", ax=axes[1], color="#DDE7EF", linewidth=0.55, fliersize=1.2)
    sns.stripplot(data=df, x="category", y="sparsity_pct", ax=axes[1], color=PALETTE["blue"], size=1.7, alpha=0.55, jitter=0.22)
    axes[1].set_title("Sparsity by dataset category", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("sparsity (%)")
    axes[1].tick_params(axis="x", rotation=30)
    set_tick_size(axes[1])

    top_cells = df.set_index("dataset")["cells"].sort_values().tail(16)
    ranked_barh(axes[2], top_cells, "Largest real datasets", "cells", colors=[PALETTE["blue"]] * len(top_cells))
    axes[2].set_xscale("log")

    type_counts = df.groupby("category")["dataset"].nunique().sort_values()
    ranked_barh(axes[3], type_counts, "Category coverage", "datasets", colors=[PALETTE["teal"]] * len(type_counts))
    source = save_panel_data(df, "Supplementary_Figure_S3_real_dataset_landscape_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S3_real_dataset_landscape_polished", source)
    decision("S3", "polished", "Reframed from generic scatter/boxplots into scale, sparsity, largest-dataset, and category-coverage evidence.", "Figure 2")
    add_legend("S3", "Real-dataset landscape.", "Supplementary Fig. S3 expands the real-data layer of Figure 2 by showing dataset scale, sparsity, largest datasets, and category coverage for the real scRNA-seq atlas.")


def figure_s4() -> None:
    df = read_csv("Supplementary_Figure_S4_simulated_parameter_landscape_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)

    param_counts = df.groupby("param_group")["dataset_label"].nunique().sort_values()
    ranked_barh(axes[0], param_counts, "Simulated perturbation-axis coverage", "configurations", colors=[PALETTE["gold"]] * len(param_counts))

    for group, sub in df.groupby("param_group"):
        axes[1].scatter(sub["cells"], sub["genes"], s=24, alpha=0.72, edgecolor="white", lw=0.3, label=group)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("genes")
    axes[1].set_title("Scale of simulated configurations", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[1].grid(True, color=PALETTE["light"], lw=0.45)
    axes[1].legend(fontsize=4.4, ncol=2, loc="lower right")
    set_tick_size(axes[1])

    sns.boxplot(data=df, x="param_group", y="sparsity_pct", ax=axes[2], color="#EEE4C9", linewidth=0.55, fliersize=1.0)
    sns.stripplot(data=df, x="param_group", y="sparsity_pct", ax=axes[2], color=PALETTE["gold"], size=1.8, alpha=0.62, jitter=0.22)
    axes[2].set_title("Sparsity by perturbation axis", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("sparsity (%)")
    axes[2].tick_params(axis="x", rotation=35)
    set_tick_size(axes[2])

    metrics = df.groupby("param_group")[["cells", "genes", "sparsity_pct", "n_batches", "n_groups"]].median().fillna(0)
    metrics = metrics.loc[param_counts.index]
    normalized = metrics.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0, axis=0)
    heatmap(axes[3], normalized, "Perturbation-axis profile", cmap=GOLD_CMAP, cbar_label="scaled median", xrot=35)
    source = save_panel_data(df, "Supplementary_Figure_S4_simulated_parameter_landscape_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S4_simulated_parameter_landscape_polished", source)
    decision("S4", "polished", "Kept simulated atlas content but replaced sparse EDA layout with axis coverage and scaled parameter profile.", "Figure 2/Figure 7")
    add_legend("S4", "Simulated-dataset parameter landscape.", "Supplementary Fig. S4 maps the simulated dataset configurations across perturbation axes, scale, sparsity, and axis-level parameter profiles.")


def figure_s5() -> None:
    df = read_csv("Supplementary_Figure_S5_full_score_matrix_source_data.csv")
    long = collapse_parent(df)
    score = long.groupby(["parent_method", "score_domain"], observed=True)["score"].median().reset_index()
    matrix = score.pivot(index="parent_method", columns="score_domain", values="score").reindex(CANONICAL)
    domain_order = [c for c in ["local", "global", "kmeans", "louvain", "spectral", "runtime_score", "memory_score", "stability_median", "overall_mean"] if c in matrix.columns]
    matrix = matrix[domain_order]

    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.45, 1.0], hspace=0.12, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a", x=-0.035)
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")

    heatmap(ax_a, matrix, "Full normalized score matrix across 26 methods", cmap=SCORE_CMAP, cbar_label="normalized score", xrot=35)

    domain_mean = matrix.mean(axis=0).sort_values()
    ranked_barh(ax_b, domain_mean, "Mean score by component", "mean score", colors=[PALETTE["teal"]] * len(domain_mean))
    ax_b.set_xlim(0, 1)

    method_mean = matrix.mean(axis=1).reindex(CANONICAL)
    y = np.arange(len(method_mean))
    ax_c.hlines(y, 0, method_mean.values, color="#D8D8D8", lw=0.7)
    ax_c.scatter(method_mean.values, y, s=18, color=[family_color(m) for m in method_mean.index], edgecolor="white", lw=0.25)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([display_method(m) for m in method_mean.index], fontsize=4.4)
    ax_c.set_xlabel("mean displayed score")
    ax_c.set_title("Method profile mean", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax_c.set_xlim(0, 1)
    ax_c.grid(axis="x", color=PALETTE["light"], lw=0.45)
    ax_c.invert_yaxis()
    source = save_panel_data(score, "Supplementary_Figure_S5_full_score_matrix_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S5_full_score_matrix_polished", source)
    decision("S5", "redesigned", "Rebuilt as a canonical 26-method score matrix with component and method-level summaries.", "Figure 3")
    add_legend("S5", "Full benchmark score matrix.", "Supplementary Fig. S5 reports the complete normalized score matrix underlying Figure 3, with rows in canonical 26-method order and scores displayed on a direction-consistent [0, 1] scale.")


def figure_s6() -> None:
    df = read_csv("Supplementary_Figure_S6_local_neighborhood_source_data.csv")
    df = df[df["method_id"].isin(CANONICAL)].set_index("method_id").reindex(CANONICAL)
    metric_cols = [c for c in df.columns if c != "method_id"]
    matrix = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    groups = {
        "KNN": [c for c in metric_cols if c.startswith("knn")],
        "NKR": [c for c in metric_cols if c.startswith("nkr")],
        "AJI": [c for c in metric_cols if c.startswith("aji")],
        "T/C": [c for c in metric_cols if c.startswith("T_") or c.startswith("C_")],
        "MRRE": [c for c in metric_cols if c.startswith("Mrre")],
        "NH": [c for c in metric_cols if c.startswith("nh")],
        "SVM": [c for c in metric_cols if c == "svm"],
    }
    group_summary = pd.DataFrame({k: matrix[v].mean(axis=1) for k, v in groups.items() if v})
    family_summary = group_summary.assign(family=[FAMILY_MAP.get(m, "other") for m in group_summary.index]).groupby("family").median()
    family_summary.index = [family_short(x) for x in family_summary.index]

    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.45, 1.0], hspace=0.12, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a", x=-0.035)
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")
    heatmap(ax_a, matrix, "Local neighborhood-retention metric matrix", cmap=SCORE_CMAP, cbar_label="metric value", xrot=45)
    heatmap(ax_b, family_summary, "Family median by local metric group", cmap=SCORE_CMAP, cbar_label="median", xrot=35, annotate=True)
    method_mean = group_summary.mean(axis=1)
    y = np.arange(len(method_mean))
    ax_c.hlines(y, 0, method_mean.values, color="#D8D8D8", lw=0.7)
    ax_c.scatter(method_mean.values, y, s=18, color=[family_color(m) for m in method_mean.index], edgecolor="white", lw=0.25)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([display_method(m) for m in method_mean.index], fontsize=4.4)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("mean local metric")
    ax_c.set_title("Method-level local summary", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax_c.set_xlim(0, 1)
    ax_c.grid(axis="x", color=PALETTE["light"], lw=0.45)
    source = save_panel_data(group_summary.reset_index(names="method_id"), "Supplementary_Figure_S6_local_neighborhood_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S6_local_neighborhood_metrics_polished", source)
    decision("S6", "redesigned", "Rebuilt as a complete local-metric matrix plus family and method summaries.", "Figure 4")
    add_legend("S6", "Local neighborhood-retention metrics.", "Supplementary Fig. S6 reports local structure-preservation metrics across the canonical 26 methods and summarizes them by metric group and method family.")


def figure_s7() -> None:
    df = read_csv("Supplementary_Figure_S7_global_geometry_source_data.csv")
    df = df[df["method_id"].isin(CANONICAL)].set_index("method_id").reindex(CANONICAL)
    metric_cols = [c for c in df.columns if c != "method_id"]
    matrix = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    family_summary = matrix.assign(family=[FAMILY_MAP.get(m, "other") for m in matrix.index]).groupby("family").median()
    family_summary.index = [family_short(x) for x in family_summary.index]

    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.3, 1.0], hspace=0.12, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a", x=-0.035)
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")
    heatmap(ax_a, matrix, "Global geometry-preservation metric matrix", cmap=SCORE_CMAP, cbar_label="metric value", xrot=35)
    heatmap(ax_b, family_summary, "Family median by global metric", cmap=SCORE_CMAP, cbar_label="median", xrot=35, annotate=True)
    metric_mean = matrix.mean(axis=0).sort_values()
    ranked_barh(ax_c, metric_mean, "Mean value by global metric", "mean metric value", colors=[PALETTE["teal"]] * len(metric_mean))
    ax_c.set_xlim(0, 1)
    source = save_panel_data(matrix.reset_index(names="method_id"), "Supplementary_Figure_S7_global_geometry_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S7_global_geometry_metrics_polished", source)
    decision("S7", "redesigned", "Rebuilt as canonical global-metric matrix with family and metric-level summaries.", "Figure 4")
    add_legend("S7", "Global geometry-preservation metrics.", "Supplementary Fig. S7 reports global geometry-preservation metrics supporting Figure 4, with canonical method ordering and family-level summaries.")


def figure_s8() -> None:
    df = read_csv("Supplementary_Figure_S8_clustering_metrics_source_data.csv")
    df = collapse_parent(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    metric_summary = df.groupby(["parent_method", "metric"], observed=True)["value"].median().reset_index()
    metric_matrix = metric_summary.pivot(index="parent_method", columns="metric", values="value").reindex(CANONICAL)
    metric_matrix = metric_matrix[[c for c in ["ARI", "NMI", "HOMO", "COMP", "SIL"] if c in metric_matrix.columns]]
    ari_alg = df[df["metric"].eq("ARI")].groupby(["parent_method", "clustering_algorithm"], observed=True)["value"].median().reset_index()
    ari_matrix = ari_alg.pivot(index="parent_method", columns="clustering_algorithm", values="value").reindex(CANONICAL)
    pair = (
        df[df["metric"].isin(["ARI", "NMI"])]
        .pivot_table(index=["parent_method", "dataset_id", "clustering_algorithm"], columns="metric", values="value", aggfunc="median")
        .reset_index()
        .dropna(subset=["ARI", "NMI"])
    )

    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.35, 1.0], hspace=0.12, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)
    heatmap(ax_a, metric_matrix, "Clustering metric medians", cmap=SCORE_CMAP, cbar_label="median", xrot=35)
    heatmap(ax_b, ari_matrix, "ARI by clustering algorithm", cmap=SCORE_CMAP, cbar_label="median ARI", xrot=35)
    for fam, group in pair.assign(family=pair["parent_method"].map(FAMILY_MAP)).groupby("family"):
        ax_c.scatter(group["ARI"], group["NMI"], s=9, alpha=0.35, color=FAMILY_COLORS.get(fam, PALETTE["gray"]), label=family_short(fam), edgecolor="none")
    ax_c.set_xlabel("ARI")
    ax_c.set_ylabel("NMI")
    ax_c.set_title("ARI-NMI agreement across tasks", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax_c.grid(True, color=PALETTE["light"], lw=0.45)
    ax_c.legend(fontsize=4.6, loc="lower right")
    set_tick_size(ax_c)
    top_freq = (
        df[df["metric"].eq("ARI")]
        .groupby(["dataset_id", "clustering_algorithm"], observed=True)
        .apply(lambda g: g.nlargest(3, "value")["parent_method"].tolist(), include_groups=False)
    )
    counts = pd.Series([m for methods in top_freq for m in methods]).value_counts().reindex(CANONICAL).fillna(0)
    ranked_barh(ax_d, counts, "Top-three ARI frequency", "task slots", colors={m: family_color(m) for m in counts.index}, top=16)
    source = save_panel_data(pd.concat([metric_summary.assign(panel="metric_medians"), ari_alg.assign(panel="ari_by_algorithm")], ignore_index=True, sort=False), "Supplementary_Figure_S8_clustering_metrics_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S8_clustering_metrics_polished", source)
    decision("S8", "redesigned", "Reduced dense raw clustering heatmap into metric medians, algorithm sensitivity, agreement, and repeated top-performance evidence.", "Figure 5")
    add_legend("S8", "Clustering and annotation-derived label concordance.", "Supplementary Fig. S8 expands Figure 5 by reporting clustering metric medians, ARI sensitivity across clustering algorithms, ARI-NMI agreement, and repeated top-three ARI frequency across dataset-algorithm tasks.")


def figure_s9() -> None:
    df = read_csv("Supplementary_Figure_S9_efficiency_scaling_source_data.csv")
    df = collapse_parent(df)
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    line_scaling(axes[0], df, "runtime_seconds", "Runtime scaling", "runtime (s)")
    line_scaling(axes[1], df, "peak_memory_gb", "Peak-memory scaling", "memory (GB)")
    completion = df.assign(run=1).pivot_table(index="parent_method", columns="n_cells", values="run", aggfunc="max").reindex(CANONICAL)
    completion = completion.reindex(sorted(completion.columns), axis=1)
    binary_matrix(axes[2], completion.fillna(0), "Run-completion matrix", xrot=35)
    endpoint = df.sort_values("n_cells").groupby("parent_method", observed=True).tail(1)
    for fam, group in endpoint.assign(family=endpoint["parent_method"].map(FAMILY_MAP)).groupby("family"):
        axes[3].scatter(group["runtime_seconds"], group["peak_memory_gb"], s=30, color=FAMILY_COLORS.get(fam, PALETTE["gray"]), alpha=0.82, edgecolor="white", lw=0.35, label=family_short(fam))
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("runtime at largest completed scale (s)")
    axes[3].set_ylabel("peak memory (GB)")
    axes[3].set_title("Largest completed endpoint", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[3].legend(fontsize=4.7, loc="lower right")
    axes[3].grid(True, color=PALETTE["light"], lw=0.45)
    set_tick_size(axes[3])
    source = save_panel_data(df, "Supplementary_Figure_S9_efficiency_scaling_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S9_efficiency_scaling_polished", source)
    decision("S9", "polished", "Aligned efficiency supplement with Figure 8 while preserving full scaling and completion evidence.", "Figure 8")
    add_legend("S9", "Efficiency scaling.", "Supplementary Fig. S9 expands Figure 8 by reporting runtime and memory trajectories, run-completion coverage, and largest completed endpoints for the evaluated methods.")


def figure_s10() -> None:
    completed_stability = ROOT / "Publication/paper/revision_figures/figure3_polish/source_data/Figure_3_completed_stability_scores_long.csv"
    if completed_stability.exists():
        df = pd.read_csv(completed_stability).rename(columns={"Method": "parent_method"})
        df["parent_method"] = df["parent_method"].replace(
            {
                "TSNE": "t-SNE",
                "SQuaD_MDS": "SQuaD-MDS",
                "ivis": "IVIS",
                "ParametricUMAP50": "Parametric UMAP 50",
                "ParametricUMAP200": "Parametric UMAP 200",
                "SQuaD_MDS_hybrid": "SQuaD-MDS hybrid",
            }
        )
    else:
        df = read_csv("Supplementary_Figure_S10_stability_source_data.csv")
    df = collapse_parent(df)
    stab = df.groupby(["parent_method", "perturbation_axis"], observed=True)["score"].median().reset_index()
    matrix = stab.pivot(index="parent_method", columns="perturbation_axis", values="score").reindex(CANONICAL)
    family_summary = matrix.assign(family=[FAMILY_MAP.get(m, "other") for m in matrix.index]).groupby("family").median()
    family_summary.index = [family_short(x) for x in family_summary.index]
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.45, 1.0], hspace=0.12, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a", x=-0.035)
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")
    heatmap(ax_a, matrix, "Stability matrix across simulated perturbations", cmap=SCORE_CMAP, cbar_label="stability score", xrot=35)
    heatmap(ax_b, family_summary, "Family median stability", cmap=SCORE_CMAP, cbar_label="median", xrot=35, annotate=True)
    method_mean = matrix.mean(axis=1).reindex(CANONICAL)
    y = np.arange(len(method_mean))
    ax_c.hlines(y, 0, method_mean.values, color="#D8D8D8", lw=0.7)
    ax_c.scatter(method_mean.values, y, s=18, color=[family_color(m) for m in method_mean.index], edgecolor="white", lw=0.25)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([display_method(m) for m in method_mean.index], fontsize=4.4)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("mean stability")
    ax_c.set_title("Method-level stability summary", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax_c.set_xlim(0, 1)
    ax_c.grid(axis="x", color=PALETTE["light"], lw=0.45)
    source = save_panel_data(stab, "Supplementary_Figure_S10_stability_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S10_stability_full_matrix_polished", source)
    decision("S10", "redesigned", "Rebuilt as canonical 26-method stability matrix plus family and method-level summaries.", "Figure 7/Figure 3")
    add_legend("S10", "Stability matrix across simulated perturbations.", "Supplementary Fig. S10 reports stability scores across simulated perturbation axes in canonical method order, supporting the stability components in Figure 3 and the robustness analysis in Figure 7.")


def box_by_numeric(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, title: str, color: str) -> None:
    plot = df.dropna(subset=[x, y]).copy()
    plot[x] = pd.Categorical(plot[x].astype(int).astype(str), categories=sorted(plot[x].dropna().astype(int).astype(str).unique(), key=lambda z: int(z)), ordered=True)
    sns.boxplot(data=plot, x=x, y=y, ax=ax, color="#DFE8EF", linewidth=0.55, fliersize=0.8)
    sns.stripplot(data=plot, x=x, y=y, ax=ax, color=color, size=1.4, alpha=0.45, jitter=0.22)
    ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold", pad=3)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    set_tick_size(ax)


def figure_s11() -> None:
    df = read_csv("Supplementary_Figure_S11_scVI_reference_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.13, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    box_by_numeric(axes[0], df, "dimension", "ari", "scVI ARI by latent dimension", PALETTE["violet"])
    box_by_numeric(axes[1], df, "dimension", "nmi", "scVI NMI by latent dimension", PALETTE["violet"])
    box_by_numeric(axes[2], df, "dimension", "trustworthiness_k30", "scVI trustworthiness", PALETTE["violet"])
    runtime = df.dropna(subset=["dimension", "runtime_seconds"])
    box_by_numeric(axes[3], runtime, "dimension", "runtime_seconds", "Runtime by latent dimension", PALETTE["violet"])
    axes[3].set_yscale("log")
    source = save_panel_data(df, "Supplementary_Figure_S11_scVI_reference_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S11_scVI_reference_analysis_polished", source)
    decision("S11", "polished", "Kept targeted scVI reference but improved metric organization and runtime scaling.", "Figure 6")
    add_legend("S11", "scVI reference analysis.", "Supplementary Fig. S11 reports the targeted scVI reference analysis across latent dimensions. scVI is shown as an additional reference and is not counted among the 26 full-benchmark methods.")


def dimension_heatmap(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, cmap=SCORE_CMAP, log: bool = False) -> pd.DataFrame:
    plot = df.dropna(subset=["parent_method", "dimension", metric]).copy()
    order = [m for m in CANONICAL if m in set(plot["parent_method"].astype(str))] + [m for m in ["scVI"] if m in set(plot["parent_method"].astype(str))]
    plot["parent_method"] = pd.Categorical(plot["parent_method"].astype(str), categories=order, ordered=True)
    table = plot.groupby(["parent_method", "dimension"], observed=True)[metric].median().reset_index()
    matrix = table.pivot(index="parent_method", columns="dimension", values=metric).reindex(order)
    matrix = matrix.reindex(sorted(matrix.columns, key=lambda x: int(x)), axis=1)
    if log:
        values = np.log10(matrix.replace(0, np.nan))
        heatmap(ax, values, title, cmap=GOLD_CMAP, vmin=None, vmax=None, cbar_label="log10 value", xrot=0)
    else:
        heatmap(ax, matrix, title, cmap=cmap, cbar_label=METRIC_LABELS.get(metric, metric), xrot=0)
    return table


def figure_s12() -> None:
    df = read_csv("Supplementary_Figure_S12_dimension_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    tables = []
    for ax, metric, title, log in [
        (axes[0], "ari", "ARI by latent dimension", False),
        (axes[1], "nmi", "NMI by latent dimension", False),
        (axes[2], "trustworthiness_k30", "Trustworthiness by latent dimension", False),
        (axes[3], "runtime_seconds", "Runtime by latent dimension", True),
    ]:
        tables.append(dimension_heatmap(ax, df, metric, title, log=log).assign(metric=metric))
    source = save_panel_data(pd.concat(tables, ignore_index=True, sort=False), "Supplementary_Figure_S12_dimension_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S12_latent_dimension_sensitivity_polished", source)
    decision("S12", "redesigned", "Converted boxplot grid into compact method-by-dimension heatmaps for easier comparison.", "Figure 6")
    add_legend("S12", "Latent-dimension sensitivity.", "Supplementary Fig. S12 reports method-level sensitivity to latent dimensionality for selected methods, separating performance metrics from runtime.")


def figure_s13() -> None:
    df = read_csv("Supplementary_Figure_S13_workflow_comparison_source_data.csv")
    metrics = ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"]
    records = []
    for metric in metrics:
        pivot = df.pivot_table(index=["parent_method", "dataset_id", "seed"], columns="workflow", values=metric, aggfunc="median").reset_index()
        if {"Direct 2D", "PCA50 to 2D"}.issubset(pivot.columns):
            if metric == "runtime_seconds":
                pivot["delta"] = np.log10((pivot["PCA50 to 2D"] + 1e-6) / (pivot["Direct 2D"] + 1e-6))
                ylab = "log10 runtime ratio"
            else:
                pivot["delta"] = pivot["PCA50 to 2D"] - pivot["Direct 2D"]
                ylab = "PCA50 - direct"
            records.append(pivot.assign(metric=metric, y_label=ylab))
    delta = pd.concat(records, ignore_index=True, sort=False)
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.14, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    for ax, metric in zip(axes, metrics):
        sub = delta[delta["metric"].eq(metric)]
        sns.boxplot(data=sub, x="parent_method", y="delta", ax=ax, color="#E3EAF1", linewidth=0.55, fliersize=0.8)
        sns.stripplot(
            data=sub,
            x="parent_method",
            y="delta",
            hue="parent_method",
            ax=ax,
            palette={m: family_color(m) for m in sub["parent_method"].unique()},
            size=1.8,
            alpha=0.55,
            jitter=0.20,
            legend=False,
        )
        ax.axhline(0, color=PALETTE["dark"], lw=0.55, ls="--")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} workflow effect", loc="left", fontsize=7.1, fontweight="bold", pad=3)
        ax.set_xlabel("")
        ax.set_ylabel(sub["y_label"].iloc[0] if not sub.empty else "delta")
        ax.tick_params(axis="x", rotation=35)
        set_tick_size(ax)
    source = save_panel_data(delta, "Supplementary_Figure_S13_workflow_comparison_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S13_visualization_workflow_comparison_polished", source)
    decision("S13", "polished", "Reframed direct-vs-PCA50 workflow as signed deltas to clarify effect direction.", "Figure 6")
    add_legend("S13", "Visualization-workflow comparison.", "Supplementary Fig. S13 compares direct two-dimensional embedding with PCA50-to-2D workflows for selected visualization methods; signed deltas show the PCA50 workflow relative to direct 2D.")


def figure_s14() -> None:
    df = read_csv("Supplementary_Figure_S14_input_gene_source_data.csv")
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    tables = []
    order = [m for m in CANONICAL if m in set(df["parent_method"].astype(str))] + [m for m in ["scVI"] if m in set(df["parent_method"].astype(str))]
    for ax, metric, title, log in [
        (axes[0], "ari", "ARI by input-gene setting", False),
        (axes[1], "nmi", "NMI by input-gene setting", False),
        (axes[2], "trustworthiness_k30", "Trustworthiness by input genes", False),
        (axes[3], "runtime_seconds", "Runtime by input genes", True),
    ]:
        plot = df.dropna(subset=["parent_method", "hvg_requested", metric]).copy()
        plot["parent_method"] = pd.Categorical(plot["parent_method"].astype(str), categories=order, ordered=True)
        table = plot.groupby(["parent_method", "hvg_requested"], observed=True)[metric].median().reset_index()
        matrix = table.pivot(index="parent_method", columns="hvg_requested", values=metric).reindex(order)
        matrix = matrix.reindex(sorted(matrix.columns, key=lambda x: int(x)), axis=1)
        if log:
            heatmap(ax, np.log10(matrix.replace(0, np.nan)), title, cmap=GOLD_CMAP, vmin=None, vmax=None, cbar_label="log10 runtime", xrot=0)
        else:
            heatmap(ax, matrix, title, cmap=SCORE_CMAP, cbar_label=METRIC_LABELS.get(metric, metric), xrot=0)
        tables.append(table.assign(metric=metric))
    source = save_panel_data(pd.concat(tables, ignore_index=True, sort=False), "Supplementary_Figure_S14_input_gene_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S14_input_gene_sensitivity_polished", source)
    decision("S14", "redesigned", "Converted boxplot grid into method-by-input-gene heatmaps for compact sensitivity comparison.", "Figure 6")
    add_legend("S14", "Input-gene sensitivity.", "Supplementary Fig. S14 reports the effect of highly variable gene settings on performance metrics and runtime for targeted sensitivity analyses.")


def figure_s15() -> None:
    df = read_csv("Supplementary_Figure_S15_reproducibility_audit_source_data.csv")
    install = df[df["source"].eq("install_manifest")].copy()
    methods = df[df["source"].eq("method_manifest")].copy()
    commits = df[df["source"].eq("source_commits")].copy()
    fig = plt.figure(figsize=(7.2, 8.4), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    if not install.empty:
        install["status_display"] = install["status"].replace(
            {
                "installed_verified": "Install verified",
                "source_import_verified": "Source import verified",
                "Source import verified": "Source import verified",
                "source_import_partial": "Source import partial",
            }
        )
        status = install.pivot_table(index="language", columns="status_display", values="method", aggfunc="count", fill_value=0)
        status_colors = {
            "Install verified": "#CFCFCF",
            "Source import verified": PALETTE["blue"],
            "Source import partial": PALETTE["rose"],
        }
        status.plot(kind="bar", stacked=True, ax=axes[0], color=[status_colors.get(c, PALETTE["gray"]) for c in status.columns], width=0.72, edgecolor="white", lw=0.4)
    axes[0].set_title("Implementation verification status", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[0].set_ylabel("methods")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=4.7)
    set_tick_size(axes[0])

    role_counts = methods["benchmark_scope"].value_counts().sort_values()
    role_color_map = {"Full benchmark": PALETTE["blue"], "Result variants": PALETTE["gold"], "Targeted analysis": PALETTE["gray"]}
    ranked_barh(axes[1], role_counts, "Benchmark-role accounting", "entries", colors=role_color_map)

    n_commits = int(commits["commit"].dropna().nunique())
    axes[2].barh(["commit hashes"], [n_commits], color=PALETTE["teal"], edgecolor="white", lw=0.4)
    axes[2].set_title("Source commit records", loc="left", fontsize=7.1, fontweight="bold", pad=3)
    axes[2].set_xlabel("records")
    axes[2].set_xlim(0, max(1, n_commits) + 2)
    axes[2].text(n_commits + 0.25, 0, str(n_commits), va="center", fontsize=5.4)
    axes[2].grid(axis="x", color=PALETTE["light"], lw=0.45)
    set_tick_size(axes[2])

    rows = []
    for path in sorted(SOURCE_IN.glob("Supplementary_Figure_S*_source_data.csv")):
        rows.append({"supplementary_figure": path.name.split("_source_data")[0].replace("Supplementary_Figure_", ""), "rows": len(pd.read_csv(path, usecols=[0]))})
    source_counts = pd.DataFrame(rows)
    ranked_barh(axes[3], source_counts.set_index("supplementary_figure")["rows"], "Source-data records by supplementary figure", "rows", colors=[PALETTE["gold"]] * len(source_counts), top=15)
    axes[3].set_xscale("log")
    source = save_panel_data(pd.concat([df.assign(panel="audit_records"), source_counts.assign(panel="source_data_rows")], ignore_index=True, sort=False), "Supplementary_Figure_S15_reproducibility_audit_polished_panel_data.csv")
    save_figure(fig, FIG_OUT / "Supplementary_Figure_S15_reproducibility_source_audit_polished", source)
    decision("S15", "redesigned", "Strengthened reproducibility audit with implementation status, role accounting, commit records, and source-data coverage.", "Figure 8/Figure 9")
    add_legend("S15", "Reproducibility and source-data audit.", "Supplementary Fig. S15 summarizes implementation verification, benchmark-role accounting, source commit records, and source-data record coverage for the supplementary figure package.")


def write_qa() -> None:
    pd.DataFrame(MANIFEST_ROWS).to_csv(QA_OUT / "figure_export_manifest.csv", index=False)
    pd.DataFrame(PANEL_DECISIONS).to_csv(QA_OUT / "supplementary_polish_decision_audit.csv", index=False)
    legend_lines = ["# Polished Supplementary Figure Legends", ""]
    for item in LEGENDS:
        legend_lines.extend([f"## Supplementary Fig. {item['figure']}. {item['title']}", item["legend"], ""])
    (QA_OUT / "polished_supplementary_figure_legends.md").write_text("\n".join(legend_lines), encoding="utf-8")

    audit_rows = []
    for source in sorted(SRC_OUT.glob("Supplementary_Figure_S*_polished_panel_data.csv")):
        df = pd.read_csv(source, nrows=2000)
        method_col = "parent_method" if "parent_method" in df.columns else "method_id" if "method_id" in df.columns else None
        if method_col:
            methods = list(dict.fromkeys(df[method_col].dropna().astype(str).tolist()))
            canonical_methods = [m for m in methods if m in set(CANONICAL)]
            extra_methods = sorted(set(methods) - set(CANONICAL))
            canonical_set = set(canonical_methods)
            if canonical_set == set(CANONICAL) and not extra_methods:
                status = "canonical_26_set_figure_reindexed"
            elif canonical_set.issubset(set(CANONICAL)) and not extra_methods:
                status = "canonical_subset_figure_reindexed"
            elif extra_methods == ["scVI"] and canonical_set == set(CANONICAL):
                status = "canonical_plus_scVI_targeted_reference"
            elif extra_methods == ["scVI"] and not canonical_set:
                status = "scVI_targeted_reference_only"
            elif extra_methods == ["scVI"] and canonical_set.issubset(set(CANONICAL)):
                status = "canonical_subset_plus_scVI_targeted_reference"
            else:
                status = "check"
        else:
            methods = []
            extra_methods = []
            status = "not_method_axis"
        audit_rows.append(
            {
                "source_data": source.name,
                "method_column": method_col or "",
                "method_count_sample": len(set(methods)),
                "extra_methods": ";".join(extra_methods),
                "order_status": status,
            }
        )
    pd.DataFrame(audit_rows).to_csv(QA_OUT / "canonical_method_order_audit.csv", index=False)


def make_contact_sheet() -> None:
    pngs = sorted(FIG_OUT.glob("*_polished.png"))
    if not pngs:
        return
    cols = 3
    rows = int(np.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.6))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, pngs):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem.replace("_polished", ""), fontsize=6.2, loc="left")
        clean_axis(ax)
    for ax in axes[len(pngs) :]:
        clean_axis(ax)
    fig.savefig(QA_OUT / "supplementary_polished_contact_sheet.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for func in [
        figure_s1,
        figure_s2,
        figure_s3,
        figure_s4,
        figure_s5,
        figure_s6,
        figure_s7,
        figure_s8,
        figure_s9,
        figure_s10,
        figure_s11,
        figure_s12,
        figure_s13,
        figure_s14,
        figure_s15,
    ]:
        func()
    write_qa()
    make_contact_sheet()
    print(f"Wrote polished supplementary figures to {FIG_OUT}")


if __name__ == "__main__":
    main()
