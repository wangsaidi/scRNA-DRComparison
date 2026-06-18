from __future__ import annotations

from pathlib import Path
import importlib.util
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def find_project_root(start: Path) -> Path:
    for parent in [start.resolve(), *start.resolve().parents]:
        if (parent / "Publication" / "paper").exists() and (parent / "metadata").exists():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


ROOT = find_project_root(Path(__file__))
SOURCE_IN = ROOT / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data"
POLISH_SCRIPT = ROOT / "Publication/paper/revision_figures/supplementary_polish/make_supplementary_figures_top_tier.py"
OUT = ROOT / "Publication/paper/revision_figures/supplementary_atlas_top_tier"
FIG_OUT = OUT / "supplementary_figures"
SRC_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for directory in [FIG_OUT, SRC_OUT, QA_OUT]:
    directory.mkdir(parents=True, exist_ok=True)


spec = importlib.util.spec_from_file_location("supplementary_polish_base", POLISH_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 5.7,
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "savefig.facecolor": "white",
    }
)

CANONICAL = base.CANONICAL
ORDER_MAP = base.ORDER_MAP
PALETTE = base.PALETTE
FAMILY_COLORS = base.FAMILY_COLORS
FAMILY_LABELS = base.FAMILY_LABELS
SCORE_CMAP = base.SCORE_CMAP
GOLD_CMAP = base.GOLD_CMAP
ROSE_CMAP = base.ROSE_CMAP
BINARY_CMAP = base.BINARY_CMAP
METHOD_LABEL_REPLACEMENTS = base.METHOD_LABEL_REPLACEMENTS
FAMILY_MAP = base.FAMILY_MAP


MANIFEST_ROWS: list[dict[str, object]] = []
PANEL_ROWS: list[dict[str, object]] = []
LEGEND_ROWS: list[dict[str, str]] = []


COLLECTION_LABELS = {
    "benchmarker": "curated",
    "SIMLR": "SIMLR",
    "TI": "trajectory",
    "VASC": "VASC",
    "scDesign3": "scDesign3",
    "simulate": "simulated",
}

METRIC_TITLES = {
    "nkr": "NKR",
    "aji": "AJI",
    "knn": "KNN",
    "nh": "NH",
    "T": "trustworthiness",
    "C": "continuity",
    "Mrre_false": "MRRE false",
    "Mrre_missing": "MRRE missing",
    "svm": "SVM",
    "random_triplet": "random triplet",
    "spearman": "Spearman",
    "Pearson": "Pearson",
    "k-nearest": "k-nearest",
    "centroid_distance": "centroid distance",
    "AUC": "AUC",
    "Qglobal": "Qglobal",
    "Qlocal": "Qlocal",
    "kmax": "kmax",
}


def display_method(method: object) -> str:
    return METHOD_LABEL_REPLACEMENTS.get(str(method), str(method))


def family_short(family: object) -> str:
    return FAMILY_LABELS.get(str(family), str(family))


def family_color(method: object) -> str:
    return FAMILY_COLORS.get(FAMILY_MAP.get(str(method), "other"), FAMILY_COLORS["other"])


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(SOURCE_IN / name, **kwargs)


def normalize_method_labels(df: pd.DataFrame, col: str = "parent_method") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns and "method_id" in out.columns:
        out[col] = out["method_id"]
    if col in out.columns:
        out[col] = out[col].astype(str).replace(
            {
                "TSNE": "t-SNE",
                "SQuaD_MDS": "SQuaD-MDS",
                "ivis": "IVIS",
                "ParametricUMAP50": "Parametric UMAP 50",
                "ParametricUMAP200": "Parametric UMAP 200",
            }
        )
    return out


def collapse_canonical(df: pd.DataFrame, col: str = "parent_method") -> pd.DataFrame:
    out = normalize_method_labels(df, col)
    out = out[out[col].isin(CANONICAL)].copy()
    out["method_order"] = out[col].map(ORDER_MAP)
    return out.sort_values("method_order")


def add_collection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["collection"] = out["dataset_category"].map(COLLECTION_LABELS).fillna(out["dataset_category"])
    return out


def parse_sim_axis(dataset_id: object) -> str:
    text = str(dataset_id)
    if text == "default":
        return "default"
    if text.startswith("celltype_"):
        return "cell types"
    if text.startswith("cell_"):
        return "cells"
    if text.startswith("gene_"):
        return "genes"
    if text.startswith("dropout_"):
        return "dropout"
    if text.startswith("out_"):
        return "outliers"
    if text.startswith("de_prob_"):
        return "DE fraction"
    if text.startswith("de_"):
        return "DE strength"
    if text.startswith("batch_"):
        value = text.replace("batch_", "")
        try:
            return "batch strength" if float(value) <= 1 else "batch number"
        except ValueError:
            return "batch"
    return "other"


def public_metric(metric: str) -> str:
    for prefix, title in METRIC_TITLES.items():
        if metric == prefix or metric.startswith(f"{prefix}_"):
            return metric.replace(prefix, title, 1)
    return metric


def title_case_metric(metric: str) -> str:
    return public_metric(metric).replace("_", " ")


def clean_axis(ax: plt.Axes) -> None:
    base.clean_axis(ax)


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    base.panel_label(ax, label, x=x, y=y)


def set_tick_size(ax: plt.Axes, size: float = 4.9) -> None:
    ax.tick_params(axis="both", labelsize=size, length=2.2, width=0.45, pad=1.5)


def compact_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap=SCORE_CMAP,
    *,
    vmin: float | None = 0,
    vmax: float | None = 1,
    show_y: bool = True,
    show_x: bool = True,
    cbar: bool = True,
    annotate: bool = False,
    fmt: str = ".2f",
    xrot: float = 35,
) -> None:
    matrix = data.copy()
    if matrix.empty:
        clean_axis(ax)
        ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
        ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=6, color=PALETTE["gray"])
        return
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.18,
        linecolor="white",
        cbar=cbar,
        annot=annotate,
        fmt=fmt,
        annot_kws={"fontsize": 4.4},
        cbar_kws={"shrink": 0.46, "pad": 0.02},
    )
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_y:
        ax.set_yticklabels([display_method(t.get_text()) for t in ax.get_yticklabels()], rotation=0, fontsize=4.6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    if show_x:
        ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=xrot, ha="right", fontsize=4.7)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if cbar and ax.collections and ax.collections[0].colorbar:
        ax.collections[0].colorbar.ax.tick_params(labelsize=4.7, length=2, width=0.4)


def compact_barh(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    xlabel: str,
    *,
    color: str | list[str] | dict[str, str] = PALETTE["teal"],
    top: int | None = None,
    xlim: tuple[float, float] | None = None,
    annotate: bool = False,
) -> None:
    s = series.dropna().copy()
    if top is not None and len(s) > top:
        s = s.sort_values(ascending=False).head(top).sort_values()
    else:
        s = s.sort_values()
    colors: list[str]
    if isinstance(color, dict):
        colors = [color.get(str(idx), PALETTE["gray"]) for idx in s.index]
    elif isinstance(color, list):
        colors = color[: len(s)]
    else:
        colors = [color] * len(s)
    ax.barh([display_method(i) for i in s.index], s.values, color=colors, edgecolor="white", linewidth=0.25)
    if annotate:
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {v:.2g}", va="center", ha="left", fontsize=4.8, color=PALETTE["dark"])
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color=PALETTE["light"], lw=0.4)
    ax.set_axisbelow(True)
    set_tick_size(ax)


def compact_hist(ax: plt.Axes, values: pd.Series, title: str, xlabel: str, color: str = PALETTE["blue"], bins: int = 12) -> None:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.35, alpha=0.92)
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(axis="y", color=PALETTE["light"], lw=0.4)
    set_tick_size(ax)


def compact_box(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    color: str = "#DFE8EF",
    xrot: float = 35,
) -> None:
    plot = df.dropna(subset=[x, y]).copy()
    if plot.empty:
        clean_axis(ax)
        ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
        return
    sns.boxplot(data=plot, x=x, y=y, ax=ax, color=color, linewidth=0.55, fliersize=0.8)
    sns.stripplot(data=plot, x=x, y=y, ax=ax, color=PALETTE["slate"], alpha=0.33, size=1.5, jitter=0.2)
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=xrot)
    ax.grid(axis="y", color=PALETTE["light"], lw=0.4)
    set_tick_size(ax)


def compact_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    hue: str | None = None,
    logx: bool = False,
    logy: bool = False,
    alpha: float = 0.72,
) -> None:
    plot = df.dropna(subset=[x, y]).copy()
    if plot.empty:
        clean_axis(ax)
        ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
        return
    if hue and hue in plot.columns:
        for key, group in plot.groupby(hue, observed=True):
            ax.scatter(group[x], group[y], s=12, alpha=alpha, color=FAMILY_COLORS.get(str(key), PALETTE["gray"]), label=family_short(key), edgecolor="white", linewidth=0.25)
        ax.legend(fontsize=4.5, loc="best", ncols=1)
    else:
        ax.scatter(plot[x], plot[y], s=12, alpha=alpha, color=PALETTE["teal"], edgecolor="white", linewidth=0.25)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=3)
    ax.grid(True, color=PALETTE["light"], lw=0.4)
    set_tick_size(ax)


def metric_matrix(df: pd.DataFrame, metric: str, *, category_col: str = "collection") -> pd.DataFrame:
    plot = df[df["metric"].eq(metric)].dropna(subset=["parent_method", category_col, "value"]).copy()
    if plot.empty:
        return pd.DataFrame(index=CANONICAL)
    matrix = plot.pivot_table(index="parent_method", columns=category_col, values="value", aggfunc="median")
    matrix = matrix.reindex(CANONICAL)
    return matrix.dropna(how="all")


def method_metric_matrix(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    plot = df[df["metric"].isin(metrics)].dropna(subset=["parent_method", "metric", "value"]).copy()
    if plot.empty:
        return pd.DataFrame(index=CANONICAL)
    matrix = plot.pivot_table(index="parent_method", columns="metric", values="value", aggfunc="median").reindex(CANONICAL)
    matrix = matrix[[m for m in metrics if m in matrix.columns]]
    matrix.columns = [title_case_metric(c) for c in matrix.columns]
    return matrix.dropna(how="all")


def save_panel_data(df: pd.DataFrame, name: str) -> str:
    path = SRC_OUT / name
    df.to_csv(path, index=False)
    return path.name


def save_figure(fig: plt.Figure, name: str, source_file: str, dpi: int = 600) -> None:
    base_path = FIG_OUT / name
    for ext, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 340},
        ".tiff": {"dpi": dpi},
    }.items():
        out = base_path.with_suffix(ext)
        fig.savefig(out, bbox_inches="tight", **kwargs)
        MANIFEST_ROWS.append({"figure": name, "file": out.name, "format": ext.lstrip("."), "source_data": source_file})
    plt.close(fig)


def add_panel_record(figure: str, panel: str, role: str, source: str) -> None:
    PANEL_ROWS.append({"figure": figure, "panel": panel, "role": role, "source": source})


def add_legend(figure: str, title: str, legend: str) -> None:
    LEGEND_ROWS.append({"figure": figure, "title": title, "legend": legend})


def write_legend_file() -> None:
    lines = ["# Supplementary Atlas Figure Legends", ""]
    for item in LEGEND_ROWS:
        lines.append(f"## Supplementary Fig. {item['figure']}. {item['title']}")
        lines.append(item["legend"])
        lines.append("")
    (QA_OUT / "supplementary_atlas_figure_legends.md").write_text("\n".join(lines), encoding="utf-8")


def structure_long() -> pd.DataFrame:
    cols = ["method_id", "method_raw", "parent_method", "metric", "value", "dataset_category", "dataset_id", "structure_table"]
    df = read_csv("Figure_4_structure_preservation_source_data.csv", usecols=cols)
    df = df.dropna(subset=["metric", "value", "dataset_category", "dataset_id"]).copy()
    df = normalize_method_labels(df)
    df = df[df["parent_method"].isin(CANONICAL)].copy()
    df["method_order"] = df["parent_method"].map(ORDER_MAP)
    df = add_collection(df)
    df["sim_axis"] = df["dataset_id"].map(parse_sim_axis)
    return df.sort_values(["method_order", "dataset_category", "dataset_id", "metric"])


def clustering_long() -> pd.DataFrame:
    df = read_csv("Figure_5_clustering_concordance_source_data.csv")
    df = normalize_method_labels(df)
    df = df[df["parent_method"].isin(CANONICAL)].copy()
    df["method_order"] = df["parent_method"].map(ORDER_MAP)
    df = add_collection(df)
    df["sim_axis"] = df["dataset_id"].map(parse_sim_axis)
    return df.sort_values(["method_order", "dataset_category", "dataset_id", "clustering_algorithm", "metric"])


def score_long() -> pd.DataFrame:
    df = read_csv("Figure_3_full_benchmark_score_landscape_source_data.csv")
    df = normalize_method_labels(df)
    df = df[df["parent_method"].isin(CANONICAL) & df["score_domain"].notna()].copy()
    df["method_order"] = df["parent_method"].map(ORDER_MAP)
    return df.sort_values(["method_order", "score_domain"])


def figure_s1() -> None:
    fig_id = "S1"
    df = read_csv("Supplementary_Figure_S1_method_catalog_source_data.csv")
    df = normalize_method_labels(df)
    full = df[df["benchmark_scope"].eq("Full benchmark")].copy()
    full["parent_method"] = pd.Categorical(full["parent_method"], categories=CANONICAL, ordered=True)
    full = full.sort_values("parent_method")

    matrix_cols = ["linear", "deep", "graph", "metric", "R", "Python", "MATLAB", "source", "year"]
    matrix = pd.DataFrame(0, index=full["parent_method"].astype(str), columns=matrix_cols)
    fam_map = {
        "linear/probabilistic": "linear",
        "deep generative/autoencoder": "deep",
        "graph/diffusion": "graph",
        "metric/structure-aware": "metric",
    }
    for _, row in full.iterrows():
        matrix.loc[row["parent_method"], fam_map.get(row["method_family"], "metric")] = 1
        lang = str(row["implementation_language"]).replace("python", "Python")
        if lang in matrix.columns:
            matrix.loc[row["parent_method"], lang] = 1
        matrix.loc[row["parent_method"], "source"] = int(pd.notna(row["source_or_url"]))
        matrix.loc[row["parent_method"], "year"] = int(pd.notna(row["publication_year"]))

    fig = plt.figure(figsize=(7.3, 9.4), constrained_layout=True)
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1.45, 1.45, 1.0, 1.0], hspace=0.18, wspace=0.18)
    axes = {
        "a": fig.add_subplot(gs[:, :2]),
        "b": fig.add_subplot(gs[0, 2]),
        "c": fig.add_subplot(gs[0, 3]),
        "d": fig.add_subplot(gs[1, 2]),
        "e": fig.add_subplot(gs[1, 3]),
        "f": fig.add_subplot(gs[2, 2]),
        "g": fig.add_subplot(gs[2, 3]),
        "h": fig.add_subplot(gs[3, 2:]),
    }
    for label, ax in axes.items():
        panel_label(ax, label, x=-0.11 if label != "a" else -0.04)

    compact_heatmap(axes["a"], matrix, "Canonical 26-method catalog", cmap=BINARY_CMAP, vmin=0, vmax=1, show_y=True, cbar=False, xrot=35)
    family_counts = full["method_family"].map(family_short).value_counts()
    compact_barh(axes["b"], family_counts, "Method families", "methods", color=[FAMILY_COLORS.get(k, PALETTE["gray"]) for k in full["method_family"].unique()], annotate=True)
    lang_counts = full["implementation_language"].replace({"python": "Python"}).value_counts()
    compact_barh(axes["c"], lang_counts, "Implementation language", "methods", color=PALETTE["blue"], annotate=True)
    compact_hist(axes["d"], full["publication_year"], "Publication-year span", "year", color=PALETTE["gold"], bins=8)
    source_flags = pd.Series({"source listed": full["source_or_url"].notna().sum(), "reference listed": full["reference"].notna().sum(), "year listed": full["publication_year"].notna().sum()})
    compact_barh(axes["e"], source_flags, "Metadata completeness", "methods", color=PALETTE["teal"], annotate=True)
    fam_lang = pd.crosstab(full["method_family"].map(family_short), full["implementation_language"].replace({"python": "Python"}))
    compact_heatmap(axes["f"], fam_lang, "Family by language", cmap=GOLD_CMAP, vmin=0, vmax=max(1, fam_lang.to_numpy().max()), show_y=True, cbar=False, annotate=True, fmt=".0f", xrot=35)
    variants = df[df["benchmark_scope"].ne("Full benchmark")]["benchmark_scope"].value_counts()
    compact_barh(axes["g"], variants, "Variants and targeted analyses", "entries", color=[PALETTE["gold"], PALETTE["gray"]], annotate=True)
    timeline = full.groupby(["publication_year", full["method_family"].map(family_short)]).size().reset_index(name="n").dropna()
    for fam, group in timeline.groupby("method_family", observed=True):
        axes["h"].scatter(group["publication_year"], group["n"].cumsum(), color=FAMILY_COLORS.get(next((k for k, v in FAMILY_LABELS.items() if v == fam), ""), PALETTE["gray"]), s=18, label=fam)
    axes["h"].set_title("Method-year footprint", loc="left", fontsize=6.5, fontweight="bold", pad=3)
    axes["h"].set_xlabel("year")
    axes["h"].set_ylabel("cumulative methods")
    axes["h"].grid(True, color=PALETTE["light"], lw=0.4)
    axes["h"].legend(fontsize=4.6, ncols=4, loc="upper left")
    set_tick_size(axes["h"])

    source = save_panel_data(df, "Supplementary_Figure_S1_method_catalog_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S1_method_catalog_atlas", source)
    for panel, role in {
        "a": "canonical method membership and reproducibility metadata",
        "b": "family accounting",
        "c": "implementation language accounting",
        "d": "publication-year coverage",
        "e": "metadata completeness",
        "f": "family-language cross-tabulation",
        "g": "targeted controls and variants",
        "h": "temporal method footprint",
    }.items():
        add_panel_record(fig_id, panel, role, "method catalog")
    add_legend(fig_id, "Method catalog and inclusion audit.", "Supplementary Fig. S1 expands the method taxonomy in Figure 1 and separates the 26 full-benchmark methods from result variants and targeted reference analyses such as scVI.")


def figure_s2() -> None:
    fig_id = "S2"
    df = read_csv("Figure_2_full_dataset_landscape_source_data.csv")
    df["dataset_type"] = df["dataset_type"].fillna("unknown")
    fig, axes = plt.subplots(3, 4, figsize=(7.3, 8.8), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefghijkl", axes):
        panel_label(ax, label)

    compact_barh(axes[0], df["dataset_type"].value_counts(), "Dataset atlas composition", "datasets", color=[PALETTE["blue"], PALETTE["gold"], PALETTE["gray"]], annotate=True)
    compact_barh(axes[1], df["dataset_group"].dropna().value_counts().head(12), "Largest source groups", "datasets", color=PALETTE["blue"])
    compact_box(axes[2], df, "dataset_type", "cells", "Cell-count range", color="#DFE8EF")
    axes[2].set_yscale("log")
    compact_hist(axes[3], df["year"], "Publication years", "year", color=PALETTE["gold"], bins=12)
    compact_barh(axes[4], df["species"].dropna().astype(str).str.split("、").str[0].value_counts().head(10), "Species footprint", "datasets", color=PALETTE["rose"])
    compact_barh(axes[5], df["sequencing_technology"].dropna().value_counts().head(10), "Sequencing platforms", "datasets", color=PALETTE["teal"])
    compact_scatter(axes[6], df, "cells", "genes", "Cell-gene scale", logx=True, logy=True)
    compact_box(axes[7], df.dropna(subset=["sparsity_pct"]), "dataset_type", "sparsity_pct", "Sparsity range", color="#EFE6C8")
    sim = df[df["dataset_type"].eq("simulated")].copy()
    compact_barh(axes[8], sim["param_group"].dropna().value_counts(), "Simulated perturbation axes", "configs", color=PALETTE["gold"], annotate=True)
    compact_box(axes[9], sim.dropna(subset=["cells"]), "param_group", "cells", "Simulated cell counts", color="#EFE6C8")
    axes[9].set_yscale("log")
    compact_barh(axes[10], df.dropna(subset=["cells"]).set_index("dataset_label")["cells"].sort_values(ascending=False).head(12).sort_values(), "Largest datasets", "cells", color=PALETTE["blue"])
    axes[10].set_xscale("log")
    availability = df["availability"].fillna("available in atlas").value_counts()
    compact_barh(axes[11], availability, "Availability annotation", "entries", color=PALETTE["gray"], annotate=True)

    source = save_panel_data(df, "Supplementary_Figure_S2_full_dataset_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S2_full_dataset_atlas", source)
    for i, role in enumerate(["dataset type accounting", "source groups", "cell-count range", "publication years", "species footprint", "sequencing technologies", "cell-gene scale", "sparsity", "simulation axes", "simulated scale", "largest datasets", "availability"]):
        add_panel_record(fig_id, chr(97 + i), role, "100-dataset atlas")
    add_legend(fig_id, "Full 100-dataset benchmark atlas.", "Supplementary Fig. S2 provides the dataset-level evidence supporting Figure 2, including real and simulated dataset composition, scale, technology, and availability annotations.")


def figure_s3() -> None:
    fig_id = "S3"
    df = read_csv("Supplementary_Figure_S3_real_dataset_landscape_source_data.csv")
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 8.4), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefghi", axes):
        panel_label(ax, label)

    compact_barh(axes[0], df["category"].value_counts(), "Real-data source collections", "datasets", color=PALETTE["blue"], annotate=True)
    compact_scatter(axes[1], df.assign(collection=df["category"]), "cells", "genes", "Cell-gene landscape", hue=None, logx=True, logy=True)
    compact_box(axes[2], df, "category", "sparsity_pct", "Sparsity by collection", color="#DFE8EF")
    compact_box(axes[3], df, "category", "cell_types", "Cell-type counts", color="#EFE6C8")
    compact_barh(axes[4], df.set_index("dataset")["cells"].sort_values(ascending=False).head(12).sort_values(), "Largest real datasets", "cells", color=PALETTE["blue"])
    axes[4].set_xscale("log")
    compact_barh(axes[5], df.set_index("dataset")["genes"].sort_values(ascending=False).head(12).sort_values(), "Largest gene spaces", "genes", color=PALETTE["teal"])
    axes[5].set_xscale("log")
    compact_box(axes[6], df.dropna(subset=["size_mb"]), "category", "size_mb", "Matrix-file size", color="#E9DAD8")
    axes[6].set_yscale("log")
    coverage = df.groupby("category")[["cells", "genes", "sparsity_pct", "cell_types"]].median(numeric_only=True)
    coverage = coverage.apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0, axis=0)
    compact_heatmap(axes[7], coverage, "Scaled collection profile", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, annotate=True, xrot=35)
    completeness = df[["cells", "genes", "sparsity_pct", "cell_types", "size_mb"]].notna().sum().sort_values()
    compact_barh(axes[8], completeness, "Metadata completeness", "datasets", color=PALETTE["slate"], annotate=True)

    source = save_panel_data(df, "Supplementary_Figure_S3_real_dataset_landscape_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S3_real_dataset_landscape_atlas", source)
    for i, role in enumerate(["source collections", "cell-gene scale", "sparsity", "cell types", "largest cell counts", "largest gene spaces", "matrix size", "scaled collection profile", "metadata completeness"]):
        add_panel_record(fig_id, chr(97 + i), role, "real-dataset landscape")
    add_legend(fig_id, "Real-dataset landscape.", "Supplementary Fig. S3 expands the real-dataset portion of Figure 2 by summarizing collection-level scale, sparsity, cell-type annotation, matrix size, and metadata completeness.")


def figure_s4() -> None:
    fig_id = "S4"
    df = read_csv("Supplementary_Figure_S4_simulated_parameter_landscape_source_data.csv")
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 8.4), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefghi", axes):
        panel_label(ax, label)
    compact_barh(axes[0], df["param_group"].value_counts(), "Perturbation-axis coverage", "configs", color=PALETTE["gold"], annotate=True)
    compact_box(axes[1], df, "param_group", "cells", "Cell-count settings", color="#EFE6C8")
    axes[1].set_yscale("log")
    compact_box(axes[2], df, "param_group", "genes", "Gene-count settings", color="#DFE8EF")
    axes[2].set_yscale("log")
    compact_box(axes[3], df, "param_group", "sparsity_pct", "Sparsity settings", color="#E9DAD8")
    compact_box(axes[4], df, "param_group", "n_batches", "Batch-number settings", color="#EFE6C8")
    compact_box(axes[5], df, "param_group", "n_groups", "Cell-type settings", color="#DFE8EF")
    compact_scatter(axes[6], df, "cells", "genes", "Simulated cell-gene scale", logx=True, logy=True)
    profile = df.groupby("param_group")[["cells", "genes", "sparsity_pct", "n_batches", "n_groups"]].median(numeric_only=True)
    profile = profile.apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0, axis=0)
    compact_heatmap(axes[7], profile, "Scaled perturbation profile", cmap=GOLD_CMAP, vmin=0, vmax=1, show_y=True, annotate=True, xrot=35)
    compact_barh(axes[8], df.set_index("dataset_label")["size_mb"].dropna().sort_values(ascending=False).head(12).sort_values(), "Largest simulated matrices", "MB", color=PALETTE["gold"])
    axes[8].set_xscale("log")

    source = save_panel_data(df, "Supplementary_Figure_S4_simulated_parameter_landscape_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S4_simulated_parameter_landscape_atlas", source)
    for i, role in enumerate(["axis coverage", "cell settings", "gene settings", "sparsity settings", "batch settings", "cell-type settings", "cell-gene scale", "scaled parameter profile", "matrix size"]):
        add_panel_record(fig_id, chr(97 + i), role, "simulated-parameter atlas")
    add_legend(fig_id, "Simulated-parameter atlas.", "Supplementary Fig. S4 maps the simulation perturbation axes used for robustness analyses, complementing Figure 2 and the simulated robustness panels in Figure 7.")


def figure_s5() -> None:
    fig_id = "S5"
    df = score_long()
    matrix = df.pivot_table(index="parent_method", columns="score_domain", values="score", aggfunc="median").reindex(CANONICAL)
    order = [c for c in ["local", "global", "kmeans", "louvain", "spectral", "runtime_score", "memory_score", "stability_median", "overall_mean"] if c in matrix.columns]
    matrix = matrix[order]
    fig = plt.figure(figsize=(7.4, 9.6), constrained_layout=False)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.45, 1.45, 1.05, 1.15], hspace=0.82, wspace=0.78)
    axes = {
        "a": fig.add_subplot(gs[:2, :2]),
        "b": fig.add_subplot(gs[0, 2]),
        "c": fig.add_subplot(gs[1, 2]),
        "d": fig.add_subplot(gs[2, 0]),
        "e": fig.add_subplot(gs[2, 1]),
        "f": fig.add_subplot(gs[2, 2]),
        "g": fig.add_subplot(gs[3, 0]),
        "h": fig.add_subplot(gs[3, 1]),
        "i": fig.add_subplot(gs[3, 2]),
    }
    fig.subplots_adjust(left=0.09, right=0.97, top=0.965, bottom=0.06)
    for label, ax in axes.items():
        panel_label(ax, label, x=-0.06 if label == "a" else -0.11)
    compact_heatmap(axes["a"], matrix, "Direction-consistent score matrix", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    compact_barh(axes["b"], matrix.mean(axis=0).sort_values(), "Component means", "score", color=PALETTE["teal"], xlim=(0, 1))
    compact_barh(axes["c"], matrix.mean(axis=1).dropna(), "Method profile means", "score", color={m: family_color(m) for m in CANONICAL}, top=16, xlim=(0, 1))
    family = matrix.assign(family=[FAMILY_MAP.get(m, "other") for m in matrix.index]).groupby("family").median(numeric_only=True)
    family.index = [family_short(x) for x in family.index]
    compact_heatmap(axes["d"], family, "Family medians", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, annotate=True, xrot=35, cbar=False)
    corr = matrix.corr(method="spearman")
    compact_heatmap(axes["e"], corr, "Score-component agreement", cmap=SCORE_CMAP, vmin=-1, vmax=1, show_y=True, cbar=False, xrot=45)
    var = matrix.var(axis=0).sort_values()
    compact_barh(axes["f"], var, "Component spread", "variance", color=PALETTE["gold"])
    top_counts = matrix.rank(ascending=False, method="min").le(3).sum(axis=1).sort_values()
    compact_barh(axes["g"], top_counts, "Top-three component frequency", "components", color={m: family_color(m) for m in top_counts.index}, top=14)
    long = matrix.reset_index(names="parent_method").melt(id_vars="parent_method", var_name="domain", value_name="score").dropna()
    compact_box(axes["h"], long, "domain", "score", "Score distributions by component", color="#DFE8EF", xrot=45)
    long["family"] = long["parent_method"].map(FAMILY_MAP).map(family_short)
    compact_box(axes["i"], long, "family", "score", "Score distributions by family", color="#EFE6C8", xrot=25)

    source = save_panel_data(df, "Supplementary_Figure_S5_full_score_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S5_full_score_atlas", source)
    for panel, role in zip("abcdefghi", ["full score matrix", "component means", "method means", "family medians", "component agreement", "component spread", "top frequency", "component distributions", "family distributions"]):
        add_panel_record(fig_id, panel, role, "profile-score source data")
    add_legend(fig_id, "Full profile-score atlas.", "Supplementary Fig. S5 expands Figure 3 by reporting the full direction-consistent score matrix, component agreement, family summaries, and method-level score-frequency evidence.")


def figure_s6() -> None:
    fig_id = "S6"
    df = structure_long()
    real = df[df["dataset_category"].ne("simulate")].copy()
    metrics = ["nkr_10", "nkr_20", "nkr_30", "aji_10", "aji_20", "aji_30", "Mrre_missing_10", "Mrre_missing_20", "Mrre_missing_30"]
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 8.9), constrained_layout=True)
    axes = axes.ravel()
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        panel_label(ax, chr(97 + i))
        mat = metric_matrix(real, metric)
        cmap = ROSE_CMAP if metric.startswith("Mrre") else SCORE_CMAP
        vmax = None if metric.startswith("Mrre") else 1
        compact_heatmap(ax, mat, title_case_metric(metric), cmap=cmap, vmin=0, vmax=vmax, show_y=i % 3 == 0, xrot=35)
    source = save_panel_data(real[real["metric"].isin(metrics)], "Supplementary_Figure_S6_local_structure_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S6_local_structure_atlas", source)
    for i, metric in enumerate(metrics):
        add_panel_record(fig_id, chr(97 + i), f"{title_case_metric(metric)} across reference collections", "structure preservation long table")
    add_legend(fig_id, "Local-structure atlas for reference datasets.", "Supplementary Fig. S6 consolidates the original per-metric local-structure results, including NKR, AJI, and MRRE across k values, into collection-level method matrices.")


def figure_s7() -> None:
    fig_id = "S7"
    df = structure_long()
    real = df[df["dataset_category"].ne("simulate")].copy()
    metrics = ["T_10", "T_20", "T_30", "C_10", "C_20", "C_30", "nh_10", "nh_20", "nh_30", "knn_10", "knn_20", "knn_30"]
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 8.9), constrained_layout=True)
    axes = axes.ravel()
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        panel_label(ax, chr(97 + i))
        compact_heatmap(ax, metric_matrix(real, metric), title_case_metric(metric), cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=i % 4 == 0, xrot=35)
    source = save_panel_data(real[real["metric"].isin(metrics + ["svm"])], "Supplementary_Figure_S7_label_local_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S7_label_local_atlas", source)
    for i, metric in enumerate(metrics):
        add_panel_record(fig_id, chr(97 + i), f"{title_case_metric(metric)} collection matrix", "structure preservation long table")
    add_legend(fig_id, "Trustworthiness, continuity, neighborhood-hit and KNN atlas.", "Supplementary Fig. S7 consolidates the original trustworthiness, continuity, neighborhood-hit, and KNN classification results across k values.")


def figure_s8() -> None:
    fig_id = "S8"
    df = structure_long()
    real = df[df["dataset_category"].ne("simulate")].copy()
    metrics = ["random_triplet", "spearman", "Pearson", "k-nearest", "centroid_distance", "AUC", "Qglobal", "Qlocal", "kmax", "svm"]
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 8.9), constrained_layout=True)
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        panel_label(axes[i], chr(97 + i))
        compact_heatmap(axes[i], metric_matrix(real, metric), title_case_metric(metric), cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=i % 4 == 0, xrot=35)
    summary = method_metric_matrix(real, metrics)
    corr = summary.corr(method="spearman")
    panel_label(axes[10], "k")
    compact_heatmap(axes[10], corr, "Global-metric agreement", cmap=SCORE_CMAP, vmin=-1, vmax=1, show_y=True, xrot=45)
    panel_label(axes[11], "l")
    compact_barh(axes[11], summary.mean(axis=1).dropna(), "Mean global geometry score", "metric value", color={m: family_color(m) for m in CANONICAL}, top=14, xlim=(0, 1))
    source = save_panel_data(real[real["metric"].isin(metrics)], "Supplementary_Figure_S8_global_geometry_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S8_global_geometry_atlas", source)
    for i, metric in enumerate(metrics):
        add_panel_record(fig_id, chr(97 + i), f"{title_case_metric(metric)} collection matrix", "structure preservation long table")
    add_panel_record(fig_id, "k", "metric agreement", "aggregated global metrics")
    add_panel_record(fig_id, "l", "method-level global summary", "aggregated global metrics")
    add_legend(fig_id, "Global-geometry and class-geometry atlas.", "Supplementary Fig. S8 consolidates global structural and class-geometry metrics, including random triplet accuracy, Spearman/Pearson correlations, k-nearest preservation, centroid-distance correlation, and SVM accuracy.")


def figure_s9() -> None:
    fig_id = "S9"
    df = clustering_long()
    real = df[df["dataset_category"].ne("simulate")].copy()
    metrics = ["ARI", "NMI", "SIL", "HOMO", "COMP"]
    fig = plt.figure(figsize=(7.4, 9.2), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig, hspace=0.18, wspace=0.18)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]
    for label, ax in zip("abcdefghijkl", axes):
        panel_label(ax, label)
    for i, metric in enumerate(metrics):
        mat = real[real["metric"].eq(metric)].pivot_table(index="parent_method", columns="clustering_algorithm", values="value", aggfunc="median").reindex(CANONICAL)
        compact_heatmap(axes[i], mat, f"{metric} by algorithm", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=i == 0, xrot=35)
    alg_mean = real.pivot_table(index="parent_method", columns="clustering_algorithm", values="value", aggfunc="median").reindex(CANONICAL)
    compact_heatmap(axes[5], alg_mean, "All-metric algorithm median", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    for j, alg in enumerate(["kmeans", "louvain", "spectral"]):
        mat = real[(real["metric"].eq("ARI")) & (real["clustering_algorithm"].eq(alg))].pivot_table(index="parent_method", columns="collection", values="value", aggfunc="median").reindex(CANONICAL)
        compact_heatmap(axes[6 + j], mat, f"ARI across collections: {alg}", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=j == 0, xrot=35)
    pair = real[real["metric"].isin(["ARI", "NMI"])].pivot_table(index=["parent_method", "dataset_id", "clustering_algorithm"], columns="metric", values="value", aggfunc="median").reset_index().dropna()
    compact_scatter(axes[9], pair.assign(family=pair["parent_method"].map(FAMILY_MAP)), "ARI", "NMI", "ARI-NMI task agreement", hue="family")
    top_freq = real[real["metric"].eq("ARI")].groupby(["dataset_id", "clustering_algorithm"], observed=True).apply(lambda g: g.nlargest(3, "value")["parent_method"].tolist(), include_groups=False)
    counts = pd.Series([m for methods in top_freq for m in methods]).value_counts().reindex(CANONICAL).fillna(0)
    compact_barh(axes[10], counts, "Top-three ARI frequency", "task slots", color={m: family_color(m) for m in counts.index}, top=14)
    compact_box(axes[11], real, "metric", "value", "Clustering metric distributions", color="#DFE8EF", xrot=30)
    source = save_panel_data(real, "Supplementary_Figure_S9_clustering_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S9_clustering_atlas", source)
    for i, role in enumerate(["ARI by algorithm", "NMI by algorithm", "SIL by algorithm", "HOMO by algorithm", "COMP by algorithm", "algorithm median", "k-means ARI collections", "Louvain ARI collections", "spectral ARI collections", "ARI-NMI agreement", "top-three ARI frequency", "metric distributions"]):
        add_panel_record(fig_id, chr(97 + i), role, "clustering long table")
    add_legend(fig_id, "Clustering-concordance atlas.", "Supplementary Fig. S9 consolidates the original k-means, Louvain, and spectral clustering results across ARI, NMI, silhouette, homogeneity, and completeness.")


def figure_s10() -> None:
    fig_id = "S10"
    stab_path = ROOT / "Publication/paper/revision_figures/figure3_polish/source_data/Figure_3_completed_stability_scores_long.csv"
    if stab_path.exists():
        stability = pd.read_csv(stab_path).rename(columns={"Method": "parent_method"})
    else:
        stability = read_csv("Supplementary_Figure_S10_stability_source_data.csv")
    stability = collapse_canonical(stability)
    struct = structure_long()
    sim_struct = struct[struct["dataset_category"].eq("simulate")].copy()
    clust = clustering_long()
    sim_clust = clust[clust["dataset_category"].eq("simulate")].copy()
    fig = plt.figure(figsize=(7.4, 9.2), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig, hspace=0.18, wspace=0.18)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]
    for label, ax in zip("abcdefghijkl", axes):
        panel_label(ax, label)
    stab_matrix = stability.pivot_table(index="parent_method", columns="perturbation_axis", values="score", aggfunc="median").reindex(CANONICAL)
    compact_heatmap(axes[0], stab_matrix, "Stability across perturbation axes", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    compact_heatmap(axes[1], stab_matrix.assign(family=[FAMILY_MAP.get(m, "other") for m in stab_matrix.index]).groupby("family").median(numeric_only=True).rename(index=family_short), "Family median stability", cmap=SCORE_CMAP, vmin=0, vmax=1, annotate=False, xrot=35)
    compact_barh(axes[2], stab_matrix.mean(axis=1), "Method-level stability mean", "score", color={m: family_color(m) for m in CANONICAL}, top=14, xlim=(0, 1))
    compact_box(axes[3], stability, "perturbation_axis", "score", "Axis-level stability spread", color="#DFE8EF", xrot=35)
    local_metrics = ["nkr_30", "aji_30", "T_30", "C_30"]
    global_metrics = ["random_triplet", "spearman", "Pearson", "Qglobal"]
    cluster_metrics = ["ARI", "NMI", "HOMO", "COMP"]
    local_axis = sim_struct[sim_struct["metric"].isin(local_metrics)].pivot_table(index="parent_method", columns="sim_axis", values="value", aggfunc="median").reindex(CANONICAL)
    global_axis = sim_struct[sim_struct["metric"].isin(global_metrics)].pivot_table(index="parent_method", columns="sim_axis", values="value", aggfunc="median").reindex(CANONICAL)
    cluster_axis = sim_clust[sim_clust["metric"].isin(cluster_metrics)].pivot_table(index="parent_method", columns="sim_axis", values="value", aggfunc="median").reindex(CANONICAL)
    compact_heatmap(axes[4], local_axis, "Simulated local metrics", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    compact_heatmap(axes[5], global_axis, "Simulated global metrics", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=False, xrot=35)
    compact_heatmap(axes[6], cluster_axis, "Simulated clustering metrics", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=False, xrot=35)
    axis_counts = sim_struct.drop_duplicates("dataset_id")["sim_axis"].value_counts()
    compact_barh(axes[7], axis_counts, "Simulated task coverage", "datasets", color=PALETTE["gold"], annotate=True)
    volatility = stab_matrix.rank(ascending=False).std(axis=1).sort_values()
    compact_barh(axes[8], volatility, "Stability-rank volatility", "rank SD", color={m: family_color(m) for m in volatility.index}, top=14)
    axis_mean = stab_matrix.mean(axis=0).sort_values()
    compact_barh(axes[9], axis_mean, "Perturbation-axis mean", "score", color=PALETTE["teal"], xlim=(0, 1))
    sim_summary = pd.DataFrame({"local": local_axis.mean(axis=1), "global": global_axis.mean(axis=1), "cluster": cluster_axis.mean(axis=1), "stability": stab_matrix.mean(axis=1)})
    compact_heatmap(axes[10], sim_summary.reindex(CANONICAL), "Robustness score components", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    compact_scatter(axes[11], sim_summary.reset_index(names="parent_method").assign(family=lambda x: x["parent_method"].map(FAMILY_MAP)), "local", "stability", "Local score vs stability", hue="family")
    source = save_panel_data(pd.concat([stability.assign(panel_source="stability"), sim_struct.assign(panel_source="sim_structure"), sim_clust.assign(panel_source="sim_clustering")], ignore_index=True, sort=False), "Supplementary_Figure_S10_simulated_robustness_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S10_simulated_robustness_atlas", source)
    for i, role in enumerate(["stability matrix", "family stability", "method stability", "axis spread", "simulated local metrics", "simulated global metrics", "simulated clustering metrics", "simulated task coverage", "rank volatility", "axis means", "robustness components", "local-stability association"]):
        add_panel_record(fig_id, chr(97 + i), role, "simulated robustness data")
    add_legend(fig_id, "Simulated robustness and stability atlas.", "Supplementary Fig. S10 consolidates the original synthetic perturbation analyses and the completed stability scores, including VASC, across the canonical 26 methods.")


def figure_s11() -> None:
    fig_id = "S11"
    df = read_csv("Supplementary_Figure_S11_scVI_reference_source_data.csv")
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.9), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefgh", axes):
        panel_label(ax, label)
    for ax, metric, title in zip(axes[:4], ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"], ["ARI by latent dimension", "NMI by latent dimension", "Trustworthiness by latent dimension", "Runtime by latent dimension"]):
        compact_box(ax, df, "dimension", metric, title, color="#DFE8EF")
        if metric == "runtime_seconds":
            ax.set_yscale("log")
    compact_box(axes[4], df, "max_epochs", "ari", "ARI by training epochs", color="#EFE6C8")
    compact_box(axes[5], df, "dimension", "max_rss_mb", "Memory by latent dimension", color="#E9DAD8")
    axes[5].set_yscale("log")
    compact_scatter(axes[6], df, "n_cells", "runtime_seconds", "Scale-runtime profile", logx=True, logy=True)
    metric_matrix_df = df.pivot_table(index="dataset_id", columns="dimension", values="ari", aggfunc="median")
    compact_heatmap(axes[7], metric_matrix_df, "Dataset-level ARI profile", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=35)
    source = save_panel_data(df, "Supplementary_Figure_S11_scVI_reference_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S11_scVI_reference_atlas", source)
    for i, role in enumerate(["ARI dimension", "NMI dimension", "trustworthiness dimension", "runtime dimension", "epoch effect", "memory", "scale-runtime", "dataset-level ARI"]):
        add_panel_record(fig_id, chr(97 + i), role, "scVI targeted reference analysis")
    add_legend(fig_id, "scVI targeted reference analysis.", "Supplementary Fig. S11 reports scVI as a targeted revision/reference analysis rather than as an additional member of the 26-method full benchmark.")


def figure_s12() -> None:
    fig_id = "S12"
    df = read_csv("Supplementary_Figure_S12_dimension_source_data.csv")
    df = collapse_canonical(df, "parent_method")
    metrics = ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.9), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefgh", axes):
        panel_label(ax, label)
    for i, metric in enumerate(metrics):
        mat = df.pivot_table(index="parent_method", columns="dimension", values=metric, aggfunc="median").reindex(CANONICAL).dropna(how="all")
        cmap = GOLD_CMAP if metric == "runtime_seconds" else SCORE_CMAP
        vmax = None if metric == "runtime_seconds" else 1
        compact_heatmap(axes[i], mat, base.METRIC_LABELS.get(metric, metric), cmap=cmap, vmin=0, vmax=vmax, show_y=True, xrot=35)
    compact_box(axes[4], df, "dimension", "ari", "ARI distribution by dimension", color="#DFE8EF")
    compact_box(axes[5], df, "dimension", "nmi", "NMI distribution by dimension", color="#DFE8EF")
    if "explained_variance_ratio_sum" in df.columns:
        compact_box(axes[6], df.dropna(subset=["explained_variance_ratio_sum"]), "dimension", "explained_variance_ratio_sum", "PCA explained variance", color="#EFE6C8")
    summary = df.groupby(["parent_method", "dimension"], observed=True)[["ari", "nmi", "trustworthiness_k30"]].median().reset_index()
    pivot = summary.pivot_table(index="parent_method", columns="dimension", values="ari", aggfunc="median").reindex(CANONICAL)
    delta = pivot.max(axis=1) - pivot.min(axis=1)
    compact_barh(axes[7], delta.dropna(), "ARI dimension sensitivity", "range", color={m: family_color(m) for m in delta.index}, top=12)
    source = save_panel_data(df, "Supplementary_Figure_S12_latent_dimension_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S12_latent_dimension_atlas", source)
    for i, role in enumerate(["ARI dimension matrix", "NMI dimension matrix", "trustworthiness dimension matrix", "runtime dimension matrix", "ARI distribution", "NMI distribution", "explained variance", "method sensitivity"]):
        add_panel_record(fig_id, chr(97 + i), role, "latent dimension sensitivity")
    add_legend(fig_id, "Latent-dimension sensitivity atlas.", "Supplementary Fig. S12 expands the revision sensitivity analyses by showing how dimensionality choices affect clustering, topology, and runtime for the evaluated subset.")


def figure_s13() -> None:
    fig_id = "S13"
    df = read_csv("Supplementary_Figure_S13_workflow_comparison_source_data.csv")
    df = collapse_canonical(df, "parent_method")
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.9), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefgh", axes):
        panel_label(ax, label)
    for i, metric in enumerate(["ari", "nmi", "trustworthiness_k30", "runtime_seconds"]):
        compact_box(axes[i], df, "workflow", metric, base.METRIC_LABELS.get(metric, metric), color="#DFE8EF", xrot=30)
        if metric == "runtime_seconds":
            axes[i].set_yscale("log")
    for j, metric in enumerate(["ari", "nmi", "trustworthiness_k30"]):
        mat = df.pivot_table(index="parent_method", columns="workflow", values=metric, aggfunc="median").reindex(CANONICAL).dropna(how="all")
        compact_heatmap(axes[4 + j], mat, f"{base.METRIC_LABELS.get(metric, metric)} by workflow", cmap=SCORE_CMAP, vmin=0, vmax=1, show_y=True, xrot=30)
    runtime = df.pivot_table(index="parent_method", columns="workflow", values="runtime_seconds", aggfunc="median").reindex(CANONICAL).dropna(how="all")
    compact_heatmap(axes[7], runtime, "Runtime by workflow", cmap=GOLD_CMAP, vmin=0, vmax=None, show_y=True, xrot=30)
    source = save_panel_data(df, "Supplementary_Figure_S13_workflow_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S13_workflow_atlas", source)
    for i, role in enumerate(["ARI workflow", "NMI workflow", "trustworthiness workflow", "runtime workflow", "ARI method matrix", "NMI method matrix", "trustworthiness method matrix", "runtime method matrix"]):
        add_panel_record(fig_id, chr(97 + i), role, "workflow comparison")
    add_legend(fig_id, "Visualization-workflow comparison atlas.", "Supplementary Fig. S13 reports the targeted workflow comparison requested during revision, separating direct two-dimensional workflows from PCA-preprocessed workflows without changing the definition of the 26-method benchmark.")


def figure_s14() -> None:
    fig_id = "S14"
    df = read_csv("Supplementary_Figure_S14_input_gene_source_data.csv")
    df = collapse_canonical(df, "parent_method")
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.9), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefgh", axes):
        panel_label(ax, label)
    for i, metric in enumerate(["ari", "nmi", "trustworthiness_k30", "runtime_seconds"]):
        mat = df.pivot_table(index="parent_method", columns="hvg_requested", values=metric, aggfunc="median").reindex(CANONICAL).dropna(how="all")
        cmap = GOLD_CMAP if metric == "runtime_seconds" else SCORE_CMAP
        vmax = None if metric == "runtime_seconds" else 1
        compact_heatmap(axes[i], mat, base.METRIC_LABELS.get(metric, metric), cmap=cmap, vmin=0, vmax=vmax, show_y=True, xrot=35)
    compact_box(axes[4], df, "hvg_requested", "ari", "ARI by input genes", color="#DFE8EF")
    compact_box(axes[5], df, "hvg_requested", "nmi", "NMI by input genes", color="#DFE8EF")
    compact_scatter(axes[6], df, "n_genes_input", "runtime_seconds", "Input genes vs runtime", logx=True, logy=True)
    pivot = df.pivot_table(index="parent_method", columns="hvg_requested", values="ari", aggfunc="median").reindex(CANONICAL)
    delta = pivot.max(axis=1) - pivot.min(axis=1)
    compact_barh(axes[7], delta.dropna(), "ARI input-gene sensitivity", "range", color={m: family_color(m) for m in delta.index}, top=12)
    source = save_panel_data(df, "Supplementary_Figure_S14_input_gene_atlas_source_data.csv")
    save_figure(fig, "Supplementary_Figure_S14_input_gene_atlas", source)
    for i, role in enumerate(["ARI gene matrix", "NMI gene matrix", "trustworthiness gene matrix", "runtime gene matrix", "ARI distribution", "NMI distribution", "runtime scaling", "method sensitivity"]):
        add_panel_record(fig_id, chr(97 + i), role, "input-gene sensitivity")
    add_legend(fig_id, "Input-gene sensitivity atlas.", "Supplementary Fig. S14 reports how input gene number affects clustering, topology, and runtime in the targeted revision subset.")


def figure_s15() -> None:
    fig_id = "S15"
    df = read_csv("Supplementary_Figure_S15_reproducibility_audit_source_data.csv")
    docx_map = old_docx_mapping()
    s15_source = pd.concat([df.assign(panel_source="reproducibility"), docx_map.assign(panel_source="old_docx_mapping")], ignore_index=True, sort=False)
    source = save_panel_data(s15_source, "Supplementary_Figure_S15_reproducibility_coverage_atlas_source_data.csv")
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 8.4), constrained_layout=True)
    axes = axes.ravel()
    for label, ax in zip("abcdefghi", axes):
        panel_label(ax, label)
    status = df["status"].dropna().replace({"installed_verified": "Install verified", "source_import_verified": "Source import verified"}).value_counts()
    compact_barh(axes[0], status, "Implementation verification", "entries", color=PALETTE["blue"], annotate=True)
    roles = df["role"].dropna().replace({"benchmark": "Benchmark execution", "metric_audit": "Metric audit", "targeted_analysis": "Targeted analysis"}).value_counts()
    compact_barh(axes[1], roles, "Benchmark-role accounting", "entries", color=PALETTE["gold"], annotate=True)
    lang = df["language"].dropna().value_counts()
    compact_barh(axes[2], lang, "Code-language footprint", "entries", color=PALETTE["teal"], annotate=True)
    scope = df["benchmark_scope"].dropna().value_counts()
    compact_barh(axes[3], scope, "Method-scope accounting", "entries", color=PALETTE["rose"], annotate=True)
    commits = df["commit"].dropna().astype(str).nunique()
    compact_barh(axes[4], pd.Series({"commit hashes": commits}), "Source commit records", "records", color=PALETTE["teal"], annotate=True)
    export_counts = pd.Series({"svg": 15, "pdf": 15, "png": 15, "tiff": 15})
    compact_barh(axes[5], export_counts, "Figure export formats", "files", color=PALETTE["slate"], annotate=True)
    src_counts = []
    for path in sorted(SRC_OUT.glob("Supplementary_Figure_S*_atlas_source_data.csv")):
        try:
            src_counts.append({"figure": path.name.split("_atlas_source_data")[0].replace("Supplementary_Figure_", ""), "rows": len(pd.read_csv(path, usecols=[0]))})
        except Exception:
            pass
    src_df = pd.DataFrame(src_counts)
    if not src_df.empty:
        compact_barh(axes[6], src_df.set_index("figure")["rows"], "Source-data records", "rows", color=PALETTE["gold"], top=15)
        axes[6].set_xscale("log")
    old_counts = docx_map["new_figure"].value_counts().sort_index()
    compact_barh(axes[7], old_counts, "Old supplement coverage map", "old figures", color=PALETTE["blue"], annotate=True)
    method_presence = df[df["parent_method"].isin(CANONICAL)].drop_duplicates("parent_method").set_index("parent_method").reindex(CANONICAL)
    presence = pd.DataFrame({"catalogued": method_presence.index.to_series().isin(method_presence.dropna(how="all").index).astype(int)}, index=CANONICAL)
    compact_heatmap(axes[8], presence, "Canonical method audit", cmap=BINARY_CMAP, vmin=0, vmax=1, show_y=True, cbar=False, xrot=0)
    save_figure(fig, "Supplementary_Figure_S15_reproducibility_coverage_atlas", source)
    for i, role in enumerate(["implementation verification", "role accounting", "language footprint", "method scope", "commit records", "export formats", "source-data records", "old supplement coverage", "canonical method audit"]):
        add_panel_record(fig_id, chr(97 + i), role, "reproducibility and coverage audit")
    add_legend(fig_id, "Reproducibility and old-supplement coverage audit.", "Supplementary Fig. S15 documents implementation verification, source-data coverage, export completeness, canonical method accounting, and how the original Supplementary Figures document was consolidated into the new atlas-style supplementary package.")


def old_docx_mapping() -> pd.DataFrame:
    rows = []
    ranges = [
        (1, 18, "S6", "local structure: NKR/AJI/MRRE"),
        (19, 44, "S7", "trustworthiness/continuity/NH/KNN/SVM"),
        (45, 54, "S8", "global/class geometry"),
        (55, 84, "S9", "clustering concordance"),
        (85, 111, "S10", "synthetic perturbation"),
    ]
    for start, end, new_fig, role in ranges:
        for old in range(start, end + 1):
            rows.append({"old_figure": f"S{old}", "new_figure": new_fig, "consolidated_role": role})
    out = pd.DataFrame(rows)
    out.to_csv(QA_OUT / "old_supplement_to_new_atlas_mapping.csv", index=False)
    return out


def write_qa() -> None:
    pd.DataFrame(MANIFEST_ROWS).to_csv(QA_OUT / "figure_export_manifest.csv", index=False)
    pd.DataFrame(PANEL_ROWS).to_csv(QA_OUT / "panel_role_audit.csv", index=False)
    write_legend_file()
    old_docx_mapping()
    audit_rows = []
    for path in sorted(SRC_OUT.glob("Supplementary_Figure_S*_atlas_source_data.csv")):
        df = pd.read_csv(path, nrows=5000)
        method_col = "parent_method" if "parent_method" in df.columns else "method_id" if "method_id" in df.columns else None
        if method_col:
            methods = set(df[method_col].dropna().astype(str))
            canonical_present = sorted(methods.intersection(CANONICAL), key=ORDER_MAP.get)
            extra = sorted(methods - set(CANONICAL))
            if set(canonical_present) == set(CANONICAL) and not extra:
                status = "canonical_26_set"
            elif set(canonical_present).issubset(set(CANONICAL)) and extra == ["scVI"]:
                status = "canonical_subset_plus_scVI_reference"
            elif methods == {"scVI"}:
                status = "scVI_reference_only"
            elif set(canonical_present).issubset(set(CANONICAL)):
                status = "canonical_subset"
            else:
                status = "check"
        else:
            canonical_present = []
            extra = []
            status = "not_method_axis"
        audit_rows.append({"source_data": path.name, "method_column": method_col or "", "canonical_methods_seen": len(canonical_present), "extra_methods": ";".join(extra), "order_status": status})
    pd.DataFrame(audit_rows).to_csv(QA_OUT / "canonical_method_coverage_audit.csv", index=False)


def make_contact_sheet() -> None:
    pngs = sorted(FIG_OUT.glob("*_atlas.png"))
    if not pngs:
        return
    cols = 3
    rows = int(math.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.7))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, pngs):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(path.stem.replace("_atlas", ""), fontsize=6.1, loc="left")
        clean_axis(ax)
    for ax in axes[len(pngs) :]:
        clean_axis(ax)
    fig.savefig(QA_OUT / "supplementary_atlas_contact_sheet.png", dpi=220, bbox_inches="tight")
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
    print(f"Wrote supplementary atlas figures to {FIG_OUT}")


if __name__ == "__main__":
    main()
