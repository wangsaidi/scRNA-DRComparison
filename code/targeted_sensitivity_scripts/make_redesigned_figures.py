from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "Publication" / "paper" / "revision_figures" / "canonical_source_tables"
OUT = ROOT / "Publication" / "paper" / "revision_figures" / "redesigned_python_figure_package"
MAIN_OUT = OUT / "main_figures"
SUPP_OUT = OUT / "supplementary_figures"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for directory in [MAIN_OUT, SUPP_OUT, SOURCE_OUT, QA_OUT]:
    directory.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
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
    "green": "#78A65A",
    "gold": "#C79B38",
    "rose": "#BF6F6B",
    "violet": "#8E6CB5",
    "slate": "#5F6B7A",
    "gray": "#A8A8A8",
    "light_gray": "#E7E7E7",
    "dark": "#2F2F2F",
    "up": "#2E9E44",
    "down": "#C94949",
}

MEMBERSHIP_CMAP = LinearSegmentedColormap.from_list("membership_cmap", ["#FAFAFA", "#DCEBE8", "#3F9C9A"])
COVERAGE_CMAP = LinearSegmentedColormap.from_list("coverage_cmap", ["#FAFAFA", "#DDEDEA", "#3F9C9A"])

FAMILY_COLORS = {
    "linear/probabilistic": PALETTE["blue"],
    "deep generative/autoencoder": PALETTE["rose"],
    "graph/diffusion": PALETTE["gold"],
    "metric/structure-aware": PALETTE["teal"],
    "other": PALETTE["gray"],
}

DATASET_COLORS = {
    "real": PALETTE["blue"],
    "simulated": PALETTE["gold"],
    "targeted": PALETTE["violet"],
    "downsampling": PALETTE["green"],
}

STATUS_COLORS = {
    "installed_verified": "#A7C7E7",
    "source_import_verified": "#D8D8D8",
    "source_import_partial": "#F0C0CC",
    "not_verified": "#F0C0CC",
    "Install verified": "#A7C7E7",
    "Source import verified": "#D8D8D8",
    "Source import partial": "#F0C0CC",
    "Not verified": "#F0C0CC",
}

METRIC_LABELS = {
    "local": "Local",
    "global": "Global",
    "kmeans": "k-means",
    "louvain": "Louvain",
    "spectral": "Spectral",
    "runtime_score": "Runtime",
    "memory_score": "Memory",
    "stability_median": "Stability",
    "overall_mean": "Overall",
    "ARI": "ARI",
    "NMI": "NMI",
    "HOMO": "Homogeneity",
    "COMP": "Completeness",
    "SIL": "Silhouette",
}

SCOPE_LABELS = {
    "full_26_method_benchmark": "Full benchmark",
    "original_result_variant": "Result variants",
    "targeted_revision_control_only": "Targeted analysis",
}

ANALYSIS_LAYER_LABELS = {
    "WP1_scVI_local": "scVI reference analysis",
    "WP2_dimension_sensitivity": "Latent-dimension sensitivity",
    "WP3_visualization_workflow": "Visualization-workflow comparison",
    "WP4_input_gene_sensitivity": "Input-gene sensitivity",
}

SOURCE_LABELS = {
    "topology_raw": "structure_metric_raw",
    "revision_subset": "targeted_sensitivity_subset",
    "revision_scvi_control": "scVI_reference_analysis",
    "revision_dimension_sensitivity": "latent_dimension_sensitivity",
    "revision_visualization_workflow": "visualization_workflow_comparison",
    "revision_input_gene_sensitivity": "input_gene_sensitivity",
}

PUBLIC_TEXT_REPLACEMENTS = [
    ("targeted_revision_control_subset", "targeted_sensitivity_subset"),
    ("targeted_revision_control_only", "targeted_analysis"),
    ("revision_scvi_control", "scVI_reference_analysis"),
    ("revision_dimension_sensitivity", "latent_dimension_sensitivity"),
    ("revision_visualization_workflow", "visualization_workflow_comparison"),
    ("revision_input_gene_sensitivity", "input_gene_sensitivity"),
    ("WP1_scVI_local", "scVI reference analysis"),
    ("WP2_dimension_sensitivity", "Latent-dimension sensitivity"),
    ("WP3_visualization_workflow", "Visualization-workflow comparison"),
    ("WP4_input_gene_sensitivity", "Input-gene sensitivity"),
    ("added_reviewer_requested_method", "targeted_reference_method"),
    ("reviewer_requested", "targeted_reference"),
    ("New method added for revision", "Targeted reference method included for sensitivity analysis"),
    ("install_verified", "Install verified"),
    ("source_import_verified", "Source import verified"),
    ("source_import_partial", "Source import partial"),
    ("not_verified", "Not verified"),
    ("targeted_reference_method", "Targeted reference method"),
    ("latent_dimension_sensitivity", "Latent-dimension sensitivity"),
    ("label_sensitive_method", "Label-sensitive method"),
    ("optional_latent_method", "Optional latent method"),
    ("legacy_visualization", "Legacy visualization"),
    ("legacy_deep_method", "Legacy deep method"),
    ("visualization_2d", "2D visualization"),
    ("baseline", "Baseline"),
    ("scrna-dr-py-modern-min", "Python modern"),
    ("scrna-dr-py-legacy", "Python legacy"),
    ("scrna-dr-r", "R benchmark"),
    ("r-benchmark", "R benchmark"),
    ("py-modern", "Python modern"),
    ("py-legacy", "Python legacy"),
    ("py36-tgplvm", "Python 3.6 tGPLVM"),
    ("py-scdh", "Python SCDH"),
    ("py-tf1", "Python TF1"),
    ("direct_2d", "Direct 2D"),
    ("pca50_to_2d", "PCA50 to 2D"),
    ("latent_dim", "Latent dimension"),
    ("raw_hvg_as_provided", "Raw HVG"),
    ("log1p_standardized_hvg_pca50", "Log1p HVG + PCA50"),
    ("log1p_standardized_hvg", "Log1p HVG"),
    ("planned_full_grid", "Planned full grid"),
    ("planned_workflow_grid", "Planned workflow grid"),
    ("targeted revision-control analyses", "targeted sensitivity analyses"),
    ("targeted revision-control analysis", "targeted sensitivity analysis"),
    ("targeted revision control", "targeted sensitivity analysis"),
    ("revision-control analyses", "targeted sensitivity analyses"),
    ("revision-control analysis", "targeted sensitivity analysis"),
    ("revision-control subset", "targeted sensitivity subset"),
    ("revision control", "targeted sensitivity analysis"),
    ("revision controls", "targeted sensitivity analyses"),
    ("revision subset", "sensitivity subset"),
    ("revision-level", "sensitivity-layer"),
    ("revised manuscript", "manuscript"),
    ("original benchmark", "full benchmark"),
    ("Original benchmark", "Full benchmark"),
    ("original result files", "benchmark result files"),
    ("original result file", "benchmark result file"),
    ("original full benchmark", "full benchmark"),
    ("topology-preservation", "structure-preservation"),
    ("topology preservation", "structure preservation"),
    ("topological structure", "neighborhood structure"),
    ("topological information", "geometric information"),
    ("topological", "structural"),
    ("topology", "structure"),
]

LOCAL_METRICS = [
    "knn_10",
    "knn_20",
    "knn_30",
    "svm",
    "nkr_10",
    "nkr_20",
    "nkr_30",
    "aji_10",
    "aji_20",
    "aji_30",
    "T_10",
    "T_20",
    "T_30",
    "C_10",
    "C_20",
    "C_30",
    "Mrre_false_10",
    "Mrre_missing_10",
    "Mrre_false_20",
    "Mrre_missing_20",
    "Mrre_false_30",
    "Mrre_missing_30",
    "nh_10",
    "nh_20",
    "nh_30",
]

GLOBAL_METRICS = [
    "random_triplet",
    "spearman",
    "k-nearest",
    "centroid_distance",
    "AUC",
    "Qglobal",
    "Pearson",
]

MANIFEST_ROWS: list[dict[str, str]] = []
CONTRACTS: list[dict[str, object]] = []
LEGENDS: list[dict[str, str]] = []


def reset_outputs() -> None:
    for directory in [MAIN_OUT, SUPP_OUT, SOURCE_OUT, QA_OUT]:
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def package_rel(path: Path) -> str:
    return str(path.relative_to(OUT))


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(CANON / name)


def load_all() -> dict[str, pd.DataFrame]:
    tables = {
        "methods": load_table("canonical_method_manifest.csv"),
        "real_atlas": load_table("canonical_real_dataset_atlas_from_supplementary_table_s2.csv"),
        "real_detail": load_table("excel_real_dataset_detail_rows.csv"),
        "sim_atlas": load_table("canonical_simulated_dataset_atlas_from_excel.csv"),
        "availability": load_table("repository_dataset_availability.csv"),
        "revision_manifest": load_table("revision_control_dataset_manifest.csv"),
        "score_long": load_table("original_score_long.csv"),
        "score_matrix": load_table("original_score_matrix.csv"),
        "topology_raw": load_table("original_topology_raw_long.csv"),
        "cluster_raw": load_table("original_clustering_raw_long.csv"),
        "efficiency": load_table("original_efficiency_scaling_long.csv"),
        "stability": load_table("original_stability_score_long.csv"),
        "scvi": load_table("revision_scvi_control.csv"),
        "dimension": load_table("revision_dimension_sensitivity.csv"),
        "workflow": load_table("revision_visualization_workflow.csv"),
        "hvg": load_table("revision_input_gene_sensitivity.csv"),
    }
    for key in ["topology_raw", "cluster_raw"]:
        tables[key]["value"] = pd.to_numeric(tables[key]["value"], errors="coerce")
    for key in ["score_long", "score_matrix", "efficiency", "stability", "scvi", "dimension", "workflow", "hvg"]:
        for col in tables[key].columns:
            if col in {"score", "runtime_seconds", "peak_memory_gb", "max_rss_mb", "ari", "nmi", "trustworthiness_k30", "silhouette_label", "dimension", "hvg_requested", "cells", "genes", "sparsity_pct"}:
                tables[key][col] = pd.to_numeric(tables[key][col], errors="coerce")
    return tables


def family_map(methods: pd.DataFrame) -> dict[str, str]:
    return dict(zip(methods["method_id"], methods["method_family"]))


def method_color(method: object, fmap: dict[str, str]) -> str:
    fam = fmap.get(str(method), "other")
    return FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])


def prepare_public_source(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    rename_columns = {}
    for col in output.columns:
        public_col = col
        for old, new in PUBLIC_TEXT_REPLACEMENTS:
            public_col = public_col.replace(old, new)
        if public_col != col:
            rename_columns[col] = public_col
    if rename_columns:
        output = output.rename(columns=rename_columns)
    if "benchmark_scope" in output.columns:
        output["benchmark_scope"] = output["benchmark_scope"].map(SCOPE_LABELS).fillna(output["benchmark_scope"])
    if "work_package" in output.columns:
        output["work_package"] = output["work_package"].map(ANALYSIS_LAYER_LABELS).fillna(output["work_package"])
    if "source" in output.columns:
        output["source"] = output["source"].replace(SOURCE_LABELS)
    for col in output.select_dtypes(include="object").columns:
        series = output[col].astype("string")
        for old, new in PUBLIC_TEXT_REPLACEMENTS:
            series = series.str.replace(old, new, regex=False)
        output[col] = series
    for col in list(output.columns):
        col_lower = col.lower()
        if (
            col_lower.endswith("path")
            or col_lower.endswith("_file")
            or col_lower in {"file", "source_file", "counts_path", "metadata_path", "embedding_path"}
        ):
            output = output.drop(columns=col)
    return output


def save_source(df: pd.DataFrame, name: str) -> str:
    path = SOURCE_OUT / name
    prepare_public_source(df).to_csv(path, index=False)
    return package_rel(path)


def save_figure(fig: plt.Figure, base_path: Path, source_file: str, dpi: int = 600) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ["svg", "pdf", "png", "tiff"]:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        MANIFEST_ROWS.append({"figure": base_path.name, "format": fmt, "path": package_rel(out), "source_data": source_file})
    plt.close(fig)


def add_contract(fig_id: str, conclusion: str, panels: list[str], source: str, legend: str) -> None:
    CONTRACTS.append(
        {
            "figure": fig_id,
            "core_conclusion": conclusion,
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib/seaborn",
            "source_data": source,
            "panels": panels,
        }
    )
    LEGENDS.append({"figure": fig_id, "legend": legend})


def write_contracts() -> None:
    lines = [
        "# Redesigned Figure Contracts",
        "",
        "Backend: Python only.",
        "Output formats: SVG, PDF, PNG, TIFF.",
        "Design rule: interpretation and caveats belong in legends, not paragraph-style text panels.",
        "",
    ]
    for item in CONTRACTS:
        lines.extend([f"## {item['figure']}", f"Core conclusion: {item['core_conclusion']}", "Panels:"])
        lines.extend([f"- {panel}" for panel in item["panels"]])
        lines.extend([f"Source data: {item['source_data']}", ""])
    (QA_OUT / "figure_contracts.md").write_text("\n".join(lines), encoding="utf-8")

    legend_lines = ["# Draft Figure Legends", ""]
    for item in LEGENDS:
        legend_lines.extend([f"## {item['figure']}", item["legend"], ""])
    (QA_OUT / "draft_figure_legends.md").write_text("\n".join(legend_lines), encoding="utf-8")


def save_manifest() -> None:
    pd.DataFrame(MANIFEST_ROWS).to_csv(QA_OUT / "figure_export_manifest.csv", index=False)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=PALETTE["dark"])


def set_ticks(ax: plt.Axes, labelsize: float = 5.2) -> None:
    ax.tick_params(axis="both", labelsize=labelsize, length=2.2, pad=1.5)


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    xrot: int = 35,
    annotate: bool = False,
) -> None:
    matrix = data.astype(float).to_numpy()
    cmap_obj = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap_obj.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=3)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([METRIC_LABELS.get(str(c), str(c)) for c in data.columns], rotation=xrot, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    set_ticks(ax, 4.7 if data.shape[0] > 22 else 5.1)
    if annotate and data.shape[0] <= 12 and data.shape[1] <= 10:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    r, g, b, _ = cmap_obj(im.norm(val))
                    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    text_color = PALETTE["dark"] if luminance > 0.62 else "white"
                    ax.text(j, i, f"{val:.0f}" if val > 3 else f"{val:.2f}", ha="center", va="center", fontsize=4.7, color=text_color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=5, length=2)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=5.3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def ranked_barh(
    ax: plt.Axes,
    values: pd.Series,
    title: str,
    xlabel: str,
    fmap: dict[str, str],
    *,
    top: int = 12,
    xlim: tuple[float, float] | None = None,
) -> None:
    vals = values.dropna().sort_values(ascending=False).head(top).iloc[::-1]
    ax.barh(vals.index, vals.values, color=[method_color(m, fmap) for m in vals.index], edgecolor="white", linewidth=0.45)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    set_ticks(ax, 5)


def draw_cards(ax: plt.Axes, cards: list[tuple[str, str, str]]) -> None:
    values = pd.Series([float(number) for number, _, _ in cards])
    labels = [label for _, label, _ in cards]
    colors = [color for _, _, color in cards]
    y = np.arange(len(cards))[::-1]
    ax.barh(y, values, color=colors, height=0.48, edgecolor="white", linewidth=0.5)
    pad = max(values.max() * 0.025, 0.8)
    for yi, value, color in zip(y, values, colors):
        ax.text(value + pad, yi, f"{value:.0f}", ha="left", va="center", fontsize=7.4, fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, values.max() * 1.22)
    ax.set_xlabel("count")
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    set_ticks(ax, 5.2)


def box_strip(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, title: str, *, hue: str | None = None, palette=None, rotate: int = 25) -> None:
    if hue is None and palette is not None:
        sns.boxplot(
            data=data,
            x=x,
            y=y,
            hue=x,
            ax=ax,
            linewidth=0.6,
            fliersize=1.2,
            palette=palette,
            legend=False,
        )
    else:
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, linewidth=0.6, fliersize=1.2, palette=palette, color=None if hue else "#E5ECF4")
    strip_data = data
    if len(strip_data) > 5000:
        strip_data = strip_data.sample(5000, random_state=7)
    sns.stripplot(data=strip_data, x=x, y=y, hue=None, ax=ax, color=PALETTE["slate"], size=1.5, alpha=0.45, jitter=0.16)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
    ax.tick_params(axis="x", rotation=rotate)
    set_ticks(ax)
    if hue:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[: len(set(data[hue].dropna()))], labels[: len(set(data[hue].dropna()))], fontsize=4.7, title="")


def topology_summaries(topo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = topo.dropna(subset=["value"]).copy()
    local = work[work["metric"].isin(LOCAL_METRICS)]
    global_ = work[work["metric"].isin(GLOBAL_METRICS)]
    local_summary = local.pivot_table(index="method_id", columns="metric", values="value", aggfunc="median")
    global_summary = global_.pivot_table(index="method_id", columns="metric", values="value", aggfunc="median")
    return local_summary, global_summary


def sort_methods_by_score(score_matrix: pd.DataFrame) -> list[str]:
    return score_matrix.sort_values("overall_mean", ascending=False)["method_id"].tolist()


def fig1(tables: dict[str, pd.DataFrame]) -> None:
    methods = tables["methods"]
    full = methods[methods["benchmark_scope"] == "full_26_method_benchmark"].copy()
    controls = methods[methods["benchmark_scope"] == "targeted_revision_control_only"].copy()
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    fam_counts = full.groupby("method_family")["method_id"].nunique().sort_values()
    axes[0].barh(fam_counts.index, fam_counts.values, color=[FAMILY_COLORS.get(f, PALETTE["gray"]) for f in fam_counts.index])
    axes[0].set_title("26 benchmark methods by family", fontsize=7, fontweight="bold", loc="left")
    axes[0].set_xlabel("methods")
    set_ticks(axes[0], 5)

    lang_counts = full.groupby("implementation_language")["method_id"].nunique().sort_values()
    axes[1].barh(lang_counts.index, lang_counts.values, color=PALETTE["slate"])
    axes[1].set_title("Implementation languages", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("methods")
    set_ticks(axes[1], 5)

    scope_counts = methods.groupby("benchmark_scope")["method_id"].nunique().reindex(
        ["full_26_method_benchmark", "original_result_variant", "targeted_revision_control_only"]
    )
    scope_plot = scope_counts.rename(index=SCOPE_LABELS)
    axes[2].bar(scope_plot.index, scope_plot.values, color=[PALETTE["blue"], PALETTE["gray"], PALETTE["violet"]])
    axes[2].set_title("Benchmark scope layers", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_ylabel("methods or variants")
    axes[2].tick_params(axis="x", rotation=25)
    set_ticks(axes[2])

    metric_domains = pd.Series({"local": 25, "global": 7, "clustering": 15, "efficiency": 2, "stability": 9})
    axes[3].bar(metric_domains.index, metric_domains.values, color=[PALETTE["teal"], PALETTE["blue"], PALETTE["gold"], PALETTE["green"], PALETTE["rose"]])
    axes[3].set_title("Metric families", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_ylabel("metrics or axes")
    axes[3].tick_params(axis="x", rotation=25)
    set_ticks(axes[3])

    variant_df = methods[methods["is_variant"].astype(str).str.lower().isin(["true", "1"])].copy()
    variant_counts = variant_df.groupby("parent_method")["method_id"].nunique().sort_values()
    axes[4].barh(variant_counts.index, variant_counts.values, color=[method_color(m, fmap) for m in variant_counts.index])
    axes[4].set_title("Shown variants", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("variants")
    set_ticks(axes[4], 5)

    draw_cards(
        axes[5],
        [
            ("26", "full methods", PALETTE["blue"]),
            ("100", "dataset atlas", PALETTE["gold"]),
            ("1", "scVI reference", PALETTE["violet"]),
            ("60", "sensitivity rows", PALETTE["green"]),
        ],
    )
    axes[5].set_title("Analysis layers", fontsize=7, fontweight="bold", loc="left")
    source = save_source(pd.concat([methods.assign(source="method_manifest"), metric_domains.rename("count").reset_index().assign(source="metric_domains")], ignore_index=True, sort=False), "Figure_1_method_taxonomy_source_data.csv")
    add_contract(
        "Figure 1",
        "The benchmark universe combines a 26-method full comparison with targeted sensitivity analyses.",
        ["a: method families", "b: implementation languages", "c: benchmark scope layers", "d: metric families", "e: method variants", "f: analysis layers"],
        source,
        "Figure 1 defines the benchmark universe. scVI is shown as a targeted reference analysis and is not counted as one of the 26 full-benchmark methods.",
    )
    save_figure(fig, MAIN_OUT / "Figure_1_method_taxonomy_and_analysis_layers", source)


def fig2(tables: dict[str, pd.DataFrame]) -> None:
    methods = tables["methods"]
    real = tables["real_atlas"]
    real_detail = tables["real_detail"]
    sim = tables["sim_atlas"]
    sensitivity = tables["revision_manifest"]
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    draw_cards(
        axes[0],
        [
            (str((methods["benchmark_scope"] == "full_26_method_benchmark").sum()), "methods", PALETTE["blue"]),
            (str(len(real)), "real datasets", PALETTE["teal"]),
            (str(len(sim)), "simulated datasets", PALETTE["gold"]),
            (str(len(sensitivity)), "sensitivity rows", PALETTE["violet"]),
        ],
    )
    axes[0].set_title("Benchmark and sensitivity layers", fontsize=7, fontweight="bold", loc="left")

    real_counts = real.groupby("dataset_group")["dataset_label"].nunique().sort_values()
    axes[1].barh(real_counts.index, real_counts.values, color=PALETTE["teal"])
    axes[1].set_title("Real atlas expansion", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("datasets")
    set_ticks(axes[1], 5)

    axes[2].scatter(real_detail["cells"], real_detail["sparsity_pct"], s=np.clip(real_detail["genes"] / 260, 8, 80), color=PALETTE["blue"], alpha=0.72, edgecolor="white", lw=0.4)
    axes[2].set_xscale("log")
    axes[2].set_title("Real data scale and sparsity", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("cells")
    axes[2].set_ylabel("sparsity (%)")
    axes[2].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[2])

    sim_counts = sim.groupby("param_group")["dataset_label"].nunique().sort_values()
    axes[3].barh(sim_counts.index, sim_counts.values, color=PALETTE["gold"])
    axes[3].set_title("Simulated parameter axes", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("configurations")
    set_ticks(axes[3], 5)

    axes[4].scatter(sim["cells"], sim["sparsity_pct"], s=np.clip(sim["genes"] / 110, 8, 120), color=PALETTE["gold"], alpha=0.74, edgecolor="white", lw=0.4)
    axes[4].set_xscale("log")
    axes[4].set_title("Simulated data scale and sparsity", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("cells")
    axes[4].set_ylabel("sparsity (%)")
    axes[4].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[4])

    use_counts = sensitivity["planned_use"].fillna("").str.split(",").explode().str.strip().replace("", np.nan).dropna().value_counts()
    axes[5].barh(use_counts.index[::-1], use_counts.values[::-1], color=PALETTE["violet"])
    axes[5].set_title("Targeted sensitivity subset", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("dataset rows")
    set_ticks(axes[5], 5)

    source = save_source(pd.concat([real.assign(source="real_atlas"), real_detail.assign(source="real_detail"), sim.assign(source="simulated_atlas"), sensitivity.assign(source="revision_subset")], ignore_index=True, sort=False), "Figure_2_full_dataset_landscape_source_data.csv")
    add_contract(
        "Figure 2",
        "The figure system separates the complete 100-dataset atlas from targeted sensitivity analyses.",
        ["a: count layer", "b: real-dataset expansion", "c: real scale/sparsity", "d: simulated axes", "e: simulated scale/sparsity", "f: targeted sensitivity subset"],
        source,
        "Figure 2 shows the full benchmark scope and targeted sensitivity subset as distinct analysis layers. Real-dataset counting follows Supplementary Table S2 expansion rules, while numeric real-data scatter uses the explicit Excel/repository detail rows.",
    )
    save_figure(fig, MAIN_OUT / "Figure_2_full_benchmark_landscape", source)


def fig3(tables: dict[str, pd.DataFrame]) -> None:
    score = tables["score_matrix"].copy()
    score = score.set_index("method_id")
    score_long = tables["score_long"]
    methods = tables["methods"]
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    heatmap(axes[0], score.drop(columns=["overall_mean"]), "Full benchmark score matrix", cmap="mako", vmin=0, vmax=1, cbar_label="normalized score")
    draw_score = score[["local", "global", "overall_mean"]].dropna()
    for method, row in draw_score.iterrows():
        axes[1].scatter(row["local"], row["global"], s=18 + 50 * row["overall_mean"], color=method_color(method, fmap), edgecolor="white", lw=0.45, alpha=0.9)
    axes[1].set_title("Local-global structure trade-off", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("local score")
    axes[1].set_ylabel("global score")
    axes[1].set_xlim(0, 1.03)
    axes[1].set_ylim(0, 1.03)
    axes[1].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[1])
    cluster = score_long[score_long["score_domain"].isin(["kmeans", "louvain", "spectral"])]
    box_strip(axes[2], cluster, "score_domain", "score", "Clustering score distributions", rotate=20)
    axes[2].set_xlabel("")
    efficiency = score[["runtime_score", "memory_score", "overall_mean"]].dropna()
    for method, row in efficiency.iterrows():
        axes[3].scatter(row["runtime_score"], row["memory_score"], s=18 + 50 * row["overall_mean"], color=method_color(method, fmap), edgecolor="white", lw=0.45, alpha=0.9)
    axes[3].set_title("Efficiency score trade-off", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("runtime score")
    axes[3].set_ylabel("memory score")
    axes[3].set_xlim(0, 1.03)
    axes[3].set_ylim(0, 1.03)
    axes[3].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[3])
    stability = score_long[score_long["score_domain"] == "stability_median"].set_index("method_id")["score"]
    ranked_barh(axes[4], stability, "Median stability score", "normalized score", fmap, top=14, xlim=(0, 1.03))
    ranked_barh(axes[5], score["overall_mean"], "Overall descriptive score", "mean normalized score", fmap, top=14, xlim=(0, 1.03))
    source = save_source(pd.concat([score.reset_index().assign(source="score_matrix"), score_long.assign(source="score_long")], ignore_index=True, sort=False), "Figure_3_full_benchmark_score_landscape_source_data.csv")
    add_contract(
        "Figure 3",
        "The full benchmark shows method-dependent trade-offs across structure preservation, clustering concordance, efficiency, and stability.",
        ["a: score matrix", "b: local-global trade-off", "c: clustering distributions", "d: runtime-memory score", "e: stability ranking", "f: descriptive overall ranking"],
        source,
        "Figure 3 summarizes the full benchmark. Composite scores are used as descriptive summaries and should not be interpreted as a universal method ranking.",
    )
    save_figure(fig, MAIN_OUT / "Figure_3_full_benchmark_score_landscape", source)


def fig4(tables: dict[str, pd.DataFrame]) -> None:
    topo = tables["topology_raw"]
    score = tables["score_matrix"].set_index("method_id")
    methods = tables["methods"]
    fmap = family_map(methods)
    local, global_ = topology_summaries(topo)
    order = sort_methods_by_score(tables["score_matrix"])
    local_sel = local.reindex(order).dropna(how="all")
    global_sel = global_.reindex(order).dropna(how="all")
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    heatmap(axes[0], local_sel[["knn_10", "knn_20", "nkr_30", "aji_30", "T_30", "C_30", "nh_30"]].head(18), "Local neighborhood metrics", cmap="viridis", vmin=0, vmax=1, cbar_label="median")
    heatmap(axes[1], global_sel[[c for c in ["random_triplet", "spearman", "k-nearest", "centroid_distance", "AUC", "Qglobal", "Pearson"] if c in global_sel.columns]].head(18), "Global geometry metrics", cmap="mako", vmin=0, vmax=1, cbar_label="median")
    ranked_barh(axes[2], score["local"], "Aggregate local score", "normalized score", fmap, top=14, xlim=(0, 1.03))
    ranked_barh(axes[3], score["global"], "Aggregate global score", "normalized score", fmap, top=14, xlim=(0, 1.03))
    work = topo[topo["metric"].isin(["T_30", "C_30", "nh_30", "random_triplet", "spearman", "Pearson"])].copy()
    work["metric_class"] = np.where(work["metric"].isin(["T_30", "C_30", "nh_30"]), "local", "global")
    family_lookup = fmap
    work["family"] = work["method_id"].map(family_lookup).fillna("other")
    box_strip(axes[4], work, "metric_class", "value", "Metric distribution", hue="family", palette=FAMILY_COLORS, rotate=0)
    draw_score = score[["local", "global", "overall_mean"]].dropna()
    for method, row in draw_score.iterrows():
        axes[5].scatter(row["local"], row["global"], s=18 + 50 * row["overall_mean"], color=method_color(method, fmap), edgecolor="white", lw=0.45, alpha=0.9)
    axes[5].set_title("Local versus global", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("local score")
    axes[5].set_ylabel("global score")
    axes[5].set_xlim(0, 1.03)
    axes[5].set_ylim(0, 1.03)
    axes[5].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[5])
    source = save_source(pd.concat([local.reset_index().assign(source="local_summary"), global_.reset_index().assign(source="global_summary"), topo.assign(source="topology_raw")], ignore_index=True, sort=False), "Figure_4_structure_preservation_source_data.csv")
    add_contract(
        "Figure 4",
        "Local neighborhood retention and global geometry preservation are distinct benchmark objectives with method- and dataset-dependent behavior.",
        ["a: local metrics", "b: global metrics", "c: local ranking", "d: global ranking", "e: raw metric distributions", "f: local-global trade-off"],
        source,
        "Figure 4 expands structure-preservation results using raw metrics from the full benchmark, separating local neighborhood retention from global geometry.",
    )
    save_figure(fig, MAIN_OUT / "Figure_4_structure_preservation_detail", source)


def fig5(tables: dict[str, pd.DataFrame]) -> None:
    cluster = tables["cluster_raw"].dropna(subset=["value"]).copy()
    methods = tables["methods"]
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    ari = cluster[cluster["metric"] == "ARI"]
    ari_mat = ari.pivot_table(index="method_id", columns="clustering_algorithm", values="value", aggfunc="median")
    ari_mat["mean"] = ari_mat.mean(axis=1, skipna=True)
    ari_mat = ari_mat.sort_values("mean", ascending=False).drop(columns=["mean"])
    heatmap(axes[0], ari_mat.head(20), "ARI by clustering algorithm", cmap="mako", vmin=0, vmax=1, cbar_label="median ARI")
    metric_alg = cluster.pivot_table(index="metric", columns="clustering_algorithm", values="value", aggfunc="median")
    heatmap(axes[1], metric_alg, "Metric-algorithm medians", cmap="viridis", vmin=0, vmax=1, cbar_label="median", annotate=True, xrot=30)
    box_strip(axes[2], cluster[cluster["metric"].isin(["ARI", "NMI", "SIL"])], "metric", "value", "Raw clustering metric spread", hue="clustering_algorithm", palette=["#DDEAF7", "#E8DFF1", "#F1DCC9"], rotate=0)
    family_lookup = fmap
    fam = ari.copy()
    fam["family"] = fam["method_id"].map(family_lookup).fillna("other")
    box_strip(axes[3], fam, "family", "value", "ARI by method family", palette=FAMILY_COLORS, rotate=30)
    axes[3].set_xlabel("")
    pair = cluster[cluster["metric"].isin(["ARI", "NMI"])].pivot_table(index=["dataset_category", "dataset_id", "method_id", "clustering_algorithm"], columns="metric", values="value", aggfunc="mean").dropna().reset_index()
    pair_plot = pair.sample(min(len(pair), 6000), random_state=7)
    axes[4].scatter(pair_plot["ARI"], pair_plot["NMI"], s=8, color=PALETTE["blue"], alpha=0.35, edgecolor="none")
    axes[4].set_title("ARI-NMI agreement", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("ARI")
    axes[4].set_ylabel("NMI")
    axes[4].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[4])
    top_frequency = ari.groupby(["dataset_id", "clustering_algorithm"]).apply(lambda x: x.nlargest(3, "value")["method_id"].tolist(), include_groups=False).explode().value_counts()
    ranked_barh(axes[5], top_frequency, "Top-three ARI frequency", "appearances", fmap, top=14)
    source = save_source(cluster, "Figure_5_clustering_concordance_source_data.csv")
    add_contract(
        "Figure 5",
        "Clustering concordance depends on the dimensionality-reduction method, clustering algorithm, metric, and label provenance.",
        ["a: ARI matrix", "b: metric-algorithm medians", "c: metric spread", "d: family-level ARI", "e: ARI-NMI agreement", "f: top-three frequency"],
        source,
        "Figure 5 uses clustering metrics from the full benchmark and describes real-dataset labels as annotation-derived concordance rather than independent labels.",
    )
    save_figure(fig, MAIN_OUT / "Figure_5_clustering_label_concordance", source)


def fig6(tables: dict[str, pd.DataFrame]) -> None:
    scvi = tables["scvi"].copy()
    dim = tables["dimension"].copy()
    workflow = tables["workflow"].copy()
    hvg = tables["hvg"].copy()
    methods = tables["methods"]
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    box_strip(axes[0], scvi, "dimension", "ari", "scVI reference: ARI", rotate=0)
    box_strip(axes[1], scvi, "dimension", "trustworthiness_k30", "scVI reference: trustworthiness", rotate=0)
    dim_mat = dim.pivot_table(index="method_id", columns="dimension", values="ari", aggfunc="median")
    dim_mat = dim_mat.loc[dim_mat.median(axis=1).sort_values(ascending=False).index]
    heatmap(axes[2], dim_mat, "Latent-dimension sensitivity", cmap="viridis", vmin=0, vmax=1, cbar_label="median ARI", annotate=True, xrot=0)
    wf_delta = workflow.pivot_table(index=["dataset_id", "method_id", "seed"], columns="workflow", values="ari", aggfunc="mean").dropna().reset_index()
    if {"direct_2d", "pca50_to_2d"}.issubset(wf_delta.columns):
        wf_delta["delta_ari"] = wf_delta["pca50_to_2d"] - wf_delta["direct_2d"]
    wf_mat = wf_delta.pivot_table(index="method_id", columns="dataset_id", values="delta_ari", aggfunc="median")
    max_abs = np.nanmax(np.abs(wf_mat.to_numpy())) if wf_mat.size else 0.1
    heatmap(axes[3], wf_mat, "Visualization workflow delta", cmap="vlag", vmin=-max_abs, vmax=max_abs, cbar_label="PCA50 minus direct", xrot=35)
    hvg_mat = hvg.pivot_table(index="method_id", columns="hvg_requested", values="ari", aggfunc="median")
    hvg_mat = hvg_mat.loc[hvg_mat.median(axis=1).sort_values(ascending=False).index]
    heatmap(axes[4], hvg_mat, "Input-gene sensitivity", cmap="mako", vmin=0, vmax=1, cbar_label="median ARI", annotate=True, xrot=0)
    runtime = pd.concat(
        [
            scvi[["method_id", "runtime_seconds"]].assign(sensitivity_layer="scVI"),
            dim[["method_id", "runtime_seconds"]].assign(sensitivity_layer="dimension"),
            workflow[["method_id", "runtime_seconds"]].assign(sensitivity_layer="workflow"),
            hvg[["method_id", "runtime_seconds"]].assign(sensitivity_layer="HVG"),
        ],
        ignore_index=True,
    )
    box_strip(axes[5], runtime, "sensitivity_layer", "runtime_seconds", "Runtime by sensitivity layer", rotate=15)
    axes[5].set_yscale("log")
    source = save_source(pd.concat([scvi.assign(source="scvi"), dim.assign(source="dimension"), workflow.assign(source="workflow"), hvg.assign(source="hvg")], ignore_index=True, sort=False), "Figure_6_sensitivity_controls_source_data.csv")
    add_contract(
        "Figure 6",
        "Targeted sensitivity analyses test scVI, latent dimensionality, visualization workflow, and input-gene selection without changing the 26-method full-benchmark count.",
        ["a: scVI ARI", "b: scVI trustworthiness", "c: latent dimension", "d: visualization workflow delta", "e: input-gene sensitivity", "f: runtime"],
        source,
        "Figure 6 evaluates targeted sensitivity analyses for scVI, latent dimensionality, visualization workflow, and input-gene selection. scVI is a targeted reference analysis and is not included as a twenty-seventh full-benchmark method.",
    )
    save_figure(fig, MAIN_OUT / "Figure_6_targeted_sensitivity_controls", source)


def fig7(tables: dict[str, pd.DataFrame]) -> None:
    stability = tables["stability"].copy()
    methods = tables["methods"]
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    mat = stability.pivot_table(index="method_id", columns="perturbation_axis", values="score", aggfunc="mean")
    mat["median"] = mat.median(axis=1)
    mat = mat.sort_values("median", ascending=False)
    heatmap(axes[0], mat.drop(columns=["median"]), "Stability across simulated perturbations", cmap="crest", vmin=0, vmax=1, cbar_label="score", xrot=35)
    ranked_barh(axes[1], mat["median"], "Median robustness ranking", "median score", fmap, top=14, xlim=(0, 1.03))
    stability["family"] = stability["method_id"].map(fmap).fillna("other")
    box_strip(axes[2], stability, "family", "score", "Family-level robustness", palette=FAMILY_COLORS, rotate=30)
    axes[2].set_xlabel("")
    axis_groups = {
        "batch": ["batch_number", "batch_strength"],
        "scale": ["cell_number", "gene_number", "celltype_number"],
        "signal": ["dropout", "de_prob", "de_strength", "out"],
    }
    for ax, (name, axes_list) in zip(axes[3:], axis_groups.items()):
        sub = stability[stability["perturbation_axis"].isin(axes_list)]
        box_strip(ax, sub, "perturbation_axis", "score", f"{name} perturbations", rotate=25)
        ax.set_xlabel("")
    source = save_source(stability, "Figure_7_simulated_robustness_source_data.csv")
    add_contract(
        "Figure 7",
        "The 50 simulated configurations reveal perturbation-specific robustness patterns rather than one universal robust method.",
        ["a: stability matrix", "b: median robustness", "c: family robustness", "d: batch perturbations", "e: scale perturbations", "f: signal perturbations"],
        source,
        "Figure 7 summarizes stability across synthetic perturbation axes from the simulated-data design.",
    )
    save_figure(fig, MAIN_OUT / "Figure_7_simulated_robustness", source)


def fig8(tables: dict[str, pd.DataFrame]) -> None:
    efficiency = tables["efficiency"].copy()
    methods = tables["methods"]
    fmap = family_map(methods)
    manifest = pd.read_csv(ROOT / "revision_benchmark" / "config" / "methods_install_manifest.csv")
    commits = pd.read_csv(ROOT / "revision_benchmark" / "config" / "source_commits.csv")
    manifest_plot = prepare_public_source(manifest)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    for method in efficiency["method_id"].dropna().unique():
        sub = efficiency[efficiency["method_id"] == method].sort_values("n_cells")
        axes[0].plot(sub["n_cells"], sub["runtime_seconds"], lw=0.8, alpha=0.65, color=method_color(method, fmap))
        axes[1].plot(sub["n_cells"], sub["peak_memory_gb"], lw=0.8, alpha=0.65, color=method_color(method, fmap))
    for ax, title, ylabel in [(axes[0], "Runtime scaling", "seconds"), (axes[1], "Memory scaling", "GB")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
        ax.set_xlabel("cells")
        ax.set_ylabel(ylabel)
        ax.grid(color=PALETTE["light_gray"], lw=0.45)
        set_ticks(ax)
    coverage = efficiency.assign(ran=1).pivot_table(index="method_id", columns="n_cells", values="ran", aggfunc="max").fillna(0)
    heatmap(axes[2], coverage, "Run coverage", cmap=COVERAGE_CMAP, vmin=0, vmax=1, cbar_label="observed", xrot=45)
    largest = efficiency.sort_values(["method_id", "n_cells"]).groupby("method_id").tail(1)
    axes[3].scatter(largest["runtime_seconds"], largest["peak_memory_gb"], s=24, color=[method_color(m, fmap) for m in largest["method_id"]], edgecolor="white", lw=0.45)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_title("Largest available endpoint", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("seconds")
    axes[3].set_ylabel("GB")
    set_ticks(axes[3])
    sns.countplot(data=manifest_plot, x="language", hue="status", ax=axes[4], palette=STATUS_COLORS)
    axes[4].set_title("Install verification", fontsize=7, fontweight="bold", loc="left")
    axes[4].legend(fontsize=4.6, title="")
    set_ticks(axes[4])
    env_counts = manifest_plot.groupby("environment")["method"].nunique().sort_values()
    axes[5].barh(env_counts.index, env_counts.values, color=PALETTE["gold"])
    axes[5].set_title("Environment mapping", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("methods")
    set_ticks(axes[5], 5)
    source = save_source(pd.concat([efficiency.assign(source="efficiency"), manifest.assign(source="install_manifest"), commits.assign(source="source_commits")], ignore_index=True, sort=False), "Figure_8_scalability_reproducibility_source_data.csv")
    add_contract(
        "Figure 8",
        "Scalability and reproducibility are method-specific and require explicit runtime, memory, run coverage, and environment tracking.",
        ["a: runtime scaling", "b: memory scaling", "c: run coverage", "d: largest endpoint", "e: install verification", "f: environments"],
        source,
        "Figure 8 connects the efficiency benchmark with reproducibility evidence from installation manifests and source commits.",
    )
    save_figure(fig, MAIN_OUT / "Figure_8_scalability_reproducibility", source)


def fig9(tables: dict[str, pd.DataFrame]) -> None:
    score = tables["score_matrix"].set_index("method_id")
    workflow = tables["workflow"]
    hvg = tables["hvg"]
    methods = tables["methods"]
    fmap = family_map(methods)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    criteria = ["local", "global", "kmeans", "runtime_score", "memory_score", "stability_median", "overall_mean"]
    rank_rows = []
    for criterion in criteria:
        if criterion in score.columns:
            top = score[criterion].dropna().sort_values(ascending=False).head(5)
            for rank, (method, value) in enumerate(top.items(), start=1):
                rank_rows.append({"criterion": criterion, "rank": rank, "method_id": method, "score": value})
    ranks = pd.DataFrame(rank_rows)
    top_frequency = ranks[ranks["rank"] <= 3]["method_id"].value_counts()
    ranked_barh(axes[0], top_frequency, "Repeated top-three methods", "appearances", fmap, top=12)
    axes[1].scatter(score["runtime_score"], score["local"], s=28 + 40 * score["stability_median"].fillna(0.5), color=[method_color(m, fmap) for m in score.index], edgecolor="white", lw=0.45)
    axes[1].set_title("Local score versus runtime", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("runtime score")
    axes[1].set_ylabel("local score")
    axes[1].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[1])
    axes[2].scatter(score["memory_score"], score["global"], s=28 + 40 * score["stability_median"].fillna(0.5), color=[method_color(m, fmap) for m in score.index], edgecolor="white", lw=0.45)
    axes[2].set_title("Global score versus memory", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("memory score")
    axes[2].set_ylabel("global score")
    axes[2].grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(axes[2])
    wf_delta = workflow.pivot_table(index=["dataset_id", "method_id", "seed"], columns="workflow", values="ari", aggfunc="mean").dropna().reset_index()
    if {"direct_2d", "pca50_to_2d"}.issubset(wf_delta.columns):
        wf_delta["delta_ari"] = wf_delta["pca50_to_2d"] - wf_delta["direct_2d"]
    wf_summary = wf_delta.groupby("method_id")["delta_ari"].median().sort_values()
    colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in wf_summary.values]
    axes[3].barh(wf_summary.index, wf_summary.values, color=colors)
    axes[3].axvline(0, color=PALETTE["slate"], lw=0.6)
    axes[3].set_title("PCA50 workflow sensitivity", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("median delta ARI")
    set_ticks(axes[3], 5)
    hvg_iqr = hvg.groupby("method_id")["ari"].agg(lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)).sort_values()
    axes[4].barh(hvg_iqr.index, hvg_iqr.values, color=[method_color(m, fmap) for m in hvg_iqr.index])
    axes[4].set_title("Input-gene sensitivity", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("ARI IQR")
    set_ticks(axes[4], 5)
    guide = ranks.pivot_table(index="criterion", columns="rank", values="score", aggfunc="max")
    heatmap(axes[5], guide, "Score of top-ranked methods", cmap="crest", vmin=0, vmax=1, cbar_label="score", annotate=True, xrot=0)
    source = save_source(pd.concat([ranks.assign(source="criterion_ranks"), wf_summary.rename("workflow_delta").reset_index().assign(source="workflow_delta"), hvg_iqr.rename("hvg_ari_iqr").reset_index().assign(source="hvg_iqr")], ignore_index=True, sort=False), "Figure_9_practical_method_selection_source_data.csv")
    add_contract(
        "Figure 9",
        "The benchmark supports task-specific method selection rather than a single universal winner.",
        ["a: repeated top-three methods", "b: local-runtime trade-off", "c: global-memory trade-off", "d: workflow sensitivity", "e: input-gene sensitivity", "f: top-rank score levels"],
        source,
        "Figure 9 converts the benchmark into a practical guide. Recommendations should be phrased as task-specific and evidence-weighted.",
    )
    save_figure(fig, MAIN_OUT / "Figure_9_practical_method_selection_guide", source)


def supplementary_figures(tables: dict[str, pd.DataFrame]) -> None:
    methods = tables["methods"]
    fmap = family_map(methods)
    score = tables["score_matrix"].set_index("method_id")
    score_long = tables["score_long"]
    topo = tables["topology_raw"]
    local, global_ = topology_summaries(topo)
    cluster = tables["cluster_raw"].dropna(subset=["value"])
    efficiency = tables["efficiency"]
    stability = tables["stability"]
    real = tables["real_atlas"]
    real_detail = tables["real_detail"]
    sim = tables["sim_atlas"]

    # S1 method catalog.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    full = methods[methods["benchmark_scope"] == "full_26_method_benchmark"]
    heatmap(axes[0], full.assign(value=1).pivot_table(index="method_id", columns="method_family", values="value", aggfunc="max").fillna(0), "26-method family membership", cmap=MEMBERSHIP_CMAP, vmin=0, vmax=1, cbar_label="member", xrot=35)
    lang = full.assign(value=1).pivot_table(index="method_id", columns="implementation_language", values="value", aggfunc="max").fillna(0)
    heatmap(axes[1], lang, "Implementation language", cmap=COVERAGE_CMAP, vmin=0, vmax=1, cbar_label="available", xrot=35)
    scope = methods.assign(value=1).pivot_table(index="method_id", columns="benchmark_scope", values="value", aggfunc="max").fillna(0).rename(columns=SCOPE_LABELS)
    heatmap(axes[2], scope, "Scope membership", cmap=MEMBERSHIP_CMAP, vmin=0, vmax=1, cbar_label="member", xrot=35)
    variants = methods[methods["benchmark_scope"].isin(["original_result_variant", "targeted_revision_control_only"])]
    axes[3].barh(variants["method_id"], np.ones(len(variants)), color=[method_color(m, fmap) for m in variants["parent_method"]])
    axes[3].set_title("Variants and targeted analyses", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("listed")
    set_ticks(axes[3], 5)
    source = save_source(methods, "Supplementary_Figure_S1_method_catalog_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S1_method_catalog", source)

    # S2 100 dataset atlas.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    real_counts = real.groupby("dataset_group")["dataset_label"].nunique().sort_values()
    axes[0].barh(real_counts.index, real_counts.values, color=PALETTE["teal"])
    axes[0].set_title("Real atlas groups", fontsize=7, fontweight="bold", loc="left")
    axes[0].set_xlabel("datasets")
    set_ticks(axes[0], 5)
    sim_counts = sim.groupby("param_group")["dataset_label"].nunique().sort_values()
    axes[1].barh(sim_counts.index, sim_counts.values, color=PALETTE["gold"])
    axes[1].set_title("Simulated atlas axes", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("configurations")
    set_ticks(axes[1], 5)
    draw_cards(axes[2], [("50", "real atlas", PALETTE["teal"]), ("50", "simulated atlas", PALETTE["gold"]), ("49", "explicit real detail", PALETTE["blue"]), ("50", "sim detail rows", PALETTE["rose"])])
    axes[2].set_title("Counting layers", fontsize=7, fontweight="bold", loc="left")
    combined = pd.concat([real[["dataset_label", "dataset_type"]], sim[["dataset_label", "dataset_type"]]], ignore_index=True)
    combined["value"] = 1
    layer_counts = combined.groupby("dataset_type")["value"].sum()
    axes[3].bar(layer_counts.index, layer_counts.values, color=[DATASET_COLORS.get(x, PALETTE["gray"]) for x in layer_counts.index])
    axes[3].set_title("100-dataset manuscript atlas", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_ylabel("datasets")
    set_ticks(axes[3])
    source = save_source(pd.concat([real.assign(source="real_atlas"), sim.assign(source="simulated_atlas")], ignore_index=True, sort=False), "Supplementary_Figure_S2_full_dataset_atlas_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S2_full_100_dataset_atlas", source)

    # S3 real dataset landscape.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    sns.scatterplot(data=real_detail, x="cells", y="sparsity_pct", hue="category", size="genes", sizes=(12, 85), ax=axes[0], linewidth=0.3, edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_title("Cells versus sparsity", fontsize=7, fontweight="bold", loc="left")
    axes[0].legend(fontsize=4.5, title="")
    set_ticks(axes[0])
    sns.scatterplot(data=real_detail, x="genes", y="cell_types", hue="category", size="cells", sizes=(12, 85), ax=axes[1], linewidth=0.3, edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Genes versus cell types", fontsize=7, fontweight="bold", loc="left")
    axes[1].legend(fontsize=4.5, title="")
    set_ticks(axes[1])
    cat_counts = real_detail.groupby("category")["dataset"].nunique().sort_values()
    axes[2].barh(cat_counts.index, cat_counts.values, color=PALETTE["blue"])
    axes[2].set_title("Explicit real-detail rows", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("rows")
    set_ticks(axes[2], 5)
    box_strip(axes[3], real_detail, "category", "sparsity_pct", "Sparsity by source group", rotate=25)
    source = save_source(real_detail, "Supplementary_Figure_S3_real_dataset_landscape_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S3_real_dataset_landscape", source)

    # S4 simulated parameter landscape.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    sns.scatterplot(data=sim, x="cells", y="sparsity_pct", hue="param_group", size="genes", sizes=(12, 85), ax=axes[0], linewidth=0.3, edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_title("Simulated scale and sparsity", fontsize=7, fontweight="bold", loc="left")
    axes[0].legend(fontsize=4.5, title="")
    set_ticks(axes[0])
    counts = sim.groupby("param_group")["dataset_label"].nunique().sort_values()
    axes[1].barh(counts.index, counts.values, color=PALETTE["gold"])
    axes[1].set_title("Configurations per axis", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("configurations")
    set_ticks(axes[1], 5)
    box_strip(axes[2], sim, "param_group", "sparsity_pct", "Sparsity by axis", rotate=25)
    box_strip(axes[3], sim, "param_group", "size_mb", "File size by axis", rotate=25)
    source = save_source(sim, "Supplementary_Figure_S4_simulated_parameter_landscape_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S4_simulated_parameter_landscape", source)

    # S5-S15.
    simple_heatmap_figure(score.drop(columns=["overall_mean"]), "Supplementary Fig. S5: full score matrix", SUPP_OUT / "Supplementary_Figure_S5_full_score_matrix", save_source(score_long, "Supplementary_Figure_S5_full_score_matrix_source_data.csv"), cmap="mako")
    simple_heatmap_figure(local.reindex(sort_methods_by_score(tables["score_matrix"])).dropna(how="all"), "Supplementary Fig. S6: local neighborhood metrics", SUPP_OUT / "Supplementary_Figure_S6_local_neighborhood_metrics", save_source(local.reset_index(), "Supplementary_Figure_S6_local_neighborhood_source_data.csv"), cmap="viridis")
    simple_heatmap_figure(global_.reindex(sort_methods_by_score(tables["score_matrix"])).dropna(how="all"), "Supplementary Fig. S7: global geometry metrics", SUPP_OUT / "Supplementary_Figure_S7_global_geometry_metrics", save_source(global_.reset_index(), "Supplementary_Figure_S7_global_geometry_source_data.csv"), cmap="mako")
    cluster_mat = cluster.pivot_table(index="method_id", columns=["clustering_algorithm", "metric"], values="value", aggfunc="median")
    simple_heatmap_figure(cluster_mat, "Supplementary Fig. S8: clustering metrics", SUPP_OUT / "Supplementary_Figure_S8_clustering_metrics", save_source(cluster, "Supplementary_Figure_S8_clustering_metrics_source_data.csv"), cmap="viridis")

    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for method in efficiency["method_id"].dropna().unique():
        sub = efficiency[efficiency["method_id"] == method].sort_values("n_cells")
        axes[0].plot(sub["n_cells"], sub["runtime_seconds"], lw=0.8, alpha=0.6, color=method_color(method, fmap))
        axes[1].plot(sub["n_cells"], sub["peak_memory_gb"], lw=0.8, alpha=0.6, color=method_color(method, fmap))
    for ax, title, ylabel in [(axes[0], "Runtime scaling", "seconds"), (axes[1], "Memory scaling", "GB")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
        ax.set_xlabel("cells")
        ax.set_ylabel(ylabel)
        set_ticks(ax)
    coverage = efficiency.assign(ran=1).pivot_table(index="method_id", columns="n_cells", values="ran", aggfunc="max").fillna(0)
    heatmap(axes[2], coverage, "Run coverage", cmap=COVERAGE_CMAP, vmin=0, vmax=1, cbar_label="observed", xrot=45)
    largest = efficiency.sort_values(["method_id", "n_cells"]).groupby("method_id").tail(1)
    axes[3].scatter(largest["runtime_seconds"], largest["peak_memory_gb"], color=[method_color(m, fmap) for m in largest["method_id"]], s=24, edgecolor="white", lw=0.45)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_title("Largest endpoint", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("seconds")
    axes[3].set_ylabel("GB")
    source = save_source(efficiency, "Supplementary_Figure_S9_efficiency_scaling_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S9_efficiency_scaling", source)

    stab_mat = stability.pivot_table(index="method_id", columns="perturbation_axis", values="score", aggfunc="mean")
    simple_heatmap_figure(stab_mat, "Supplementary Fig. S10: stability matrix", SUPP_OUT / "Supplementary_Figure_S10_stability_full_matrix", save_source(stability, "Supplementary_Figure_S10_stability_source_data.csv"), cmap="crest")
    revision_detail_figure(tables["scvi"], "dimension", "Supplementary Fig. S11: scVI reference analysis", SUPP_OUT / "Supplementary_Figure_S11_scVI_reference_analysis", "Supplementary_Figure_S11_scVI_reference_source_data.csv")
    revision_detail_figure(tables["dimension"], "dimension", "Supplementary Fig. S12: latent-dimension sensitivity", SUPP_OUT / "Supplementary_Figure_S12_latent_dimension_sensitivity", "Supplementary_Figure_S12_dimension_source_data.csv")
    workflow_detail_figure(tables["workflow"], SUPP_OUT / "Supplementary_Figure_S13_visualization_workflow_comparison")
    revision_detail_figure(tables["hvg"], "hvg_requested", "Supplementary Fig. S14: input-gene sensitivity", SUPP_OUT / "Supplementary_Figure_S14_input_gene_sensitivity", "Supplementary_Figure_S14_input_gene_source_data.csv")
    reproducibility_supp_figure(methods)


def simple_heatmap_figure(data: pd.DataFrame, title: str, base: Path, source: str, *, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    heatmap(ax, data, title, cmap=cmap, vmin=0, vmax=1, cbar_label="median/score")
    add_panel_label(ax, "a", x=-0.05)
    save_figure(fig, base, source)


def revision_detail_figure(df: pd.DataFrame, x: str, title: str, base: Path, source_name: str) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    metrics = [m for m in ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"] if m in df.columns]
    for ax, metric in zip(axes, metrics):
        box_strip(ax, df, x, metric, f"{title}: {metric}", rotate=0)
        if metric == "runtime_seconds":
            ax.set_yscale("log")
    source = save_source(df, source_name)
    save_figure(fig, base, source)


def workflow_detail_figure(df: pd.DataFrame, base: Path) -> None:
    plot_df = prepare_public_source(df)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for ax, metric in zip(axes, ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"]):
        box_strip(ax, plot_df, "method_id", metric, f"Workflow comparison: {metric}", hue="workflow", palette=["#DDEAF7", "#F1DCC9"], rotate=25)
        if metric == "runtime_seconds":
            ax.set_yscale("log")
    source = save_source(df, "Supplementary_Figure_S13_workflow_comparison_source_data.csv")
    save_figure(fig, base, source)


def reproducibility_supp_figure(methods: pd.DataFrame) -> None:
    manifest = pd.read_csv(ROOT / "revision_benchmark" / "config" / "methods_install_manifest.csv")
    commits = pd.read_csv(ROOT / "revision_benchmark" / "config" / "source_commits.csv")
    manifest_plot = prepare_public_source(manifest)
    commits_plot = prepare_public_source(commits)
    export_counts = pd.DataFrame(MANIFEST_ROWS)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    sns.countplot(data=manifest_plot, x="language", hue="status", ax=axes[0], palette=STATUS_COLORS)
    axes[0].set_title("Install status", fontsize=7, fontweight="bold", loc="left")
    axes[0].legend(fontsize=4.6, title="")
    role = manifest_plot.groupby("role")["method"].nunique().sort_values()
    axes[1].barh(role.index, role.values, color=PALETTE["slate"])
    axes[1].set_title("Manifest roles", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("methods")
    set_ticks(axes[1], 5)
    axes[2].barh(commits_plot["source"].head(12), np.ones(min(12, len(commits_plot))), color=PALETTE["teal"])
    axes[2].set_title("Source commit records", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("recorded")
    set_ticks(axes[2], 4.8)
    source_counts = export_counts.groupby("format")["figure"].nunique() if not export_counts.empty else pd.Series(dtype=int)
    axes[3].bar(source_counts.index, source_counts.values, color=PALETTE["gold"])
    axes[3].set_title("Figure exports before S15", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_ylabel("figures")
    set_ticks(axes[3])
    source = save_source(pd.concat([manifest.assign(source="install_manifest"), commits.assign(source="source_commits"), methods.assign(source="method_manifest")], ignore_index=True, sort=False), "Supplementary_Figure_S15_reproducibility_audit_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S15_reproducibility_source_audit", source)


def write_qa() -> None:
    files = sorted(list(MAIN_OUT.glob("*.png")) + list(SUPP_OUT.glob("*.png")))
    pd.DataFrame([{"file": package_rel(p), "bytes": p.stat().st_size} for p in files]).to_csv(QA_OUT / "png_file_size_check.csv", index=False)
    notes = [
        "# Redesigned Figure QA Notes",
        "",
        "- Backend exclusivity: Python/matplotlib/seaborn only.",
        "- SVG text: svg.fonttype = none.",
        "- PDF text: pdf.fonttype = 42.",
        "- Exports: SVG, PDF, PNG, TIFF for each figure.",
        "- scVI is treated as a targeted reference analysis, not as a full 100-dataset benchmark method.",
        "- Figure legends carry interpretive caveats; panels avoid paragraph-style text.",
    ]
    (QA_OUT / "qa_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    reset_outputs()
    tables = load_all()
    fig1(tables)
    fig2(tables)
    fig3(tables)
    fig4(tables)
    fig5(tables)
    fig6(tables)
    fig7(tables)
    fig8(tables)
    fig9(tables)
    supplementary_figures(tables)
    write_contracts()
    save_manifest()
    write_qa()
    print(f"Generated redesigned figure package: {OUT}")
    print(f"Main figures: {len(list(MAIN_OUT.glob('*.svg')))} SVG files")
    print(f"Supplementary figures: {len(list(SUPP_OUT.glob('*.svg')))} SVG files")
    print(f"Manifest: {QA_OUT / 'figure_export_manifest.csv'}")


if __name__ == "__main__":
    main()
