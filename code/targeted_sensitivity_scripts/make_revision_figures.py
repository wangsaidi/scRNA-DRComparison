from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.gridspec import GridSpec


# Mandatory editable SVG text settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 6.2
plt.rcParams["axes.linewidth"] = 0.55
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.45
plt.rcParams["ytick.major.width"] = 0.45
plt.rcParams["savefig.facecolor"] = "white"


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Publication" / "paper" / "revision_figures" / "python_figure_package"
MAIN_OUT = OUT / "main_figures"
SUPP_OUT = OUT / "supplementary_figures"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for directory in [MAIN_OUT, SUPP_OUT, SOURCE_OUT, QA_OUT]:
    directory.mkdir(parents=True, exist_ok=True)


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
    "red": "#B94D4A",
    "up": "#2E9E44",
    "down": "#C94949",
}

STATUS_PALETTE = {
    "installed_verified": "#A7C7E7",
    "source_import_verified": "#D8D8D8",
    "source_import_partial": "#F0C0CC",
    "not_verified": "#F0C0CC",
}

FAMILY_COLORS = {
    "linear/probabilistic": PALETTE["blue"],
    "deep generative": PALETTE["rose"],
    "deep generative + scVI": PALETTE["violet"],
    "deep autoencoder": PALETTE["green"],
    "graph autoencoder": "#7BAAA7",
    "visualization/graph": PALETTE["gold"],
    "label-sensitive": "#B98AC7",
    "other": PALETTE["gray"],
}

METRIC_LABELS = {
    "local": "Local score",
    "global": "Global score",
    "kmeans": "k-means score",
    "louvain": "Louvain score",
    "spectral": "Spectral score",
    "runtime_score": "Runtime score",
    "memory_score": "Memory score",
    "stability_median": "Stability median",
    "overall_mean": "Mean score",
}

MANIFEST_ROWS: list[dict[str, str]] = []
CONTRACTS: list[dict[str, object]] = []


def norm_method(method: object) -> str:
    text = str(method)
    mapping = {
        "TSNE": "t-SNE",
        "ivis": "IVIS",
        "SQuaD_MDS": "SQuaD-MDS",
        "SQuaD_MDS_hybrid": "SQuaD-MDS hybrid",
        "ParametricUMAP50": "Parametric UMAP 50",
        "ParametricUMAP200": "Parametric UMAP 200",
    }
    return mapping.get(text, text)


def label_family(raw_family: object) -> str:
    text = str(raw_family)
    mapping = {
        "linear_probabilistic": "linear/probabilistic",
        "deep_generative": "deep generative",
        "deep_generative_added": "deep generative + scVI",
        "deep_autoencoder": "deep autoencoder",
        "graph_autoencoder": "graph autoencoder",
        "visualization_or_graph": "visualization/graph",
        "label_sensitive_method": "label-sensitive",
    }
    return mapping.get(text, "other")


def add_contract(
    fig_id: str,
    conclusion: str,
    archetype: str,
    panels: list[str],
    source_data: str,
    reviewer_risk: str,
) -> None:
    CONTRACTS.append(
        {
            "figure": fig_id,
            "core_conclusion": conclusion,
            "archetype": archetype,
            "backend": "Python/matplotlib/seaborn",
            "target_output": "Communications Biology revision; double-column figure package",
            "final_size": "183 mm wide; dense multi-panel page",
            "panels": panels,
            "source_data": source_data,
            "reviewer_risk": reviewer_risk,
        }
    )


def save_contracts() -> None:
    lines: list[str] = [
        "# Revision Figure Contracts",
        "",
        "Backend: Python only for plotting, preview, and export.",
        "Output formats: SVG primary, plus PDF, PNG preview, and TIFF.",
        "",
    ]
    for item in CONTRACTS:
        lines.extend(
            [
                f"## {item['figure']}",
                f"Core conclusion: {item['core_conclusion']}",
                f"Figure archetype: {item['archetype']}",
                f"Target/output: {item['target_output']}",
                f"Backend: {item['backend']}",
                f"Final size: {item['final_size']}",
                "Panel map:",
            ]
        )
        for panel in item["panels"]:
            lines.append(f"- {panel}")
        lines.extend(
            [
                f"Source data needed: {item['source_data']}",
                "Image-integrity notes: quantitative panels are script-generated from project CSV files; no image adjustment.",
                f"Reviewer risk: {item['reviewer_risk']}",
                "",
            ]
        )
    (QA_OUT / "figure_contracts.md").write_text("\n".join(lines), encoding="utf-8")


def save_manifest() -> None:
    pd.DataFrame(MANIFEST_ROWS).to_csv(QA_OUT / "figure_export_manifest.csv", index=False)


def save_source(df: pd.DataFrame, name: str) -> str:
    path = SOURCE_OUT / name
    df.to_csv(path, index=False)
    return str(path.relative_to(ROOT))


def save_figure(fig: plt.Figure, base_path: Path, source_file: str, dpi: int = 600) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    formats = ["svg", "pdf", "png", "tiff"]
    for fmt in formats:
        out_path = base_path.with_suffix(f".{fmt}")
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        MANIFEST_ROWS.append(
            {
                "figure": base_path.name,
                "format": fmt,
                "path": str(out_path.relative_to(ROOT)),
                "source_data": source_file,
            }
        )
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["dark"],
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def wrap_text(text: str, width: int = 36) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    fontsize: float = 6.0,
    lw: float = 0.8,
) -> patches.FancyBboxPatch:
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor=PALETTE["dark"],
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["dark"],
    )
    return rect


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PALETTE["slate"]),
    )


def text_panel(
    ax: plt.Axes,
    title: str,
    bullets: list[str],
    *,
    title_color: str = PALETTE["dark"],
    width: int = 42,
) -> None:
    clean_axis(ax)
    ax.text(0, 0.98, title, ha="left", va="top", fontsize=7.4, fontweight="bold", color=title_color)
    y = 0.82
    for bullet in bullets:
        ax.text(0.03, y, f"- {wrap_text(bullet, width)}", ha="left", va="top", fontsize=5.7, linespacing=1.25)
        y -= 0.18 if len(bullet) < 70 else 0.23


def set_small_ticks(ax: plt.Axes, labelsize: float = 5.2) -> None:
    ax.tick_params(axis="both", labelsize=labelsize, length=2.2, pad=1.5)


def method_color(method: object, family_map: dict[str, str]) -> str:
    family = family_map.get(norm_method(method), "other")
    return FAMILY_COLORS.get(family, FAMILY_COLORS["other"])


def load_method_grid() -> tuple[pd.DataFrame, dict[str, str]]:
    path = ROOT / "revision_benchmark" / "experiments" / "method_dimension_grid.csv"
    grid = pd.read_csv(path)
    grid["method_norm"] = grid["method"].map(norm_method)
    grid["family_label"] = grid["method_family"].map(label_family)
    family_map = dict(zip(grid["method_norm"], grid["family_label"]))
    extra = {
        "Parametric UMAP 50": "visualization/graph",
        "Parametric UMAP 200": "visualization/graph",
        "SQuaD-MDS hybrid": "visualization/graph",
        "SQuaD-MDS": "visualization/graph",
        "SSNMDI": "label-sensitive",
        "IVIS": "visualization/graph",
        "t-SNE": "visualization/graph",
    }
    family_map.update(extra)
    return grid, family_map


def load_datasets() -> pd.DataFrame:
    path = ROOT / "revision_benchmark" / "experiments" / "datasets_manifest.csv"
    df = pd.read_csv(path)
    df["n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce")
    df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce")
    return df


def load_score_long() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    score_map = {
        "local": ROOT / "metric" / "score" / "local" / "local.csv",
        "global": ROOT / "metric" / "score" / "global" / "global.csv",
        "kmeans": ROOT / "metric" / "score" / "cluster" / "kmeans.csv",
        "louvain": ROOT / "metric" / "score" / "cluster" / "louvain.csv",
        "spectral": ROOT / "metric" / "score" / "cluster" / "spectral.csv",
        "runtime_score": ROOT / "metric" / "score" / "efficiency" / "time.csv",
        "memory_score": ROOT / "metric" / "score" / "efficiency" / "memory.csv",
    }
    for metric, path in score_map.items():
        df = pd.read_csv(path)
        df["method"] = df["Method"].map(norm_method)
        df["metric"] = metric
        rows.append(df[["method", "metric", "score"]])
    stability_rows: list[pd.DataFrame] = []
    for path in sorted((ROOT / "metric" / "score" / "stability").glob("*.csv")):
        df = pd.read_csv(path)
        df["method"] = df["Method"].map(norm_method)
        df["metric"] = path.stem
        df["perturbation"] = path.stem
        stability_rows.append(df[["method", "metric", "perturbation", "score"]])
    stability_long = pd.concat(stability_rows, ignore_index=True)
    stability_median = (
        stability_long.groupby("method", as_index=False)["score"].median().assign(metric="stability_median")
    )
    score_long = pd.concat(rows + [stability_median[["method", "metric", "score"]]], ignore_index=True)
    score_matrix = score_long.pivot_table(index="method", columns="metric", values="score", aggfunc="mean")
    preferred = ["local", "global", "kmeans", "louvain", "spectral", "runtime_score", "memory_score", "stability_median"]
    score_matrix = score_matrix.reindex(columns=[c for c in preferred if c in score_matrix.columns])
    score_matrix["overall_mean"] = score_matrix.mean(axis=1, skipna=True)
    score_matrix = score_matrix.sort_values("overall_mean", ascending=False)
    return score_long, score_matrix, stability_long


def load_efficiency() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "metric" / "efficiency").glob("cell_*.csv")):
        df = pd.read_csv(path)
        n = int(path.stem.split("_")[1])
        df["n_cells"] = n
        df["method"] = df["Method"].map(norm_method)
        df = df.rename(columns={"PeakMemory(gb)": "peak_memory_gb", "Time(s)": "time_s"})
        rows.append(df[["method", "n_cells", "peak_memory_gb", "time_s"]])
    out = pd.concat(rows, ignore_index=True)
    out["peak_memory_gb"] = pd.to_numeric(out["peak_memory_gb"], errors="coerce")
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    return out


def load_revision_tables() -> dict[str, pd.DataFrame]:
    paths = {
        "wp1": ROOT / "revision_benchmark" / "results" / "metrics" / "WP1_scVI_local_scVI_metrics.csv",
        "wp2": ROOT / "revision_benchmark" / "results" / "source_data" / "wp2_dimension_sensitivity_long.csv",
        "wp3": ROOT / "revision_benchmark" / "results" / "source_data" / "wp3_visualization_workflow_long.csv",
        "wp4": ROOT / "revision_benchmark" / "results" / "source_data" / "wp4_input_gene_sensitivity_long.csv",
    }
    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    for df in tables.values():
        if "method" in df.columns:
            df["method"] = df["method"].map(norm_method)
        for col in ["dimension", "hvg_requested", "runtime_seconds", "max_rss_mb"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return tables


def heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cbar_label: str = "score",
    annotate: bool = False,
    xrot: int = 35,
) -> None:
    matrix = data.astype(float).to_numpy()
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#F1F1F1")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=3)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([METRIC_LABELS.get(str(c), str(c)) for c in data.columns], rotation=xrot, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    set_small_ticks(ax, 4.8 if data.shape[0] > 20 else 5.3)
    if annotate and data.shape[0] <= 10 and data.shape[1] <= 8:
        finite = np.isfinite(matrix)
        int_like = finite.any() and np.all(np.isclose(matrix[finite], np.round(matrix[finite]))) and np.nanmax(matrix) > 1.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = matrix[i, j]
                if not np.isfinite(val):
                    continue
                label = f"{val:.0f}" if int_like else f"{val:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=4.8, color="white" if val > 0.55 else "black")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=5, length=2)
    cbar.set_label(cbar_label, fontsize=5.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def ranked_barh(
    ax: plt.Axes,
    values: pd.Series,
    title: str,
    xlabel: str,
    family_map: dict[str, str],
    *,
    top: int = 12,
    xlim: tuple[float, float] | None = None,
) -> None:
    vals = values.dropna().sort_values(ascending=False).head(top).iloc[::-1]
    colors = [method_color(method, family_map) for method in vals.index]
    ax.barh(vals.index, vals.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    set_small_ticks(ax, 5)
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)


def line_with_interval(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    family_map: dict[str, str],
    *,
    xlabel: str = "",
    max_groups: int | None = None,
    legend: bool = True,
    direct_labels: bool = False,
) -> None:
    work = df[[x, y, group]].dropna().copy()
    order = sorted(work[x].dropna().unique())
    group_scores = work.groupby(group)[y].median().sort_values(ascending=False)
    groups = list(group_scores.index)
    if max_groups:
        groups = groups[:max_groups]
    label_positions = []
    for name in groups:
        sub = work[work[group] == name]
        summary = sub.groupby(x)[y].agg(["mean", "sem"]).reindex(order)
        color = method_color(name, family_map)
        ax.plot(order, summary["mean"], marker="o", ms=3.2, lw=1.1, color=color, label=name)
        sem = summary["sem"].fillna(0)
        ax.fill_between(order, summary["mean"] - sem, summary["mean"] + sem, color=color, alpha=0.12, lw=0)
        if direct_labels:
            last_valid = summary["mean"].dropna()
            if not last_valid.empty:
                label_positions.append((name, last_valid.index[-1], float(last_valid.iloc[-1]), color))
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(order)
    set_small_ticks(ax)
    ax.grid(axis="y", color=PALETTE["light_gray"], lw=0.45)
    if direct_labels and label_positions:
        x_min, x_max = min(order), max(order)
        x_pad = (x_max - x_min) * 0.08 if x_max != x_min else 1
        ax.set_xlim(x_min, x_max + x_pad)
        used: list[float] = []
        for name, x_pos, y_pos, color in sorted(label_positions, key=lambda t: t[2]):
            y_label = y_pos
            for prior in used:
                if abs(y_label - prior) < 0.035:
                    y_label += 0.035
            used.append(y_label)
            ax.text(x_pos + x_pad * 0.18, y_label, name, fontsize=4.7, va="center", color=color)
    elif legend:
        ax.legend(fontsize=4.7, ncol=2, loc="best", handlelength=1.4, columnspacing=0.8)


def family_legend(ax: plt.Axes, family_map: dict[str, str]) -> None:
    seen = sorted(set(family_map.values()))
    handles = [
        patches.Patch(facecolor=FAMILY_COLORS.get(fam, FAMILY_COLORS["other"]), edgecolor="none", label=fam)
        for fam in seen
        if fam in FAMILY_COLORS
    ]
    ax.legend(handles=handles, fontsize=4.9, ncol=2, loc="lower left", bbox_to_anchor=(0, -0.05))


def draw_score_scatter(
    ax: plt.Axes,
    score_matrix: pd.DataFrame,
    family_map: dict[str, str],
    x: str,
    y: str,
    title: str,
    *,
    label_top: int = 8,
) -> None:
    work = score_matrix[[x, y, "overall_mean"]].dropna().copy()
    for method, row in work.iterrows():
        ax.scatter(
            row[x],
            row[y],
            s=18 + 42 * float(row.get("overall_mean", 0.5)),
            color=method_color(method, family_map),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
        )
    top_names = list(work["overall_mean"].sort_values(ascending=False).head(label_top).index)
    anchors = [name for name in ["PCA", "UMAP", "t-SNE", "scGBM", "scVI"] if name in work.index]
    label_names = []
    for name in top_names + anchors:
        if name not in label_names:
            label_names.append(name)
    offsets = [(5, 5), (5, -7), (-22, 6), (-22, -8), (6, 12), (-30, 12), (8, -14), (-34, -14)]
    for i, method in enumerate(label_names[: max(label_top + 3, 6)]):
        row = work.loc[method]
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            method,
            xy=(row[x], row[y]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.5,
            color=PALETTE["dark"],
            arrowprops=dict(arrowstyle="-", color=PALETTE["gray"], lw=0.35, shrinkA=0, shrinkB=2),
        )
    ax.set_title(title, fontsize=7, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel(METRIC_LABELS.get(x, x))
    ax.set_ylabel(METRIC_LABELS.get(y, y))
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    set_small_ticks(ax)
    ax.grid(color=PALETTE["light_gray"], lw=0.45)


def fig1_taxonomy(grid: pd.DataFrame, datasets: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.1, 1.0], width_ratios=[1.05, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1:])
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label, x=-0.035 if ax is ax_a else -0.08)

    ax = ax_a
    clean_axis(ax)
    ax.set_title("Taxonomy and common notation for single-cell dimensionality reduction benchmarking", fontsize=7.6, fontweight="bold", loc="left")
    steps = [
        ("Expression matrix\nX in R^(n x p)", "#DFEAF8"),
        ("Preprocessing\nHVG / log / scale", "#E7F2EC"),
        ("DR operator\nf_theta(X)", "#F4E5DC"),
        ("Embedding\nZ in R^(n x k)", "#EAE4F4"),
        ("Evaluation\nM(Z, labels, X)", "#ECECEC"),
    ]
    xs = np.linspace(0.035, 0.79, len(steps))
    for (text, color), x in zip(steps, xs):
        draw_box(ax, (x, 0.48), 0.16, 0.25, text, color, fontsize=5.8)
    for i in range(len(xs) - 1):
        draw_arrow(ax, (xs[i] + 0.16, 0.61), (xs[i + 1], 0.61))
    ax.plot([0.35, 0.88], [0.30, 0.30], color=PALETTE["slate"], lw=0.7)
    ax.text(
        0.04,
        0.31,
        "Revision stance",
        fontsize=6.1,
        fontweight="bold",
        color=PALETTE["dark"],
        ha="left",
        va="center",
    )
    ax.text(
        0.17,
        0.31,
        "common notation and method taxonomy; no claim that all methods share one mathematical theory",
        fontsize=5.9,
        color=PALETTE["slate"],
        ha="left",
        va="center",
    )

    ax = ax_b
    fam_counts = grid.groupby("family_label")["method_norm"].nunique().sort_values()
    ax.barh(fam_counts.index, fam_counts.values, color=[FAMILY_COLORS.get(f, PALETTE["gray"]) for f in fam_counts.index])
    ax.set_title("Method families", fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel("methods")
    set_small_ticks(ax, 5)
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    ax.set_xlim(0, max(fam_counts.values) * 1.18)
    for y, v in enumerate(fam_counts.values):
        ax.text(v + 0.12, y, str(int(v)), va="center", fontsize=5.3)

    ax = ax_c
    clean_axis(ax)
    ax.set_title("Two distinct tasks", fontsize=7, fontweight="bold", loc="left")
    draw_box(ax, (0.08, 0.64), 0.78, 0.16, "Latent summarization", "#E1ECF7", fontsize=6.0)
    ax.text(0.10, 0.52, "k = 2, 5, 10, 20, 50\nPCA, scVI, GLMPCA, pCMF,\nscGBM, SAUCIE", fontsize=5.2, va="top")
    draw_box(ax, (0.08, 0.23), 0.78, 0.16, "2D visualization endpoint", "#F3E7D5", fontsize=6.0)
    ax.text(0.10, 0.11, "direct 2D or PCA50 -> 2D\nUMAP, t-SNE, PaCMAP, PHATE", fontsize=5.2, va="top")

    ax = ax_d
    clean_axis(ax)
    ax.set_title("Evaluation families", fontsize=7, fontweight="bold", loc="left")
    metrics = [
        ("Local", "neighborhood preservation"),
        ("Global", "large-scale geometry"),
        ("Clustering", "label concordance"),
        ("Efficiency", "runtime and memory"),
        ("Stability", "perturbation robustness"),
    ]
    for i, (head, body) in enumerate(metrics):
        y = 0.82 - i * 0.16
        draw_box(ax, (0.03, y), 0.30, 0.105, head, "#F0F0F0", fontsize=5.5, lw=0.5)
        ax.text(0.38, y + 0.052, body, fontsize=5.3, va="center", ha="left")

    ax = ax_e
    clean_axis(ax)
    ax.set_title("Revision evidence scale", fontsize=7, fontweight="bold", loc="left")
    evidence = [
        ("methods", grid["method_norm"].nunique()),
        ("manifest rows", len(datasets)),
        ("dataset groups", datasets["dataset_group"].nunique()),
        ("work packages", 4),
    ]
    positions = [(0.05, 0.60), (0.53, 0.60), (0.05, 0.22), (0.53, 0.22)]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["violet"]]
    for (label, value), (x, y), color in zip(evidence, positions, colors):
        draw_box(ax, (x, y), 0.37, 0.25, "", "#F5F5F5", fontsize=5.0, lw=0.55)
        ax.text(x + 0.05, y + 0.15, f"{value}", fontsize=14, fontweight="bold", color=color, ha="left", va="center")
        ax.text(x + 0.05, y + 0.065, label, fontsize=5.4, color=PALETTE["dark"], ha="left", va="center")

    ax = ax_f
    clean_axis(ax)
    ax.set_title("Reviewer-risk guardrails encoded in the figures", fontsize=7, fontweight="bold", loc="left")
    guard = [
        ("Labels", "simulation labels / annotation concordance"),
        ("Scores", "distribution and source data, not opaque winner claims"),
        ("Runtime", "method-specific environments"),
        ("Scope", "targeted WP2/WP4 controls, not full factorial reruns"),
    ]
    y0 = 0.74
    for i, (head, body) in enumerate(guard):
        y = y0 - i * 0.18
        ax.add_patch(patches.Circle((0.05, y), 0.022, facecolor=PALETTE["teal"], edgecolor="none"))
        ax.text(0.05, y, str(i + 1), ha="center", va="center", fontsize=4.6, color="white", fontweight="bold")
        ax.text(0.10, y, head, fontsize=5.9, fontweight="bold", va="center")
        ax.text(0.25, y, body, fontsize=5.6, va="center", color=PALETTE["slate"])

    source = save_source(
        pd.DataFrame(
            {
                "figure": ["Figure 1"],
                "n_methods": [grid["method_norm"].nunique()],
                "n_dataset_manifest_rows": [len(datasets)],
                "n_work_packages": [4],
            }
        ),
        "Figure_1_taxonomy_source_data.csv",
    )
    add_contract(
        "Figure 1",
        "The revised manuscript presents a taxonomy/common-notation framework that separates latent summarization, visualization endpoints, and evaluation tasks.",
        "schematic-led composite",
        [
            "a: common notation workflow",
            "b: method-family counts",
            "c: latent dimension versus visualization endpoint distinction",
            "d: evaluation-family map",
            "e: revision evidence scale",
            "f: reviewer-risk guardrails",
        ],
        source,
        "Do not overclaim a unified mathematical framework; keep panel text as taxonomy/common notation.",
    )
    save_figure(fig, MAIN_OUT / "Figure_1_taxonomy_common_notation", source)


def fig2_design(grid: pd.DataFrame, datasets: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    ax = axes[0]
    colors = datasets["dataset_group"].map({"synthetic": PALETTE["gold"], "downsampling": PALETTE["teal"]}).fillna(PALETTE["gray"])
    ax.scatter(datasets["n_cells"], datasets["n_genes"], c=colors, s=28, edgecolor="white", lw=0.45, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Local revision dataset scale", fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel("cells")
    ax.set_ylabel("genes")
    handles = [
        patches.Patch(color=PALETTE["gold"], label="synthetic"),
        patches.Patch(color=PALETTE["teal"], label="downsampling"),
    ]
    ax.legend(handles=handles, fontsize=4.8, loc="upper left")
    set_small_ticks(ax)
    ax.grid(color=PALETTE["light_gray"], lw=0.45)

    ax = axes[1]
    group_counts = datasets.groupby("dataset_group")["dataset_id"].nunique().sort_values(ascending=True)
    bars = ax.barh(group_counts.index, group_counts.values, color=[PALETTE["teal"], PALETTE["gold"]][: len(group_counts)])
    ax.set_title("Dataset composition", fontsize=7, fontweight="bold", loc="left")
    ax.set_xlabel("unique dataset ids")
    for bar, val in zip(bars, group_counts.values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, str(int(val)), va="center", fontsize=5.5)
    ax.set_xlim(0, max(group_counts.values) * 1.25)
    set_small_ticks(ax, 5.3)
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)

    ax = axes[2]
    clean_axis(ax)
    ax.set_title("Evaluation path and traceability", fontsize=7, fontweight="bold", loc="left")
    steps = ["input", "embed.", "metrics", "CSV", "figure"]
    subtitles = ["X", "Z", "ARI/NMI/\ngeometry", "CSV", "SVG/PDF/\nTIFF"]
    xs = np.linspace(0.035, 0.835, len(steps))
    for i, (text, x) in enumerate(zip(steps, xs)):
        draw_box(ax, (x, 0.54), 0.125, 0.18, text, ["#E1ECF7", "#F3E7D5", "#E7F2EC", "#ECECEC", "#EAE4F4"][i], fontsize=5.4)
        ax.text(x + 0.0625, 0.42, subtitles[i], ha="center", va="top", fontsize=4.8, color=PALETTE["slate"])
        if i < len(steps) - 1:
            draw_arrow(ax, (x + 0.125, 0.63), (xs[i + 1], 0.63))
    ax.text(0.04, 0.18, "All quantitative panels in this package are regenerated from project CSV files.", fontsize=5.6, color=PALETTE["slate"])

    ax = axes[3]
    role_matrix = grid.pivot_table(index="family_label", columns="analysis_role", values="method_norm", aggfunc="nunique", fill_value=0)
    role_matrix = role_matrix.loc[role_matrix.sum(axis=1).sort_values(ascending=False).index]
    role_matrix.columns = [c.replace("_", "\n") for c in role_matrix.columns]
    heatmap(ax, role_matrix, "Method families by analysis role", cmap="Blues", vmin=0, vmax=max(1, float(role_matrix.values.max())), cbar_label="methods", annotate=True, xrot=25)

    ax = axes[4]
    clean_axis(ax)
    ax.set_title("Revision work-package map", fontsize=7, fontweight="bold", loc="left")
    wp_rows = [
        ("WP1", "scVI", "15 datasets\n5 dimensions"),
        ("WP2", "dimension", "6 methods\n8 datasets"),
        ("WP3", "workflow", "4 visualizers\n2 workflows"),
        ("WP4", "HVG input", "6 methods\n500-3000 HVGs"),
    ]
    for i, (wp, issue, design) in enumerate(wp_rows):
        y = 0.73 - i * 0.17
        draw_box(ax, (0.035, y - 0.045), 0.10, 0.09, wp, "#E1ECF7", fontsize=5.8, lw=0.5)
        ax.text(0.20, y, issue, fontsize=5.7, fontweight="bold", va="center")
        ax.text(0.48, y, design, fontsize=5.2, va="center", color=PALETTE["slate"])
        ax.plot([0.03, 0.93], [y - 0.085, y - 0.085], color=PALETTE["light_gray"], lw=0.5)

    ax = axes[5]
    clean_axis(ax)
    ax.set_title("Design choices that answer the review", fontsize=7, fontweight="bold", loc="left")
    choices = [
        ("dimension", "latent k grid separates summarization from 2D endpoints"),
        ("workflow", "direct 2D compared with PCA50 -> 2D"),
        ("source data", "run-level CSVs linked to each figure"),
        ("code", "method-specific environment and source manifest"),
    ]
    for i, (head, body) in enumerate(choices):
        y = 0.78 - i * 0.18
        ax.text(0.04, y, f"{i + 1}", fontsize=7.2, fontweight="bold", color=PALETTE["blue"], va="center")
        ax.text(0.14, y, head, fontsize=5.9, fontweight="bold", va="center")
        ax.text(0.37, y, body, fontsize=5.4, color=PALETTE["slate"], va="center")

    source = save_source(
        pd.concat(
            [
                datasets.assign(source_table="datasets_manifest"),
                grid.rename(columns={"method_norm": "dataset_id"}).assign(source_table="method_dimension_grid"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Figure_2_benchmark_design_source_data.csv",
    )
    add_contract(
        "Figure 2",
        "The revised benchmark design makes dataset scope, method roles, work packages, and source-data flow explicit.",
        "schematic-led composite",
        [
            "a: local revision dataset scale",
            "b: dataset composition",
            "c: evaluation path and traceability",
            "d: method family by analysis-role matrix",
            "e: WP1-WP4 work-package map",
            "f: reviewer-facing design choices",
        ],
        source,
        "Dataset manifest is local revision material; avoid implying it exhaustively documents the original full 100-dataset corpus unless separately verified.",
    )
    save_figure(fig, MAIN_OUT / "Figure_2_benchmark_design_data_landscape", source)


def fig3_benchmark_landscape(score_long: pd.DataFrame, score_matrix: pd.DataFrame, stability_long: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.55), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.45, 1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    axes = [ax_a] + [fig.add_subplot(gs[i, j]) for i in range(1, 4) for j in range(2)]
    for label, ax in zip("abcdefg", axes):
        add_panel_label(ax, label)

    heatmap(ax_a, score_matrix.drop(columns=["overall_mean"]), "Original benchmark score landscape", cmap="mako", cbar_label="normalized score")
    draw_score_scatter(axes[1], score_matrix, family_map, "local", "global", "Local versus global structure scores")

    cluster = score_long[score_long["metric"].isin(["kmeans", "louvain", "spectral"])].copy()
    sns.boxplot(data=cluster, x="metric", y="score", ax=axes[2], color="#D7E4F3", linewidth=0.6, fliersize=1.8)
    sns.stripplot(data=cluster, x="metric", y="score", ax=axes[2], color=PALETTE["blue"], size=2, alpha=0.55, jitter=0.18)
    axes[2].set_title("Clustering metric distributions", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("normalized score")
    axes[2].set_xticks(range(3), labels=["k-means", "Louvain", "Spectral"])
    set_small_ticks(axes[2])

    draw_score_scatter(axes[3], score_matrix, family_map, "runtime_score", "memory_score", "Efficiency score trade-off", label_top=6)

    stab_wide = stability_long.pivot_table(index="method", columns="perturbation", values="score", aggfunc="mean")
    sns.boxplot(data=stability_long, x="perturbation", y="score", ax=axes[4], color="#E8DFF1", linewidth=0.6, fliersize=1.5)
    axes[4].set_title("Perturbation stability scores", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("normalized score")
    axes[4].tick_params(axis="x", rotation=45)
    set_small_ticks(axes[4], 4.8)

    ranked_barh(axes[5], score_matrix["overall_mean"], "Mean across score families", "mean normalized score", family_map, top=12, xlim=(0, 1))

    text_panel(
        axes[6],
        "Interpretation guardrail",
        [
            "This panel summarizes score patterns; it should not be described as a single definitive winner.",
            "Runtime and memory panels use normalized efficiency scores where higher is better.",
            "Clustering panels should be reported as label-concordance or simulation-label agreement, depending on dataset provenance.",
        ],
        width=42,
    )

    source = save_source(score_long.merge(score_matrix["overall_mean"].rename("overall_mean"), left_on="method", right_index=True, how="left"), "Figure_3_original_benchmark_score_source_data.csv")
    save_source(stab_wide.reset_index(), "Figure_3_stability_matrix_source_data.csv")
    add_contract(
        "Figure 3",
        "The original benchmark shows broad method-dependent trade-offs across structure preservation, clustering concordance, efficiency, and stability.",
        "quantitative grid",
        [
            "a: all-method score heatmap",
            "b: local-global scatter",
            "c: clustering score distributions",
            "d: runtime-memory score trade-off",
            "e: perturbation stability distributions",
            "f: mean score ranking",
            "g: interpretation guardrail",
        ],
        source,
        "Reviewers criticized composite score opacity; present this as descriptive score landscape, not a singular aggregate conclusion.",
    )
    save_figure(fig, MAIN_OUT / "Figure_3_original_benchmark_landscape", source)


def fig4_structure(score_matrix: pd.DataFrame, wp2: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    ranked_barh(axes[0], score_matrix["local"], "Local structure score ranking", "normalized score", family_map, top=12, xlim=(0, 1))
    ranked_barh(axes[1], score_matrix["global"], "Global structure score ranking", "normalized score", family_map, top=12, xlim=(0, 1))
    draw_score_scatter(axes[2], score_matrix, family_map, "local", "global", "Local-global method trade-off", label_top=10)

    trust = wp2.pivot_table(index="method", columns="dimension", values="trustworthiness_k30", aggfunc="median")
    trust = trust.loc[trust.median(axis=1).sort_values(ascending=False).index]
    heatmap(axes[3], trust, "WP2 trustworthiness by latent dimension", cmap="crest", cbar_label="trustworthiness", annotate=True)

    sil = wp2.pivot_table(index="method", columns="dimension", values="silhouette_label", aggfunc="median")
    sil = sil.loc[trust.index]
    heatmap(axes[4], sil, "WP2 silhouette by latent dimension", cmap="viridis", vmin=float(np.nanmin(sil.values)), vmax=float(np.nanmax(sil.values)), cbar_label="silhouette", annotate=True)

    work = wp2[["method", "dimension", "ari", "trustworthiness_k30"]].dropna()
    for dim in sorted(work["dimension"].unique()):
        sub = work[work["dimension"] == dim]
        axes[5].scatter(sub["trustworthiness_k30"], sub["ari"], s=13, alpha=0.55, label=f"k={int(dim)}")
    axes[5].set_title("Label concordance versus local geometry", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("trustworthiness k=30")
    axes[5].set_ylabel("ARI")
    axes[5].legend(fontsize=4.8, ncol=2)
    axes[5].grid(color=PALETTE["light_gray"], lw=0.45)
    set_small_ticks(axes[5])

    source = save_source(wp2, "Figure_4_structure_preservation_source_data.csv")
    add_contract(
        "Figure 4",
        "Local and global structure preservation differ by method and by latent dimensionality, so the revision should avoid one-dimensional method claims.",
        "quantitative grid",
        [
            "a: local score ranking",
            "b: global score ranking",
            "c: local-global trade-off",
            "d: WP2 trustworthiness heatmap",
            "e: WP2 silhouette heatmap",
            "f: ARI versus trustworthiness",
        ],
        source,
        "Do not describe t-SNE/UMAP behavior as universally local/global; keep dimension and workflow qualifiers visible.",
    )
    save_figure(fig, MAIN_OUT / "Figure_4_structure_preservation_detail", source)


def fig5_clustering(score_long: pd.DataFrame, wp2: pd.DataFrame, wp3: pd.DataFrame, wp4: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    cluster = score_long[score_long["metric"].isin(["kmeans", "louvain", "spectral"])].copy()
    cluster_wide = cluster.pivot_table(index="method", columns="metric", values="score", aggfunc="mean")
    cluster_wide["mean"] = cluster_wide.mean(axis=1)
    cluster_wide = cluster_wide.sort_values("mean", ascending=False).head(20).drop(columns=["mean"])
    heatmap(axes[0], cluster_wide, "Clustering concordance score matrix", cmap="mako", cbar_label="normalized score", xrot=30)

    sns.boxplot(data=cluster, x="metric", y="score", ax=axes[1], color="#E3EDF7", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=cluster, x="metric", y="score", ax=axes[1], color=PALETTE["blue"], size=2, jitter=0.18, alpha=0.55)
    axes[1].set_title("Scores across clustering algorithms", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("normalized score")
    axes[1].set_xticks(range(3), labels=["k-means", "Louvain", "Spectral"])
    set_small_ticks(axes[1])

    line_with_interval(axes[2], wp2, "dimension", "ari", "method", "WP2 ARI across latent dimensions", "ARI", family_map, xlabel="latent dimension", legend=False, direct_labels=True)
    line_with_interval(axes[3], wp4, "hvg_requested", "ari", "method", "WP4 ARI across HVG cutoffs", "ARI", family_map, xlabel="requested HVGs", legend=False, direct_labels=True)

    work = wp3.copy()
    sns.boxplot(data=work, x="method", y="ari", hue="workflow", ax=axes[4], linewidth=0.6, fliersize=1.5, palette=["#DDEAF7", "#F1DCC9"])
    axes[4].set_title("WP3 workflow-dependent ARI", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("ARI")
    axes[4].tick_params(axis="x", rotation=25)
    axes[4].legend(fontsize=4.8, title="", loc="lower right")
    set_small_ticks(axes[4])

    text_panel(
        axes[5],
        "Label terminology for the revision",
        [
            "Synthetic datasets: use known simulation labels.",
            "Real datasets: use annotation-concordance targets unless independent label provenance is documented.",
            "Avoid phrasing that implies labels discovered by DR/clustering are independent ground truth.",
            "Report ARI/NMI/HOM/COM with run-level source data.",
        ],
        title_color=PALETTE["red"],
        width=42,
    )

    source = save_source(
        pd.concat(
            [
                cluster.assign(source="original_cluster_scores"),
                wp2.assign(source="wp2_dimension_sensitivity"),
                wp3.assign(source="wp3_visualization_workflow"),
                wp4.assign(source="wp4_input_gene_sensitivity"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Figure_5_clustering_concordance_source_data.csv",
    )
    add_contract(
        "Figure 5",
        "Clustering/label-concordance performance depends on algorithm, embedding dimension, visualization workflow, and input-gene settings.",
        "quantitative grid",
        [
            "a: clustering score heatmap",
            "b: clustering algorithm distributions",
            "c: WP2 ARI by latent dimension",
            "d: WP4 ARI by HVG cutoff",
            "e: WP3 workflow-dependent ARI",
            "f: label terminology guardrail",
        ],
        source,
        "Avoid circular ground-truth language for real-data annotations; state label provenance explicitly.",
    )
    save_figure(fig, MAIN_OUT / "Figure_5_clustering_annotation_concordance", source)


def fig6_revision_controls(tables: dict[str, pd.DataFrame], family_map: dict[str, str]) -> None:
    wp1, wp2, wp3, wp4 = tables["wp1"], tables["wp2"], tables["wp3"], tables["wp4"]
    fig = plt.figure(figsize=(7.2, 8.75), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    for label, ax in zip("abcdefghi", axes):
        add_panel_label(ax, label)

    sns.boxplot(data=wp1, x="dimension", y="ari", ax=axes[0], color="#E8DFF1", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=wp1, x="dimension", y="ari", ax=axes[0], color=PALETTE["violet"], size=2, alpha=0.55, jitter=0.18)
    axes[0].set_title("WP1 scVI ARI by dimension", fontsize=7, fontweight="bold", loc="left")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("ARI")
    set_small_ticks(axes[0])

    sns.boxplot(data=wp1, x="dimension", y="trustworthiness_k30", ax=axes[1], color="#E4EEF1", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=wp1, x="dimension", y="trustworthiness_k30", ax=axes[1], color=PALETTE["teal"], size=2, alpha=0.5, jitter=0.18)
    axes[1].set_title("WP1 scVI local geometry", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("latent dimension")
    axes[1].set_ylabel("trustworthiness k=30")
    set_small_ticks(axes[1])

    sc = axes[2].scatter(wp1["runtime_seconds"], wp1["max_rss_mb"], c=wp1["dimension"], cmap="viridis", s=16, edgecolor="white", lw=0.4, alpha=0.8)
    axes[2].set_title("WP1 scVI runtime-memory trace", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("runtime (s)")
    axes[2].set_ylabel("max RSS (MB)")
    axes[2].set_xscale("log")
    cbar = fig.colorbar(sc, ax=axes[2], fraction=0.04, pad=0.02)
    cbar.set_label("latent k", fontsize=5.2)
    cbar.ax.tick_params(labelsize=4.8, length=2)
    set_small_ticks(axes[2])
    axes[2].grid(color=PALETTE["light_gray"], lw=0.45)

    ari = wp2.pivot_table(index="method", columns="dimension", values="ari", aggfunc="median")
    ari = ari.loc[ari.median(axis=1).sort_values(ascending=False).index]
    heatmap(axes[3], ari, "WP2 dimension sensitivity", cmap="viridis", cbar_label="median ARI", annotate=True)

    delta = (
        wp3.pivot_table(index=["dataset_id", "method", "seed"], columns="workflow", values="ari", aggfunc="mean")
        .dropna()
        .reset_index()
    )
    delta["delta_pca50_minus_direct"] = delta["pca50_to_2d"] - delta["direct_2d"]
    order = delta.groupby("method")["delta_pca50_minus_direct"].median().sort_values(ascending=False).index
    sns.boxplot(data=delta, x="method", y="delta_pca50_minus_direct", order=order, ax=axes[4], color="#F1DCC9", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=delta, x="method", y="delta_pca50_minus_direct", order=order, ax=axes[4], color=PALETTE["gold"], size=2, alpha=0.55, jitter=0.16)
    axes[4].axhline(0, color=PALETTE["slate"], lw=0.7, ls="--")
    axes[4].set_title("WP3 PCA50 workflow effect", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("delta ARI")
    axes[4].tick_params(axis="x", rotation=25)
    set_small_ticks(axes[4])

    line_with_interval(axes[5], wp4, "hvg_requested", "ari", "method", "WP4 input-gene sensitivity", "ARI", family_map, xlabel="requested HVGs", legend=False, direct_labels=True)

    runtime = pd.concat(
        [
            wp2[["method", "runtime_seconds"]].assign(work_package="WP2"),
            wp3[["method", "runtime_seconds"]].assign(work_package="WP3"),
            wp4[["method", "runtime_seconds"]].assign(work_package="WP4"),
        ],
        ignore_index=True,
    )
    sns.boxplot(data=runtime, x="work_package", y="runtime_seconds", ax=axes[6], color="#E6E6E6", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=runtime, x="work_package", y="runtime_seconds", ax=axes[6], color=PALETTE["slate"], size=1.7, alpha=0.4, jitter=0.18)
    axes[6].set_yscale("log")
    axes[6].set_title("Runtime distribution in new controls", fontsize=7, fontweight="bold", loc="left")
    axes[6].set_xlabel("")
    axes[6].set_ylabel("runtime (s, log)")
    set_small_ticks(axes[6])

    counts = pd.DataFrame(
        {
            "work_package": ["WP1 scVI", "WP2 dimension", "WP3 workflow", "WP4 HVG"],
            "rows": [len(wp1), len(wp2), len(wp3), len(wp4)],
            "datasets": [wp1["dataset_id"].nunique(), wp2["dataset_id"].nunique(), wp3["dataset_id"].nunique(), wp4["dataset_id"].nunique()],
            "methods": [wp1["method"].nunique(), wp2["method"].nunique(), wp3["method"].nunique(), wp4["method"].nunique()],
        }
    )
    axes[7].bar(counts["work_package"], counts["rows"], color=[PALETTE["violet"], PALETTE["blue"], PALETTE["gold"], PALETTE["green"]])
    axes[7].set_title("Run-level source-data rows", fontsize=7, fontweight="bold", loc="left")
    axes[7].set_ylabel("rows")
    axes[7].tick_params(axis="x", rotation=25)
    set_small_ticks(axes[7])

    text_panel(
        axes[8],
        "Scope caveats retained",
        [
            "WP2/WP4 are targeted controls, not full 26-method factorial reruns.",
            "WP4 is capped at 3000 HVGs because the local matrices contain 3000 genes.",
            "scVI results use locally provided matrices; raw count status should be reported conservatively.",
            "GLMPCA/scGBM bounded-iteration settings should be disclosed in Methods.",
        ],
        title_color=PALETTE["red"],
        width=36,
    )

    source = save_source(
        pd.concat(
            [
                wp1.assign(source="WP1_scVI_local"),
                wp2.assign(source="WP2_dimension_sensitivity"),
                wp3.assign(source="WP3_visualization_workflow"),
                wp4.assign(source="WP4_input_gene_sensitivity"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Figure_6_revision_controls_source_data.csv",
    )
    save_source(counts, "Figure_6_revision_controls_inclusion_counts.csv")
    add_contract(
        "Figure 6",
        "The new WP1-WP4 controls directly address the major revision critiques about scVI, 2D-only benchmarking, visualization workflows, and input-gene sensitivity.",
        "quantitative grid",
        [
            "a: WP1 scVI ARI",
            "b: WP1 scVI trustworthiness",
            "c: WP1 scVI runtime-memory trace",
            "d: WP2 dimension sensitivity",
            "e: WP3 PCA50 workflow effect",
            "f: WP4 input-gene sensitivity",
            "g: runtime distributions",
            "h: inclusion counts",
            "i: scope caveats",
        ],
        source,
        "Keep caveats visible so reviewers see rigor rather than overextension.",
    )
    save_figure(fig, MAIN_OUT / "Figure_6_revision_controls_WP1_WP4", source)


def fig7_robustness(stability_long: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    stab = stability_long.copy()
    stab_wide = stab.pivot_table(index="method", columns="perturbation", values="score", aggfunc="mean")
    stab_wide["median"] = stab_wide.median(axis=1)
    heatmap(axes[0], stab_wide.sort_values("median", ascending=False).drop(columns=["median"]).head(20), "Robustness score heatmap", cmap="crest", cbar_label="score", xrot=45)

    stab["family"] = stab["method"].map(lambda x: family_map.get(x, "other"))
    short_family = {
        "linear/probabilistic": "linear",
        "visualization/graph": "visualization",
        "deep generative": "generative",
        "deep generative + scVI": "gen. + scVI",
        "deep autoencoder": "autoencoder",
        "graph autoencoder": "graph AE",
        "label-sensitive": "label-sensitive",
        "other": "other",
    }
    stab["family_short"] = stab["family"].map(short_family).fillna(stab["family"])
    palette_short = {short_family.get(k, k): v for k, v in FAMILY_COLORS.items()}
    sns.boxplot(data=stab, x="family_short", y="score", hue="family_short", ax=axes[1], palette=palette_short, legend=False, linewidth=0.6, fliersize=1.5)
    axes[1].set_title("Stability by method family", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("normalized score")
    axes[1].tick_params(axis="x", rotation=25)
    set_small_ticks(axes[1], 4.8)

    ranked_barh(axes[2], stab_wide["median"], "Median stability ranking", "median score", family_map, top=12, xlim=(0, 1))

    batch = stab[stab["perturbation"].isin(["batch_number", "batch_strength"])]
    sns.boxplot(data=batch, x="perturbation", y="score", ax=axes[3], color="#E2EEF0", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=batch, x="perturbation", y="score", ax=axes[3], color=PALETTE["teal"], size=2, alpha=0.45, jitter=0.18)
    axes[3].set_title("Batch-related perturbations", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("score")
    axes[3].set_xticks(range(2), labels=["batch number", "batch strength"])
    set_small_ticks(axes[3])

    bio = stab[stab["perturbation"].isin(["cell_number", "gene_number", "celltype_number"])]
    sns.boxplot(data=bio, x="perturbation", y="score", ax=axes[4], color="#E8ECF6", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=bio, x="perturbation", y="score", ax=axes[4], color=PALETTE["blue"], size=2, alpha=0.45, jitter=0.18)
    axes[4].set_title("Scale and cell-type complexity", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("score")
    axes[4].set_xticks(range(3), labels=["cells", "genes", "cell types"])
    set_small_ticks(axes[4])

    stress = stab[stab["perturbation"].isin(["dropout", "de_prob", "de_strength", "out"])]
    sns.boxplot(data=stress, x="perturbation", y="score", ax=axes[5], color="#F1E4DD", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=stress, x="perturbation", y="score", ax=axes[5], color=PALETTE["rose"], size=2, alpha=0.45, jitter=0.18)
    axes[5].set_title("Dropout, DE, and outlier perturbations", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("")
    axes[5].set_ylabel("score")
    axes[5].set_xticks(range(4), labels=["dropout", "DE prob", "DE strength", "outliers"], rotation=25)
    set_small_ticks(axes[5])

    source = save_source(stability_long, "Figure_7_robustness_stability_source_data.csv")
    add_contract(
        "Figure 7",
        "Robustness differs across perturbation classes and method families, supporting a context-dependent interpretation of benchmark performance.",
        "quantitative grid",
        [
            "a: stability heatmap",
            "b: family-level stability",
            "c: median stability ranking",
            "d: batch perturbations",
            "e: scale/cell-type perturbations",
            "f: dropout/DE/outlier perturbations",
        ],
        source,
        "Do not overemphasize batch-effect benchmarking; present batch as one perturbation class among several.",
    )
    save_figure(fig, MAIN_OUT / "Figure_7_robustness_complexity", source)


def fig8_scalability(efficiency: pd.DataFrame, grid: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    selected = ["PCA", "UMAP", "t-SNE", "PHATE", "PaCMAP", "GLMPCA", "scGBM", "SAUCIE"]
    for method in selected:
        sub = efficiency[efficiency["method"] == method].sort_values("n_cells")
        if sub.empty:
            continue
        axes[0].plot(sub["n_cells"], sub["time_s"], marker="o", ms=3, lw=1.1, label=method, color=method_color(method, family_map))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Runtime scaling", fontsize=7, fontweight="bold", loc="left")
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel("time (s, log)")
    axes[0].legend(fontsize=4.7, ncol=2)
    axes[0].grid(color=PALETTE["light_gray"], lw=0.45)
    set_small_ticks(axes[0])

    for method in selected:
        sub = efficiency[efficiency["method"] == method].sort_values("n_cells")
        if sub.empty:
            continue
        axes[1].plot(sub["n_cells"], sub["peak_memory_gb"], marker="o", ms=3, lw=1.1, label=method, color=method_color(method, family_map))
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Memory scaling", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("cells")
    axes[1].set_ylabel("peak memory (GB, log)")
    axes[1].grid(color=PALETTE["light_gray"], lw=0.45)
    set_small_ticks(axes[1])

    avail = efficiency.assign(ran=1).pivot_table(index="method", columns="n_cells", values="ran", aggfunc="max").fillna(0)
    avail["run_count"] = avail.sum(axis=1)
    avail = avail.sort_values("run_count", ascending=False).drop(columns=["run_count"])
    heatmap(axes[2], avail.head(26), "Completed run grid by cell number", cmap="Blues", vmin=0, vmax=1, cbar_label="observed", xrot=45)

    max_cell = efficiency["n_cells"].max()
    max_df = efficiency[efficiency["n_cells"] == max_cell].dropna(subset=["time_s", "peak_memory_gb"])
    for _, row in max_df.iterrows():
        axes[3].scatter(row["time_s"], row["peak_memory_gb"], s=28, color=method_color(row["method"], family_map), edgecolor="white", lw=0.4)
    label_methods = []
    for name in list(max_df.sort_values("time_s").head(4)["method"]) + list(max_df.sort_values("peak_memory_gb", ascending=False).head(2)["method"]):
        if name not in label_methods:
            label_methods.append(name)
    offsets = [(5, 5), (5, -8), (-28, 6), (-30, -8), (7, 12), (-34, 12)]
    for i, (_, row) in enumerate(max_df[max_df["method"].isin(label_methods)].iterrows()):
        dx, dy = offsets[i % len(offsets)]
        axes[3].annotate(
            row["method"],
            xy=(row["time_s"], row["peak_memory_gb"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.7,
            arrowprops=dict(arrowstyle="-", color=PALETTE["gray"], lw=0.35, shrinkA=0, shrinkB=2),
        )
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_title(f"Runtime-memory at {int(max_cell):,} cells", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("time (s, log)")
    axes[3].set_ylabel("peak memory (GB, log)")
    axes[3].grid(color=PALETTE["light_gray"], lw=0.45)
    set_small_ticks(axes[3])

    manifest = pd.read_csv(ROOT / "revision_benchmark" / "config" / "methods_install_manifest.csv")
    status_counts = manifest.groupby(["language", "status"])["method"].nunique().reset_index()
    sns.barplot(data=status_counts, x="language", y="method", hue="status", ax=axes[4], palette=STATUS_PALETTE)
    axes[4].set_title("Method installation evidence", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("methods")
    axes[4].legend(fontsize=4.7, title="", loc="upper right")
    set_small_ticks(axes[4])

    text_panel(
        axes[5],
        "Reproducibility language",
        [
            "Report method-specific environments and package/source manifests.",
            "Provide a minimal working example and benchmark orchestration scripts.",
            "Avoid claiming one universal environment runs all legacy and modern methods.",
            "Upper scalability evidence in this package reaches 73,233 cells.",
        ],
        width=42,
    )

    source = save_source(pd.concat([efficiency.assign(source="efficiency_scaling"), manifest.assign(source="methods_install_manifest")], ignore_index=True, sort=False), "Figure_8_scalability_reproducibility_source_data.csv")
    add_contract(
        "Figure 8",
        "Scalability and reproducibility are method-specific, requiring explicit runtime/memory traces and environment manifests.",
        "quantitative grid",
        [
            "a: runtime scaling",
            "b: memory scaling",
            "c: run coverage by cell number",
            "d: runtime-memory at largest local cell count",
            "e: installation evidence",
            "f: reproducibility language",
        ],
        source,
        "Reviewer asked for reproducible code/tutorials; figure evidence should be accompanied by actual repository cleanup.",
    )
    save_figure(fig, MAIN_OUT / "Figure_8_scalability_reproducibility", source)


def fig9_practical_guide(score_matrix: pd.DataFrame, wp3: pd.DataFrame, wp4: pd.DataFrame, family_map: dict[str, str]) -> None:
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)

    guide_rows = []
    criteria = {
        "local structure": "local",
        "global structure": "global",
        "clustering": "kmeans",
        "runtime": "runtime_score",
        "memory": "memory_score",
        "stability": "stability_median",
        "balanced score": "overall_mean",
    }
    for label, metric in criteria.items():
        if metric in score_matrix:
            top = score_matrix[metric].dropna().sort_values(ascending=False).head(5)
            for rank, (method, score) in enumerate(top.items(), 1):
                guide_rows.append({"criterion": label, "rank": rank, "method": method, "score": score})
    guide = pd.DataFrame(guide_rows)
    top_counts = guide[guide["rank"] <= 3].groupby("method")["criterion"].nunique().sort_values(ascending=False)

    table = guide[guide["rank"] <= 3].pivot_table(index="criterion", columns="rank", values="method", aggfunc="first")
    clean_axis(axes[0])
    axes[0].set_title("Top methods by evidence criterion", fontsize=7, fontweight="bold", loc="left")
    y = 0.86
    for criterion, row in table.iterrows():
        axes[0].text(0.02, y, criterion, fontsize=5.7, fontweight="bold", va="center")
        for i, rank in enumerate([1, 2, 3]):
            axes[0].text(0.40 + i * 0.18, y, str(row.get(rank, "")), fontsize=5.5, va="center")
        y -= 0.11
    axes[0].text(0.40, 0.96, "rank 1", fontsize=5.2, fontweight="bold")
    axes[0].text(0.58, 0.96, "rank 2", fontsize=5.2, fontweight="bold")
    axes[0].text(0.76, 0.96, "rank 3", fontsize=5.2, fontweight="bold")

    ranked_barh(axes[1], top_counts, "Repeated top-three appearances", "criteria count", family_map, top=12)

    work = score_matrix.copy()
    work["structure_mean"] = work[["local", "global"]].mean(axis=1)
    work["efficiency_mean"] = work[["runtime_score", "memory_score"]].mean(axis=1)
    for method, row in work.dropna(subset=["structure_mean", "efficiency_mean"]).iterrows():
        stability_size = row.get("stability_median", 0.3)
        if not np.isfinite(stability_size):
            stability_size = 0.3
        axes[2].scatter(
            row["efficiency_mean"],
            row["structure_mean"],
            s=22 + 60 * stability_size,
            color=method_color(method, family_map),
            edgecolor="white",
            lw=0.45,
        )
    offsets = [(5, 5), (-30, 5), (5, -8), (-30, -8), (8, 12), (-34, 12)]
    for i, (method, row) in enumerate(work.sort_values("overall_mean", ascending=False).head(6).iterrows()):
        dx, dy = offsets[i % len(offsets)]
        axes[2].annotate(
            method,
            xy=(row["efficiency_mean"], row["structure_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.5,
            color=PALETTE["dark"],
            arrowprops=dict(arrowstyle="-", color=PALETTE["gray"], lw=0.35, shrinkA=0, shrinkB=2),
        )
    axes[2].set_title("Structure-efficiency-stability trade-off", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("mean efficiency score")
    axes[2].set_ylabel("mean structure score")
    axes[2].set_xlim(0, 1.03)
    axes[2].set_ylim(0, 1.03)
    axes[2].grid(color=PALETTE["light_gray"], lw=0.45)
    set_small_ticks(axes[2])

    workflow_delta = (
        wp3.pivot_table(index=["dataset_id", "method", "seed"], columns="workflow", values="ari", aggfunc="mean")
        .dropna()
        .reset_index()
    )
    workflow_delta["delta"] = workflow_delta["pca50_to_2d"] - workflow_delta["direct_2d"]
    delta_summary = workflow_delta.groupby("method")["delta"].median().sort_values(ascending=False)
    colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in delta_summary.values]
    axes[3].bar(delta_summary.index, delta_summary.values, color=colors)
    axes[3].axhline(0, color=PALETTE["slate"], lw=0.7)
    axes[3].set_title("When PCA50 preprocessing helps visualization", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_ylabel("median delta ARI")
    axes[3].tick_params(axis="x", rotation=25)
    set_small_ticks(axes[3])

    hvg_var = wp4.groupby("method")["ari"].agg(lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)).sort_values(ascending=False)
    ranked_barh(axes[4], hvg_var, "Input-gene sensitivity spread", "ARI IQR across runs", family_map, top=8)

    text_panel(
        axes[5],
        "How to write this guide",
        [
            "Frame recommendations as evidence-weighted choices, not universal rankings.",
            "Separate visualization-only workflows from latent summarization.",
            "Mention where source data are targeted controls rather than full benchmark reruns.",
            "Use this figure as an end-of-results practical guide if the final manuscript keeps nine main figures.",
        ],
        width=42,
    )

    source = save_source(pd.concat([guide.assign(source="criterion_top_methods"), workflow_delta.assign(source="wp3_workflow_delta"), hvg_var.rename("ari_iqr").reset_index().assign(source="wp4_hvg_iqr")], ignore_index=True, sort=False), "Figure_9_practical_guide_source_data.csv")
    add_contract(
        "Figure 9",
        "A practical method guide should be evidence-weighted and task-specific, not a universal winner table.",
        "asymmetric mixed-modality figure",
        [
            "a: top methods by criterion",
            "b: repeated top-three appearances",
            "c: structure-efficiency-stability trade-off",
            "d: PCA50 workflow effect",
            "e: input-gene sensitivity spread",
            "f: writing guardrails",
        ],
        source,
        "If the journal/editor pushes for fewer display items, this figure is the first candidate to move to Supplementary Information.",
    )
    save_figure(fig, MAIN_OUT / "Figure_9_practical_method_selection_guide", source)


def supplementary_figures(
    score_long: pd.DataFrame,
    score_matrix: pd.DataFrame,
    stability_long: pd.DataFrame,
    efficiency: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    datasets: pd.DataFrame,
    grid: pd.DataFrame,
    family_map: dict[str, str],
) -> None:
    # S1 all score matrix.
    fig, ax = plt.subplots(figsize=(7.2, 8.4), constrained_layout=True)
    heatmap(ax, score_matrix.drop(columns=["overall_mean"]), "Supplementary Fig. S1: all normalized score families", cmap="mako", cbar_label="normalized score")
    add_panel_label(ax, "a", x=-0.05)
    source = save_source(score_long, "Supplementary_Figure_S1_score_matrix_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S1_score_matrix", source)

    # S2 WP2 full.
    wp2 = tables["wp2"]
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for ax, metric, title in zip(axes[:3], ["ari", "nmi", "trustworthiness_k30"], ["ARI", "NMI", "Trustworthiness k=30"]):
        mat = wp2.pivot_table(index="method", columns="dimension", values=metric, aggfunc="median")
        mat = mat.loc[mat.median(axis=1).sort_values(ascending=False).index]
        heatmap(ax, mat, f"WP2 {title}", cmap="viridis", cbar_label=title, annotate=True)
    line_with_interval(axes[3], wp2, "dimension", "runtime_seconds", "method", "WP2 runtime by dimension", "runtime (s)", family_map, xlabel="latent dimension")
    axes[3].set_yscale("log")
    source = save_source(wp2, "Supplementary_Figure_S2_WP2_dimension_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S2_WP2_dimension_sensitivity", source)

    # S3 WP3 full.
    wp3 = tables["wp3"]
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for ax, metric, title in zip(axes[:3], ["ari", "nmi", "trustworthiness_k30"], ["ARI", "NMI", "Trustworthiness"]):
        sns.boxplot(data=wp3, x="method", y=metric, hue="workflow", ax=ax, linewidth=0.6, fliersize=1.5, palette=["#DDEAF7", "#F1DCC9"])
        ax.set_title(f"WP3 {title}", fontsize=7, fontweight="bold", loc="left")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=4.8, title="")
        set_small_ticks(ax)
    sns.boxplot(data=wp3, x="method", y="runtime_seconds", hue="workflow", ax=axes[3], linewidth=0.6, fliersize=1.5, palette=["#DDEAF7", "#F1DCC9"])
    axes[3].set_yscale("log")
    axes[3].set_title("WP3 runtime", fontsize=7, fontweight="bold", loc="left")
    axes[3].tick_params(axis="x", rotation=25)
    axes[3].legend(fontsize=4.8, title="")
    set_small_ticks(axes[3])
    source = save_source(wp3, "Supplementary_Figure_S3_WP3_workflow_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S3_WP3_visualization_workflow", source)

    # S4 WP4 full.
    wp4 = tables["wp4"]
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for ax, metric, title in zip(axes[:3], ["ari", "nmi", "trustworthiness_k30"], ["ARI", "NMI", "Trustworthiness"]):
        line_with_interval(ax, wp4, "hvg_requested", metric, "method", f"WP4 {title}", title, family_map, xlabel="requested HVGs")
    line_with_interval(axes[3], wp4, "hvg_requested", "runtime_seconds", "method", "WP4 runtime", "runtime (s)", family_map, xlabel="requested HVGs")
    axes[3].set_yscale("log")
    source = save_source(wp4, "Supplementary_Figure_S4_WP4_HVG_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S4_WP4_input_gene_sensitivity", source)

    # S5 WP1 full.
    wp1 = tables["wp1"]
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for ax, metric, title in zip(axes, ["ari", "nmi", "trustworthiness_k30", "runtime_seconds"], ["ARI", "NMI", "Trustworthiness", "Runtime"]):
        sns.boxplot(data=wp1, x="dimension", y=metric, ax=ax, color="#E8DFF1", linewidth=0.6, fliersize=1.5)
        sns.stripplot(data=wp1, x="dimension", y=metric, ax=ax, color=PALETTE["violet"], size=2, alpha=0.5, jitter=0.18)
        if metric == "runtime_seconds":
            ax.set_yscale("log")
        ax.set_title(f"WP1 scVI {title}", fontsize=7, fontweight="bold", loc="left")
        set_small_ticks(ax)
    source = save_source(wp1, "Supplementary_Figure_S5_WP1_scVI_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S5_WP1_scVI_local", source)

    # S6 efficiency full.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    for method in efficiency["method"].dropna().unique():
        sub = efficiency[efficiency["method"] == method].sort_values("n_cells")
        axes[0].plot(sub["n_cells"], sub["time_s"], lw=0.8, alpha=0.65, color=method_color(method, family_map))
        axes[1].plot(sub["n_cells"], sub["peak_memory_gb"], lw=0.8, alpha=0.65, color=method_color(method, family_map))
    for ax, title, ylabel in [(axes[0], "All-method runtime scaling", "time (s)"), (axes[1], "All-method memory scaling", "peak memory (GB)")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
        ax.set_xlabel("cells")
        ax.set_ylabel(ylabel)
        ax.grid(color=PALETTE["light_gray"], lw=0.45)
        set_small_ticks(ax)
    max_rows = efficiency.sort_values(["n_cells", "time_s"]).groupby("method").tail(1)
    axes[2].scatter(max_rows["time_s"], max_rows["peak_memory_gb"], c=[method_color(m, family_map) for m in max_rows["method"]], s=24, edgecolor="white", lw=0.4)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Latest available runtime-memory point", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("peak memory (GB)")
    set_small_ticks(axes[2])
    avail = efficiency.assign(ran=1).pivot_table(index="method", columns="n_cells", values="ran", aggfunc="max").fillna(0)
    heatmap(axes[3], avail, "All-method run coverage", cmap="Greys", vmin=0, vmax=1, cbar_label="ran", xrot=45)
    source = save_source(efficiency, "Supplementary_Figure_S6_efficiency_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S6_efficiency_scaling", source)

    # S7 stability full.
    fig, ax = plt.subplots(figsize=(7.2, 8.2), constrained_layout=True)
    stab_full = stability_long.pivot_table(index="method", columns="perturbation", values="score", aggfunc="mean")
    stab_full["median"] = stab_full.median(axis=1)
    heatmap(ax, stab_full.sort_values("median", ascending=False).drop(columns=["median"]), "Supplementary Fig. S7: stability across perturbations", cmap="crest", cbar_label="score", xrot=45)
    add_panel_label(ax, "a", x=-0.05)
    source = save_source(stability_long, "Supplementary_Figure_S7_stability_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S7_stability_all_perturbations", source)

    # S8 environment manifest.
    manifest = pd.read_csv(ROOT / "revision_benchmark" / "config" / "methods_install_manifest.csv")
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    sns.countplot(data=manifest, x="language", hue="status", ax=axes[0], palette=STATUS_PALETTE)
    axes[0].set_title("Install status by language", fontsize=7, fontweight="bold", loc="left")
    axes[0].legend(fontsize=4.7, title="")
    set_small_ticks(axes[0])
    role_counts = manifest.groupby("role")["method"].nunique().sort_values()
    axes[1].barh(role_counts.index, role_counts.values, color=PALETTE["slate"])
    axes[1].set_title("Manifest roles", fontsize=7, fontweight="bold", loc="left")
    axes[1].set_xlabel("methods")
    set_small_ticks(axes[1], 4.8)
    channel_counts = manifest.groupby("install_channel")["method"].nunique().sort_values()
    axes[2].barh(channel_counts.index, channel_counts.values, color=PALETTE["teal"])
    axes[2].set_title("Install channels", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("methods")
    set_small_ticks(axes[2], 4.8)
    text_panel(axes[3], "Manifest use", ["Cite this as evidence of method-specific reproducibility work.", "Keep environment export files and source commit table adjacent to the code repository.", "Do not imply all methods run under one environment."], width=40)
    source = save_source(manifest, "Supplementary_Figure_S8_environment_manifest_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S8_environment_manifest", source)

    # S9 dataset manifest.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for label, ax in zip("abcd", axes):
        add_panel_label(ax, label)
    sns.histplot(data=datasets, x="n_cells", hue="dataset_group", ax=axes[0], bins=20, multiple="stack", palette={"synthetic": PALETTE["gold"], "downsampling": PALETTE["teal"]})
    axes[0].set_xscale("log")
    axes[0].set_title("Cell-count distribution", fontsize=7, fontweight="bold", loc="left")
    set_small_ticks(axes[0])
    sns.histplot(data=datasets, x="n_genes", hue="dataset_group", ax=axes[1], bins=20, multiple="stack", palette={"synthetic": PALETTE["gold"], "downsampling": PALETTE["teal"]})
    axes[1].set_xscale("log")
    axes[1].set_title("Gene-count distribution", fontsize=7, fontweight="bold", loc="left")
    set_small_ticks(axes[1])
    use_counts = datasets["planned_use"].fillna("not specified").str.split(",").explode().str.strip().value_counts().head(10)
    axes[2].barh(use_counts.index[::-1], use_counts.values[::-1], color=PALETTE["blue"])
    axes[2].set_title("Planned uses in manifest", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("rows")
    set_small_ticks(axes[2], 4.8)
    text_panel(axes[3], "Dataset-language caveat", ["Use this manifest for local revision evidence.", "Original manuscript dataset counts should be reported only from the verified original data table.", "Synthetic labels may be treated as known simulation labels."], width=40)
    source = save_source(datasets, "Supplementary_Figure_S9_dataset_manifest_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S9_dataset_manifest", source)

    # S10 original structure and efficiency detail.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    ranked_barh(axes[0], score_matrix["local"], "Original local preservation score", "normalized score", family_map, top=14, xlim=(0, 1.03))
    ranked_barh(axes[1], score_matrix["global"], "Original global preservation score", "normalized score", family_map, top=14, xlim=(0, 1.03))
    draw_score_scatter(axes[2], score_matrix, family_map, "local", "global", "Local-global trade-off", label_top=9)
    ranked_barh(axes[3], score_matrix["runtime_score"], "Runtime score detail", "normalized score", family_map, top=14, xlim=(0, 1.03))
    ranked_barh(axes[4], score_matrix["memory_score"], "Memory score detail", "normalized score", family_map, top=14, xlim=(0, 1.03))
    structure_eff = score_long[score_long["metric"].isin(["local", "global", "runtime_score", "memory_score"])].copy()
    metric_order = ["local", "global", "runtime_score", "memory_score"]
    sns.boxplot(data=structure_eff, x="metric", y="score", order=metric_order, ax=axes[5], color="#E5ECF4", linewidth=0.6, fliersize=1.3)
    sns.stripplot(data=structure_eff, x="metric", y="score", order=metric_order, ax=axes[5], color=PALETTE["slate"], size=1.8, alpha=0.55, jitter=0.18)
    axes[5].set_title("Distribution across original methods", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("")
    axes[5].set_ylabel("normalized score")
    axes[5].set_xticks(axes[5].get_xticks(), labels=[METRIC_LABELS.get(m, m) for m in metric_order], rotation=25, ha="right")
    set_small_ticks(axes[5])
    source = save_source(
        pd.concat(
            [
                score_matrix.reset_index().assign(record_type="score_matrix"),
                structure_eff.assign(record_type="score_long"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Supplementary_Figure_S10_original_structure_efficiency_source_data.csv",
    )
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S10_original_structure_efficiency_detail", source)

    # S11 original clustering detail.
    cluster = score_long[score_long["metric"].isin(["kmeans", "louvain", "spectral"])].copy()
    cluster_wide = cluster.pivot_table(index="method", columns="metric", values="score", aggfunc="mean")
    cluster_wide = cluster_wide.reindex(columns=["kmeans", "louvain", "spectral"])
    cluster_wide["cluster_mean"] = cluster_wide.mean(axis=1, skipna=True)
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    ranked_barh(axes[0], cluster_wide["kmeans"], "k-means concordance", "normalized score", family_map, top=14, xlim=(0, 1.03))
    ranked_barh(axes[1], cluster_wide["louvain"], "Louvain concordance", "normalized score", family_map, top=14, xlim=(0, 1.03))
    ranked_barh(axes[2], cluster_wide["spectral"], "Spectral concordance", "normalized score", family_map, top=14, xlim=(0, 1.03))
    for ax, pair, title in [
        (axes[3], ("kmeans", "louvain"), "k-means versus Louvain"),
        (axes[4], ("kmeans", "spectral"), "k-means versus spectral"),
    ]:
        work = cluster_wide[list(pair) + ["cluster_mean"]].dropna()
        for method, row in work.iterrows():
            ax.scatter(row[pair[0]], row[pair[1]], s=18 + 40 * row["cluster_mean"], color=method_color(method, family_map), edgecolor="white", linewidth=0.45, alpha=0.9)
        for method in work["cluster_mean"].sort_values(ascending=False).head(7).index:
            ax.text(work.loc[method, pair[0]], work.loc[method, pair[1]] + 0.018, method, fontsize=4.6, ha="center")
        ax.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=0.6, ls="--")
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(METRIC_LABELS.get(pair[0], pair[0]))
        ax.set_ylabel(METRIC_LABELS.get(pair[1], pair[1]))
        ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
        ax.grid(color=PALETTE["light_gray"], lw=0.45)
        set_small_ticks(ax)
    sns.boxplot(data=cluster, x="metric", y="score", order=["kmeans", "louvain", "spectral"], ax=axes[5], color="#E3EDF7", linewidth=0.6, fliersize=1.5)
    sns.stripplot(data=cluster, x="metric", y="score", order=["kmeans", "louvain", "spectral"], ax=axes[5], color=PALETTE["blue"], size=1.9, jitter=0.18, alpha=0.55)
    axes[5].set_title("Score spread by clustering algorithm", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel("")
    axes[5].set_ylabel("normalized score")
    axes[5].set_xticks(axes[5].get_xticks(), labels=["k-means", "Louvain", "spectral"], rotation=20, ha="right")
    set_small_ticks(axes[5])
    source = save_source(
        pd.concat([cluster.assign(record_type="score_long"), cluster_wide.reset_index().assign(record_type="cluster_wide")], ignore_index=True, sort=False),
        "Supplementary_Figure_S11_original_clustering_source_data.csv",
    )
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S11_original_clustering_detail", source)

    # S12 WP2 dimension sensitivity detail.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    wp2_order = wp2.groupby("method")["ari"].median().sort_values(ascending=False).index
    wp2_inclusion = wp2.assign(run=1).pivot_table(index="method", columns="dimension", values="run", aggfunc="sum").reindex(wp2_order).fillna(0)
    heatmap(axes[0], wp2_inclusion, "WP2 included runs", cmap="Greys", vmin=0, vmax=None, cbar_label="runs", annotate=True, xrot=0)
    runtime_mat = np.log10(wp2.pivot_table(index="method", columns="dimension", values="runtime_seconds", aggfunc="median").reindex(wp2_order) + 1.0)
    heatmap(axes[1], runtime_mat, "WP2 median runtime", cmap="magma", vmin=None, vmax=None, cbar_label="log10(seconds+1)", xrot=0)
    memory_mat = np.log10(wp2.pivot_table(index="method", columns="dimension", values="max_rss_mb", aggfunc="median").reindex(wp2_order) + 1.0)
    heatmap(axes[2], memory_mat, "WP2 median memory", cmap="rocket", vmin=None, vmax=None, cbar_label="log10(MB+1)", xrot=0)
    ari_mat = wp2.pivot_table(index="method", columns="dimension", values="ari", aggfunc="median").reindex(wp2_order)
    if 2 in ari_mat.columns and 50 in ari_mat.columns:
        dim_delta = ari_mat[50] - ari_mat[2]
        delta_label = "ARI delta, 50D minus 2D"
    else:
        dim_delta = ari_mat.max(axis=1) - ari_mat.min(axis=1)
        delta_label = "ARI max-min across dimensions"
    vals = dim_delta.dropna().sort_values()
    axes[3].barh(vals.index, vals.values, color=[method_color(m, family_map) for m in vals.index], edgecolor="white", linewidth=0.45)
    axes[3].axvline(0, color=PALETTE["slate"], lw=0.6)
    axes[3].set_title("Dimension effect size", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel(delta_label)
    set_small_ticks(axes[3], 5)
    dataset_dim = wp2.pivot_table(index="dataset_id", columns="dimension", values="ari", aggfunc="median")
    heatmap(axes[4], dataset_dim, "Dataset-level ARI by dimension", cmap="viridis", cbar_label="median ARI", xrot=0)
    text_panel(
        axes[5],
        "WP2 interpretation",
        [
            "The full dimension grid is kept as a control rather than only the best dimension.",
            "Runtime and memory are shown beside accuracy metrics to avoid overclaiming a single winner.",
            "Dimension effects should be described as method- and dataset-dependent.",
        ],
        width=39,
    )
    source = save_source(
        pd.concat(
            [
                wp2.assign(record_type="wp2_long"),
                wp2_inclusion.reset_index().assign(record_type="wp2_included_runs"),
                dim_delta.rename("ari_dimension_delta").reset_index().assign(record_type="wp2_dimension_delta"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Supplementary_Figure_S12_WP2_dimension_detail_source_data.csv",
    )
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S12_WP2_dimension_detail", source)

    # S13 WP3 visualization workflow pairwise deltas.
    delta_frames = []
    delta_mats: dict[str, pd.DataFrame] = {}
    for metric in ["ari", "nmi", "trustworthiness_k30", "runtime_seconds", "max_rss_mb"]:
        wide = (
            wp3.pivot_table(index=["dataset_id", "method", "seed"], columns="workflow", values=metric, aggfunc="mean")
            .dropna()
            .reset_index()
        )
        if {"direct_2d", "pca50_to_2d"}.issubset(wide.columns):
            wide["delta"] = wide["pca50_to_2d"] - wide["direct_2d"]
            wide["metric"] = metric
            delta_frames.append(wide[["dataset_id", "method", "seed", "metric", "delta", "direct_2d", "pca50_to_2d"]])
            delta_mats[metric] = wide.pivot_table(index="method", columns="dataset_id", values="delta", aggfunc="median")
    wp3_delta = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    for ax, metric, title in [
        (axes[0], "ari", "ARI delta"),
        (axes[1], "nmi", "NMI delta"),
        (axes[2], "trustworthiness_k30", "Trustworthiness delta"),
    ]:
        mat = delta_mats.get(metric, pd.DataFrame())
        max_abs = float(np.nanmax(np.abs(mat.to_numpy()))) if not mat.empty and np.isfinite(mat.to_numpy()).any() else 0.1
        heatmap(ax, mat, f"WP3 {title}", cmap="vlag", vmin=-max_abs, vmax=max_abs, cbar_label="PCA50 minus direct", xrot=35)
    runtime_delta = wp3_delta[wp3_delta["metric"] == "runtime_seconds"].groupby("method")["delta"].median().sort_values()
    axes[3].barh(runtime_delta.index, runtime_delta.values, color=[PALETTE["up"] if v <= 0 else PALETTE["down"] for v in runtime_delta.values], edgecolor="white", linewidth=0.45)
    axes[3].axvline(0, color=PALETTE["slate"], lw=0.6)
    axes[3].set_title("Runtime delta", fontsize=7, fontweight="bold", loc="left")
    axes[3].set_xlabel("PCA50 minus direct (s)")
    set_small_ticks(axes[3], 5)
    ari_delta = wp3_delta[wp3_delta["metric"] == "ari"].copy()
    sns.boxplot(data=ari_delta, x="method", y="delta", ax=axes[4], color="#E5ECF4", linewidth=0.6, fliersize=1.4)
    sns.stripplot(data=ari_delta, x="method", y="delta", ax=axes[4], color=PALETTE["blue"], size=1.8, alpha=0.55, jitter=0.18)
    axes[4].axhline(0, color=PALETTE["slate"], lw=0.6)
    axes[4].set_title("Pairwise ARI delta distribution", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_ylabel("PCA50 minus direct")
    axes[4].tick_params(axis="x", rotation=25)
    set_small_ticks(axes[4])
    text_panel(
        axes[5],
        "WP3 interpretation",
        [
            "Positive deltas indicate PCA50 preprocessing improved the visualization endpoint.",
            "Negative runtime deltas indicate PCA50 preprocessing reduced runtime.",
            "This figure separates workflow choice from claims about a method's latent model.",
        ],
        width=38,
    )
    source = save_source(wp3_delta, "Supplementary_Figure_S13_WP3_workflow_delta_source_data.csv")
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S13_WP3_workflow_pairwise_deltas", source)

    # S14 WP4 input-gene sensitivity detail.
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    wp4_order = wp4.groupby("method")["ari"].median().sort_values(ascending=False).index
    hvg_inclusion = wp4.assign(run=1).pivot_table(index="method", columns="hvg_requested", values="run", aggfunc="sum").reindex(wp4_order).fillna(0)
    heatmap(axes[0], hvg_inclusion, "WP4 included runs", cmap="Greys", vmin=0, vmax=None, cbar_label="runs", annotate=True, xrot=0)
    actual_hvg = wp4.copy()
    actual_hvg["hvg_actual"] = pd.to_numeric(actual_hvg["hvg_actual"], errors="coerce")
    actual_mat = actual_hvg.pivot_table(index="method", columns="hvg_requested", values="hvg_actual", aggfunc="median").reindex(wp4_order)
    heatmap(axes[1], actual_mat, "Actual genes entering model", cmap="crest", vmin=None, vmax=None, cbar_label="median genes", xrot=0)
    ari_hvg = wp4.pivot_table(index="method", columns="hvg_requested", values="ari", aggfunc="median").reindex(wp4_order)
    heatmap(axes[2], ari_hvg, "ARI across input-gene cutoffs", cmap="viridis", cbar_label="median ARI", xrot=0)
    nmi_hvg = wp4.pivot_table(index="method", columns="hvg_requested", values="nmi", aggfunc="median").reindex(wp4_order)
    heatmap(axes[3], nmi_hvg, "NMI across input-gene cutoffs", cmap="mako", cbar_label="median NMI", xrot=0)
    ari_iqr = wp4.groupby("method")["ari"].agg(lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)).sort_values()
    axes[4].barh(ari_iqr.index, ari_iqr.values, color=[method_color(m, family_map) for m in ari_iqr.index], edgecolor="white", linewidth=0.45)
    axes[4].set_title("Input-gene ARI spread", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("ARI IQR across all WP4 runs")
    set_small_ticks(axes[4], 5)
    runtime_hvg = wp4.pivot_table(index="method", columns="hvg_requested", values="runtime_seconds", aggfunc="median").reindex(wp4_order)
    low_hvg, high_hvg = min(runtime_hvg.columns), max(runtime_hvg.columns)
    runtime_delta = (runtime_hvg[high_hvg] - runtime_hvg[low_hvg]).dropna().sort_values()
    axes[5].barh(runtime_delta.index, runtime_delta.values, color=[method_color(m, family_map) for m in runtime_delta.index], edgecolor="white", linewidth=0.45)
    axes[5].axvline(0, color=PALETTE["slate"], lw=0.6)
    axes[5].set_title("Runtime change with more input genes", fontsize=7, fontweight="bold", loc="left")
    axes[5].set_xlabel(f"seconds, {int(high_hvg)} minus {int(low_hvg)} HVGs")
    set_small_ticks(axes[5], 5)
    source = save_source(
        pd.concat(
            [
                wp4.assign(record_type="wp4_long"),
                hvg_inclusion.reset_index().assign(record_type="wp4_included_runs"),
                ari_iqr.rename("ari_iqr").reset_index().assign(record_type="wp4_ari_iqr"),
                runtime_delta.rename("runtime_delta_seconds").reset_index().assign(record_type="wp4_runtime_delta"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Supplementary_Figure_S14_WP4_input_gene_detail_source_data.csv",
    )
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S14_WP4_input_gene_detail", source)

    # S15 reproducibility and source tracking detail.
    manifest = pd.read_csv(ROOT / "revision_benchmark" / "config" / "methods_install_manifest.csv")
    commits_path = ROOT / "revision_benchmark" / "config" / "source_commits.csv"
    commits = pd.read_csv(commits_path) if commits_path.exists() else pd.DataFrame(columns=["source", "commit"])
    fig = plt.figure(figsize=(7.2, 8.35), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for label, ax in zip("abcdef", axes):
        add_panel_label(ax, label)
    role_status = manifest.assign(value=1).pivot_table(index="role", columns="status", values="method", aggfunc="nunique").fillna(0)
    heatmap(axes[0], role_status, "Method roles by verification status", cmap="Blues", vmin=0, vmax=None, cbar_label="methods", annotate=True, xrot=30)
    sns.countplot(data=manifest, x="language", hue="status", ax=axes[1], palette=STATUS_PALETTE)
    axes[1].set_title("Language and install status", fontsize=7, fontweight="bold", loc="left")
    axes[1].legend(fontsize=4.7, title="")
    set_small_ticks(axes[1])
    channel_counts = manifest.groupby("install_channel")["method"].nunique().sort_values()
    axes[2].barh(channel_counts.index, channel_counts.values, color=PALETTE["teal"], edgecolor="white", linewidth=0.45)
    axes[2].set_title("Install channels", fontsize=7, fontweight="bold", loc="left")
    axes[2].set_xlabel("methods")
    set_small_ticks(axes[2], 4.8)
    clean_axis(axes[3])
    axes[3].set_title("Source commit records", fontsize=7, fontweight="bold", loc="left")
    display_commits = commits.copy()
    display_commits["short_commit"] = display_commits["commit"].astype(str).str.slice(0, 8)
    y = 0.9
    for _, row in display_commits.head(12).iterrows():
        axes[3].text(0.02, y, f"{wrap_text(row['source'], 18)}", fontsize=5.3, ha="left", va="top")
        axes[3].text(0.78, y, row["short_commit"], fontsize=5.3, ha="left", va="top", color=PALETTE["slate"])
        y -= 0.07
    if len(display_commits) > 12:
        axes[3].text(0.02, y, f"+ {len(display_commits) - 12} additional sources", fontsize=5.3, ha="left", va="top", color=PALETTE["slate"])
    env_counts = manifest.groupby("environment")["method"].nunique().sort_values()
    axes[4].barh(env_counts.index, env_counts.values, color=PALETTE["gold"], edgecolor="white", linewidth=0.45)
    axes[4].set_title("Environment mapping", fontsize=7, fontweight="bold", loc="left")
    axes[4].set_xlabel("methods")
    set_small_ticks(axes[4], 4.8)
    text_panel(
        axes[5],
        "Reproducibility statement",
        [
            "Record method-specific environments rather than implying one shared environment.",
            "Pair source commits with method manifests in the revision package.",
            "Report figure source CSVs and export manifests as audit artifacts.",
        ],
        width=38,
    )
    source = save_source(
        pd.concat(
            [
                manifest.assign(record_type="method_install_manifest"),
                commits.assign(record_type="source_commits"),
                grid.assign(record_type="method_dimension_grid"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "Supplementary_Figure_S15_reproducibility_tracking_source_data.csv",
    )
    save_figure(fig, SUPP_OUT / "Supplementary_Figure_S15_reproducibility_tracking", source)


def write_qa_notes() -> None:
    files = sorted(list(MAIN_OUT.glob("*.png")) + list(SUPP_OUT.glob("*.png")))
    rows = []
    for path in files:
        rows.append({"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(QA_OUT / "png_file_size_check.csv", index=False)
    notes = [
        "# Figure QA Notes",
        "",
        "- Backend exclusivity: all generated figures were drawn and exported by Python/matplotlib/seaborn.",
        "- SVG text rule: `svg.fonttype = none` was set before figure creation.",
        "- PDF text rule: `pdf.fonttype = 42` was set before figure creation.",
        "- Exports: each figure was saved as SVG, PDF, PNG, and TIFF.",
        "- Quantitative source data: source CSVs were written under `source_data/` and linked in `qa/figure_export_manifest.csv`.",
        "- Remaining manual QA: inspect SVG/PDF in Illustrator/Inkscape or the journal production workflow for final label spacing after manuscript-specific caption edits.",
        "",
    ]
    (QA_OUT / "qa_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    grid, family_map = load_method_grid()
    datasets = load_datasets()
    score_long, score_matrix, stability_long = load_score_long()
    efficiency = load_efficiency()
    tables = load_revision_tables()

    save_source(grid, "method_dimension_grid_used.csv")
    save_source(datasets, "datasets_manifest_used.csv")
    save_source(score_long, "all_original_score_long_used.csv")
    save_source(score_matrix.reset_index(), "all_original_score_matrix_used.csv")
    save_source(stability_long, "all_stability_long_used.csv")
    save_source(efficiency, "all_efficiency_scaling_used.csv")

    fig1_taxonomy(grid, datasets, family_map)
    fig2_design(grid, datasets, family_map)
    fig3_benchmark_landscape(score_long, score_matrix, stability_long, family_map)
    fig4_structure(score_matrix, tables["wp2"], family_map)
    fig5_clustering(score_long, tables["wp2"], tables["wp3"], tables["wp4"], family_map)
    fig6_revision_controls(tables, family_map)
    fig7_robustness(stability_long, family_map)
    fig8_scalability(efficiency, grid, family_map)
    fig9_practical_guide(score_matrix, tables["wp3"], tables["wp4"], family_map)
    supplementary_figures(score_long, score_matrix, stability_long, efficiency, tables, datasets, grid, family_map)

    save_contracts()
    save_manifest()
    write_qa_notes()
    print(f"Generated revision figure package: {OUT}")
    print(f"Main figures: {len(list(MAIN_OUT.glob('*.svg')))} SVG files")
    print(f"Supplementary figures: {len(list(SUPP_OUT.glob('*.svg')))} SVG files")
    print(f"Manifest: {QA_OUT / 'figure_export_manifest.csv'}")


if __name__ == "__main__":
    main()
