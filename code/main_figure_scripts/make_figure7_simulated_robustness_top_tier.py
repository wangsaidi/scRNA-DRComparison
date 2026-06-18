from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure7_polish"
SOURCE_DIR = OUT / "source_data"
PLOT_OUT = OUT / "polished"
QA_OUT = OUT / "qa"
PLOT_OUT.mkdir(parents=True, exist_ok=True)
QA_OUT.mkdir(parents=True, exist_ok=True)

SOURCE = SOURCE_DIR / "Figure_7_parameter_level_stability_source_data.csv"
AVAILABILITY = SOURCE_DIR / "Figure_7_parameter_level_availability_summary.csv"
FINAL_CANON = (
    ROOT
    / "Publication/paper/submission_package_communications_biology_20260609/07_source_data_and_code_availability/github_release/source_data/main_figures/canonical_source_tables"
)
METHODS = FINAL_CANON / "canonical_method_manifest.csv"
VALIDATION = SOURCE_DIR / "Figure_7_recomputed_vs_published_axis_score_check.csv"

FIGURE_BASENAME = "Figure_7_simulated_robustness_parameter_landscape_polished"

AXIS_ORDER = [
    "cell_number",
    "gene_number",
    "celltype_number",
    "batch_number",
    "batch_strength",
    "dropout",
    "de_prob",
    "de_strength",
    "out",
]

DISPLAY_DATASETS = {
    "cell_number": ["cell_100", "cell_500", "cell_1k", "cell_5k", "cell_1w", "cell_2w", "cell_3w", "cell_4w", "cell_5w"],
    "gene_number": ["gene_5k", "gene_2w", "gene_3w", "gene_4w", "gene_5w"],
    "celltype_number": ["celltype_7", "celltype_9", "celltype_11", "celltype_13", "celltype_15"],
    "batch_number": ["batch_2", "batch_4", "batch_6", "batch_8", "batch_10"],
    "batch_strength": ["batch_0.2", "batch_0.4", "batch_0.6", "batch_0.8", "batch_1.0"],
    "dropout": ["dropout_-1", "dropout_0", "dropout_1", "dropout_2", "dropout_3"],
    "de_prob": ["de_prob_0.05", "de_prob_0.15", "de_prob_0.2", "de_prob_0.25", "de_prob_0.3"],
    "de_strength": ["de_0.2", "de_0.4", "de_0.6", "de_0.8", "de_1.0"],
    "out": ["out_0.1", "out_0.2", "out_0.3", "out_0.4", "out_0.5"],
}

AXIS_LABEL = {
    "cell_number": "Cell no.",
    "gene_number": "Gene no.",
    "celltype_number": "Cell-type no.",
    "batch_number": "Batch no.",
    "batch_strength": "Batch strength",
    "dropout": "Dropout",
    "de_prob": "DE prob.",
    "de_strength": "DE strength",
    "out": "Outlier",
}

HEATMAP_AXIS_LABEL = {
    **AXIS_LABEL,
    "celltype_number": "Cell types",
    "batch_strength": "Batch str.",
    "de_prob": "DE prob.",
    "de_strength": "DE str.",
}

SHORT_AXIS_LABEL = {
    "cell_number": "Cell",
    "gene_number": "Gene",
    "celltype_number": "Type",
    "batch_number": "Batch N",
    "batch_strength": "Batch S",
    "dropout": "Drop",
    "de_prob": "DE p",
    "de_strength": "DE s",
    "out": "Out",
}

VALUE_LABELS = {
    "cell_100": "100",
    "cell_500": "500",
    "cell_1k": "1k",
    "cell_5k": "5k",
    "cell_1w": "10k",
    "cell_2w": "20k",
    "cell_3w": "30k",
    "cell_4w": "40k",
    "cell_5w": "50k",
    "gene_5k": "5k",
    "gene_2w": "20k",
    "gene_3w": "30k",
    "gene_4w": "40k",
    "gene_5w": "50k",
    "celltype_7": "7",
    "celltype_9": "9",
    "celltype_11": "11",
    "celltype_13": "13",
    "celltype_15": "15",
    "batch_2": "2",
    "batch_4": "4",
    "batch_6": "6",
    "batch_8": "8",
    "batch_10": "10",
    "batch_0.2": "0.2",
    "batch_0.4": "0.4",
    "batch_0.6": "0.6",
    "batch_0.8": "0.8",
    "batch_1.0": "1.0",
    "dropout_-1": "-1",
    "dropout_0": "0",
    "dropout_1": "1",
    "dropout_2": "2",
    "dropout_3": "3",
    "de_prob_0.05": "0.05",
    "de_prob_0.15": "0.15",
    "de_prob_0.2": "0.20",
    "de_prob_0.25": "0.25",
    "de_prob_0.3": "0.30",
    "de_0.2": "0.2",
    "de_0.4": "0.4",
    "de_0.6": "0.6",
    "de_0.8": "0.8",
    "de_1.0": "1.0",
    "out_0.1": "0.1",
    "out_0.2": "0.2",
    "out_0.3": "0.3",
    "out_0.4": "0.4",
    "out_0.5": "0.5",
}

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

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "simulated_robustness_score",
    ["#F2EFE6", "#CAD9D7", "#88BBB6", "#3F9C9A", "#214D6C"],
)
SCORE_CMAP.set_bad("#D7D7D7")

COMPONENT_CMAP = LinearSegmentedColormap.from_list(
    "component_balance",
    ["#F3F0E6", "#D7C69B", "#C79B38", "#7F5F1D"],
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 5.8,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
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
    }.get(name, name)


def panel_label(ax: plt.Axes, label: str, dx: float = -0.12, dy: float = 1.04) -> None:
    ax.text(
        dx,
        dy,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#171717",
    )


def panel_header(
    ax: plt.Axes,
    label: str,
    title: str,
    dx: float = -0.08,
    dy: float = 1.045,
    title_dx: float = 0.085,
) -> None:
    ax.text(
        dx,
        dy,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#171717",
    )
    ax.text(
        dx + title_dx,
        dy,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="normal",
        color="#171717",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compact_parameter_labels(axis_name: str, order: list[str]) -> list[str]:
    labels = [VALUE_LABELS[d] for d in order]
    if axis_name == "cell_number":
        keep = {0, 2, 4, 6, 8}
        return [label if idx in keep else "" for idx, label in enumerate(labels)]
    return labels


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long = pd.read_csv(SOURCE)
    availability = pd.read_csv(AVAILABILITY)
    manifest = pd.read_csv(METHODS)
    full_methods = (
        manifest[
            manifest["benchmark_scope"].eq("full_26_method_benchmark")
            & (~manifest["is_variant"].astype(bool))
        ]
        .sort_values("method_order")["method_id"]
        .tolist()
    )
    family_lookup = manifest.set_index("method_id")["method_family"].to_dict()
    long = long[long["method_id"].isin(full_methods)].copy()
    long["method_id"] = pd.Categorical(long["method_id"], categories=full_methods, ordered=True)
    long["family_short"] = long["method_family"].map(FAMILY_RENAME)
    long["family_short"] = long["family_short"].fillna(long["method_id"].map(family_lookup).map(FAMILY_RENAME))
    long["perturbation_axis"] = pd.Categorical(long["perturbation_axis"], categories=AXIS_ORDER, ordered=True)
    return long, availability, manifest


def draw_score_validation(ax: plt.Axes, method_order: list[str]) -> pd.DataFrame:
    check = pd.read_csv(VALIDATION)
    check = check[
        check["method_id"].isin(method_order)
        & check["perturbation_axis"].isin(AXIS_ORDER)
        & check["published_axis_score"].notna()
        & check["recomputed_axis_score"].notna()
    ].copy()
    check["perturbation_axis"] = pd.Categorical(check["perturbation_axis"], categories=AXIS_ORDER, ordered=True)
    check["method_id"] = pd.Categorical(check["method_id"], categories=method_order, ordered=True)
    check = check.sort_values(["perturbation_axis", "method_id"])
    check["abs_diff"] = (check["published_axis_score"] - check["recomputed_axis_score"]).abs()

    axis_colors = {
        axis: color
        for axis, color in zip(
            AXIS_ORDER,
            ["#3D6FB6", "#7396C8", "#A8BEDB", "#BF6F6B", "#D69B97", "#C79B38", "#DAB969", "#E6CF8E", "#3F9C9A"],
        )
    }
    for axis in AXIS_ORDER:
        sub = check[check["perturbation_axis"].astype(str).eq(axis)]
        if sub.empty:
            continue
        ax.scatter(
            sub["published_axis_score"],
            sub["recomputed_axis_score"],
            s=9,
            color=axis_colors[axis],
            alpha=0.62,
            edgecolor="white",
            linewidth=0.20,
        )
    ax.plot([0, 1], [0, 1], color="#2B2B2B", linewidth=0.7, linestyle=(0, (3, 2)))
    med = check["abs_diff"].median()
    p90 = check["abs_diff"].quantile(0.90)
    ax.text(
        0.045,
        0.955,
        f"median |diff|={med:.3f}\n90th pct={p90:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.8,
        color="#2B2B2B",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("published axis score")
    ax.set_ylabel("recomputed axis score")
    ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    panel_header(ax, "a", "Robustness-score validation", dx=-0.16, dy=1.055, title_dx=0.14)
    clean_axis(ax)
    return check[
        [
            "method_id",
            "perturbation_axis",
            "published_axis_score",
            "recomputed_axis_score",
            "abs_diff",
        ]
    ]


def draw_parameter_heatmap(fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    method_order = list(df["method_id"].cat.categories)
    columns = [dataset for axis in AXIS_ORDER for dataset in DISPLAY_DATASETS[axis]]
    heat = (
        df.pivot_table(index="method_id", columns="dataset_id", values="score", aggfunc="mean", observed=False)
        .reindex(index=method_order, columns=columns)
    )
    data = np.ma.masked_invalid(heat.to_numpy(dtype=float))
    im = ax.imshow(data, aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlim(-1.35, len(columns) - 0.5)
    ax.set_ylim(len(method_order) - 0.5, -0.5)

    method_family = (
        df.drop_duplicates("method_id")
        .set_index("method_id")["family_short"]
        .reindex(method_order)
        .fillna("linear")
    )
    for y, fam in enumerate(method_family):
        ax.add_patch(Rectangle((-1.25, y - 0.48), 0.52, 0.96, facecolor=FAMILY_COLORS[fam], edgecolor="none", clip_on=False))

    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels([display_method(m) for m in method_order], fontsize=4.9)
    ax.tick_params(axis="y", pad=1.5, length=0)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([VALUE_LABELS[c] for c in columns], rotation=90, ha="center", va="top", fontsize=4.5)
    ax.tick_params(axis="x", pad=1.5, length=0)

    start = 0
    for axis in AXIS_ORDER:
        width = len(DISPLAY_DATASETS[axis])
        center = start + (width - 1) / 2
        if start > 0:
            ax.axvline(start - 0.5, color="white", linewidth=1.05)
        ax.text(center, -1.35, HEATMAP_AXIS_LABEL[axis], ha="center", va="bottom", fontsize=4.85, fontweight="bold", clip_on=False)
        start += width

    for y in np.arange(0.5, len(method_order), 1):
        ax.axhline(y, color="white", linewidth=0.18, alpha=0.62)

    missing_mask = heat.isna().to_numpy()
    for y, x in np.argwhere(missing_mask):
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="#D7D7D7",
                edgecolor="#AFAFAF",
                linewidth=0.12,
                hatch="///",
            )
        )

    panel_header(ax, "b", "Parameter-level robustness landscape across 26 methods", dx=-0.070, dy=1.125, title_dx=0.070)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = ax.inset_axes([1.012, 0.66, 0.014, 0.28])
    cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(labelsize=4.5, length=1.5, pad=0.8)
    cax.set_title("score", fontsize=4.9, pad=2.0)
    ax.add_patch(
        Rectangle(
            (1.012, 0.595),
            0.014,
            0.030,
            transform=ax.transAxes,
            facecolor="#D7D7D7",
            edgecolor="#AFAFAF",
            linewidth=0.25,
            hatch="///",
            clip_on=False,
        )
    )
    ax.text(
        1.034,
        0.610,
        "not generated",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.7,
        color="#444444",
        clip_on=False,
    )
    return heat.reset_index()


def draw_family_axis_heatmap(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    fam_axis = (
        df[df["score_available"]]
        .groupby(["family_short", "perturbation_axis"], observed=True)["score"]
        .median()
        .unstack("perturbation_axis")
        .reindex(index=FAMILY_ORDER, columns=AXIS_ORDER)
    )
    im = ax.imshow(fam_axis.to_numpy(), aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels(FAMILY_ORDER)
    ax.set_xticks(np.arange(len(AXIS_ORDER)))
    ax.set_xticklabels([SHORT_AXIS_LABEL[a] for a in AXIS_ORDER], rotation=35, ha="right")
    for i in range(fam_axis.shape[0]):
        for j in range(fam_axis.shape[1]):
            val = fam_axis.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.8, color="#111111" if val < 0.66 else "white")
    panel_header(ax, "c", "Family median by perturbation", dx=-0.15, dy=1.055, title_dx=0.13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.inset_axes([1.035, 0.18, 0.035, 0.62])
    cbar = plt.colorbar(im, cax=cb, orientation="vertical", ticks=[0, 0.5, 1])
    cbar.outline.set_linewidth(0.35)
    cb.tick_params(labelsize=4.4, length=1.5, pad=0.5)
    return fam_axis.reset_index()


def draw_component_balance(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    comp = df[df["score_available"]].melt(
        id_vars=["method_id", "family_short"],
        value_vars=["local_score", "global_score", "cluster_score"],
        var_name="component",
        value_name="component_score",
    )
    comp["component"] = comp["component"].map({"local_score": "Local", "global_score": "Global", "cluster_score": "Cluster"})
    table = (
        comp.groupby(["family_short", "component"], observed=True)["component_score"]
        .median()
        .unstack("component")
        .reindex(index=FAMILY_ORDER, columns=["Local", "Global", "Cluster"])
    )
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap=COMPONENT_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels(FAMILY_ORDER)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(table.columns)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.0, color="#111111" if val < 0.58 else "white")
    panel_header(ax, "d", "Metric-domain balance", dx=-0.16, dy=1.055, title_dx=0.13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.inset_axes([1.035, 0.18, 0.035, 0.62])
    cbar = plt.colorbar(im, cax=cb, orientation="vertical", ticks=[0, 0.5, 1])
    cbar.outline.set_linewidth(0.35)
    cb.tick_params(labelsize=4.4, length=1.5, pad=0.5)
    return table.reset_index()


def draw_ranking(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    method_order = list(df["method_id"].cat.categories)
    rank = (
        df[df["score_available"]]
        .groupby(["method_id", "family_short"], observed=True)["score"]
        .agg(median="median", q25=lambda x: np.quantile(x, 0.25), q75=lambda x: np.quantile(x, 0.75), n="count")
        .reset_index()
    )
    rank["method_id"] = pd.Categorical(rank["method_id"], categories=method_order, ordered=True)
    rank = rank.sort_values("method_id")
    y = np.arange(len(rank))
    for yi, row in rank.iterrows():
        ax.plot([row["q25"], row["q75"]], [yi, yi], color="#C5C5C5", linewidth=1.2, solid_capstyle="round")
        ax.scatter(row["median"], yi, s=15, color=FAMILY_COLORS[row["family_short"]], edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(str(m)) for m in rank["method_id"]], fontsize=4.45)
    ax.invert_yaxis()
    ax.set_xlim(0.12, 0.74)
    ax.set_xlabel("median score with IQR")
    panel_header(ax, "e", "Overall method robustness", dx=-0.14, dy=1.055, title_dx=0.12)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)
    return rank


def draw_family_distribution(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    plot = df[df["score_available"]].copy()
    groups = [plot[plot["family_short"].eq(f)]["score"].dropna().values for f in FAMILY_ORDER]
    positions = np.arange(len(FAMILY_ORDER))
    vp = ax.violinplot(groups, positions=positions, widths=0.82, showmeans=False, showmedians=False, showextrema=False)
    for body, fam in zip(vp["bodies"], FAMILY_ORDER):
        body.set_facecolor(FAMILY_COLORS[fam])
        body.set_edgecolor("none")
        body.set_alpha(0.24)
    bp = ax.boxplot(groups, positions=positions, widths=0.26, patch_artist=True, showfliers=False)
    for patch, fam in zip(bp["boxes"], FAMILY_ORDER):
        patch.set_facecolor(FAMILY_COLORS[fam])
        patch.set_edgecolor("#343434")
        patch.set_linewidth(0.45)
        patch.set_alpha(0.78)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#343434")
            artist.set_linewidth(0.45)
    rng = np.random.default_rng(20260605)
    for x, fam in enumerate(FAMILY_ORDER):
        vals = plot[plot["family_short"].eq(fam)]["score"].dropna().sample(frac=0.22, random_state=13).values
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(np.full_like(vals, x, dtype=float) + jitter, vals, s=3.0, color=FAMILY_COLORS[fam], alpha=0.18, linewidths=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(FAMILY_ORDER)
    ax.set_ylim(0.1, 0.85)
    ax.set_ylabel("score")
    panel_header(ax, "f", "Score distributions by family", dx=-0.17, dy=1.055, title_dx=0.14)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)
    return plot[["method_id", "family_short", "dataset_id", "perturbation_axis", "score"]]


def axis_family_median(df: pd.DataFrame, axis_name: str) -> pd.DataFrame:
    subset = df[df["perturbation_axis"].eq(axis_name) & df["score_available"]].copy()
    order = DISPLAY_DATASETS[axis_name]
    subset["dataset_id"] = pd.Categorical(subset["dataset_id"], categories=order, ordered=True)
    out = (
        subset.groupby(["family_short", "dataset_id"], observed=True)["score"]
        .median()
        .reset_index()
        .sort_values(["family_short", "dataset_id"])
    )
    out["x"] = out["dataset_id"].cat.codes
    return out


def draw_trajectory_group(
    fig: plt.Figure,
    spec,
    df: pd.DataFrame,
    label: str,
    title: str,
    axes: list[str],
    y_limits: tuple[float, float] = (0.12, 0.78),
) -> pd.DataFrame:
    parent = fig.add_subplot(spec)
    parent.axis("off")
    panel_header(parent, label, title, dx=-0.080, dy=1.065, title_dx=0.090)
    sub = GridSpecFromSubplotSpec(1, len(axes), subplot_spec=spec, wspace=0.26)
    records = []
    for idx, axis_name in enumerate(axes):
        ax = fig.add_subplot(sub[0, idx])
        table = axis_family_median(df, axis_name)
        records.append(table.assign(panel=label, trajectory_axis=axis_name))
        for fam in FAMILY_ORDER:
            vals = table[table["family_short"].eq(fam)].copy()
            if vals.empty:
                continue
            ax.plot(
                vals["x"],
                vals["score"],
                color=FAMILY_COLORS[fam],
                linewidth=1.1,
                marker="o",
                markersize=2.0,
                markeredgecolor="white",
                markeredgewidth=0.25,
            )
        ax.set_title(AXIS_LABEL[axis_name], fontsize=5.9, pad=2.0, y=1.01)
        order = DISPLAY_DATASETS[axis_name]
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(compact_parameter_labels(axis_name, order), rotation=45, ha="right", fontsize=4.5)
        ax.set_ylim(*y_limits)
        ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.45)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.set_ylabel("median score")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        clean_axis(ax)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def add_family_legend(fig: plt.Figure) -> None:
    handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f) for f in FAMILY_ORDER]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.010),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.0,
        handleheight=0.7,
        fontsize=5.8,
        title="Method family",
        title_fontsize=6.0,
    )


def write_qa(
    panel_tables: dict[str, pd.DataFrame],
    figure_path: Path,
    df: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    png_path = figure_path.with_suffix(".png")
    with Image.open(png_path) as img:
        arr = np.asarray(img.convert("RGB"))
        pixel_std = float(arr.std())
        width, height = img.size
    missing = df[~df["score_available"]]
    qa = {
        "figure": figure_path.name,
        "full_26_methods": int(df["method_id"].nunique()),
        "parameter_datasets": int(df["dataset_id"].nunique()),
        "complete_parameter_datasets": int(availability["complete_score_inputs"].sum()),
        "incomplete_parameter_datasets": int((~availability["complete_score_inputs"]).sum()),
        "score_rows_available": int(df["score_available"].sum()),
        "score_rows_missing": int((~df["score_available"]).sum()),
        "missing_datasets": ";".join(sorted(missing["dataset_id"].astype(str).unique())),
        "png_width_px": width,
        "png_height_px": height,
        "png_pixel_std": pixel_std,
        "nonblank_png": bool(pixel_std > 5),
        "score_min": float(df["score"].min(skipna=True)),
        "score_max": float(df["score"].max(skipna=True)),
    }
    pd.DataFrame([qa]).to_csv(QA_OUT / "Figure_7_visual_qa_summary.csv", index=False)

    for name, table in panel_tables.items():
        table.to_csv(SOURCE_DIR / f"Figure_7_panel_{name}_plotted_values.csv", index=False)

    checklist = f"""# Figure 7 Polished Visual QA

Generated by: `Publication/paper/revision_figures/figure7_polish/make_figure7_simulated_robustness_top_tier.py`

## Figure Contract

- Core claim: simulated perturbation robustness is method- and stress-axis dependent across the full 26-method benchmark.
- Evidence hierarchy: panel b preserves the original parameter-level Figure 7 layer; panel a validates the derived robustness axis scores against recomputed source metrics; panels c-j summarize family structure, method ranking, and stress trajectories.
- Unit of analysis: method x simulated parameter-level dataset score.
- Missingness: cell_4w and cell_5w are shown as not generated/not scored; they are not averaged into summaries.

## QA

- Full benchmark methods plotted: {qa['full_26_methods']}.
- Parameter datasets shown: {qa['parameter_datasets']}; complete: {qa['complete_parameter_datasets']}; incomplete: {qa['incomplete_parameter_datasets']}.
- Available score rows: {qa['score_rows_available']}; missing score rows: {qa['score_rows_missing']}.
- Missing datasets: {qa['missing_datasets']}.
- Score range: {qa['score_min']:.4f}-{qa['score_max']:.4f}.
- PNG dimensions: {qa['png_width_px']} x {qa['png_height_px']} px; nonblank: {qa['nonblank_png']}.

## Panel Map

- a, recomputed robustness axis scores versus published axis-score summaries across methods and perturbation axes.
- b, 26-method parameter-level robustness heatmap, retaining the original Figure 7 information layer and explicit not-generated/not-scored cells.
- c, family median robustness by perturbation axis.
- d, local/global/clustering component balance by method family.
- e, all-method median robustness ranking with IQR across parameter levels.
- f, per-parameter score distributions grouped by method family.
- g, scale stress trajectories for cell number, gene number, and cell-type number.
- h, batch stress trajectories for batch number and batch strength.
- i, technical-noise trajectories for dropout and outlier proportion.
- j, biological-signal trajectories for DE probability and DE strength.
"""
    (QA_OUT / "Figure_7_polished_visual_qa_checklist.md").write_text(checklist, encoding="utf-8")


def main() -> None:
    df, availability, _ = load_data()

    fig = plt.figure(figsize=(9.20, 10.20))
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[0.88, 3.55, 2.38, 2.22],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.50,
        wspace=0.64,
        left=0.064,
        right=0.982,
        bottom=0.074,
        top=0.976,
    )

    panel_tables: dict[str, pd.DataFrame] = {}
    panel_tables["a"] = draw_score_validation(fig.add_subplot(gs[0, 0:2]), list(df["method_id"].cat.categories))
    panel_tables["c"] = draw_family_axis_heatmap(fig.add_subplot(gs[0, 2:4]), df)
    panel_tables["d"] = draw_component_balance(fig.add_subplot(gs[0, 4:6]), df)
    panel_tables["b"] = draw_parameter_heatmap(fig, fig.add_subplot(gs[1, 0:6]), df)
    panel_tables["e"] = draw_ranking(fig.add_subplot(gs[2, 0:2]), df)
    panel_tables["f"] = draw_family_distribution(fig.add_subplot(gs[2, 2:4]), df)
    panel_tables["g"] = draw_trajectory_group(fig, gs[2, 4:6], df, "g", "Scale stress response", ["cell_number", "gene_number", "celltype_number"])
    panel_tables["h"] = draw_trajectory_group(fig, gs[3, 0:2], df, "h", "Batch stress response", ["batch_number", "batch_strength"])
    panel_tables["i"] = draw_trajectory_group(fig, gs[3, 2:4], df, "i", "Technical-noise response", ["dropout", "out"])
    panel_tables["j"] = draw_trajectory_group(fig, gs[3, 4:6], df, "j", "Biological-signal response", ["de_prob", "de_strength"])

    add_family_legend(fig)

    base = PLOT_OUT / FIGURE_BASENAME
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_qa(panel_tables, base.with_suffix(".png"), df, availability)


if __name__ == "__main__":
    main()
