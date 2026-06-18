from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[4]
CANON = ROOT / "Publication" / "paper" / "revision_figures" / "canonical_source_tables"
FIG4_SOURCE = (
    ROOT
    / "Publication"
    / "paper"
    / "revision_figures"
    / "redesigned_python_figure_package"
    / "source_data"
    / "Figure_4_structure_preservation_source_data.csv"
)
OUT = ROOT / "Publication" / "paper" / "revision_figures" / "figure4_polish"
PLOT_OUT = OUT / "polished"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
for path in [PLOT_OUT, SOURCE_OUT, QA_OUT]:
    path.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6.1,
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
    "gold": "#C79B38",
    "rose": "#BF6F6B",
    "slate": "#5F6B7A",
    "gray": "#A8A8A8",
    "light_gray": "#E7E7E7",
    "dark": "#2F2F2F",
}

FAMILY_COLORS = {
    "linear/probabilistic": PALETTE["blue"],
    "deep generative/autoencoder": PALETTE["rose"],
    "graph/diffusion": PALETTE["gold"],
    "metric/structure-aware": PALETTE["teal"],
    "other": PALETTE["gray"],
}

LOCAL_DISPLAY = {
    "knn_10": "KNN 10",
    "knn_30": "KNN 30",
    "nkr_30": "NKR 30",
    "aji_30": "AJI 30",
    "T_30": "Trust. 30",
    "C_30": "Cont. 30",
    "nh_30": "NH 30",
    "svm": "SVM",
}

GLOBAL_DISPLAY = {
    "random_triplet": "Triplet",
    "spearman": "Spearman",
    "k-nearest": "k-nearest",
    "centroid_distance": "Centroid",
    "AUC": "AUC",
    "Qglobal": "Q global",
    "Pearson": "Pearson",
}


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.085, y: float = 1.025) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.3,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["dark"],
    )


def set_ticks(ax: plt.Axes, labelsize: float = 5.2) -> None:
    ax.tick_params(axis="both", labelsize=labelsize, length=2.2, pad=1.5)


def method_family_map() -> dict[str, str]:
    methods = pd.read_csv(CANON / "canonical_method_manifest.csv")
    return dict(zip(methods["method_id"], methods["method_family"]))


def canonical_full_methods() -> list[str]:
    methods = pd.read_csv(CANON / "canonical_method_manifest.csv")
    full = methods[methods["benchmark_scope"].eq("full_26_method_benchmark")].copy()
    return full.sort_values("method_order")["method_id"].tolist()


def method_color(method: str, fmap: dict[str, str]) -> str:
    return FAMILY_COLORS.get(fmap.get(method, "other"), FAMILY_COLORS["other"])


def family_label(family: str) -> str:
    labels = {
        "linear/probabilistic": "linear/probabilistic",
        "deep generative/autoencoder": "deep generative/autoencoder",
        "graph/diffusion": "graph/diffusion",
        "metric/structure-aware": "metric/structure-aware",
    }
    return labels.get(family, family)


def draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    label_map: dict[str, str],
    cmap: str,
    cbar_label: str,
) -> None:
    matrix = data.astype(float).to_numpy()
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap_obj, vmin=0, vmax=1)
    ax.set_title(title, fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([label_map.get(c, c) for c in data.columns], rotation=38, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    set_ticks(ax, 4.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=2)
    cbar.set_label(cbar_label, fontsize=5.1)


def draw_ranked_bar(
    ax: plt.Axes,
    values: pd.Series,
    title: str,
    xlabel: str,
    fmap: dict[str, str],
    top: int | None = None,
) -> pd.DataFrame:
    vals_desc = values.dropna().sort_values(ascending=False)
    if top is not None:
        vals_desc = vals_desc.head(top)
    vals = vals_desc.iloc[::-1]
    colors = [method_color(str(m), fmap) for m in vals.index]
    ax.barh(vals.index, vals.values, color=colors, edgecolor="white", linewidth=0.38, height=0.62)
    ax.set_title(title, fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.03)
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    ax.tick_params(axis="y", labelsize=4.15, length=0, pad=1.0)
    ax.tick_params(axis="x", labelsize=4.8, length=2.0, pad=1.3)
    return pd.DataFrame({"method_id": vals.index[::-1], "score": vals.values[::-1], "rank": np.arange(1, len(vals) + 1)})


def draw_distribution(ax: plt.Axes, raw: pd.DataFrame, fmap: dict[str, str]) -> pd.DataFrame:
    selected = raw[raw["metric"].isin(["knn_30", "T_30", "C_30", "nh_30", "random_triplet", "spearman", "Pearson"])].copy()
    selected["metric_class"] = np.where(
        selected["metric"].isin(["knn_30", "T_30", "C_30", "nh_30"]), "Local", "Global"
    )
    selected["family"] = selected["method_id"].map(fmap).fillna("other")
    selected["family_label"] = selected["family"].map(family_label)
    selected["value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected.dropna(subset=["value", "metric_class", "family_label"])
    sns.boxplot(
        data=selected,
        x="metric_class",
        y="value",
        hue="family_label",
        palette=FAMILY_COLORS,
        linewidth=0.55,
        fliersize=0.8,
        ax=ax,
    )
    sample = selected.sample(min(len(selected), 3600), random_state=11)
    sns.stripplot(
        data=sample,
        x="metric_class",
        y="value",
        color=PALETTE["slate"],
        size=1.05,
        alpha=0.18,
        jitter=0.22,
        ax=ax,
    )
    ax.set_title("Raw metric distributions", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("raw metric value")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color=PALETTE["light_gray"], lw=0.45)
    set_ticks(ax, 5.1)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    legend_items = [
        ("linear", FAMILY_COLORS["linear/probabilistic"]),
        ("deep", FAMILY_COLORS["deep generative/autoencoder"]),
        ("graph", FAMILY_COLORS["graph/diffusion"]),
        ("metric", FAMILY_COLORS["metric/structure-aware"]),
    ]
    x0 = 0.18
    for i, (label, color) in enumerate(legend_items):
        x = x0 + i * 0.16
        ax.scatter(x, -0.17, s=17, color=color, edgecolor="white", lw=0.35, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.018, -0.17, label, transform=ax.transAxes, fontsize=4.8, va="center", ha="left", color=PALETTE["dark"])
    return selected


def draw_tradeoff(ax: plt.Axes, score: pd.DataFrame, fmap: dict[str, str]) -> pd.DataFrame:
    work = score[["method_id", "local", "global", "overall_mean"]].dropna().copy()
    work["family"] = work["method_id"].map(fmap).fillna("other")
    x_med = work["local"].median()
    y_med = work["global"].median()
    ax.axline((0, 0), slope=1, color="#D0D0D0", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axvline(x_med, color="#D9D9D9", lw=0.65, ls=(0, (2, 2)), zorder=0)
    ax.axhline(y_med, color="#D9D9D9", lw=0.65, ls=(0, (2, 2)), zorder=0)
    for _, row in work.iterrows():
        ax.scatter(
            row["local"],
            row["global"],
            s=22 + 80 * row["overall_mean"],
            color=method_color(row["method_id"], fmap),
            edgecolor="white",
            lw=0.45,
            alpha=0.92,
            zorder=3,
        )
    label_offsets = {
        "SQuaD-MDS": (0.012, 0.020),
        "scvis": (0.012, 0.018),
        "SPDR": (0.026, -0.006),
        "TriMap": (0.012, 0.000),
        "t-SNE": (0.010, -0.040),
        "PCA": (0.010, 0.020),
        "VASC": (-0.068, -0.006),
    }
    for method, (dx, dy) in label_offsets.items():
        row = work.loc[work["method_id"].eq(method)]
        if row.empty:
            continue
        x = float(row["local"].iloc[0])
        y = float(row["global"].iloc[0])
        ax.text(x + dx, y + dy, method, fontsize=5.1, color=PALETTE["dark"], ha="left", va="center")
    ax.text(0.035, 0.965, "global-high", transform=ax.transAxes, fontsize=5.1, color=PALETTE["slate"], va="top")
    ax.text(0.965, 0.035, "local-high", transform=ax.transAxes, fontsize=5.1, color=PALETTE["slate"], ha="right")
    corr = work["local"].corr(work["global"])
    ax.text(0.965, 0.965, f"r = {corr:.2f}", transform=ax.transAxes, fontsize=5.4, color=PALETTE["dark"], ha="right", va="top")
    ax.set_title("Local versus global scores", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("local score")
    ax.set_ylabel("global score")
    ax.set_xlim(0.15, 0.84)
    ax.set_ylim(0.25, 0.88)
    ax.grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(ax, 5.1)
    return work


def draw_dataset_rank_stability(
    ax: plt.Axes,
    raw: pd.DataFrame,
    canonical_methods: list[str],
    fmap: dict[str, str],
) -> pd.DataFrame:
    metrics = list(LOCAL_DISPLAY) + list(GLOBAL_DISPLAY)
    work = raw[raw["metric"].isin(metrics)].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["dataset_category", "dataset_id", "method_id", "value"])
    dataset_cols = ["dataset_category", "dataset_id"]
    score = (
        work.groupby(dataset_cols + ["method_id"], observed=False)["value"]
        .mean()
        .reset_index(name="structure_score")
    )
    coverage = score.groupby(dataset_cols, observed=False)["method_id"].nunique().reset_index(name="methods")
    complete_datasets = coverage.loc[coverage["methods"].ge(20), dataset_cols]
    score = score.merge(complete_datasets, on=dataset_cols, how="inner")
    score["rank"] = score.groupby(dataset_cols, observed=False)["structure_score"].rank(
        method="average", ascending=False
    )
    summary = (
        score.groupby("method_id", observed=False)
        .agg(
            median_rank=("rank", "median"),
            q1_rank=("rank", lambda x: float(np.quantile(x, 0.25))),
            q3_rank=("rank", lambda x: float(np.quantile(x, 0.75))),
            mean_structure_score=("structure_score", "mean"),
            top3_fraction=("rank", lambda x: float(np.mean(x <= 3))),
            dataset_instances=("rank", "size"),
        )
        .reindex(canonical_methods)
        .reset_index()
    )
    y = np.arange(len(summary))
    colors = [method_color(method, fmap) for method in summary["method_id"]]
    for yi, (_, row), color in zip(y, summary.iterrows(), colors):
        ax.hlines(yi, row["q1_rank"], row["q3_rank"], color=color, lw=2.0, alpha=0.76)
        ax.scatter(
            row["median_rank"],
            yi,
            s=14 + 50 * row["top3_fraction"],
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(summary["method_id"])
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(canonical_methods) + 0.5)
    ax.set_title("Dataset-level rank stability", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("within-dataset rank (lower better)")
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    ax.tick_params(axis="y", labelsize=4.35, length=0, pad=1.2)
    ax.tick_params(axis="x", labelsize=5.0, length=2.2, pad=1.5)
    return summary


def draw_family_structure_profile(ax: plt.Axes, score: pd.DataFrame, fmap: dict[str, str]) -> pd.DataFrame:
    work = score[["method_id", "local", "global", "kmeans"]].copy()
    work["family"] = work["method_id"].map(fmap).fillna("other")
    family_order = [
        "linear/probabilistic",
        "deep generative/autoencoder",
        "graph/diffusion",
        "metric/structure-aware",
    ]
    family_short = {
        "linear/probabilistic": "linear",
        "deep generative/autoencoder": "deep",
        "graph/diffusion": "graph",
        "metric/structure-aware": "metric",
    }
    profile = (
        work.groupby("family", observed=False)[["local", "global", "kmeans"]]
        .median()
        .reindex(family_order)
        .rename(index=family_short, columns={"local": "Local", "global": "Global", "kmeans": "Cluster"})
    )
    cmap_obj = plt.get_cmap("YlGnBu").copy()
    im = ax.imshow(profile.to_numpy(dtype=float), aspect="auto", cmap=cmap_obj, vmin=0.25, vmax=0.75)
    ax.set_title("Family-level structure profile", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xticks(np.arange(profile.shape[1]))
    ax.set_xticklabels(profile.columns)
    ax.set_yticks(np.arange(profile.shape[0]))
    ax.set_yticklabels(profile.index)
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            value = profile.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.1, color="white" if value > 0.58 else PALETTE["dark"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=2)
    cbar.set_label("median score", fontsize=5.1)
    return profile.reset_index(names="family")


def draw_rank_discordance(ax: plt.Axes, score: pd.DataFrame, fmap: dict[str, str]) -> pd.DataFrame:
    work = score[["method_id", "local", "global"]].dropna().copy()
    work["local_rank"] = work["local"].rank(method="average", ascending=False)
    work["global_rank"] = work["global"].rank(method="average", ascending=False)
    work["rank_discordance"] = (work["local_rank"] - work["global_rank"]).abs()
    work["family"] = work["method_id"].map(fmap).fillna("other")

    ax.plot([1, 26], [1, 26], color="#CFCFCF", lw=0.7, ls=(0, (3, 2)), zorder=0)
    for _, row in work.iterrows():
        ax.scatter(
            row["local_rank"],
            row["global_rank"],
            s=20 + 3.8 * row["rank_discordance"],
            color=method_color(row["method_id"], fmap),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.92,
            zorder=3,
        )
    label_rows = work.sort_values("rank_discordance", ascending=False).head(7)
    for _, row in label_rows.iterrows():
        ax.text(
            row["local_rank"] + 0.45,
            row["global_rank"],
            str(row["method_id"]),
            fontsize=4.7,
            va="center",
            ha="left",
            color=PALETTE["dark"],
        )
    ax.set_title("Local-global rank discordance", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("local rank")
    ax.set_ylabel("global rank")
    ax.set_xlim(0.4, 26.9)
    ax.set_ylim(26.9, 0.4)
    ax.grid(color=PALETTE["light_gray"], lw=0.45)
    set_ticks(ax, 5.0)
    return work


def draw_metric_heterogeneity(
    ax: plt.Axes,
    local_heat: pd.DataFrame,
    global_heat: pd.DataFrame,
    fmap: dict[str, str],
) -> pd.DataFrame:
    combined = pd.concat(
        [
            local_heat.add_prefix("local_"),
            global_heat.add_prefix("global_"),
        ],
        axis=1,
    )
    values = combined.apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "method_id": values.index,
            "metric_median": values.median(axis=1, skipna=True).values,
            "metric_iqr": (
                values.quantile(0.75, axis=1, interpolation="linear")
                - values.quantile(0.25, axis=1, interpolation="linear")
            ).values,
            "metric_range": (values.max(axis=1, skipna=True) - values.min(axis=1, skipna=True)).values,
            "n_metrics": values.notna().sum(axis=1).values,
        }
    ).dropna(subset=["metric_iqr"])
    plot = summary.sort_values("metric_iqr", ascending=True)
    y = np.arange(len(plot))
    colors = [method_color(method, fmap) for method in plot["method_id"]]
    ax.hlines(y, 0, plot["metric_iqr"], color=colors, lw=1.7, alpha=0.78)
    ax.scatter(plot["metric_iqr"], y, s=14, color=colors, edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["method_id"])
    ax.set_xlim(0, max(0.05, float(plot["metric_iqr"].max()) * 1.18))
    ax.set_title("Metric-level heterogeneity", fontsize=7.2, fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("IQR across structure metrics")
    ax.grid(axis="x", color=PALETTE["light_gray"], lw=0.45)
    ax.tick_params(axis="y", labelsize=4.15, length=0, pad=1.0)
    ax.tick_params(axis="x", labelsize=4.8, length=2.0, pad=1.3)
    return summary


def main() -> None:
    fmap = method_family_map()
    canonical_methods = canonical_full_methods()
    fig4_source = pd.read_csv(FIG4_SOURCE)
    local_summary = fig4_source[fig4_source["source"].eq("local_summary")].set_index("method_id")
    global_summary = fig4_source[fig4_source["source"].eq("global_summary")].set_index("method_id")
    raw = fig4_source[fig4_source["source"].eq("structure_metric_raw")].copy()
    score = pd.read_csv(CANON / "original_score_matrix.csv")
    local_summary = local_summary.loc[local_summary.index.intersection(canonical_methods)]
    global_summary = global_summary.loc[global_summary.index.intersection(canonical_methods)]
    raw = raw[raw["method_id"].isin(canonical_methods)].copy()
    score = score[score["method_id"].isin(canonical_methods)].copy()

    local_cols = list(LOCAL_DISPLAY)
    global_cols = list(GLOBAL_DISPLAY)
    local_order = canonical_methods
    global_order = canonical_methods
    local_heat = local_summary.reindex(local_order)[local_cols]
    global_heat = global_summary.reindex(global_order)[global_cols]
    score_indexed = score.set_index("method_id")

    fig = plt.figure(figsize=(9.60, 10.95), constrained_layout=False)
    gs = GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[2.30, 2.52, 2.12, 2.36],
        hspace=0.52,
        wspace=0.40,
    )
    top_grid = gs[0, :].subgridspec(1, 2, wspace=0.28)
    axes = {
        "a": fig.add_subplot(top_grid[0, 0]),
        "b": fig.add_subplot(top_grid[0, 1]),
        "c": fig.add_subplot(gs[1, 0:2]),
        "d": fig.add_subplot(gs[1, 2:4]),
        "e": fig.add_subplot(gs[1, 4:6]),
        "f": fig.add_subplot(gs[2, 0:2]),
        "g": fig.add_subplot(gs[2, 2:4]),
        "h": fig.add_subplot(gs[2, 4:6]),
        "i": fig.add_subplot(gs[3, 0:3]),
        "j": fig.add_subplot(gs[3, 3:6]),
    }
    for label, ax in axes.items():
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.982, bottom=0.065)

    draw_heatmap(axes["a"], local_heat, "Local neighborhood-retention metrics", LOCAL_DISPLAY, "viridis", "median")
    draw_heatmap(axes["b"], global_heat, "Global geometry-preservation metrics", GLOBAL_DISPLAY, "mako", "median")
    local_rank = draw_ranked_bar(axes["c"], score_indexed["local"], "Aggregate local score", "normalized score", fmap)
    global_rank = draw_ranked_bar(axes["d"], score_indexed["global"], "Aggregate global score", "normalized score", fmap)
    dist_data = draw_distribution(axes["e"], raw, fmap)
    tradeoff = draw_tradeoff(axes["f"], score, fmap)
    rank_stability = draw_dataset_rank_stability(axes["g"], raw, canonical_methods, fmap)
    family_profile = draw_family_structure_profile(axes["h"], score, fmap)
    rank_discordance = draw_rank_discordance(axes["i"], score, fmap)
    metric_heterogeneity = draw_metric_heterogeneity(axes["j"], local_heat, global_heat, fmap)

    source_parts = [
        local_heat.reset_index().assign(panel="a_local_heatmap"),
        global_heat.reset_index().assign(panel="b_global_heatmap"),
        local_rank.assign(panel="c_local_rank"),
        global_rank.assign(panel="d_global_rank"),
        dist_data.assign(panel="e_metric_distribution"),
        tradeoff.assign(panel="f_local_global_tradeoff"),
        rank_stability.assign(panel="g_dataset_rank_stability"),
        family_profile.assign(panel="h_family_structure_profile"),
        rank_discordance.assign(panel="i_rank_discordance"),
        metric_heterogeneity.assign(panel="j_metric_heterogeneity"),
    ]
    panel_source = pd.concat(source_parts, ignore_index=True, sort=False)
    panel_source = panel_source.rename(
        columns={
            "metric_class": "structure_metric_class",
            "value": "raw_metric_value",
        }
    )
    panel_source.to_csv(SOURCE_OUT / "Figure_4_structure_preservation_top_tier_panel_data.csv", index=False)

    base = PLOT_OUT / "Figure_4_structure_preservation_top_tier"
    for fmt in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(base.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=600)
    plt.close(fig)

    qa = {
        "panel_count": 10,
        "canonical_full_methods": int(len(canonical_methods)),
        "local_heatmap_methods": int(local_heat.shape[0]),
        "global_heatmap_methods": int(global_heat.shape[0]),
        "structure_raw_methods": int(raw["method_id"].nunique()),
        "structure_dataset_instances": int(
            raw.dropna(subset=["dataset_category", "dataset_id"])
            .drop_duplicates(["dataset_category", "dataset_id"])
            .shape[0]
        ),
        "raw_distribution_rows": int(len(dist_data)),
        "tradeoff_methods": int(len(tradeoff)),
        "rank_stability_methods": int(rank_stability["method_id"].nunique()),
        "local_global_score_correlation": float(tradeoff["local"].corr(tradeoff["global"])),
        "top_local_method": str(score.sort_values("local", ascending=False)["method_id"].iloc[0]),
        "top_global_method": str(score.sort_values("global", ascending=False)["method_id"].iloc[0]),
        "max_local_global_rank_discordance": float(rank_discordance["rank_discordance"].max()),
        "highest_metric_heterogeneity_method": str(
            metric_heterogeneity.sort_values("metric_iqr", ascending=False)["method_id"].iloc[0]
        ),
    }
    pd.DataFrame([qa]).to_csv(QA_OUT / "Figure_4_structure_preservation_top_tier_qa.csv", index=False)
    print(f"Wrote {base.with_suffix('.png')}")
    for key, value in qa.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
