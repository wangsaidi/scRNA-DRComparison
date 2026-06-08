from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure6_polish"
PLOT_OUT = OUT / "polished"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"

for path in (PLOT_OUT, SOURCE_OUT, QA_OUT):
    path.mkdir(parents=True, exist_ok=True)

SOURCE = (
    ROOT
    / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_6_sensitivity_controls_source_data.csv"
)
METHOD_SOURCE = (
    ROOT
    / "Publication/paper/revision_figures/redesigned_python_figure_package/source_data/Figure_1_method_taxonomy_source_data.csv"
)

DIMENSIONS = [2, 5, 10, 20, 50]
HVGS = [500, 1000, 2000, 3000]
DIM_METHOD_ORDER = ["PCA", "scGBM", "scVI", "GLMPCA", "SAUCIE", "pCMF"]
HVG_METHOD_ORDER = ["PCA", "scGBM", "scVI", "GLMPCA", "UMAP", "t-SNE"]
FAMILY_ORDER = ["linear", "deep", "graph", "metric"]
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
CONTROL_LABELS = {
    "scvi": "scVI ref.",
    "dimension": "latent dim.",
    "workflow": "workflow",
    "hvg": "input genes",
}
WORKFLOW_LABELS = {"Direct 2D": "direct 2D", "PCA50 to 2D": "PCA50 to 2D"}
DATASET_LABELS = {
    "batch_1.0": "batch 1",
    "celltype_11": "11 types",
    "celltype_15": "15 types",
    "celltype_7": "7 types",
    "default": "default",
    "dropout_0": "dropout 0",
    "gene_5k": "5k genes",
    "gene_5w": "50k genes",
}

SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_teal", ["#F3F1E9", "#C8DEDA", "#71B9AD", "#2E807F", "#123D58"]
)
DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_balance", ["#2C6DAA", "#F7F6F2", "#B9504B"]
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 6.05,
        "axes.titlesize": 7.15,
        "axes.labelsize": 6.1,
        "xtick.labelsize": 5.55,
        "ytick.labelsize": 5.55,
        "axes.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "savefig.facecolor": "white",
    }
)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.095, y: float = 1.035) -> None:
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


def setup_axis(ax: plt.Axes) -> None:
    ax.tick_params(width=0.45, length=2.2, pad=1.4)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.55)
        ax.spines[side].set_color("#222222")


def display_method(method: object) -> str:
    text = str(method)
    return {"t-SNE": "t-SNE", "scVI": "scVI", "scGBM": "scGBM", "pCMF": "pCMF"}.get(text, text)


def fmt_value(value: float, digits: int = 2) -> str:
    value = float(value)
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def fmt_delta(value: float, digits: int = 2) -> str:
    value = float(value)
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:+.{digits}f}"


def load_family_map() -> dict[str, str]:
    taxonomy = pd.read_csv(METHOD_SOURCE)
    taxonomy["family_short"] = taxonomy["method_family"].map(FAMILY_RENAME).fillna("unknown")
    return taxonomy.drop_duplicates("parent_method").set_index("parent_method")["family_short"].to_dict()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    numeric_cols = [
        "ari",
        "nmi",
        "homogeneity",
        "completeness",
        "silhouette_label",
        "trustworthiness_k30",
        "dimension",
        "seed",
        "max_epochs",
        "n_cells",
        "n_genes",
        "runtime_seconds",
        "max_rss_mb",
        "hvg_requested",
        "hvg_actual",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_family_strip(ax: plt.Axes, methods: list[str], family_map: dict[str, str], x: float = -0.72) -> None:
    for y, method in enumerate(methods):
        family = family_map.get(method, "unknown")
        ax.add_patch(
            Rectangle(
                (x, y - 0.42),
                0.12,
                0.84,
                facecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["unknown"]),
                edgecolor="none",
                clip_on=False,
            )
        )
    ax.set_xlim(x - 0.02, ax.get_xlim()[1])


def draw_matrix(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    family_map: dict[str, str] | None = None,
    annotate: bool = True,
    cmap=SCORE_CMAP,
    norm=None,
    vmin: float | None = 0,
    vmax: float | None = 1,
    panel_label: str | None = None,
) -> None:
    arr = matrix.astype(float).values
    im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, vmin=vmin if norm is None else None, vmax=vmax if norm is None else None)
    ax.set_title(title, loc="left", pad=2.5, fontweight="bold")
    if panel_label:
        add_panel_label(ax, panel_label)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([display_method(x) for x in matrix.index])
    ax.tick_params(length=0, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if family_map is not None:
        add_family_strip(ax, [str(x) for x in matrix.index], family_map)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if pd.isna(val):
                    continue
                color = "white" if (norm(val) if norm is not None else val) > 0.56 else "#262626"
                ax.text(j, i, fmt_value(val), ha="center", va="center", fontsize=5.25, color=color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=1.8, width=0.45)
    cbar.set_label(cbar_label, fontsize=5.15, labelpad=1.4)
    cbar.outline.set_linewidth(0.45)


def draw_scvi_reference(ax: plt.Axes, scvi: pd.DataFrame) -> pd.DataFrame:
    records = []
    colors = {"ARI": "#3D6FB6", "Trustworthiness": "#3F9C9A"}
    offsets = {"ARI": -0.095, "Trustworthiness": 0.095}
    metrics = {"ARI": "ari", "Trustworthiness": "trustworthiness_k30"}
    x = np.arange(len(DIMENSIONS))
    rng = np.random.default_rng(20260604)
    for label, col in metrics.items():
        grouped = scvi.groupby("dimension")[col].agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            n="count",
        )
        grouped = grouped.reindex(DIMENSIONS)
        xpos = x + offsets[label]
        for i, dim in enumerate(DIMENSIONS):
            values = scvi.loc[scvi["dimension"].eq(dim), col].dropna().clip(0, 1).values
            jitter = rng.normal(0, 0.018, size=len(values))
            ax.scatter(
                np.full(len(values), xpos[i]) + jitter,
                values,
                s=5.8,
                color=colors[label],
                alpha=0.24,
                linewidths=0,
                rasterized=True,
            )
        ax.vlines(
            xpos,
            grouped["q25"].values,
            grouped["q75"].values,
            color=colors[label],
            linewidth=1.0,
            alpha=0.78,
        )
        ax.plot(
            xpos,
            grouped["median"].values,
            color=colors[label],
            marker="o",
            markersize=3.6,
            linewidth=1.25,
            label=label,
        )
        tmp = grouped.reset_index().rename(columns={"index": "dimension"})
        tmp["metric"] = label
        records.append(tmp)

    ax.set_title("scVI latent-dimensional response", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "a")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in DIMENSIONS])
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    ax.legend(loc="upper left", fontsize=5.4, handlelength=1.6, borderaxespad=0.2)
    return pd.concat(records, ignore_index=True).assign(panel="a")


def dimension_matrix(dimension: pd.DataFrame) -> pd.DataFrame:
    heat = dimension.groupby(["method", "dimension"])["ari"].median().unstack().reindex(columns=DIMENSIONS)
    ordered = [m for m in DIM_METHOD_ORDER if m in heat.index]
    remaining = [m for m in heat.mean(axis=1).sort_values(ascending=False).index.tolist() if m not in ordered]
    order = ordered + remaining
    return heat.loc[order]


def draw_dimension_delta(
    ax: plt.Axes, dim_heat: pd.DataFrame, family_map: dict[str, str]
) -> pd.DataFrame:
    endpoints = dim_heat[[2, 50]].copy()
    endpoints["delta_50d_minus_2d"] = endpoints[50] - endpoints[2]
    y = np.arange(len(endpoints))
    for pos, (method, row) in enumerate(endpoints.iterrows()):
        color = FAMILY_COLORS.get(family_map.get(method, "unknown"), FAMILY_COLORS["unknown"])
        ax.plot(
            [row[2], row[50]],
            [pos, pos],
            color=color,
            linewidth=1.55,
            alpha=0.78,
            solid_capstyle="round",
        )
        ax.scatter(
            row[2],
            pos,
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            row[50],
            pos,
            s=25,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        ax.text(
            1.045,
            pos,
            fmt_delta(row["delta_50d_minus_2d"]),
            ha="left",
            va="center",
            fontsize=5.25,
            color="#333333",
            clip_on=False,
        )
    ax.axvline(0, color="#CFCFCF", linewidth=0.5)
    ax.axvline(1, color="#E4E4E4", linewidth=0.45)
    ax.set_title("2D-to-50D endpoint shift", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "c")
    ax.set_xlabel("median ARI")
    ax.set_xlim(-0.035, 1.14)
    ax.set_ylim(-0.62, len(endpoints) - 0.38)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([display_method(m) for m in endpoints.index])
    ax.scatter(0.785, -0.45, s=20, facecolor="white", edgecolor="#555555", linewidth=0.85, clip_on=False, zorder=5)
    ax.text(0.818, -0.45, "2D", ha="left", va="center", fontsize=5.0, color="#333333", clip_on=False)
    ax.scatter(0.91, -0.45, s=22, facecolor="#555555", edgecolor="white", linewidth=0.45, clip_on=False, zorder=5)
    ax.text(0.943, -0.45, "50D", ha="left", va="center", fontsize=5.0, color="#333333", clip_on=False)
    ax.text(1.045, -0.45, r"$\Delta$ARI", ha="left", va="center", fontsize=5.25, color="#333333", clip_on=False)
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.45)
    ax.tick_params(axis="y", length=0)
    setup_axis(ax)
    return (
        endpoints.reset_index()
        .rename(columns={"index": "method", 2: "median_ari_2d", 50: "median_ari_50d"})
        .assign(panel="c")
    )


def workflow_delta_matrix(workflow: pd.DataFrame) -> pd.DataFrame:
    piv = workflow.pivot_table(
        index=["method", "dataset_id", "seed"],
        columns="workflow",
        values="ari",
        aggfunc="median",
    )
    if "PCA50 to 2D" not in piv.columns or "Direct 2D" not in piv.columns:
        raise ValueError("Workflow comparison requires Direct 2D and PCA50 to 2D records")
    piv["delta"] = piv["PCA50 to 2D"] - piv["Direct 2D"]
    delta = piv.reset_index().groupby(["method", "dataset_id"])["delta"].median().unstack()
    method_order = delta.mean(axis=1).sort_values(ascending=False).index.tolist()
    dataset_order = delta.median(axis=0).sort_values(ascending=False).index.tolist()
    delta = delta.loc[method_order, dataset_order]
    delta.columns = [DATASET_LABELS.get(c, c) for c in delta.columns]
    return delta


def draw_workflow_effect(
    ax: plt.Axes,
    wf_delta: pd.DataFrame,
    family_map: dict[str, str],
) -> pd.DataFrame:
    plot = wf_delta.copy()
    plot["median"] = wf_delta.median(axis=1)
    arr = plot.astype(float).values
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    norm = TwoSlopeNorm(vmin=min(vmin, -0.15), vcenter=0, vmax=max(vmax, 0.75))
    im = ax.imshow(arr, aspect="auto", cmap=DELTA_CMAP, norm=norm)
    ax.set_title("Visualization workflow effect", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "d")
    ax.set_xticks(np.arange(plot.shape[1]))
    ax.set_xticklabels([str(c) for c in plot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(plot.shape[0]))
    ax.set_yticklabels([display_method(x) for x in plot.index])
    ax.tick_params(length=0, pad=1.2)
    ax.axvline(plot.shape[1] - 1.5, color="white", linewidth=1.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_family_strip(ax, [str(x) for x in plot.index], family_map)
    median_col = plot.shape[1] - 1
    for i, value in enumerate(plot["median"].values):
        text_color = "white" if norm(value) > 0.62 else "#262626"
        ax.text(median_col, i, fmt_delta(value), ha="center", va="center", fontsize=5.0, color=text_color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.018)
    cbar.ax.tick_params(labelsize=4.8, length=1.8, width=0.45)
    cbar.set_label("PCA50 minus direct ARI", fontsize=5.15, labelpad=1.4)
    cbar.outline.set_linewidth(0.45)
    return plot.reset_index().melt(
        id_vars="method", var_name="dataset_or_summary", value_name="pca50_minus_direct_ari"
    ).assign(panel="d")


def hvg_matrix(hvg: pd.DataFrame) -> pd.DataFrame:
    heat = hvg.groupby(["method", "hvg_requested"])["ari"].median().unstack().reindex(columns=HVGS)
    ordered = [m for m in HVG_METHOD_ORDER if m in heat.index]
    remaining = [m for m in heat[3000].sort_values(ascending=False).index.tolist() if m not in ordered]
    order = ordered + remaining
    return heat.loc[order]


def draw_runtime(ax: plt.Axes, runtime: pd.DataFrame) -> pd.DataFrame:
    order = ["scvi", "dimension", "workflow", "hvg"]
    plot = runtime[runtime["source"].isin(order)].dropna(subset=["runtime_seconds"]).copy()
    plot["control_layer"] = pd.Categorical(plot["source"], categories=order, ordered=True)
    colors = ["#3F9C9A", "#3D6FB6", "#C79B38", "#BF6F6B"]
    rng = np.random.default_rng(20260604)
    for i, src in enumerate(order):
        vals = plot[plot["control_layer"].eq(src)]["runtime_seconds"].values
        draw_vals = vals
        if len(draw_vals) > 260:
            draw_vals = rng.choice(draw_vals, size=260, replace=False)
        ax.scatter(
            np.full(len(draw_vals), i) + rng.normal(0, 0.062, size=len(draw_vals)),
            draw_vals,
            s=4.4,
            color="#5F6D78",
            alpha=0.22,
            linewidths=0,
            rasterized=True,
        )
        q25, med, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.vlines(i, q25, q75, color=colors[i], linewidth=5.2, alpha=0.34, zorder=3)
        ax.hlines(med, i - 0.24, i + 0.24, color=colors[i], linewidth=1.55, zorder=4)
        ax.scatter(i, med, s=15, facecolor=colors[i], edgecolor="white", linewidth=0.45, zorder=5)
    ax.set_yscale("log")
    ax.set_title("Runtime cost of targeted controls", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "f")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([CONTROL_LABELS[o] for o in order], rotation=18, ha="right")
    ax.set_ylabel("runtime (s)")
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.45, which="major")
    ax.set_xlim(-0.5, len(order) - 0.5)
    setup_axis(ax)
    return plot.assign(panel="f")


def build_control_sensitivity(
    scvi_plot: pd.DataFrame,
    dimension: pd.DataFrame,
    workflow: pd.DataFrame,
    hvg: pd.DataFrame,
    family_map: dict[str, str],
) -> pd.DataFrame:
    records = []
    if not scvi_plot.empty:
        scvi_range = (
            scvi_plot.groupby(["method", "dataset_id", "seed"], observed=False)["ari"]
            .agg(lambda s: s.max() - s.min())
            .reset_index(name="sensitivity")
        )
        scvi_range["source"] = "scvi"
        records.append(scvi_range)

    if not dimension.empty:
        dim_range = (
            dimension.groupby(["method", "dataset_id", "seed"], observed=False)["ari"]
            .agg(lambda s: s.max() - s.min())
            .reset_index(name="sensitivity")
        )
        dim_range["source"] = "dimension"
        records.append(dim_range)

    if not workflow.empty:
        piv = workflow.pivot_table(
            index=["method", "dataset_id", "seed"],
            columns="workflow",
            values="ari",
            aggfunc="median",
        )
        if {"Direct 2D", "PCA50 to 2D"}.issubset(set(piv.columns)):
            wf_delta = (piv["PCA50 to 2D"] - piv["Direct 2D"]).abs().reset_index(name="sensitivity")
            wf_delta["source"] = "workflow"
            records.append(wf_delta)

    if not hvg.empty:
        hvg_range = (
            hvg.groupby(["method", "dataset_id", "seed"], observed=False)["ari"]
            .agg(lambda s: s.max() - s.min())
            .reset_index(name="sensitivity")
        )
        hvg_range["source"] = "hvg"
        records.append(hvg_range)

    if not records:
        return pd.DataFrame(columns=["method", "dataset_id", "seed", "sensitivity", "source", "family"])
    out = pd.concat(records, ignore_index=True, sort=False)
    out["family"] = out["method"].map(family_map).fillna("unknown")
    return out.dropna(subset=["sensitivity"])


def draw_control_sensitivity_distribution(ax: plt.Axes, sensitivity: pd.DataFrame) -> pd.DataFrame:
    order = ["scvi", "dimension", "workflow", "hvg"]
    plot = sensitivity[sensitivity["source"].isin(order)].copy()
    plot["source"] = pd.Categorical(plot["source"], categories=order, ordered=True)
    colors = ["#8A8A8A", "#3D6FB6", "#C79B38", "#BF6F6B"]
    groups = [plot[plot["source"].eq(src)]["sensitivity"].dropna().values for src in order]
    positions = np.arange(len(order))
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.54,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 0.75},
        whiskerprops={"color": "#555555", "linewidth": 0.55},
        capprops={"color": "#555555", "linewidth": 0.55},
        boxprops={"linewidth": 0.55, "edgecolor": "#333333"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
    rng = np.random.default_rng(20260605)
    for pos, src in zip(positions, order):
        vals = plot[plot["source"].eq(src)]["sensitivity"].dropna().values
        if len(vals) > 300:
            vals = rng.choice(vals, size=300, replace=False)
        ax.scatter(
            np.full(len(vals), pos) + rng.normal(0, 0.055, len(vals)),
            vals,
            s=4.0,
            color="#5F6D78",
            alpha=0.20,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title("Control-layer ARI sensitivity", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "g")
    ax.set_xticks(positions)
    ax.set_xticklabels([CONTROL_LABELS[src] for src in order], rotation=18, ha="right")
    ax.set_ylabel("ARI range or |delta|")
    ymax = max(0.10, float(plot["sensitivity"].quantile(0.98)) * 1.15)
    ax.set_ylim(-0.01, min(1.02, ymax))
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.45)
    setup_axis(ax)
    summary = (
        plot.groupby("source", observed=False)
        .agg(records=("sensitivity", "size"), methods=("method", "nunique"), median_sensitivity=("sensitivity", "median"))
        .reset_index()
    )
    summary["panel"] = "g"
    return summary


def draw_method_sensitivity_profile(ax: plt.Axes, sensitivity: pd.DataFrame) -> pd.DataFrame:
    order = ["scvi", "dimension", "workflow", "hvg"]
    layer_colors = {"scvi": "#8A8A8A", "dimension": "#3D6FB6", "workflow": "#C79B38", "hvg": "#BF6F6B"}
    med = (
        sensitivity[sensitivity["source"].isin(order)]
        .groupby(["method", "family", "source"], observed=False)["sensitivity"]
        .median()
        .reset_index(name="median_sensitivity")
    )
    method_order = (
        med.groupby("method", observed=False)["median_sensitivity"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    method_order = method_order[:10]
    med = med[med["method"].isin(method_order)].copy()
    y_lookup = {method: i for i, method in enumerate(method_order)}
    offsets = {"scvi": -0.18, "dimension": -0.06, "workflow": 0.06, "hvg": 0.18}
    for src in order:
        sub = med[med["source"].eq(src)]
        if sub.empty:
            continue
        ax.scatter(
            sub["median_sensitivity"],
            [y_lookup[m] + offsets[src] for m in sub["method"]],
            s=26,
            color=layer_colors[src],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.92,
            label=CONTROL_LABELS[src],
            zorder=3,
        )
    for method in method_order:
        fam = med.loc[med["method"].eq(method), "family"].dropna()
        family = fam.iloc[0] if len(fam) else "unknown"
        ax.add_patch(
            Rectangle(
                (0.006, y_lookup[method] - 0.31),
                0.010,
                0.62,
                transform=ax.get_yaxis_transform(),
                facecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["unknown"]),
                edgecolor="none",
                clip_on=True,
                zorder=1,
            )
        )
    ax.set_title("Method-level control response", loc="left", pad=2.5, fontweight="bold")
    add_panel_label(ax, "h")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels([display_method(m) for m in method_order])
    ax.invert_yaxis()
    ax.set_xlabel("median ARI sensitivity")
    ax.grid(axis="x", color="#E9E9E9", linewidth=0.45)
    ax.axvline(0, color="#D8D8D8", linewidth=0.55, zorder=0)
    ax.set_xlim(-0.035, max(0.12, float(med["median_sensitivity"].max()) * 1.18))
    setup_axis(ax)
    ax.tick_params(axis="y", length=0, pad=2.8)
    ax.legend(
        loc="lower right",
        fontsize=4.7,
        handletextpad=0.25,
        borderaxespad=0.15,
        labelspacing=0.25,
    )
    med["panel"] = "h"
    return med


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


def build() -> None:
    df = load_data()
    family_map = load_family_map()
    df["family"] = df["parent_method"].map(family_map).fillna("unknown")

    scvi_all = df[df["source"].eq("scvi")].copy()
    scvi_plot = scvi_all[np.isclose(scvi_all["max_epochs"], 20.0, equal_nan=False)].copy()
    dimension = df[df["source"].eq("dimension")].copy()
    workflow = df[df["source"].eq("workflow")].copy()
    hvg = df[df["source"].eq("hvg")].copy()
    runtime = pd.concat([scvi_plot, dimension, workflow, hvg], ignore_index=True, sort=False)
    sensitivity = build_control_sensitivity(scvi_plot, dimension, workflow, hvg, family_map)

    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_OUT / "Figure_6_targeted_sensitivity_controls_full_source_data.csv", index=False)
    scvi_plot.to_csv(SOURCE_OUT / "Figure_6_scVI_reference_plotted_records.csv", index=False)

    dim_heat = dimension_matrix(dimension)
    wf_delta = workflow_delta_matrix(workflow)
    gene_heat = hvg_matrix(hvg)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(7.2, 9.55),
        gridspec_kw={"height_ratios": [0.88, 1.00, 1.00, 0.82], "wspace": 0.42, "hspace": 0.50},
    )

    panel_a = draw_scvi_reference(axes[0, 0], scvi_plot)
    draw_matrix(
        axes[0, 1],
        dim_heat,
        "Latent-dimension sensitivity",
        "median ARI",
        family_map=family_map,
        panel_label="b",
    )
    panel_c = draw_dimension_delta(axes[1, 0], dim_heat, family_map)
    panel_d = draw_workflow_effect(axes[1, 1], wf_delta, family_map)
    draw_matrix(
        axes[2, 0],
        gene_heat,
        "Input-gene sensitivity",
        "median ARI",
        family_map=family_map,
        panel_label="e",
    )
    panel_f = draw_runtime(axes[2, 1], runtime)
    panel_g = draw_control_sensitivity_distribution(axes[3, 0], sensitivity)
    panel_h = draw_method_sensitivity_profile(axes[3, 1], sensitivity)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=4.8,
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
        columnspacing=1.15,
        handletextpad=0.35,
        fontsize=5.55,
        title="Method family",
        title_fontsize=5.75,
    )
    fig.subplots_adjust(left=0.08, right=0.972, top=0.975, bottom=0.078)

    basename = "Figure_6_targeted_sensitivity_controls_polished"
    save_outputs(fig, basename)
    plt.close(fig)

    panel_b = dim_heat.reset_index().melt(
        id_vars="method", var_name="dimension", value_name="median_ari"
    ).assign(panel="b")
    panel_e = gene_heat.reset_index().melt(
        id_vars="method", var_name="hvg_requested", value_name="median_ari"
    ).assign(panel="e")
    panel_data = pd.concat(
        [
            panel_a,
            panel_b,
            panel_c,
            panel_d,
            panel_e,
            panel_f[
                [
                    "panel",
                    "source",
                    "work_package",
                    "dataset_id",
                    "method",
                    "dimension",
                    "seed",
                    "runtime_seconds",
                ]
            ],
            panel_g,
            panel_h,
        ],
        ignore_index=True,
        sort=False,
    )
    panel_data.to_csv(SOURCE_OUT / f"{basename}_panel_data.csv", index=False)

    qa = {
        "panel_count": 8,
        "input_rows": len(df),
        "scvi_reference_rows_all": len(scvi_all),
        "scvi_reference_rows_plotted_max_epochs_20": len(scvi_plot),
        "scvi_reference_rows_excluded_non_20_epochs": len(scvi_all) - len(scvi_plot),
        "scvi_reference_configurations_plotted": scvi_plot["dataset_id"].nunique(),
        "scvi_reference_dimensions": ",".join(map(str, sorted(scvi_plot["dimension"].dropna().astype(int).unique()))),
        "dimension_rows": len(dimension),
        "dimension_methods": dimension["method"].nunique(),
        "dimension_datasets": dimension["dataset_id"].nunique(),
        "dimension_seeds_min_per_method_dimension": int(
            dimension.groupby(["method", "dimension"]).size().min()
        ),
        "workflow_rows": len(workflow),
        "workflow_methods": workflow["method"].nunique(),
        "workflow_datasets": workflow["dataset_id"].nunique(),
        "hvg_rows": len(hvg),
        "hvg_methods": hvg["method"].nunique(),
        "hvg_datasets": hvg["dataset_id"].nunique(),
        "hvg_requested_values": ",".join(map(str, HVGS)),
        "runtime_rows_plotted": len(runtime),
        "control_sensitivity_rows": len(sensitivity),
        "control_sensitivity_layers": sensitivity["source"].nunique(),
        "control_sensitivity_methods": sensitivity["method"].nunique(),
        "top_dimension_delta_method": str(panel_c.sort_values("delta_50d_minus_2d", ascending=False)["method"].iloc[0]),
        "top_dimension_delta": float(panel_c["delta_50d_minus_2d"].max()),
        "workflow_delta_median": float(np.nanmedian(wf_delta.values)),
        "input_gene_3000_top_method": str(gene_heat[3000].idxmax()),
        "input_gene_3000_top_median_ari": float(gene_heat[3000].max()),
        "max_median_control_response_method": str(
            panel_h.sort_values("median_sensitivity", ascending=False)["method"].iloc[0]
        ),
        "max_median_control_response": float(panel_h["median_sensitivity"].max()),
    }
    pd.DataFrame([qa]).to_csv(SOURCE_OUT / f"{basename}_qa_summary.csv", index=False)

    checklist = f"""# Figure 6 Legend And Checklist

Generated by: `Publication/paper/revision_figures/figure6_polish/make_figure6_targeted_sensitivity_controls.py`

## Figure Role

Figure 6 is a new revision figure, not a replacement for the original scalability
figure. It summarizes targeted sensitivity controls for scVI, latent dimensionality,
visualization workflow, and requested input genes. These analyses do not change the
26-method full-benchmark count.

## Panel Logic

- a, scVI reference analysis across latent dimensions, using comparable `max_epochs = 20`
  records. Individual runs, interquartile intervals, and median trajectories are shown.
- b, Median ARI across selected latent dimensionalities for the targeted method subset.
- c, Median ARI endpoint shift from 2D to 50D for the same latent-dimension subset,
  using the same method order as panel b.
- d, Median ARI difference between PCA50-to-2D and direct-2D visualization workflows,
  with an additional row-median summary column.
- e, Median ARI across requested HVG cutoffs for the input-gene sensitivity subset.
- f, Runtime distributions for the plotted targeted control layers, shown as jittered
  runs with median and interquartile summaries.
- g, Control-layer ARI sensitivity distributions. Latent-dimension and input-gene
  controls use within-task ARI ranges; workflow uses absolute PCA50-minus-direct ARI.
- h, Method-level median control response across observed targeted-control layers,
  avoiding blank missing-data heatmap cells.

## Data QA

- Input rows: {qa['input_rows']}.
- scVI reference rows plotted: {qa['scvi_reference_rows_plotted_max_epochs_20']};
  excluded non-20-epoch pilot rows: {qa['scvi_reference_rows_excluded_non_20_epochs']}.
- scVI reference configurations plotted: {qa['scvi_reference_configurations_plotted']}.
- Latent-dimension control: {qa['dimension_methods']} methods, {qa['dimension_datasets']}
  datasets, dimensions {qa['scvi_reference_dimensions']}.
- Workflow control: {qa['workflow_methods']} methods, {qa['workflow_datasets']} datasets.
- Input-gene control: {qa['hvg_methods']} methods, {qa['hvg_datasets']} datasets,
  requested HVGs {qa['hvg_requested_values']}.
- Runtime rows plotted: {qa['runtime_rows_plotted']}.
- Control-sensitivity rows: {qa['control_sensitivity_rows']} across
  {qa['control_sensitivity_layers']} control layers and
  {qa['control_sensitivity_methods']} methods.

## Manuscript Wording Guardrails

- Use "targeted sensitivity analyses" or "targeted controls", not "full rerun".
- State that scVI is a targeted reference analysis and is not counted as a 27th
  full-benchmark method.
- State that latent-dimension, workflow, and input-gene controls were run on selected
  methods and datasets chosen to answer reviewer concerns.
- Do not imply that these targeted controls replace the 26-method, 100-dataset
  benchmark landscape in Figures 1-3.

## Output Files

- `polished/{basename}.svg`
- `polished/{basename}.pdf`
- `polished/{basename}.png`
- `polished/{basename}.tiff`
- `source_data/Figure_6_targeted_sensitivity_controls_full_source_data.csv`
- `source_data/Figure_6_scVI_reference_plotted_records.csv`
- `source_data/{basename}_panel_data.csv`
- `source_data/{basename}_qa_summary.csv`
"""
    (QA_OUT / f"{basename}_legend_and_checklist.md").write_text(checklist, encoding="utf-8")


if __name__ == "__main__":
    build()
