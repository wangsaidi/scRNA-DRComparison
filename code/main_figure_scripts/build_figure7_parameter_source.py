from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Publication/paper/revision_figures/figure7_polish"
SOURCE_OUT = OUT / "source_data"
QA_OUT = OUT / "qa"
SOURCE_OUT.mkdir(parents=True, exist_ok=True)
QA_OUT.mkdir(parents=True, exist_ok=True)

TOPO_ROOT = ROOT / "metric/topo/simulate"
CLUSTER_ROOT = ROOT / "metric/cluster/simulate"
STABILITY_AXIS_ROOT = ROOT / "metric/score/stability"
ATLAS = ROOT / "Publication/paper/revision_figures/canonical_source_tables/canonical_simulated_dataset_atlas_from_excel.csv"
METHODS = ROOT / "Publication/paper/revision_figures/canonical_source_tables/canonical_method_manifest.csv"
VASC_COMPLETED_RAW = ROOT / "Publication/paper/revision_figures/figure3_polish/source_data/Figure_3_completed_stability_raw_metrics.csv"

EXCLUDED_METRICS = {
    "AUC",
    "Qlocal",
    "Qglobal",
    "kmax",
    "knn_10",
    "knn_20",
    "nkr_10",
    "nkr_20",
    "aji_10",
    "aji_20",
    "T_10",
    "T_20",
    "C_10",
    "C_20",
    "nh_10",
    "nh_20",
    "Mrre_false_10",
    "Mrre_false_20",
    "Mrre_missing_10",
    "Mrre_missing_20",
}

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

AXIS_LABELS = {
    "cell_number": "cell number",
    "gene_number": "gene number",
    "celltype_number": "cell-type number",
    "batch_number": "batch number",
    "batch_strength": "batch strength",
    "dropout": "dropout",
    "de_prob": "DE probability",
    "de_strength": "DE strength",
    "out": "outlier proportion",
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

CANONICAL_METHOD = {
    "TSNE": "t-SNE",
    "SQuaD_MDS": "SQuaD-MDS",
    "SQuaD_MDS_hybrid": "SQuaD-MDS hybrid",
    "ParametricUMAP50": "Parametric UMAP 50",
    "ParametricUMAP200": "Parametric UMAP 200",
    "ivis": "IVIS",
}

PARENT_METHOD = {
    "Parametric UMAP 50": "UMAP",
    "Parametric UMAP 200": "UMAP",
    "SQuaD-MDS hybrid": "SQuaD-MDS",
}


def canonical_method(method: object) -> str:
    text = str(method)
    return CANONICAL_METHOD.get(text, text)


def parent_method(method_id: str) -> str:
    return PARENT_METHOD.get(method_id, method_id)


def load_method_metadata() -> pd.DataFrame:
    meta = pd.read_csv(METHODS)
    meta = meta[["method_id", "parent_method", "method_family", "method_order"]].drop_duplicates("method_id")
    extra = pd.DataFrame(
        [
            {"method_id": "Parametric UMAP 50", "parent_method": "UMAP", "method_family": "graph/diffusion", "method_order": 17.1},
            {"method_id": "Parametric UMAP 200", "parent_method": "UMAP", "method_family": "graph/diffusion", "method_order": 17.2},
            {"method_id": "SQuaD-MDS hybrid", "parent_method": "SQuaD-MDS", "method_family": "metric/structure-aware", "method_order": 25.1},
        ]
    )
    return pd.concat([meta, extra], ignore_index=True).drop_duplicates("method_id", keep="last")


def dataset_to_axis(dataset_id: str) -> str:
    for axis, datasets in DISPLAY_DATASETS.items():
        if dataset_id in datasets:
            return axis
    if dataset_id == "default":
        return "default"
    raise ValueError(f"Unmapped simulated dataset: {dataset_id}")


def required_paths(dataset_id: str) -> dict[str, Path]:
    return {
        "dr1": TOPO_ROOT / dataset_id / "dr1.csv",
        "dr2": TOPO_ROOT / dataset_id / "dr2.csv",
        "dr3": TOPO_ROOT / dataset_id / "dr3.csv",
        "kmeans_ARI": CLUSTER_ROOT / dataset_id / "indicators/kmeans_ARI.csv",
        "kmeans_NMI": CLUSTER_ROOT / dataset_id / "indicators/kmeans_NMI.csv",
        "kmeans_SIL": CLUSTER_ROOT / dataset_id / "indicators/kmeans_SIL.csv",
        "kmeans_COMP": CLUSTER_ROOT / dataset_id / "indicators/kmeans_COMP.csv",
        "kmeans_HOMO": CLUSTER_ROOT / dataset_id / "indicators/kmeans_HOMO.csv",
    }


def read_metric_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Method" not in df.columns:
        raise ValueError(f"Missing Method column in {path}")
    return df


def score_dataset(dataset_id: str, method_universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = required_paths(dataset_id)
    availability_rows = [
        {"dataset_id": dataset_id, "file_role": role, "path": str(path.relative_to(ROOT)), "exists": path.exists()}
        for role, path in paths.items()
    ]
    dfs = [read_metric_file(path) for path in paths.values()]
    complete = all(df is not None for df in dfs)
    if not complete:
        scores = pd.DataFrame({"method_raw": method_universe})
        scores["dataset_id"] = dataset_id
        scores["score"] = np.nan
        scores["local_score"] = np.nan
        scores["global_score"] = np.nan
        scores["cluster_score"] = np.nan
        scores["n_score_metrics"] = 0
        scores["complete_score_inputs"] = False
        components = pd.DataFrame()
        return scores, pd.DataFrame(availability_rows), components

    merged = reduce(lambda left, right: pd.merge(left, right, on="Method", how="outer"), dfs)  # type: ignore[arg-type]
    metric_cols = [c for c in merged.columns if c != "Method" and c not in EXCLUDED_METRICS]
    local_cols = [c for c in ["knn_30", "svm", "nkr_30", "aji_30", "T_30", "C_30", "nh_30", "Mrre_false_30", "Mrre_missing_30"] if c in merged.columns]
    global_cols = [c for c in ["random_triplet", "spearman", "k-nearest", "centroid_distance", "Pearson"] if c in merged.columns]
    cluster_cols = [c for c in ["ARI", "NMI", "SIL", "COMP", "HOMO"] if c in merged.columns]

    scores = pd.DataFrame()
    scores["method_raw"] = merged["Method"]
    scores["dataset_id"] = dataset_id
    scores["score"] = merged[metric_cols].mean(axis=1, skipna=True)
    scores["local_score"] = merged[local_cols].mean(axis=1, skipna=True)
    scores["global_score"] = merged[global_cols].mean(axis=1, skipna=True)
    scores["cluster_score"] = merged[cluster_cols].mean(axis=1, skipna=True)
    scores["n_score_metrics"] = merged[metric_cols].notna().sum(axis=1)
    scores["complete_score_inputs"] = True
    scores["source_layer"] = "standard_metric_files"

    component_rows = []
    for _, row in merged.iterrows():
        for col in metric_cols:
            component_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method_raw": row["Method"],
                    "metric": col,
                    "metric_value": row[col],
                    "component_domain": "cluster" if col in cluster_cols else "global" if col in global_cols else "local",
                }
            )
    return scores, pd.DataFrame(availability_rows), pd.DataFrame(component_rows)


def score_completed_vasc(datasets: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not VASC_COMPLETED_RAW.exists():
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.read_csv(VASC_COMPLETED_RAW)
    raw = raw[raw["Method"].eq("VASC")].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_cols = [
        c
        for c in raw.columns
        if c not in {"Method", "Dataset", "perturbation_axis"} and c not in EXCLUDED_METRICS
    ]
    local_cols = [c for c in ["knn_30", "svm", "nkr_30", "aji_30", "T_30", "C_30", "nh_30", "Mrre_false_30", "Mrre_missing_30"] if c in raw.columns]
    global_cols = [c for c in ["random_triplet", "spearman", "k-nearest", "centroid_distance", "Pearson"] if c in raw.columns]
    cluster_cols = [c for c in ["ARI", "NMI", "SIL", "COMP", "HOMO"] if c in raw.columns]

    rows = []
    component_rows = []
    for item in datasets:
        dataset_id = str(item["dataset_id"])
        axis = str(item["perturbation_axis"])
        axis_order = int(item["axis_order"])
        match = raw[raw["Dataset"].eq(dataset_id)]
        base = {
            "method_raw": "VASC",
            "dataset_id": dataset_id,
            "method_id": "VASC",
            "parent_method": "VASC",
            "perturbation_axis": axis,
            "perturbation_axis_label": AXIS_LABELS[axis],
            "axis_order": axis_order,
            "parameter_label": VALUE_LABELS.get(dataset_id, dataset_id),
        }
        if match.empty:
            rows.append(
                {
                    **base,
                    "score": np.nan,
                    "local_score": np.nan,
                    "global_score": np.nan,
                    "cluster_score": np.nan,
                    "n_score_metrics": 0,
                    "complete_score_inputs": False,
                    "source_layer": "figure3_completed_vasc_missing_for_dataset",
                }
            )
            continue

        row = match.iloc[0]
        rows.append(
            {
                **base,
                "score": row[metric_cols].mean(skipna=True),
                "local_score": row[local_cols].mean(skipna=True),
                "global_score": row[global_cols].mean(skipna=True),
                "cluster_score": row[cluster_cols].mean(skipna=True),
                "n_score_metrics": int(row[metric_cols].notna().sum()),
                "complete_score_inputs": True,
                "source_layer": "figure3_completed_vasc_raw_metrics",
            }
        )
        for col in metric_cols:
            component_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method_raw": "VASC",
                    "metric": col,
                    "metric_value": row[col],
                    "component_domain": "cluster" if col in cluster_cols else "global" if col in global_cols else "local",
                    "perturbation_axis": axis,
                    "axis_order": axis_order,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(component_rows)


def build() -> None:
    atlas = pd.read_csv(ATLAS)
    metadata = load_method_metadata()

    method_order_lookup = metadata.set_index("method_id")["method_order"].to_dict()
    method_universe = sorted(
        pd.read_csv(TOPO_ROOT / "cell_100/dr1.csv")["Method"].unique().tolist(),
        key=lambda x: (method_order_lookup.get(canonical_method(x), 999), canonical_method(x)),
    )

    records = []
    availability = []
    components = []
    datasets = []
    for axis, axis_datasets in DISPLAY_DATASETS.items():
        for order, dataset_id in enumerate(axis_datasets, start=1):
            datasets.append({"dataset_id": dataset_id, "perturbation_axis": axis, "axis_order": order})

    for item in datasets:
        scores, avail, comp = score_dataset(item["dataset_id"], method_universe)
        scores["method_id"] = scores["method_raw"].map(canonical_method)
        scores["parent_method"] = scores["method_id"].map(parent_method)
        scores["perturbation_axis"] = item["perturbation_axis"]
        scores["perturbation_axis_label"] = AXIS_LABELS[item["perturbation_axis"]]
        scores["axis_order"] = item["axis_order"]
        scores["parameter_label"] = scores["dataset_id"].map(VALUE_LABELS)
        records.append(scores)
        avail["perturbation_axis"] = item["perturbation_axis"]
        avail["axis_order"] = item["axis_order"]
        availability.append(avail)
        if not comp.empty:
            comp["perturbation_axis"] = item["perturbation_axis"]
            comp["axis_order"] = item["axis_order"]
            components.append(comp)

    long = pd.concat(records, ignore_index=True)
    vasc_scores, vasc_components = score_completed_vasc(datasets)
    if not vasc_scores.empty:
        long = pd.concat([long, vasc_scores], ignore_index=True)
    long = long.merge(metadata, on=["method_id", "parent_method"], how="left")

    atlas_subset = atlas.rename(columns={"dataset_label": "atlas_dataset_label"}).copy()
    atlas_subset["dataset_id"] = atlas_subset["atlas_dataset_label"].str.replace("_default", "", regex=False)
    atlas_subset.loc[atlas_subset["atlas_dataset_label"].eq("default_default"), "dataset_id"] = "default"
    long = long.merge(
        atlas_subset[
            [
                "dataset_id",
                "param_group",
                "param_value",
                "cells",
                "genes",
                "sparsity_pct",
                "n_batches",
                "n_groups",
                "size_mb",
            ]
        ],
        on="dataset_id",
        how="left",
    )
    long["score_available"] = long["score"].notna()

    component_long = pd.concat(components, ignore_index=True) if components else pd.DataFrame()
    if not vasc_components.empty:
        component_long = pd.concat([component_long, vasc_components], ignore_index=True)
    if not component_long.empty:
        component_long["method_id"] = component_long["method_raw"].map(canonical_method)
        component_long["parent_method"] = component_long["method_id"].map(parent_method)
        component_long["parameter_label"] = component_long["dataset_id"].map(VALUE_LABELS)
        component_long = component_long.merge(metadata, on=["method_id", "parent_method"], how="left")

    availability_long = pd.concat(availability, ignore_index=True)
    availability_summary = (
        availability_long.groupby(["dataset_id", "perturbation_axis", "axis_order"], as_index=False)
        .agg(required_files=("file_role", "nunique"), files_present=("exists", "sum"), complete_score_inputs=("exists", "all"))
    )
    availability_summary["parameter_label"] = availability_summary["dataset_id"].map(VALUE_LABELS)

    axis = (
        long[long["score_available"]]
        .groupby(["method_raw", "method_id", "parent_method", "perturbation_axis"], as_index=False)
        .agg(score=("score", "mean"), parameter_levels=("dataset_id", "nunique"))
    )
    axis = axis.merge(metadata, on=["method_id", "parent_method"], how="left")

    comparisons = []
    for path in sorted(STABILITY_AXIS_ROOT.glob("*.csv")):
        perturbation_axis = path.stem
        published = pd.read_csv(path).rename(columns={"Method": "method_raw", "score": "published_axis_score"})
        published["method_id"] = published["method_raw"].map(canonical_method)
        calc = axis[axis["perturbation_axis"].eq(perturbation_axis)][["method_id", "score"]].rename(columns={"score": "recomputed_axis_score"})
        comp = published.merge(calc, on="method_id", how="outer")
        comp["perturbation_axis"] = perturbation_axis
        comp["abs_diff"] = (comp["published_axis_score"] - comp["recomputed_axis_score"]).abs()
        comparisons.append(comp)
    comparison = pd.concat(comparisons, ignore_index=True)

    long.to_csv(SOURCE_OUT / "Figure_7_parameter_level_stability_source_data.csv", index=False)
    component_long.to_csv(SOURCE_OUT / "Figure_7_parameter_level_metric_components.csv", index=False)
    axis.to_csv(SOURCE_OUT / "Figure_7_recomputed_axis_stability_source_data.csv", index=False)
    availability_summary.to_csv(SOURCE_OUT / "Figure_7_parameter_level_availability_summary.csv", index=False)
    availability_long.to_csv(SOURCE_OUT / "Figure_7_required_file_availability_long.csv", index=False)
    comparison.to_csv(SOURCE_OUT / "Figure_7_recomputed_vs_published_axis_score_check.csv", index=False)

    qa = {
        "parameter_level_rows": len(long),
        "metric_component_rows": len(component_long),
        "method_raw_count": long["method_raw"].nunique(),
        "method_id_count": long["method_id"].nunique(),
        "parent_method_count": long["parent_method"].nunique(),
        "perturbation_axis_count": long["perturbation_axis"].nunique(),
        "parameter_dataset_count": long["dataset_id"].nunique(),
        "complete_parameter_dataset_count": int(availability_summary["complete_score_inputs"].sum()),
        "incomplete_parameter_dataset_count": int((~availability_summary["complete_score_inputs"]).sum()),
        "available_score_rows": int(long["score_available"].sum()),
        "missing_score_rows": int((~long["score_available"]).sum()),
        "max_abs_axis_score_diff_vs_published": float(comparison["abs_diff"].max(skipna=True)),
        "mean_abs_axis_score_diff_vs_published": float(comparison["abs_diff"].mean(skipna=True)),
    }
    pd.DataFrame([qa]).to_csv(SOURCE_OUT / "Figure_7_parameter_source_qa_summary.csv", index=False)

    missing_datasets = availability_summary[~availability_summary["complete_score_inputs"]][["dataset_id", "perturbation_axis", "files_present", "required_files"]]
    checklist = f"""# Figure 7 Parameter-Level Source Data QA

Generated by: `Publication/paper/revision_figures/figure7_polish/build_figure7_parameter_source.py`

## Purpose

This source package reconstructs parameter-level simulated robustness scores from
topology and k-means clustering metric files, following the original Figure 7
notebook logic. It restores the parameter-gradient layer that was lost in the
axis-aggregated redesigned Figure 7. VASC parameter-level rows are appended from
the Figure 3 completed VASC metric cache so the plotted main panel retains the
full 26-method benchmark universe.

## QA Summary

- Parameter-level rows: {qa['parameter_level_rows']}.
- Metric component rows: {qa['metric_component_rows']}.
- Raw method variants: {qa['method_raw_count']}; canonical method IDs: {qa['method_id_count']};
  parent methods: {qa['parent_method_count']}.
- Perturbation axes: {qa['perturbation_axis_count']}; parameter-level datasets: {qa['parameter_dataset_count']}.
- Complete parameter datasets: {qa['complete_parameter_dataset_count']};
  incomplete parameter datasets: {qa['incomplete_parameter_dataset_count']}.
- Score rows available: {qa['available_score_rows']}; missing score rows: {qa['missing_score_rows']}.
- Maximum absolute difference from published axis-level stability scores:
  {qa['max_abs_axis_score_diff_vs_published']:.6g}.

The published axis-level stability files are retained as a diagnostic comparison only:
they are not expected to equal a simple mean of the parameter-level values plotted
in Figure 7.

## Incomplete Parameter Datasets

{missing_datasets.to_string(index=False) if not missing_datasets.empty else 'None'}

## Output Files

- `source_data/Figure_7_parameter_level_stability_source_data.csv`
- `source_data/Figure_7_parameter_level_metric_components.csv`
- `source_data/Figure_7_recomputed_axis_stability_source_data.csv`
- `source_data/Figure_7_parameter_level_availability_summary.csv`
- `source_data/Figure_7_required_file_availability_long.csv`
- `source_data/Figure_7_recomputed_vs_published_axis_score_check.csv`
- `source_data/Figure_7_parameter_source_qa_summary.csv`
"""
    (QA_OUT / "Figure_7_parameter_source_qa_checklist.md").write_text(checklist, encoding="utf-8")


if __name__ == "__main__":
    build()
