#!/usr/bin/env python3
"""Promote completed WP1 scVI synthetic runs into the WP2 dimension grid."""
from pathlib import Path

import pandas as pd


WP2_DATASETS = [
    "default",
    "celltype_7",
    "celltype_11",
    "celltype_15",
    "dropout_0",
    "batch_1.0",
    "gene_5k",
    "gene_5w",
]
DIMENSIONS = [2, 5, 10, 20, 50]
SEEDS = [0, 1, 2]


def main() -> int:
    source = Path("revision_benchmark/results/metrics/WP1_scVI_local_scVI_metrics.csv")
    out = Path("revision_benchmark/results/metrics/WP2_dimension_sensitivity_scVI_metrics.csv")
    summary_out = Path("revision_benchmark/experiments/wp2_scvi_dimension_summary.csv")
    results_out = Path("revision_benchmark/experiments/wp2_scvi_dimension_results.csv")

    df = pd.read_csv(source)
    df = df[
        df["dataset_id"].isin(WP2_DATASETS)
        & df["dimension"].isin(DIMENSIONS)
        & df["seed"].isin(SEEDS)
        & df["max_epochs"].eq(20)
    ].copy()
    df = df.drop_duplicates(["dataset_id", "method", "dimension", "seed", "max_epochs"], keep="last")
    df["work_package"] = "WP2_dimension_sensitivity"
    df = df.sort_values(["dataset_id", "dimension", "seed"])

    expected = len(WP2_DATASETS) * len(DIMENSIONS) * len(SEEDS)
    if len(df) != expected:
        missing = []
        have = {(r.dataset_id, int(r.dimension), int(r.seed)) for r in df.itertuples()}
        for dataset_id in WP2_DATASETS:
            for dim in DIMENSIONS:
                for seed in SEEDS:
                    key = (dataset_id, dim, seed)
                    if key not in have:
                        missing.append(key)
        raise SystemExit(f"Expected {expected} scVI WP2 rows, found {len(df)}; missing={missing[:20]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df.to_csv(results_out, index=False)

    summary = df.groupby(["dataset_id", "dimension"], as_index=False).agg(
        runs=("seed", "count"),
        ari_mean=("ari", "mean"),
        ari_sd=("ari", "std"),
        nmi_mean=("nmi", "mean"),
        nmi_sd=("nmi", "std"),
        trustworthiness_k30_mean=("trustworthiness_k30", "mean"),
        trustworthiness_k30_sd=("trustworthiness_k30", "std"),
        silhouette_label_mean=("silhouette_label", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        max_rss_mb_max=("max_rss_mb", "max"),
    )
    summary.to_csv(summary_out, index=False)
    print({
        "rows": len(df),
        "datasets": df["dataset_id"].nunique(),
        "dimensions": sorted(df["dimension"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "metrics": str(out),
        "summary": str(summary_out),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
