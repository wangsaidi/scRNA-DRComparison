#!/usr/bin/env python3
"""Run visualization workflow controls for direct 2D vs PCA50-to-2D embeddings."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "default": ("datasets/simulate/default/counts_matrix.csv", "datasets/simulate/default/cell_metadata.csv"),
    "celltype_7": ("datasets/simulate/celltype_7/counts_matrix.csv", "datasets/simulate/celltype_7/cell_metadata.csv"),
    "celltype_11": ("datasets/simulate/celltype_11/counts_matrix.csv", "datasets/simulate/celltype_11/cell_metadata.csv"),
    "celltype_15": ("datasets/simulate/celltype_15/counts_matrix.csv", "datasets/simulate/celltype_15/cell_metadata.csv"),
    "dropout_0": ("datasets/simulate/dropout_0/counts_matrix.csv", "datasets/simulate/dropout_0/cell_metadata.csv"),
    "batch_1.0": ("datasets/simulate/batch_1.0/counts_matrix.csv", "datasets/simulate/batch_1.0/cell_metadata.csv"),
    "gene_5k": ("datasets/simulate/gene_5k/counts_matrix.csv", "datasets/simulate/gene_5k/cell_metadata.csv"),
    "gene_5w": ("datasets/simulate/gene_5w/counts_matrix.csv", "datasets/simulate/gene_5w/cell_metadata.csv"),
}


def parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df.astype(np.float32)


def read_labels(path: Path, cells) -> pd.Series:
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.astype(str)
    if "Group" in meta.columns:
        labels = meta["Group"].astype(str)
    elif "cell_type" in meta.columns:
        labels = meta["cell_type"].astype(str)
    else:
        candidates = [c for c in meta.columns if c.lower() in {"label", "labels", "group", "celltype", "cell_type"}]
        if not candidates:
            raise ValueError(f"No label column found in {path}")
        labels = meta[candidates[0]].astype(str)
    labels = labels.reindex(cells)
    if labels.isna().any():
        raise ValueError(f"Label file {path} is missing {int(labels.isna().sum())} cells")
    return labels


def preprocess(x: pd.DataFrame) -> np.ndarray:
    values = np.log1p(x.to_numpy(dtype=np.float32))
    return StandardScaler(with_mean=True, with_std=True).fit_transform(values).astype(np.float32)


def pca50(values: np.ndarray, seed: int) -> np.ndarray:
    n_components = min(50, values.shape[0] - 1, values.shape[1])
    return PCA(n_components=n_components, random_state=seed, svd_solver="randomized").fit_transform(values).astype(np.float32)


def embed(method: str, values: np.ndarray, seed: int) -> np.ndarray:
    method_key = method.lower()
    if method_key == "umap":
        import umap

        model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed)
        return model.fit_transform(values)
    if method_key == "pacmap":
        import pacmap

        model = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=seed)
        return model.fit_transform(values, init="pca")
    if method_key == "trimap":
        import trimap

        model = trimap.TRIMAP(n_dims=2, n_inliers=10, n_outliers=5, n_random=5)
        return model.fit_transform(values)
    if method_key == "phate":
        import phate

        model = phate.PHATE(n_components=2, knn=15, random_state=seed, n_jobs=1, verbose=0)
        return model.fit_transform(values)
    if method_key in {"tsne", "t-sne"}:
        model = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed)
        return model.fit_transform(values)
    raise ValueError(f"Unsupported visualization method: {method}")


def compute_metrics(x_input: np.ndarray, z: np.ndarray, labels: pd.Series, seed: int, metric_sample_size: int) -> dict:
    out = {}
    n = z.shape[0]
    if metric_sample_size and n > metric_sample_size:
        rng = np.random.default_rng(seed)
        metric_idx = np.sort(rng.choice(n, size=metric_sample_size, replace=False))
        out["metric_sample_size"] = int(metric_sample_size)
    else:
        metric_idx = np.arange(n)
        out["metric_sample_size"] = int(n)

    y = labels.to_numpy()
    unique = np.unique(y)
    n_clusters = len(unique)
    out["n_labels"] = int(n_clusters)
    if n_clusters > 1 and n_clusters < n:
        pred = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(z)
        out["ari"] = float(adjusted_rand_score(y, pred))
        out["nmi"] = float(normalized_mutual_info_score(y, pred))
        out["homogeneity"] = float(homogeneity_score(y, pred))
        out["completeness"] = float(completeness_score(y, pred))
        try:
            out["silhouette_label"] = float(silhouette_score(z[metric_idx], y[metric_idx]))
        except Exception as exc:
            out["silhouette_label_error"] = str(exc)

    if n > 10:
        k = min(30, n - 1, len(metric_idx) - 1)
        try:
            out["trustworthiness_k30"] = float(trustworthiness(x_input[metric_idx], z[metric_idx], n_neighbors=k))
        except Exception as exc:
            out["trustworthiness_error"] = str(exc)
    return out


def run_one(dataset_id: str, method: str, workflow: str, seed: int, output_root: Path, metric_sample_size: int) -> dict:
    matrix_path = Path(DATASETS[dataset_id][0])
    label_path = Path(DATASETS[dataset_id][1])
    start = time.time()

    x = read_matrix(matrix_path)
    labels = read_labels(label_path, x.index)
    x_values = preprocess(x)
    if workflow == "direct_2d":
        emb_input = x_values
        pca_components = 0
    elif workflow == "pca50_to_2d":
        emb_input = pca50(x_values, seed)
        pca_components = int(emb_input.shape[1])
    else:
        raise ValueError(f"Unsupported workflow: {workflow}")

    z = np.asarray(embed(method, emb_input, seed), dtype=np.float64)
    if z.shape != (x.shape[0], 2):
        raise ValueError(f"Unexpected embedding shape for {method}/{workflow}: {z.shape}")

    emb_dir = output_root / "embeddings" / "WP3_visualization_workflow" / dataset_id / method / workflow
    metric_dir = output_root / "metrics"
    log_dir = output_root / "logs" / "WP3_visualization_workflow"
    emb_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"seed_{seed}.csv"
    pd.DataFrame(z, index=x.index, columns=[f"{method}_1", f"{method}_2"]).to_csv(emb_path)

    metrics = compute_metrics(x_values, z, labels, seed, metric_sample_size)
    metrics.update({
        "work_package": "WP3_visualization_workflow",
        "dataset_id": dataset_id,
        "method": method,
        "environment": "scrna-dr-py-modern-min",
        "workflow": workflow,
        "dimension": 2,
        "seed": int(seed),
        "pca_components": pca_components,
        "metric_sample_size_requested": int(metric_sample_size),
        "n_cells": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "runtime_seconds": float(time.time() - start),
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "matrix_path": str(matrix_path),
        "label_path": str(label_path),
        "embedding_path": str(emb_path),
    })

    metric_path = metric_dir / "WP3_visualization_workflow_metrics.csv"
    row = pd.DataFrame([metrics])
    if metric_path.exists():
        old = pd.read_csv(metric_path)
        row = pd.concat([old, row], ignore_index=True)
        row = row.drop_duplicates(subset=["dataset_id", "method", "workflow", "seed"], keep="last")
    row.to_csv(metric_path, index=False)

    log_path = log_dir / f"{dataset_id}_{method}_{workflow}_seed{seed}.json"
    log_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS.keys()))
    parser.add_argument("--methods", default="UMAP,PaCMAP,TriMap,PHATE")
    parser.add_argument("--workflows", default="direct_2d,pca50_to_2d")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-root", default="revision_benchmark/results")
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    workflows = parse_csv(args.workflows)
    seeds = parse_csv(args.seeds, int)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset IDs: {', '.join(unknown)}")

    failed_path = Path(args.output_root) / "logs" / "WP3_visualization_workflow_failed_runs.jsonl"
    for dataset_id in datasets:
        for method in methods:
            for workflow in workflows:
                for seed in seeds:
                    print(f"RUN WP3 dataset={dataset_id} method={method} workflow={workflow} seed={seed}", flush=True)
                    try:
                        metrics = run_one(dataset_id, method, workflow, seed, Path(args.output_root), args.metric_sample_size)
                        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        failed_path.parent.mkdir(parents=True, exist_ok=True)
                        failed = {"dataset_id": dataset_id, "method": method, "workflow": workflow, "seed": seed, "error": repr(exc)}
                        with failed_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(failed, sort_keys=True) + "\n")
                        print(json.dumps(failed, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
