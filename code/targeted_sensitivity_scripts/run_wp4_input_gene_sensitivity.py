#!/usr/bin/env python3
"""Run input-gene/HVG sensitivity controls for reviewer response.

The selected local synthetic matrices contain 3,000 genes. This runner varies
the top variable input genes across 500, 1000, 2000, and 3000, avoiding any
misleading unavailable 5,000-gene condition.
"""
import argparse
import json
import os
import resource
import subprocess
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
RSCRIPT = Path("/home/saidi/anaconda3/envs/scrna-dr-r/bin/Rscript")
GLMPCA_HELPER = Path("revision_benchmark/scripts/run_glmpca_embedding.R")
SCGBM_HELPER = Path("revision_benchmark/scripts/run_scgbm_embedding.R")
WORK_PACKAGE = "WP4_input_gene_sensitivity"


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


def select_top_variable_genes(x: pd.DataFrame, requested: int) -> pd.DataFrame:
    actual = min(int(requested), x.shape[1])
    values = np.log1p(x.to_numpy(dtype=np.float32))
    order = np.argsort(values.var(axis=0))[::-1][:actual]
    return x.iloc[:, order].copy()


def preprocess_standardized(x: pd.DataFrame) -> np.ndarray:
    values = np.log1p(x.to_numpy(dtype=np.float32))
    return StandardScaler(with_mean=True, with_std=True).fit_transform(values).astype(np.float32)


def pca50(values: np.ndarray, seed: int) -> np.ndarray:
    n_components = min(50, values.shape[0] - 1, values.shape[1])
    return PCA(n_components=n_components, random_state=seed, svd_solver="randomized").fit_transform(values).astype(np.float32)


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
    if 1 < n_clusters < n:
        pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(z)
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


def run_pca(x_hvg: pd.DataFrame, seed: int, dimension: int) -> tuple[np.ndarray, dict]:
    values = preprocess_standardized(x_hvg)
    z = PCA(n_components=dimension, random_state=seed, svd_solver="randomized").fit_transform(values)
    return z, {"workflow": "latent_dim", "dimension": int(dimension), "pca_components": 0, "model_input": "log1p_standardized_hvg"}


def run_umap(x_hvg: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    import umap

    values = preprocess_standardized(x_hvg)
    pca_input = pca50(values, seed)
    model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=seed)
    z = model.fit_transform(pca_input)
    return z, {"workflow": "pca50_to_2d", "dimension": 2, "pca_components": int(pca_input.shape[1]), "model_input": "log1p_standardized_hvg_pca50"}


def run_tsne(x_hvg: pd.DataFrame, seed: int) -> tuple[np.ndarray, dict]:
    values = preprocess_standardized(x_hvg)
    pca_input = pca50(values, seed)
    model = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed)
    z = model.fit_transform(pca_input)
    return z, {"workflow": "pca50_to_2d", "dimension": 2, "pca_components": int(pca_input.shape[1]), "model_input": "log1p_standardized_hvg_pca50"}


def run_scvi(x_hvg: pd.DataFrame, seed: int, dimension: int, max_epochs: int) -> tuple[np.ndarray, dict]:
    import anndata as ad
    import torch
    import scvi

    torch.set_num_threads(int(os.environ.get("SCVI_TORCH_THREADS", "4")))
    scvi.settings.seed = seed
    adata = ad.AnnData(X=x_hvg.to_numpy(dtype=np.float32))
    adata.obs_names = x_hvg.index.astype(str)
    adata.var_names = x_hvg.columns.astype(str)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=dimension)
    model.train(max_epochs=max_epochs, accelerator="cpu", devices=1, enable_progress_bar=False)
    z = model.get_latent_representation()
    return z, {"workflow": "latent_dim", "dimension": int(dimension), "max_epochs": int(max_epochs), "pca_components": 0, "model_input": "raw_hvg_as_provided"}


def run_r_method(
    method: str,
    x_hvg: pd.DataFrame,
    dataset_id: str,
    hvg_requested: int,
    seed: int,
    dimension: int,
    output_root: Path,
    max_iter: int,
    min_iter: int,
) -> tuple[np.ndarray, dict]:
    helper = GLMPCA_HELPER if method == "GLMPCA" else SCGBM_HELPER
    tmp_dir = output_root / "tmp" / WORK_PACKAGE / dataset_id / f"hvg_{hvg_requested}" / method
    tmp_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = tmp_dir / f"matrix_seed_{seed}.csv"
    emb_path = tmp_dir / f"embedding_dim_{dimension}_seed_{seed}.csv"
    x_hvg.to_csv(matrix_path)
    cmd = [
        str(RSCRIPT),
        str(helper),
        "--matrix",
        str(matrix_path),
        "--embedding",
        str(emb_path),
        "--dimension",
        str(dimension),
        "--seed",
        str(seed),
        "--max-iter",
        str(max_iter),
        "--min-iter",
        str(min_iter),
    ]
    subprocess.run(cmd, check=True)
    emb = pd.read_csv(emb_path, index_col=0)
    emb.index = emb.index.astype(str)
    emb = emb.reindex(x_hvg.index)
    if emb.isna().any().any():
        raise ValueError(f"{method} embedding does not align to matrix cells")
    return emb.to_numpy(dtype=np.float64), {"workflow": "latent_dim", "dimension": int(dimension), "max_iter": int(max_iter), "min_iter": int(min_iter), "pca_components": 0, "model_input": "raw_hvg_as_provided"}


def save_result(
    output_root: Path,
    dataset_id: str,
    method: str,
    hvg_requested: int,
    seed: int,
    x_full: pd.DataFrame,
    x_hvg: pd.DataFrame,
    labels: pd.Series,
    z: np.ndarray,
    extra: dict,
    elapsed: float,
    metric_sample_size: int,
    matrix_path: Path,
    label_path: Path,
) -> dict:
    emb_dir = output_root / "embeddings" / WORK_PACKAGE / dataset_id / method / f"hvg_{hvg_requested}" / f"seed_{seed}"
    metric_dir = output_root / "metrics"
    log_dir = output_root / "logs" / WORK_PACKAGE
    emb_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / "embedding.csv"
    pd.DataFrame(z, index=x_hvg.index, columns=[f"{method}_{i + 1}" for i in range(z.shape[1])]).to_csv(emb_path)

    x_metric = np.log1p(x_hvg.to_numpy(dtype=np.float32))
    metrics = compute_metrics(x_metric, z, labels, seed, metric_sample_size)
    metrics.update({
        "work_package": WORK_PACKAGE,
        "dataset_id": dataset_id,
        "method": method,
        "environment": "scrna-dr-py-modern-min" if method in {"PCA", "scVI", "UMAP", "t-SNE"} else "scrna-dr-r",
        "hvg_requested": int(hvg_requested),
        "hvg_actual": int(x_hvg.shape[1]),
        "seed": int(seed),
        "n_cells": int(x_hvg.shape[0]),
        "n_genes_total": int(x_full.shape[1]),
        "n_genes_input": int(x_hvg.shape[1]),
        "metric_sample_size_requested": int(metric_sample_size),
        "runtime_seconds": float(elapsed),
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "matrix_path": str(matrix_path),
        "label_path": str(label_path),
        "embedding_path": str(emb_path),
        **extra,
    })
    metric_path = metric_dir / f"{WORK_PACKAGE}_metrics.csv"
    row = pd.DataFrame([metrics])
    if metric_path.exists():
        old = pd.read_csv(metric_path)
        row = pd.concat([old, row], ignore_index=True)
        row = row.drop_duplicates(subset=["dataset_id", "method", "hvg_requested", "seed", "workflow", "dimension"], keep="last")
    row.to_csv(metric_path, index=False)
    log_path = log_dir / f"{dataset_id}_{method}_hvg{hvg_requested}_seed{seed}.json"
    log_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def run_one(
    dataset_id: str,
    method: str,
    hvg_requested: int,
    seed: int,
    output_root: Path,
    metric_sample_size: int,
    latent_dim: int,
    scvi_epochs: int,
    max_iter: int,
    min_iter: int,
) -> dict:
    matrix_path = Path(DATASETS[dataset_id][0])
    label_path = Path(DATASETS[dataset_id][1])
    start = time.time()
    np.random.seed(seed)
    x = read_matrix(matrix_path)
    labels = read_labels(label_path, x.index)
    x_hvg = select_top_variable_genes(x, hvg_requested)

    if method == "PCA":
        z, extra = run_pca(x_hvg, seed, latent_dim)
    elif method == "UMAP":
        z, extra = run_umap(x_hvg, seed)
    elif method == "t-SNE":
        z, extra = run_tsne(x_hvg, seed)
    elif method == "scVI":
        z, extra = run_scvi(x_hvg, seed, latent_dim, scvi_epochs)
    elif method in {"GLMPCA", "scGBM"}:
        z, extra = run_r_method(method, x_hvg, dataset_id, hvg_requested, seed, latent_dim, output_root, max_iter, min_iter)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return save_result(
        output_root,
        dataset_id,
        method,
        hvg_requested,
        seed,
        x,
        x_hvg,
        labels,
        np.asarray(z, dtype=np.float64),
        extra,
        time.time() - start,
        metric_sample_size,
        matrix_path,
        label_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS.keys()))
    parser.add_argument("--methods", default="PCA,scVI,GLMPCA,scGBM,UMAP,t-SNE")
    parser.add_argument("--hvg-sizes", default="500,1000,2000,3000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--scvi-epochs", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--min-iter", type=int, default=5)
    parser.add_argument("--output-root", default="revision_benchmark/results")
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    methods = parse_csv(args.methods)
    hvg_sizes = parse_csv(args.hvg_sizes, int)
    seeds = parse_csv(args.seeds, int)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset IDs: {', '.join(unknown)}")
    if any(m in {"GLMPCA", "scGBM"} for m in methods) and not RSCRIPT.exists():
        raise SystemExit(f"Missing Rscript: {RSCRIPT}")

    output_root = Path(args.output_root)
    failed_path = output_root / "logs" / f"{WORK_PACKAGE}_failed_runs.jsonl"
    for dataset_id in datasets:
        for hvg in hvg_sizes:
            for method in methods:
                for seed in seeds:
                    print(f"RUN {WORK_PACKAGE} dataset={dataset_id} hvg={hvg} method={method} seed={seed}", flush=True)
                    try:
                        metrics = run_one(
                            dataset_id,
                            method,
                            hvg,
                            seed,
                            output_root,
                            args.metric_sample_size,
                            args.latent_dim,
                            args.scvi_epochs,
                            args.max_iter,
                            args.min_iter,
                        )
                        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        failed_path.parent.mkdir(parents=True, exist_ok=True)
                        failed = {"dataset_id": dataset_id, "hvg_requested": hvg, "method": method, "seed": seed, "error": repr(exc)}
                        with failed_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(failed, sort_keys=True) + "\n")
                        print(json.dumps(failed, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
