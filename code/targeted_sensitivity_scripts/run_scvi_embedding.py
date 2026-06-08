#!/usr/bin/env python3
"""Run scVI on a local benchmark CSV dataset and write embedding + metrics.

This is intentionally narrow: it supports the local datasets/ layout used for
reviewer-response WP1. It does not download data and does not mutate inputs.
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, silhouette_score


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df.astype(np.float32)


def read_labels(path: Path | None, cells) -> pd.Series | None:
    if path is None:
        return None
    meta = pd.read_csv(path, index_col=0)
    meta.index = meta.index.astype(str)
    if "Group" in meta.columns:
        labels = meta["Group"].astype(str)
    elif "cell_type" in meta.columns:
        labels = meta["cell_type"].astype(str)
    else:
        candidates = [c for c in meta.columns if c.lower() in {"label", "labels", "group", "celltype", "cell_type"}]
        if not candidates:
            return None
        labels = meta[candidates[0]].astype(str)
    labels = labels.reindex(cells)
    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(f"Label file is missing {missing} cells present in the matrix")
    return labels


def compute_metrics(x_input: np.ndarray, z: np.ndarray, labels: pd.Series | None, seed: int, metric_sample_size: int) -> dict:
    out = {}
    n = z.shape[0]
    if metric_sample_size and n > metric_sample_size:
        rng = np.random.default_rng(seed)
        metric_idx = np.sort(rng.choice(n, size=metric_sample_size, replace=False))
        out["metric_sample_size"] = int(metric_sample_size)
    else:
        metric_idx = np.arange(n)
        out["metric_sample_size"] = int(n)
    if labels is not None:
        y = labels.to_numpy()
        unique = np.unique(y)
        n_clusters = len(unique)
        out["n_labels"] = int(n_clusters)
        if n_clusters > 1 and n_clusters < n:
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
        k = min(30, n - 1)
        try:
            out["trustworthiness_k30"] = float(trustworthiness(x_input[metric_idx], z[metric_idx], n_neighbors=min(k, len(metric_idx) - 1)))
        except Exception as exc:
            out["trustworthiness_error"] = str(exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--output-root", default="revision_benchmark/results")
    parser.add_argument("--metric-sample-size", type=int, default=5000, help="Sample this many cells for O(n^2) metrics on large datasets; use 0 for all cells.")
    parser.add_argument("--work-package", default="WP1_scVI_local")
    args = parser.parse_args()

    start = time.time()
    np.random.seed(args.seed)

    import torch
    import scvi

    torch.set_num_threads(int(os.environ.get("SCVI_TORCH_THREADS", "4")))
    scvi.settings.seed = args.seed

    matrix_path = Path(args.matrix)
    label_path = Path(args.labels) if args.labels else None
    x = read_matrix(matrix_path)
    labels = read_labels(label_path, x.index) if label_path else None

    adata = ad.AnnData(X=x.to_numpy(dtype=np.float32))
    adata.obs_names = x.index.astype(str)
    adata.var_names = x.columns.astype(str)
    if labels is not None:
        adata.obs["label"] = labels.to_numpy()

    # scVI expects count-like nonnegative input. Local benchmark matrices are
    # nonnegative; use them as provided to keep reviewer reruns traceable.
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=args.dimension)
    model.train(max_epochs=args.max_epochs, accelerator="cpu", devices=1, enable_progress_bar=False)
    z = model.get_latent_representation()

    out_root = Path(args.output_root)
    emb_dir = out_root / "embeddings" / args.work_package / args.dataset_id / "scVI" / f"dim_{args.dimension}"
    metric_dir = out_root / "metrics"
    log_dir = out_root / "logs" / args.work_package
    emb_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    emb = pd.DataFrame(z, index=x.index, columns=[f"scVI_{i+1}" for i in range(z.shape[1])])
    emb_path = emb_dir / f"seed_{args.seed}.csv"
    emb.to_csv(emb_path)

    metrics = compute_metrics(x.to_numpy(dtype=np.float32), z, labels, args.seed, args.metric_sample_size)
    elapsed = time.time() - start
    metrics.update({
        "work_package": args.work_package,
        "dataset_id": args.dataset_id,
        "method": "scVI",
        "environment": "scrna-dr-py-modern-min",
        "dimension": args.dimension,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "metric_sample_size_requested": args.metric_sample_size,
        "n_cells": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "runtime_seconds": float(elapsed),
        "max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "matrix_path": str(matrix_path),
        "label_path": str(label_path) if label_path else "",
        "embedding_path": str(emb_path),
    })
    metric_path = metric_dir / f"{args.work_package}_scVI_metrics.csv"
    row = pd.DataFrame([metrics])
    if metric_path.exists():
        old = pd.read_csv(metric_path)
        row = pd.concat([old, row], ignore_index=True)
        row = row.drop_duplicates(subset=["dataset_id", "method", "dimension", "seed", "max_epochs"], keep="last")
    row.to_csv(metric_path, index=False)

    log_path = log_dir / f"{args.dataset_id}_scVI_dim{args.dimension}_seed{args.seed}.json"
    log_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
