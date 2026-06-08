from __future__ import annotations

import json
import math
from functools import reduce
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, Wedge
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "Publication" / "paper" / "revision_figures" / "figure3_polish" / "polished_trial"
SOURCE_DIR = ROOT / "Publication" / "paper" / "revision_figures" / "figure3_polish" / "source_data"

RAW_DIR = SOURCE_DIR / "vasc_stability_raw_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260604

AXES_DATASETS = {
    "cell_number": ["cell_100", "cell_500", "cell_1k", "cell_5k", "cell_1w", "cell_2w", "cell_3w"],
    "gene_number": ["gene_5k", "gene_2w", "gene_3w", "gene_4w", "gene_5w"],
    "celltype_number": ["celltype_7", "celltype_9", "celltype_11", "celltype_13", "celltype_15"],
    "dropout": ["dropout_-1", "dropout_0", "dropout_1", "dropout_2", "dropout_3"],
    "batch_number": ["batch_2", "batch_4", "batch_6", "batch_8", "batch_10"],
    "batch_strength": ["batch_0.2", "batch_0.4", "batch_0.6", "batch_0.8", "batch_1.0"],
    "de_prob": ["de_prob_0.05", "de_prob_0.15", "de_prob_0.2", "de_prob_0.25", "de_prob_0.3"],
    "de_strength": ["de_0.2", "de_0.4", "de_0.6", "de_0.8", "de_1.0"],
    "out": ["out_0.1", "out_0.2", "out_0.3", "out_0.4", "out_0.5"],
}

CANONICAL_METHODS = [
    "PCA",
    "GLMPCA",
    "pCMF",
    "ZIFA",
    "scGBM",
    "SSNMDI",
    "tGPLVM",
    "VAE",
    "scvis",
    "VASC",
    "SAUCIE",
    "scScope",
    "SCDRHA",
    "scGAE",
    "DREAM",
    "DRA",
    "UMAP",
    "SIMLR",
    "PHATE",
    "EDGE",
    "SPDR",
    "IVIS",
    "PaCMAP",
    "t-SNE",
    "TriMap",
    "SQuaD-MDS",
]

ORIGINAL_STYLE_GROUPS = {
    "Category 1": ["PCA", "GLMPCA", "pCMF", "ZIFA", "scGBM", "SSNMDI", "tGPLVM"],
    "Category 2": ["VAE", "scvis", "VASC", "SAUCIE", "scScope", "SCDRHA", "scGAE", "DREAM", "DRA"],
    "Category 3": ["UMAP", "SIMLR", "PHATE", "EDGE", "SPDR"],
    "Category 4": ["IVIS", "PaCMAP", "t-SNE", "TriMap", "SQuaD-MDS"],
}

RAW_TO_CANONICAL = {
    "ivis": "IVIS",
    "TSNE": "t-SNE",
    "SQuaD_MDS": "SQuaD-MDS",
}

FAMILY = {
    "PCA": "linear/probabilistic",
    "GLMPCA": "linear/probabilistic",
    "pCMF": "linear/probabilistic",
    "ZIFA": "linear/probabilistic",
    "scGBM": "linear/probabilistic",
    "SSNMDI": "linear/probabilistic",
    "tGPLVM": "linear/probabilistic",
    "VAE": "deep generative/autoencoder",
    "scvis": "deep generative/autoencoder",
    "VASC": "deep generative/autoencoder",
    "SAUCIE": "deep generative/autoencoder",
    "scScope": "deep generative/autoencoder",
    "SCDRHA": "deep generative/autoencoder",
    "scGAE": "deep generative/autoencoder",
    "DREAM": "deep generative/autoencoder",
    "DRA": "deep generative/autoencoder",
    "UMAP": "graph/diffusion",
    "SIMLR": "graph/diffusion",
    "PHATE": "graph/diffusion",
    "EDGE": "graph/diffusion",
    "SPDR": "graph/diffusion",
    "IVIS": "metric/structure-aware",
    "PaCMAP": "metric/structure-aware",
    "t-SNE": "metric/structure-aware",
    "TriMap": "metric/structure-aware",
    "SQuaD-MDS": "metric/structure-aware",
}

FAMILY_COLORS = {
    "linear/probabilistic": "#5B7FA6",
    "deep generative/autoencoder": "#B65A5C",
    "graph/diffusion": "#4E9A7B",
    "metric/structure-aware": "#9C7A3C",
}

DOMAIN_COLORS = {
    "Structure preservation": "#5E81AC",
    "Clustering concordance": "#C76F5A",
    "Efficiency": "#D9A441",
    "Robustness": "#5F9E72",
}

DOMAIN_HEADER_LABELS = {
    "Structure preservation": "Structure",
    "Clustering concordance": "Clustering",
    "Efficiency": "Efficiency",
    "Robustness": "Robustness",
}

DOMAIN_LIGHT = {
    "Structure preservation": "#CFE4EF",
    "Clustering concordance": "#F8CDC7",
    "Efficiency": "#FBE2BF",
    "Robustness": "#CCE8D2",
}

DOMAIN_CMAPS = {
    "Structure preservation": ("#07599B", "#C8E0EF"),
    "Clustering concordance": ("#CE1F1B", "#F7A37D"),
    "Efficiency": ("#F5AA2C", "#FFFCE8"),
    "Robustness": ("#00602F", "#A8DCA4"),
}

PANEL_LABEL_STYLE = {
    "ha": "left",
    "va": "bottom",
    "fontsize": 10,
    "fontweight": "bold",
    "color": "#111111",
}

MATRIX_COLUMNS = [
    ("local", "Local", "Structure preservation"),
    ("global", "Global", "Structure preservation"),
    ("kmeans", "K-means", "Clustering concordance"),
    ("louvain", "Louvain", "Clustering concordance"),
    ("spectral", "Spectral", "Clustering concordance"),
    ("runtime_score", "Runtime", "Efficiency"),
    ("memory_score", "Memory", "Efficiency"),
    ("cell_number", "Cell no.", "Robustness"),
    ("gene_number", "Gene no.", "Robustness"),
    ("celltype_number", "Cell-type no.", "Robustness"),
    ("dropout", "Dropout", "Robustness"),
    ("batch_number", "Batch no.", "Robustness"),
    ("batch_strength", "Batch str.", "Robustness"),
    ("de_prob", "DE prop.", "Robustness"),
    ("de_strength", "DE str.", "Robustness"),
    ("out", "Outlier", "Robustness"),
]

RAW_VARIABILITY_TOPOLOGY_METRICS = {
    "Local": [
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
        "random_triplet",
        "spearman",
        "k-nearest",
        "centroid_distance",
    ],
    "Global": ["T_10", "T_20", "T_30", "C_10", "C_20", "C_30", "AUC", "Qlocal", "Qglobal", "Pearson"],
}

RAW_VARIABILITY_CLUSTERING_METRICS = ["ARI", "NMI", "COMP", "HOMO", "SIL"]


def canonical_method(value: str) -> str:
    return RAW_TO_CANONICAL.get(value, value)


def read_embedding(dataset: str, method: str = "VASC") -> np.ndarray:
    path = ROOT / "results" / "simulate" / "datasets" / dataset / f"{method}_2.csv"
    return pd.read_csv(path, index_col=0).values.astype(np.float64)


def read_counts_and_labels(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = ROOT / "datasets" / "simulate" / dataset
    x = pd.read_csv(data_dir / "counts_matrix.csv", index_col=0).values.astype(np.float32)
    labels_raw = pd.read_csv(data_dir / "cell_metadata.csv", index_col=0)["Group"].values
    labels = LabelEncoder().fit_transform(labels_raw)
    return x, labels


def metric_sample_indices(n: int, max_n: int = 800) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(SEED + n)
    return np.sort(rng.choice(np.arange(n), size=max_n, replace=False))


def knn_eval(z: np.ndarray, labels: np.ndarray, n_neighbors: int) -> float:
    skf = StratifiedKFold(n_splits=10)
    scores = []
    for train_idx, test_idx in skf.split(z, labels):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(z[train_idx], labels[train_idx])
        scores.append(clf.score(z[test_idx], labels[test_idx]))
    return float(np.mean(scores))


def svm_eval(z: np.ndarray, labels: np.ndarray) -> float:
    z_scaled = scale(z)
    if z.shape[0] > 1500:
        rng = np.random.default_rng(SEED + z.shape[0])
        idx = np.sort(rng.choice(np.arange(z.shape[0]), size=1500, replace=False))
        z_scaled = z_scaled[idx]
        labels = labels[idx]
    skf = StratifiedKFold(n_splits=10)
    scores = []
    for train_idx, test_idx in skf.split(z_scaled, labels):
        clf = SVC()
        clf.fit(z_scaled[train_idx], labels[train_idx])
        scores.append(clf.score(z_scaled[test_idx], labels[test_idx]))
    return float(np.mean(scores))


def neighbor_kept_ratio(x: np.ndarray, z: np.ndarray, k: int) -> float:
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(x)
    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(z)
    high = nn_high.kneighbors(x, return_distance=False)[:, 1:]
    low = nn_low.kneighbors(z, return_distance=False)[:, 1:]
    kept = sum(len(np.intersect1d(high[i], low[i])) for i in range(x.shape[0]))
    return float(kept / (k * x.shape[0]))


def average_jaccard_index(x: np.ndarray, z: np.ndarray, k: int) -> float:
    high = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    low = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]
    vals = []
    for i in range(x.shape[0]):
        set_high = set(high[i])
        set_low = set(low[i])
        vals.append(len(set_high & set_low) / len(set_high | set_low))
    return float(np.mean(vals))


def neighborhood_overlap_metrics(x: np.ndarray, z: np.ndarray, ks: list[int]) -> dict[str, float]:
    max_k = max(ks)
    high = NearestNeighbors(n_neighbors=max_k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    low = NearestNeighbors(n_neighbors=max_k + 1).fit(z).kneighbors(z, return_distance=False)[:, 1:]
    out: dict[str, float] = {}
    for k in ks:
        kept = 0
        jaccard = []
        for i in range(x.shape[0]):
            set_high = set(high[i, :k])
            set_low = set(low[i, :k])
            inter = len(set_high & set_low)
            kept += inter
            jaccard.append(inter / len(set_high | set_low))
        out[f"nkr_{k}"] = float(kept / (k * x.shape[0]))
        out[f"aji_{k}"] = float(np.mean(jaccard))
    return out


def random_triplet_eval(x: np.ndarray, z: np.ndarray, num_triplets: int = 5) -> float:
    rng = np.random.default_rng(SEED + x.shape[0])
    n = x.shape[0]
    anchors = np.arange(n)
    triplets = rng.choice(anchors, (n, num_triplets, 2))
    correct = 0
    for i in range(n):
        anchor_x = x[i]
        anchor_z = z[i]
        for left, right in triplets[i]:
            high_label = np.linalg.norm(anchor_x - x[left]) < np.linalg.norm(anchor_x - x[right])
            low_label = np.linalg.norm(anchor_z - z[left]) < np.linalg.norm(anchor_z - z[right])
            correct += int(high_label == low_label)
    return float(correct / (n * num_triplets))


def spearman_correlation_eval(x: np.ndarray, z: np.ndarray, n_points: int = 500) -> float:
    rng = np.random.default_rng(100)
    size = min(n_points, x.shape[0])
    idx = np.sort(rng.choice(np.arange(x.shape[0]), size=size, replace=False))
    dist_high = distance_matrix(x[idx], x[idx]).reshape(-1)
    dist_low = distance_matrix(z[idx], z[idx]).reshape(-1)
    corr = scipy.stats.spearmanr(dist_high, dist_low).correlation
    return 0.0 if pd.isna(corr) else float(corr)


def centroid_knn_eval(x: np.ndarray, z: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    categories = np.unique(labels)
    high_centers = np.vstack([x[labels == cat].mean(axis=0) for cat in categories])
    low_centers = np.vstack([z[labels == cat].mean(axis=0) for cat in categories])
    k = min(k, len(categories) - 1)
    high_idx = NearestNeighbors(n_neighbors=k + 1).fit(high_centers).kneighbors(high_centers, return_distance=False)[:, 1:]
    low_idx = NearestNeighbors(n_neighbors=k + 1).fit(low_centers).kneighbors(low_centers, return_distance=False)[:, 1:]
    kept = sum(high_idx[i, j] in low_idx[i, :] for i in range(len(categories)) for j in range(k))
    return float(kept / (k * len(categories)))


def centroid_corr_eval(x: np.ndarray, z: np.ndarray, labels: np.ndarray) -> float:
    categories = np.unique(labels)
    high_centers = np.vstack([x[labels == cat].mean(axis=0) for cat in categories])
    low_centers = np.vstack([z[labels == cat].mean(axis=0) for cat in categories])
    dist_high = distance_matrix(high_centers, high_centers).reshape(-1)
    dist_low = distance_matrix(low_centers, low_centers).reshape(-1)
    corr = scipy.stats.spearmanr(dist_high, dist_low).correlation
    return 0.0 if pd.isna(corr) else float(corr)


def safe_silhouette(z: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(pred)) < 2:
        return 0.0
    return float(silhouette_score(z, pred))


def compute_vasc_metrics_for_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dr1 = RAW_DIR / f"{dataset}_VASC_dr1.csv"
    cache_dr2 = RAW_DIR / f"{dataset}_VASC_dr2.csv"
    cache_cluster = RAW_DIR / f"{dataset}_VASC_kmeans.csv"
    if cache_dr1.exists() and cache_dr2.exists() and cache_cluster.exists():
        return pd.read_csv(cache_dr1), pd.read_csv(cache_dr2), pd.read_csv(cache_cluster)

    print(f"Computing VASC metrics for {dataset}", flush=True)
    x, labels = read_counts_and_labels(dataset)
    z = read_embedding(dataset)
    if x.shape[0] != z.shape[0]:
        raise ValueError(f"{dataset}: counts rows {x.shape[0]} != embedding rows {z.shape[0]}")

    idx = metric_sample_indices(x.shape[0])
    x_metric = x[idx]
    z_metric = z[idx]
    labels_metric = labels[idx]

    dr1 = {"Method": "VASC"}
    for k in [10, 20, 30]:
        dr1[f"knn_{k}"] = knn_eval(z, labels, k)
    dr1["svm"] = svm_eval(z, labels)
    dr1.update(neighborhood_overlap_metrics(x_metric, z_metric, [10, 20, 30]))
    dr1["random_triplet"] = random_triplet_eval(x_metric, z_metric)
    dr1["spearman"] = spearman_correlation_eval(x_metric, z_metric)
    dr1["k-nearest"] = centroid_knn_eval(x, z, labels, k=1)
    dr1["centroid_distance"] = centroid_corr_eval(x, z, labels)

    dr2 = {"Method": "VASC", "AUC": np.nan, "Qlocal": np.nan, "Qglobal": np.nan}
    for k in [10, 20, 30]:
        kk = min(k, z_metric.shape[0] - 1)
        dr2[f"T_{k}"] = trustworthiness(x_metric, z_metric, n_neighbors=kk)
    for k in [10, 20, 30]:
        kk = min(k, z_metric.shape[0] - 1)
        dr2[f"C_{k}"] = trustworthiness(z_metric, x_metric, n_neighbors=kk)
    dr2["kmax"] = np.nan

    n_cluster = len(np.unique(labels))
    pred = KMeans(n_clusters=n_cluster, random_state=0, n_init=50).fit(z).labels_
    cluster = {
        "Method": "VASC",
        "ARI": round(adjusted_rand_score(labels, pred), 2),
        "NMI": round(normalized_mutual_info_score(labels, pred), 2),
        "SIL": round(safe_silhouette(z, pred), 2),
        "COMP": round(completeness_score(labels, pred), 2),
        "HOMO": round(homogeneity_score(labels, pred), 2),
    }

    dr1_df = pd.DataFrame([dr1]).round(3)
    dr2_df = pd.DataFrame([dr2]).round(3)
    cluster_df = pd.DataFrame([cluster])
    dr1_df.to_csv(cache_dr1, index=False)
    dr2_df.to_csv(cache_dr2, index=False)
    cluster_df.to_csv(cache_cluster, index=False)
    return dr1_df, dr2_df, cluster_df


def existing_axis_table(dataset: str) -> pd.DataFrame:
    topo_dir = ROOT / "metric" / "topo" / "simulate" / dataset
    cluster_dir = ROOT / "metric" / "cluster" / "simulate" / dataset / "indicators"
    dr1 = pd.read_csv(topo_dir / "dr1.csv")
    dr2 = pd.read_csv(topo_dir / "dr2.csv")
    cluster_paths = [
        cluster_dir / "kmeans_ARI.csv",
        cluster_dir / "kmeans_NMI.csv",
        cluster_dir / "kmeans_SIL.csv",
        cluster_dir / "kmeans_COMP.csv",
        cluster_dir / "kmeans_HOMO.csv",
    ]
    clusters = [pd.read_csv(path) for path in cluster_paths]
    merged = reduce(lambda left, right: pd.merge(left, right, on="Method", how="outer"), [dr1, dr2, *clusters])
    merged.insert(1, "Dataset", dataset)
    return merged


def vasc_axis_table(dataset: str) -> pd.DataFrame:
    dr1, dr2, cluster = compute_vasc_metrics_for_dataset(dataset)
    merged = reduce(lambda left, right: pd.merge(left, right, on="Method", how="outer"), [dr1, dr2, cluster])
    merged.insert(1, "Dataset", dataset)
    return merged


def compute_stability_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames = []
    axis_scores = []
    for axis, datasets in AXES_DATASETS.items():
        frames = []
        for dataset in datasets:
            existing = existing_axis_table(dataset)
            existing = existing[existing["Method"] != "VASC"]
            vasc = vasc_axis_table(dataset)
            frames.append(pd.concat([existing, vasc], ignore_index=True, sort=False))
        axis_raw = pd.concat(frames, ignore_index=True, sort=False)
        axis_raw["perturbation_axis"] = axis
        raw_frames.append(axis_raw)

        metric_cols = [
            col
            for col in axis_raw.columns
            if col not in {"Method", "Dataset", "perturbation_axis", "AUC", "Qlocal", "Qglobal", "kmax"}
        ]
        scaled = axis_raw.copy()
        scaled[metric_cols] = MinMaxScaler().fit_transform(scaled[metric_cols])
        scaled["Accuracy_subscore"] = scaled[metric_cols].mean(axis=1)
        scores = scaled.groupby("Method", as_index=False)["Accuracy_subscore"].mean()
        scores = scores.rename(columns={"Accuracy_subscore": "score"})
        scores["perturbation_axis"] = axis
        axis_scores.append(scores)
        scores.to_csv(SOURCE_DIR / f"Figure_3_completed_stability_{axis}.csv", index=False)

    raw_all = pd.concat(raw_frames, ignore_index=True, sort=False)
    score_all = pd.concat(axis_scores, ignore_index=True, sort=False)
    raw_all.to_csv(SOURCE_DIR / "Figure_3_completed_stability_raw_metrics.csv", index=False)
    score_all.to_csv(SOURCE_DIR / "Figure_3_completed_stability_scores_long.csv", index=False)
    return raw_all, score_all


def load_score_matrix(stability_long: pd.DataFrame) -> pd.DataFrame:
    score_path = ROOT / "Publication" / "paper" / "revision_figures" / "canonical_source_tables" / "original_score_matrix.csv"
    score = pd.read_csv(score_path)
    score["method_id"] = score["method_id"].map(canonical_method)
    score = score[score["method_id"].isin(CANONICAL_METHODS)].copy()
    score = score.drop_duplicates(subset=["method_id"], keep="first")

    stability_long = stability_long.copy()
    stability_long["method_id"] = stability_long["Method"].map(canonical_method)
    stability_piv = stability_long.pivot_table(index="method_id", columns="perturbation_axis", values="score", aggfunc="mean")
    stability_piv = stability_piv.reindex(CANONICAL_METHODS)
    stability_piv["stability_median"] = stability_piv[list(AXES_DATASETS.keys())].median(axis=1, skipna=True)

    score = score.set_index("method_id")
    for col in list(AXES_DATASETS.keys()) + ["stability_median"]:
        score[col] = stability_piv[col]

    mean_cols = [col for col, _, _ in MATRIX_COLUMNS if col in score.columns]
    score["overall_mean"] = score[mean_cols].mean(axis=1, skipna=True)
    score = score.reindex(CANONICAL_METHODS).reset_index()
    score["family"] = score["method_id"].map(FAMILY)
    score.to_csv(SOURCE_DIR / "Figure_3_completed_score_matrix.csv", index=False)
    return score


def save_provenance(raw_all: pd.DataFrame, stability_long: pd.DataFrame, score: pd.DataFrame) -> None:
    vasc_scores = stability_long[stability_long["Method"] == "VASC"].copy()
    provenance = {
        "figure": "Figure 3 trial with completed VASC stability",
        "seed": SEED,
        "vasc_axes_completed": vasc_scores[["perturbation_axis", "score"]].to_dict(orient="records"),
        "raw_metric_rows": int(raw_all.shape[0]),
        "methods_in_plot": CANONICAL_METHODS,
        "notes": [
            "Original metric tables were not overwritten.",
            "VASC raw stability metrics were computed from local synthetic count matrices and VASC_2.csv embeddings.",
            "High-dimensional neighborhood, trustworthiness, continuity and triplet metrics use a fixed sample of up to 800 cells; distance Spearman uses up to 500 cells; low-dimensional KNN and k-means use the available embedding rows; SVM uses up to 1500 embedding rows.",
            "If k-means returns fewer than two distinct clusters, silhouette is undefined and is encoded as 0 for the score table.",
            "Scores are min-max scaled per perturbation axis using the combined existing rows plus the newly computed VASC row.",
        ],
    }
    (SOURCE_DIR / "Figure_3_completed_vasc_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


def setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def interpolate_color(low: str, high: str, value: float) -> tuple[float, float, float, float]:
    value = float(np.clip(value, 0, 1))
    low_rgba = np.array(mpl.colors.to_rgba(low))
    high_rgba = np.array(mpl.colors.to_rgba(high))
    return tuple(low_rgba * (1 - value) + high_rgba * value)


def raw_variability_summary(score: pd.DataFrame) -> pd.DataFrame:
    canonical_dir = ROOT / "Publication" / "paper" / "revision_figures" / "canonical_source_tables"
    methods = [method for group in ORIGINAL_STYLE_GROUPS.values() for method in group]
    frames: list[pd.DataFrame] = []

    topo = pd.read_csv(canonical_dir / "original_topology_raw_long.csv")
    topo["method_id"] = topo["method_id"].map(canonical_method)
    for domain, metrics in RAW_VARIABILITY_TOPOLOGY_METRICS.items():
        subset = topo[topo["method_id"].isin(methods) & topo["metric"].isin(metrics)].copy()
        subset["domain"] = domain
        frames.append(subset[["method_id", "dataset_category", "dataset_id", "domain", "metric", "value"]])

    clustering = pd.read_csv(canonical_dir / "original_clustering_raw_long.csv")
    clustering["method_id"] = clustering["method_id"].map(canonical_method)
    clustering = clustering[
        clustering["method_id"].isin(methods) & clustering["metric"].isin(RAW_VARIABILITY_CLUSTERING_METRICS)
    ].copy()
    clustering["domain"] = "Clustering"
    frames.append(clustering[["method_id", "dataset_category", "dataset_id", "domain", "metric", "value"]])

    raw = pd.concat(frames, ignore_index=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"])

    def normalize_metric(values: pd.Series) -> pd.Series:
        min_value = values.min()
        max_value = values.max()
        if math.isclose(float(max_value), float(min_value)):
            return pd.Series(np.full(values.shape[0], 0.5), index=values.index)
        return (values - min_value) / (max_value - min_value)

    raw["normalized_raw_score"] = raw.groupby(["domain", "metric"], group_keys=False)["value"].apply(normalize_metric)
    raw = raw[raw["normalized_raw_score"].between(0, 1, inclusive="both")]

    summary = (
        raw.groupby("method_id")["normalized_raw_score"]
        .agg(
            raw_median="median",
            raw_q1=lambda s: s.quantile(0.25),
            raw_q3=lambda s: s.quantile(0.75),
            raw_mean="mean",
            raw_observations="count",
        )
        .reset_index()
    )
    summary = summary.merge(score[["method_id", "overall_mean", "family"]], on="method_id", how="left")
    summary.to_csv(SOURCE_DIR / "Figure_3_completed_raw_variability_summary.csv", index=False)
    return summary


def original_style_positions() -> tuple[list[str], dict[str, float], dict[str, float], float]:
    methods: list[str] = []
    ypos: dict[str, float] = {}
    category_y: dict[str, float] = {}
    y = 0.0
    for category, group_methods in ORIGINAL_STYLE_GROUPS.items():
        category_y[category] = y
        y += 1.04
        for method in group_methods:
            methods.append(method)
            ypos[method] = y
            y += 1.02
        y += 0.48
    return methods, ypos, category_y, y - 0.48


def add_header_block(
    ax: plt.Axes,
    x0: float,
    x1: float,
    color: str,
    light_color: str,
    title: str,
    subtitle: str | None = None,
    title_size: float = 8.2,
    subtitle_size: float = 7.0,
) -> None:
    ax.add_patch(Rectangle((x0, -4.24), x1 - x0, 0.78, facecolor=color, edgecolor="white", linewidth=0.8, zorder=4))
    ax.text(
        (x0 + x1) / 2,
        -3.85,
        title,
        color="white",
        fontsize=title_size,
        ha="center",
        va="center",
        linespacing=0.86,
        zorder=5,
    )
    if subtitle:
        subtitle_y = -3.38
        subtitle_h = 1.20
        ax.add_patch(
            Rectangle(
                (x0, subtitle_y),
                x1 - x0,
                subtitle_h,
                facecolor=light_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            subtitle_y + subtitle_h / 2,
            subtitle,
            color="#111111",
            fontsize=subtitle_size,
            ha="center",
            va="center",
            linespacing=1.02,
            zorder=5,
        )


def draw_bubble_matrix(ax: plt.Axes, score: pd.DataFrame) -> None:
    cols = MATRIX_COLUMNS
    n_cols = len(cols)
    methods, ypos, category_y, max_y = original_style_positions()
    score_idx = score.set_index("method_id")

    ax.set_xlim(-3.04, n_cols - 0.25)
    ax.set_ylim(max_y + 2.05, -4.38)
    ax.set_facecolor("white")

    # Original-style colored header bands.
    add_header_block(ax, -2.82, -1.08, "#555555", "#E8E8E8", "Method", title_size=7.9)
    add_header_block(
        ax,
        -0.65,
        1.55,
        DOMAIN_COLORS["Structure preservation"],
        DOMAIN_LIGHT["Structure preservation"],
        "Structure Preservation",
        "Local neighborhood\nand global geometry",
        title_size=5.25,
        subtitle_size=6.20,
    )
    add_header_block(
        ax,
        1.55,
        4.85,
        "#F13A2A",
        DOMAIN_LIGHT["Clustering concordance"],
        "Cluster Accuracy",
        "K-means, Louvain,\nspectral",
        title_size=7.25,
        subtitle_size=6.20,
    )
    add_header_block(
        ax,
        4.85,
        6.65,
        "#F7941E",
        DOMAIN_LIGHT["Efficiency"],
        "Efficiency",
        "Runtime and\npeak memory",
        title_size=7.6,
        subtitle_size=6.75,
    )
    add_header_block(
        ax,
        6.65,
        15.35,
        "#41A85F",
        DOMAIN_LIGHT["Robustness"],
        "Stability",
        "Cell, gene, cell-type,\nbatch, dropout, DE, outlier",
        title_size=7.9,
        subtitle_size=6.95,
    )

    row_index = 0
    for method in methods:
        y = ypos[method]
        if row_index % 2 == 0:
            ax.axhspan(y - 0.43, y + 0.43, color="#E3E3E3", zorder=0)
        row_index += 1

    for category, y in category_y.items():
        ax.text(-2.72, y, category, ha="left", va="center", fontsize=7.3, color="#111111")

    for method in methods:
        ax.text(-2.28, ypos[method], method, ha="left", va="center", fontsize=7.3, color="#111111")

    for x, (col, label, domain) in enumerate(cols):
        ax.plot([x, x], [-0.47, -0.30], color="#111111", lw=0.55, zorder=3)
        ax.text(x - 0.02, -0.58, label, rotation=31, ha="left", va="bottom", fontsize=5.75, color="#111111")
        low, high = DOMAIN_CMAPS[domain]
        for method in methods:
            value = score_idx.loc[method, col]
            if pd.isna(value):
                continue
            y = ypos[method]
            size = 24 + 150 * float(np.clip(value, 0, 1)) ** 1.45
            face = interpolate_color(low, high, float(value))
            marker = "s" if domain == "Efficiency" else "o"
            edge = "#6B6B6B" if domain == "Efficiency" else "#5A5A5A"
            linewidth = 0.45 if domain == "Efficiency" else 0.35
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                color=face,
                edgecolors=edge,
                linewidths=linewidth,
                zorder=2,
            )

    # Original-style score-size legend inside panel a.
    legend_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    legend_x0 = -2.76
    legend_y = max_y + 1.22
    for i, value in enumerate(legend_vals):
        x = legend_x0 + i * 0.50
        marker = "s" if math.isclose(value, 1.0) else "o"
        size = 13 + 86 * value**1.45
        ax.scatter(x, legend_y, s=size, marker=marker, color="#BFBFBF", edgecolors="#555555", linewidths=0.35, zorder=5)
        ax.text(x, legend_y + 0.72, f"{value:g}", ha="center", va="center", fontsize=5.7, color="#222222")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _flow_arrow(ax: plt.Axes, x0: float, x1: float, y: float, color: str = "#6F7780") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=8.6,
            lw=0.75,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def _step_label(ax: plt.Axes, x: float, number: int, title: str, color: str, y: float = 0.88) -> None:
    ax.scatter(x, y, s=86, facecolor="white", edgecolor=color, linewidth=0.85, zorder=4)
    ax.text(x, y, str(number), ha="center", va="center", fontsize=6.3, color=color, fontweight="bold", zorder=5)
    ax.text(
        x + 0.020,
        y,
        title,
        ha="left",
        va="center",
        fontsize=6.55,
        color="#111111",
        fontweight="bold",
        zorder=4,
    )


def draw_score_demo_panel(ax: plt.Axes, score: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    example_method = "VASC" if "VASC" in set(score["method_id"]) else str(score.iloc[0]["method_id"])
    domain_map = {
        "Structure": ["local", "global"],
        "Clustering": ["kmeans", "louvain", "spectral"],
        "Efficiency": ["runtime_score", "memory_score"],
        "Stability": list(AXES_DATASETS.keys()),
    }
    component_labels = {
        "local": "Local",
        "global": "Global",
        "kmeans": "K-means",
        "louvain": "Louvain",
        "spectral": "Spectral",
        "runtime_score": "Runtime",
        "memory_score": "Memory",
        "cell_number": "Cell no.",
        "gene_number": "Gene no.",
        "celltype_number": "Cell-type",
        "dropout": "Dropout",
        "batch_number": "Batch no.",
        "batch_strength": "Batch str.",
        "de_prob": "DE prop.",
        "de_strength": "DE str.",
        "out": "Outlier",
    }
    domain_palette = {
        "Structure": DOMAIN_COLORS["Structure preservation"],
        "Clustering": DOMAIN_COLORS["Clustering concordance"],
        "Efficiency": DOMAIN_COLORS["Efficiency"],
        "Stability": DOMAIN_COLORS["Robustness"],
    }
    component_cols = [col for cols in domain_map.values() for col in cols]
    example_row = score.set_index("method_id").loc[example_method]
    component_values = example_row[component_cols].astype(float)
    domain_values = {
        domain: float(component_values[cols].mean())
        for domain, cols in domain_map.items()
    }
    profile_value = float(component_values.mean())
    stored_profile = float(example_row["overall_mean"])
    if not np.isclose(profile_value, stored_profile, atol=1e-12):
        raise ValueError(f"Profile score mismatch for {example_method}: {profile_value} vs {stored_profile}")
    total_components = len(component_cols)
    contributions = {
        domain: len(cols) / total_components * domain_values[domain]
        for domain, cols in domain_map.items()
    }
    if not np.isclose(sum(contributions.values()), stored_profile, atol=1e-12):
        raise ValueError(f"Contribution mismatch for {example_method}: {sum(contributions.values())} vs {stored_profile}")

    demo_rows = [
        {"method_id": example_method, "level": "component", "domain": domain, "metric": col, "score": float(component_values[col])}
        for domain, cols in domain_map.items()
        for col in cols
    ]
    demo_rows.extend(
        {"method_id": example_method, "level": "domain_mean", "domain": domain, "metric": "domain_mean", "score": value}
        for domain, value in domain_values.items()
    )
    demo_rows.extend(
        {
            "method_id": example_method,
            "level": "profile_contribution",
            "domain": domain,
            "metric": "n_over_16_times_domain_mean",
            "score": value,
        }
        for domain, value in contributions.items()
    )
    demo_rows.append(
        {"method_id": example_method, "level": "profile_mean", "domain": "all_displayed_components", "metric": "overall_mean", "score": profile_value}
    )
    pd.DataFrame(demo_rows).to_csv(SOURCE_DIR / "Figure_3_score_construction_demo_values.csv", index=False)

    ax.text(0.035, 0.875, "Metric-to-score transformation", ha="left", va="center", fontsize=6.7, fontweight="bold", color="#111111")
    ax.text(0.330, 0.875, f"{example_method} normalized component profile", ha="left", va="center", fontsize=6.7, fontweight="bold", color="#111111")
    ax.text(0.725, 0.875, "Aggregation audit", ha="left", va="center", fontsize=6.7, fontweight="bold", color="#111111")

    # Direction-aware normalization rule; raw distributions are summarized by the formula, not fabricated values.
    ax.text(0.045, 0.705, "benefit metric", ha="left", va="center", fontsize=5.25, color="#0B867D", fontweight="bold")
    ax.text(0.045, 0.455, "cost metric", ha="left", va="center", fontsize=5.25, color="#9B6512", fontweight="bold")
    ax.text(0.052, 0.650, r"$s=(x-x_{\min})/(x_{\max}-x_{\min})$", ha="left", va="center", fontsize=5.4, color="#252A30")
    ax.text(0.052, 0.400, r"$s=(x_{\max}-x)/(x_{\max}-x_{\min})$", ha="left", va="center", fontsize=5.4, color="#252A30")
    ax.text(0.045, 0.135, "larger score always indicates better performance", ha="left", va="center", fontsize=4.8, color="#5C6570")
    for y0, color, flip in [(0.495, "#0B867D", False), (0.245, "#D9A441", True)]:
        xs = np.linspace(0.055, 0.230, 34)
        curve = 0.016 + 0.056 * np.exp(-((xs - (0.168 if not flip else 0.118)) / 0.045) ** 2)
        if flip:
            curve = curve[::-1]
        ax.fill_between(xs, y0, y0 + curve, color=color, alpha=0.18, lw=0)
        ax.plot(xs, y0 + curve, color=color, lw=0.85, alpha=0.78)
        ax.plot([0.055, 0.230], [y0, y0], color="#C8CED4", lw=0.65)
        _flow_arrow(ax, 0.235, 0.265, y0 + 0.035, color=color)

    # Actual VASC component values shown as grouped point estimates on the normalized [0,1] scale.
    x0, x1 = 0.350, 0.625
    ax.plot([x0, x1], [0.748, 0.748], color="#343A40", lw=0.55)
    for tick, label in [(0, "0"), (0.5, "0.5"), (1, "1")]:
        xt = x0 + (x1 - x0) * tick
        ax.plot([xt, xt], [0.728, 0.768], color="#343A40", lw=0.45)
        ax.text(xt, 0.786, label, ha="center", va="bottom", fontsize=4.8, color="#343A40")
    ax.text((x0 + x1) / 2, 0.692, "direction-consistent score", ha="center", va="center", fontsize=4.9, color="#1F63B5")

    row_y = {"Structure": 0.600, "Clustering": 0.485, "Efficiency": 0.370, "Stability": 0.255}
    for domain, y0 in row_y.items():
        vals = component_values[domain_map[domain]].to_numpy(dtype=float)
        color = domain_palette[domain]
        ax.plot([x0, x1], [y0, y0], color="#D8DDE2", lw=1.1, solid_capstyle="round", zorder=1)
        offsets = (np.arange(len(vals)) % 3 - 1) * 0.018
        xs = x0 + (x1 - x0) * vals
        ax.scatter(xs, y0 + offsets, s=21, color=color, edgecolors="white", linewidths=0.38, alpha=0.94, zorder=4)
        mean_x = x0 + (x1 - x0) * domain_values[domain]
        ax.scatter(mean_x, y0, s=62, marker="D", color=color, edgecolors="#FFFFFF", linewidths=0.5, zorder=5)
        ax.text(x0 - 0.018, y0, domain, ha="right", va="center", fontsize=5.15, color=color, fontweight="bold")
        ax.text(x1 + 0.010, y0, f"n={len(vals)} mean={domain_values[domain]:.2f}", ha="left", va="center", fontsize=4.55, color="#424A53")
    ax.text((x0 + x1) / 2, 0.145, "small dots: displayed components; diamonds: domain means", ha="center", va="center", fontsize=4.65, color="#5C6570")

    # Aggregation is equivalent to a component-count weighted sum of domain means.
    bar_x0, bar_x1 = 0.724, 0.850
    bar_y = 0.592
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="#D8DDE2", lw=7.5, solid_capstyle="round", zorder=1)
    cursor = bar_x0
    for domain, value in contributions.items():
        color = domain_palette[domain]
        width = (bar_x1 - bar_x0) * value
        ax.plot([cursor, cursor + width], [bar_y, bar_y], color=color, lw=7.5, solid_capstyle="butt", zorder=2)
        cursor += width
    ax.plot([bar_x0, bar_x1], [bar_y - 0.085, bar_y - 0.085], color="#D8DDE2", lw=0.8)
    for tick, label in [(0, "0"), (0.5, "0.5"), (1, "1")]:
        xt = bar_x0 + (bar_x1 - bar_x0) * tick
        ax.plot([xt, xt], [bar_y - 0.105, bar_y - 0.065], color="#7B838C", lw=0.45)
        ax.text(xt, bar_y - 0.126, label, ha="center", va="top", fontsize=4.6, color="#5C6570")
    y_base = 0.382
    for i, (domain, contribution) in enumerate(contributions.items()):
        color = domain_palette[domain]
        y0 = y_base - i * 0.058
        ax.scatter(0.724, y0, s=14, color=color, edgecolors="white", linewidths=0.35)
        ax.text(0.738, y0, f"{len(domain_map[domain])}/16 x {domain_values[domain]:.2f}", ha="left", va="center", fontsize=4.35, color="#39414A")
        ax.text(0.842, y0, f"{contribution:.2f}", ha="right", va="center", fontsize=4.35, color="#39414A")
    ax.text((bar_x0 + bar_x1) / 2, 0.705, r"$\sum_d (n_d/16)\bar{s}_d$", ha="center", va="center", fontsize=6.3, color="#252A30")

    gauge_center = (0.945, 0.525)
    radius = 0.058
    ax.add_patch(Wedge(gauge_center, radius, 90, 450, width=0.013, facecolor="#E8E3F7", edgecolor="none", zorder=1))
    ax.add_patch(Wedge(gauge_center, radius, 90, 90 + 360 * profile_value, width=0.013, facecolor="#4E2A91", edgecolor="none", zorder=2))
    ax.text(gauge_center[0], gauge_center[1] + 0.012, f"{profile_value:.2f}", ha="center", va="center", fontsize=10.6, color="#4E2A91", fontweight="bold")
    ax.text(gauge_center[0], gauge_center[1] - 0.068, "profile score", ha="center", va="center", fontsize=5.2, color="#4E2A91", fontweight="bold")


def draw_overall_panel(ax: plt.Axes, score: pd.DataFrame) -> None:
    plot_df = score.set_index("method_id").reindex(CANONICAL_METHODS[::-1]).reset_index()
    colors = plot_df["family"].map(FAMILY_COLORS)
    ax.barh(plot_df["method_id"], plot_df["overall_mean"], color=colors, height=0.62, alpha=0.88)
    ax.set_xlim(0, max(0.78, float(plot_df["overall_mean"].max()) + 0.04))
    ax.set_xlabel("Mean profile score", fontsize=6.4)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=4.8, length=0, pad=1.2)
    ax.tick_params(axis="x", labelsize=5.7, length=2)
    ax.grid(axis="x", color="#E1E4E8", lw=0.45)
    ax.set_title("Overall profile", fontsize=7.2, pad=3)


def draw_domain_panel(ax: plt.Axes, score: pd.DataFrame) -> None:
    domain_map = {
        "Structure": ["local", "global"],
        "Clustering": ["kmeans", "louvain", "spectral"],
        "Efficiency": ["runtime_score", "memory_score"],
        "Robustness": list(AXES_DATASETS.keys()),
    }
    rows = []
    for _, row in score.iterrows():
        for domain, cols in domain_map.items():
            values = [row[col] for col in cols if col in row and not pd.isna(row[col])]
            if values:
                rows.append(
                    {
                        "method_id": row["method_id"],
                        "domain": domain,
                        "score": float(np.mean(values)),
                        "family": row["family"],
                    }
                )
    domain_df = pd.DataFrame(rows)
    order = ["Structure", "Clustering", "Efficiency", "Robustness"]
    palette = {
        "Structure": DOMAIN_COLORS["Structure preservation"],
        "Clustering": DOMAIN_COLORS["Clustering concordance"],
        "Efficiency": DOMAIN_COLORS["Efficiency"],
        "Robustness": DOMAIN_COLORS["Robustness"],
    }
    sns.boxplot(
        data=domain_df,
        x="domain",
        y="score",
        hue="domain",
        order=order,
        hue_order=order,
        ax=ax,
        palette=palette,
        width=0.55,
        fliersize=0,
        linewidth=0.6,
        saturation=0.82,
        legend=False,
    )
    sns.stripplot(
        data=domain_df,
        x="domain",
        y="score",
        order=order,
        hue="family",
        palette=FAMILY_COLORS,
        ax=ax,
        size=2.5,
        alpha=0.72,
        jitter=0.18,
        linewidth=0.25,
        edgecolor="white",
    )
    if ax.legend_:
        ax.legend_.remove()
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("")
    ax.set_ylabel("Domain score", fontsize=6.4)
    ax.tick_params(axis="x", labelsize=5.8, rotation=18, length=0)
    ax.tick_params(axis="y", labelsize=5.7, length=2)
    ax.grid(axis="y", color="#E1E4E8", lw=0.45)
    ax.set_title("Score distributions", fontsize=7.2, pad=3)


def domain_score_long(score: pd.DataFrame) -> pd.DataFrame:
    domain_map = {
        "Structure": ["local", "global"],
        "Clustering": ["kmeans", "louvain", "spectral"],
        "Efficiency": ["runtime_score", "memory_score"],
        "Robustness": list(AXES_DATASETS.keys()),
    }
    rows = []
    for _, row in score.iterrows():
        for domain, cols in domain_map.items():
            values = [row[col] for col in cols if col in row and not pd.isna(row[col])]
            if not values:
                continue
            rows.append(
                {
                    "method_id": row["method_id"],
                    "family": row["family"],
                    "domain": domain,
                    "domain_score": float(np.mean(values)),
                    "component_count": len(values),
                    "weighted_contribution": float(np.mean(values) * len(values) / len(MATRIX_COLUMNS)),
                }
            )
    return pd.DataFrame(rows)


def draw_family_domain_profile(ax: plt.Axes, score: pd.DataFrame) -> None:
    long = domain_score_long(score)
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
    domain_order = ["Structure", "Clustering", "Efficiency", "Robustness"]
    mat = (
        long.groupby(["family", "domain"], observed=False)["domain_score"]
        .median()
        .unstack()
        .reindex(index=family_order, columns=domain_order)
        .rename(index=family_short)
    )
    cmap = LinearSegmentedColormap.from_list("family_domain", ["#F6F3EA", "#88C5B8", "#24547A"])
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0.15, vmax=0.95)
    ax.set_title("Family-domain profile", fontsize=7.2, pad=3)
    ax.set_xticks(np.arange(len(domain_order)))
    ax.set_xticklabels(domain_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.tick_params(length=0, labelsize=4.9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=4.9, color="white" if value > 0.62 else "#2A2A2A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    mat.reset_index().to_csv(SOURCE_DIR / "Figure_3_family_domain_profile.csv", index=False)


def draw_component_top_frequency(ax: plt.Axes, score: pd.DataFrame) -> None:
    rows = []
    for col, _, domain in MATRIX_COLUMNS:
        if col not in score.columns:
            continue
        top = score[["method_id", "family", col]].dropna().sort_values(col, ascending=False).head(3)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append({"method_id": row["method_id"], "family": row["family"], "component": col, "domain": domain, "rank": rank})
    top = pd.DataFrame(rows)
    counts = top.groupby(["method_id", "family"], observed=False).size().reset_index(name="top3_components")
    counts = counts.sort_values(["top3_components", "method_id"], ascending=[False, True]).head(10).iloc[::-1]
    colors = counts["family"].map(FAMILY_COLORS)
    ax.barh(counts["method_id"], counts["top3_components"], color=colors, height=0.62, alpha=0.90)
    ax.set_title("Top-three component frequency", fontsize=7.2, pad=3)
    ax.set_xlabel("components")
    ax.set_xlim(0, max(5, int(counts["top3_components"].max()) + 1))
    ax.tick_params(axis="y", labelsize=4.8, length=0, pad=1.2)
    ax.tick_params(axis="x", labelsize=5.4, length=2)
    ax.grid(axis="x", color="#E1E4E8", lw=0.45)
    counts.to_csv(SOURCE_DIR / "Figure_3_component_top3_frequency.csv", index=False)


def draw_contribution_panel(ax: plt.Axes, score: pd.DataFrame) -> None:
    long = domain_score_long(score)
    top_methods = score.sort_values("overall_mean", ascending=False)["method_id"].head(8).tolist()
    plot = long[long["method_id"].isin(top_methods)].copy()
    plot["method_id"] = pd.Categorical(plot["method_id"], categories=top_methods[::-1], ordered=True)
    plot = plot.sort_values(["method_id", "domain"])
    domain_order = ["Structure", "Clustering", "Efficiency", "Robustness"]
    colors = {
        "Structure": DOMAIN_COLORS["Structure preservation"],
        "Clustering": DOMAIN_COLORS["Clustering concordance"],
        "Efficiency": DOMAIN_COLORS["Efficiency"],
        "Robustness": DOMAIN_COLORS["Robustness"],
    }
    y_lookup = {method: i for i, method in enumerate(top_methods[::-1])}
    left = {method: 0.0 for method in top_methods}
    for domain in domain_order:
        sub = plot[plot["domain"].eq(domain)]
        for _, row in sub.iterrows():
            method = str(row["method_id"])
            ax.barh(
                y_lookup[method],
                row["weighted_contribution"],
                left=left[method],
                color=colors[domain],
                edgecolor="white",
                linewidth=0.35,
                height=0.62,
                label=domain,
            )
            left[method] += row["weighted_contribution"]
    ax.set_yticks(np.arange(len(top_methods)))
    ax.set_yticklabels(top_methods[::-1])
    ax.set_title("Weighted domain contributions", fontsize=7.2, pad=3)
    ax.set_xlabel("profile-score contribution")
    ax.set_xlim(0, max(0.78, max(left.values()) + 0.04))
    ax.tick_params(axis="y", labelsize=4.8, length=0, pad=1.2)
    ax.tick_params(axis="x", labelsize=5.4, length=2)
    ax.grid(axis="x", color="#E1E4E8", lw=0.45)
    plot.to_csv(SOURCE_DIR / "Figure_3_weighted_domain_contributions.csv", index=False)


def draw_raw_variability_panel(ax: plt.Axes, score: pd.DataFrame) -> None:
    summary = raw_variability_summary(score)
    order = CANONICAL_METHODS[::-1]
    plot_df = summary.set_index("method_id").reindex(order).reset_index()

    for y, row in plot_df.iterrows():
        if pd.isna(row["raw_median"]):
            continue
        color = FAMILY_COLORS.get(row["family"], "#777777")
        ax.hlines(y, row["raw_q1"], row["raw_q3"], color=color, lw=2.2, alpha=0.72)
        ax.scatter(
            row["raw_median"],
            y,
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["method_id"], fontsize=4.7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Normalized raw score", fontsize=6.4)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=5.7, length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E1E4E8", lw=0.45)
    ax.set_title("Raw-score variability", fontsize=7.2, pad=3)


def add_panel_labels(fig: plt.Figure, axes: list[tuple[str, plt.Axes]]) -> None:
    for label, ax in axes:
        x = -0.035 if label in {"a", "b", "h"} else -0.055
        ax.text(x, 1.004, label, transform=ax.transAxes, clip_on=False, **PANEL_LABEL_STYLE)


def draw_figure(score: pd.DataFrame, compact: bool = False) -> None:
    setup_mpl()
    if compact:
        fig_size = (7.25, 14.85)
        height_ratios = [6.60, 0.35, 1.64, 0.40, 1.84, 0.70, 1.45, 0.42, 1.80]
        content_rows = [0, 2, 4, 6, 8]
        nrows = 9
        top = 0.976
        bottom = 0.064
        hspace = 0.0
    else:
        fig_size = (7.25, 17.35)
        height_ratios = [8.50, 2.02, 2.24, 1.75, 2.25]
        content_rows = [0, 1, 2, 3, 4]
        nrows = 5
        top = 0.972
        bottom = 0.060
        hspace = 0.220

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=6,
        height_ratios=height_ratios,
        left=0.070,
        right=0.985,
        top=top,
        bottom=bottom,
        wspace=0.42,
        hspace=hspace,
    )
    ax_matrix = fig.add_subplot(gs[content_rows[0], :])
    ax_score_demo = fig.add_subplot(gs[content_rows[1], :])
    ax_rank = fig.add_subplot(gs[content_rows[2], 0:3])
    ax_domain = fig.add_subplot(gs[content_rows[2], 3:6])
    ax_family_domain = fig.add_subplot(gs[content_rows[3], 0:2])
    ax_component_frequency = fig.add_subplot(gs[content_rows[3], 2:4])
    ax_contribution = fig.add_subplot(gs[content_rows[3], 4:6])
    ax_variability = fig.add_subplot(gs[content_rows[4], :])

    draw_bubble_matrix(ax_matrix, score)
    draw_score_demo_panel(ax_score_demo, score)
    draw_overall_panel(ax_rank, score)
    draw_domain_panel(ax_domain, score)
    draw_family_domain_profile(ax_family_domain, score)
    draw_component_top_frequency(ax_component_frequency, score)
    draw_contribution_panel(ax_contribution, score)
    draw_raw_variability_panel(ax_variability, score)
    add_panel_labels(
        fig,
        [
            ("a", ax_matrix),
            ("b", ax_score_demo),
            ("c", ax_rank),
            ("d", ax_domain),
            ("e", ax_family_domain),
            ("f", ax_component_frequency),
            ("g", ax_contribution),
            ("h", ax_variability),
        ],
    )

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markeredgecolor="none", label=family, markersize=5.5)
        for family, color in FAMILY_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.013),
        ncol=4,
        fontsize=5.8,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    pd.DataFrame(
        [
            {
                "panel_count": 8,
                "layout": "compact" if compact else "full",
                "figure_width_in": fig_size[0],
                "figure_height_in": fig_size[1],
                "method_count": int(score["method_id"].nunique()),
                "component_count": len(MATRIX_COLUMNS),
                "top_profile_method": str(score.sort_values("overall_mean", ascending=False)["method_id"].iloc[0]),
                "top_profile_score": float(score["overall_mean"].max()),
            }
        ]
    ).to_csv(
        SOURCE_DIR / (
            "Figure_3_completed_compact_visual_qa_summary.csv"
            if compact
            else "Figure_3_completed_visual_qa_summary.csv"
        ),
        index=False,
    )

    stems = (
        ["Figure_3_vasc_completed_compact_8panel_trial"]
        if compact
        else [
            "Figure_3_vasc_completed_score_demo_top_tier",
            "Figure_3_vasc_completed_tight_top_tier",
            "Figure_3_vasc_completed_reviewer_optimized_trial",
            "Figure_3_vasc_completed_original_style_trial",
            "Figure_3_vasc_completed_trial",
        ]
    )
    for stem in stems:
        base = OUT_DIR / stem
        fig.savefig(base.with_suffix(".png"), dpi=450)
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def main() -> None:
    raw_all, stability_long = compute_stability_scores()
    score = load_score_matrix(stability_long)
    save_provenance(raw_all, stability_long, score)
    draw_figure(score, compact=True)

    vasc_summary = (
        stability_long[stability_long["Method"] == "VASC"]
        .sort_values("perturbation_axis")
        [["perturbation_axis", "score"]]
    )
    print("Completed VASC stability scores:")
    print(vasc_summary.to_string(index=False))
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
