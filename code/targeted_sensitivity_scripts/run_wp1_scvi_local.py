#!/usr/bin/env python3
"""Batch launcher for WP1 scVI local reruns.

Uses only datasets already present in the project `datasets/` directory.
Calls `run_scvi_embedding.py` for each dataset/dimension/seed.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_scvi_embedding.py"
PYTHON = "/home/saidi/anaconda3/envs/scrna-dr-py-modern-min/bin/python"

DATASETS = {
    "cell_100": ("datasets/downsampling/cell_100.csv", "datasets/downsampling/label/cell_100.csv"),
    "cell_500": ("datasets/downsampling/cell_500.csv", "datasets/downsampling/label/cell_500.csv"),
    "cell_1000": ("datasets/downsampling/cell_1000.csv", "datasets/downsampling/label/cell_1000.csv"),
    "cell_5000": ("datasets/downsampling/cell_5000.csv", "datasets/downsampling/label/cell_5000.csv"),
    "cell_10000": ("datasets/downsampling/cell_10000.csv", "datasets/downsampling/label/cell_10000.csv"),
    "cell_73233": ("datasets/downsampling/cell_73233.csv", "datasets/downsampling/label/cell_73233.csv"),
    "default": ("datasets/simulate/default/counts_matrix.csv", "datasets/simulate/default/cell_metadata.csv"),
    "celltype_7": ("datasets/simulate/celltype_7/counts_matrix.csv", "datasets/simulate/celltype_7/cell_metadata.csv"),
    "celltype_11": ("datasets/simulate/celltype_11/counts_matrix.csv", "datasets/simulate/celltype_11/cell_metadata.csv"),
    "celltype_15": ("datasets/simulate/celltype_15/counts_matrix.csv", "datasets/simulate/celltype_15/cell_metadata.csv"),
    "dropout_0": ("datasets/simulate/dropout_0/counts_matrix.csv", "datasets/simulate/dropout_0/cell_metadata.csv"),
    "dropout_2": ("datasets/simulate/dropout_2/counts_matrix.csv", "datasets/simulate/dropout_2/cell_metadata.csv"),
    "batch_0.2": ("datasets/simulate/batch_0.2/counts_matrix.csv", "datasets/simulate/batch_0.2/cell_metadata.csv"),
    "batch_1.0": ("datasets/simulate/batch_1.0/counts_matrix.csv", "datasets/simulate/batch_1.0/cell_metadata.csv"),
    "gene_5k": ("datasets/simulate/gene_5k/counts_matrix.csv", "datasets/simulate/gene_5k/cell_metadata.csv"),
    "gene_5w": ("datasets/simulate/gene_5w/counts_matrix.csv", "datasets/simulate/gene_5w/cell_metadata.csv"),
}

PRESETS = {
    "smoke": ["cell_100"],
    "core_small": ["default", "cell_1000"],
    "wp1_local": ["default", "celltype_7", "celltype_11", "celltype_15", "dropout_0", "dropout_2", "batch_0.2", "batch_1.0", "gene_5k", "gene_5w", "cell_1000", "cell_5000", "cell_10000", "cell_73233"],
}


def parse_csv(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def embedding_path(dataset: str, dim: int, seed: int) -> Path:
    return Path("revision_benchmark/results/embeddings/WP1_scVI_local") / dataset / "scVI" / f"dim_{dim}" / f"seed_{seed}.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="core_small")
    ap.add_argument("--datasets", help="Comma-separated dataset IDs; overrides --preset")
    ap.add_argument("--dimensions", default="2,5,10,20,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--failed-log", default="revision_benchmark/results/logs/WP1_scVI_local_failed_runs.jsonl")
    args = ap.parse_args()

    datasets = parse_csv(args.datasets) if args.datasets else PRESETS[args.preset]
    dims = parse_csv(args.dimensions, int)
    seeds = parse_csv(args.seeds, int)

    for ds in datasets:
        if ds not in DATASETS:
            raise SystemExit(f"Unknown dataset {ds}. Known: {', '.join(sorted(DATASETS))}")
        matrix, labels = DATASETS[ds]
        for dim in dims:
            for seed in seeds:
                out = embedding_path(ds, dim, seed)
                if args.skip_existing and out.exists():
                    print(f"SKIP existing {out}")
                    continue
                cmd = [
                    PYTHON, str(SCRIPT),
                    "--matrix", matrix,
                    "--labels", labels,
                    "--dataset-id", ds,
                    "--dimension", str(dim),
                    "--seed", str(seed),
                    "--max-epochs", str(args.max_epochs),
                ]
                print("RUN", " ".join(cmd), flush=True)
                if not args.dry_run:
                    started = time.time()
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as exc:
                        record = {
                            "dataset_id": ds,
                            "dimension": dim,
                            "seed": seed,
                            "max_epochs": args.max_epochs,
                            "returncode": exc.returncode,
                            "cmd": cmd,
                            "elapsed_seconds": time.time() - started,
                        }
                        failed_log = Path(args.failed_log)
                        failed_log.parent.mkdir(parents=True, exist_ok=True)
                        with failed_log.open("a") as fh:
                            fh.write(json.dumps(record, sort_keys=True) + "\n")
                        if not args.continue_on_error:
                            raise
                        print(f"FAILED but continuing: {record}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
