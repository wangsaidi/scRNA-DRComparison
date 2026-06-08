# Reproducibility Guide

This guide describes the intended rerun levels for the benchmark companion repository.

## 1. Inspect Final Outputs

Use these files when the goal is to inspect the evidence supporting the manuscript:

- `figures/main/`
- `figures/supplementary/`
- `source_data/main_figures/`
- `source_data/supplementary_figures/`
- `source_data/supplementary_tables/`
- `tables/All_Supplementary_Tables_with_Introduction_20260608.xlsx`

This level does not require rerunning the dimensionality-reduction methods.

## 2. Rebuild Figure and Table Assemblies

Install the minimal Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The scripts in this release preserve the project-root layout used during manuscript assembly. Before running them from a standalone GitHub clone, update path constants so that they point to the local clone's `source_data/` and output directories.

Representative provenance scripts include:

```bash
code/supplementary_table_code/make_final_supplementary_tables.py
code/supplementary_table_code/make_supplementary_table_review.py
code/supplementary_figure_code/make_supplementary_atlas_top_tier.py
```

Main-figure scripts are stored in `code/main_figure_scripts/`. Each script is paired with source-data files under `source_data/main_figures/`.

## 3. Rerun Targeted Sensitivity Analyses

Targeted sensitivity scripts are stored in `code/targeted_sensitivity_scripts/`. These analyses include scVI reference analysis, latent-dimension sensitivity, visualization-workflow controls and input-gene sensitivity.

Use the environment records in `config/benchmark_environments/` to identify method-specific dependencies. A single universal environment is not expected because the benchmark spans multiple software ecosystems.

## 4. Rerun the Full Benchmark

The full benchmark requires the complete processed dataset archive and method-specific environments. The original project package reports the processed dataset and result archives through Figshare DOI `10.6084/m9.figshare.32064900`.

Before a full rerun, confirm:

- dataset archive location and checksums;
- method-specific software versions;
- local paths for input matrices and metadata;
- compute resources for large datasets;
- final Data Availability and Code Availability wording.
