# scRNA-DRComparison

Companion code, figure, source-data and supplementary-table package for a systematic benchmark of dimensionality-reduction methods for single-cell RNA-seq analysis.

This repository extends the original `scRNA-DRComparison` project package into a manuscript-facing release: it keeps the study scope centered on 26 representative dimensionality-reduction methods, documents the full 100-dataset benchmark landscape, and provides final figures, source data, supplementary tables and scripts for inspecting the reported analyses.

## Study Overview

Dimensionality reduction is a central step in single-cell RNA-seq analysis, but method choice can affect visualization, clustering, structure preservation, robustness and computational efficiency. This benchmark evaluates representative methods across real and simulated single-cell settings and organizes the results into a structured evidence framework.

The benchmark has two analytical layers:

- **Full benchmark layer**: 26 prespecified dimensionality-reduction methods evaluated across the complete benchmark landscape.
- **Targeted sensitivity layer**: additional sensitivity and reference analyses, including latent-dimensionality checks, visualization-workflow controls, input-gene sensitivity and a targeted scVI reference analysis. These analyses support interpretation but do not redefine the 26-method benchmark scope.

The data landscape includes 50 real scRNA-seq datasets and 50 simulated datasets. The original public project README described downloadable processed datasets and full result archives through Figshare (`10.6084/m9.figshare.32064900`). This release focuses on the final figure/table source data, manuscript figures and reproducibility materials that accompany the benchmark.

## Repository Layout

```text
.
|-- README.md
|-- CODE_AND_DATA_NOTES.md
|-- FILE_MANIFEST.csv
|-- figures/
|   |-- main/
|   |   |-- png/
|   |   |-- pdf/
|   |   `-- svg/
|   `-- supplementary/
|       |-- png/
|       |-- pdf/
|       `-- svg/
|-- source_data/
|   |-- main_figures/
|   |-- supplementary_figures/
|   `-- supplementary_tables/
|-- tables/
|-- docs/
|-- code/
|   |-- main_figure_scripts/
|   |-- supplementary_figure_code/
|   |-- supplementary_table_code/
|   `-- targeted_sensitivity_scripts/
`-- config/
```

High-resolution TIFF files for journal submission are intentionally not stored in this GitHub-ready folder because several files exceed common GitHub size limits. They are retained in the full local materials package under `submission_high_resolution_figures/`.

## Main Figures

The final main figures are available in PNG, PDF and SVG formats under `figures/main/`.

| Figure | File prefix | Main evidence |
|---|---|---|
| Figure 1 | `Figure_1_conceptual_overview` | Conceptual overview and categories of the 26 dimensionality-reduction methods |
| Figure 2 | `Figure_2_workflow` | Benchmark workflow and dataset/metric organization |
| Figure 3 | `Figure_3_profile_score_audit` | Unified profile-score calculation and method-level performance audit |
| Figure 4 | `Figure_4_structure_preservation` | Local and global structure-preservation performance |
| Figure 5 | `Figure_5_clustering_concordance` | Clustering and annotation-concordance behavior |
| Figure 6 | `Figure_6_targeted_sensitivity_controls` | Targeted sensitivity controls and scVI reference analysis |
| Figure 7 | `Figure_7_simulated_robustness` | Robustness and stability across simulated perturbations |
| Figure 8 | `Figure_8_scalability_reproducibility` | Runtime, memory, completion and reproducibility audit |
| Figure 9 | `Figure_9_practical_selection` | Practical method-selection synthesis |

Source data used to assemble these figures are stored in `source_data/main_figures/`. Figure-generation scripts are stored in `code/main_figure_scripts/`.

## Supplementary Figures

The supplementary figure atlas is available in PNG, PDF and SVG formats under `figures/supplementary/`.

| Supplementary figure | File prefix | Purpose |
|---|---|---|
| S1 | `Supplementary_Figure_S1_method_catalog_atlas` | Method catalog, benchmark scope and implementation metadata |
| S2 | `Supplementary_Figure_S2_full_dataset_atlas` | Full 100-dataset benchmark landscape |
| S3 | `Supplementary_Figure_S3_real_dataset_landscape_atlas` | Real-dataset metadata landscape |
| S4 | `Supplementary_Figure_S4_simulated_parameter_landscape_atlas` | Simulated-parameter atlas |
| S5 | `Supplementary_Figure_S5_full_score_atlas` | Full profile-score atlas |
| S6 | `Supplementary_Figure_S6_local_structure_atlas` | Local-neighborhood structure preservation |
| S7 | `Supplementary_Figure_S7_label_local_atlas` | Trustworthiness, continuity and neighborhood metrics |
| S8 | `Supplementary_Figure_S8_global_geometry_atlas` | Global geometry and class-geometry metrics |
| S9 | `Supplementary_Figure_S9_clustering_atlas` | Clustering-concordance metrics |
| S10 | `Supplementary_Figure_S10_simulated_robustness_atlas` | Simulated robustness and stability |
| S11 | `Supplementary_Figure_S11_scVI_reference_atlas` | scVI targeted reference analysis |
| S12 | `Supplementary_Figure_S12_latent_dimension_atlas` | Latent-dimension sensitivity |
| S13 | `Supplementary_Figure_S13_workflow_atlas` | Visualization-workflow comparison |
| S14 | `Supplementary_Figure_S14_input_gene_atlas` | Input-gene sensitivity |
| S15 | `Supplementary_Figure_S15_reproducibility_coverage_atlas` | Reproducibility and evidence-layer coverage audit |

Detailed supplementary-figure descriptions are provided in `docs/Supplementary_Figures_Detailed_Descriptions_20260608.docx`. Source data for the supplementary figure atlas are stored in `source_data/supplementary_figures/`, and the atlas-generation scripts are stored in `code/supplementary_figure_code/`.

## Supplementary Tables

Machine-readable supplementary tables are provided as individual CSV files in `source_data/supplementary_tables/` and as compiled workbooks in `tables/`.

| Table | CSV file | Scope |
|---|---|---|
| S1 | `S1_MethodCatalog.csv` | Method catalog and family assignment |
| S2 | `S2_Dataset100Atlas.csv` | Full 100-dataset benchmark atlas |
| S3 | `S3_RealDatasetDetail.csv` | Real-dataset metadata details |
| S4 | `S4_SimulatedAtlas.csv` | Simulated benchmark atlas |
| S5 | `S5_MetricInventory.csv` | Metric inventory and score direction |
| S6 | `S6_ProfileScoreMatrix.csv` | Profile-score matrix across methods |
| S7 | `S7_StructureCoverage.csv` | Structure-preservation coverage |
| S8 | `S8_ClusteringCoverage.csv` | Clustering-concordance coverage |
| S9 | `S9_RobustnessStability.csv` | Robustness and stability coverage |
| S10 | `S10_ScalabilityAudit.csv` | Scalability and efficiency audit |
| S11 | `S11_TargetedSensitivity.csv` | Targeted sensitivity and reference results |
| S12 | `S12_SuppFigurePanelMap.csv` | Supplementary-figure panel map |
| S13 | `S13_SuppEvidenceCoverage.csv` | Supplementary evidence-layer coverage map |
| S14 | `S14_SourceFileManifest.csv` | Source-file manifest |
| S15 | `S15_DataRouteAudit.csv` | Data-route and reproducibility audit |

`tables/All_Supplementary_Tables_with_Introduction_20260608.xlsx` contains an introduction sheet, table index and S1-S15 tables in a single workbook.

## Reproducibility

The repository is organized for three levels of reproducibility.

1. **Inspect final evidence**: use `figures/`, `source_data/`, `tables/` and `docs/` to inspect the exact figure/table outputs and their source data.
2. **Regenerate figure and table assemblies**: use scripts in `code/main_figure_scripts/`, `code/supplementary_figure_code/` and `code/supplementary_table_code/` with the included source-data files. Some scripts preserve project-root path constants from the manuscript assembly workspace; update these paths before running them from a standalone clone.
3. **Rerun targeted sensitivity analyses**: use scripts in `code/targeted_sensitivity_scripts/` together with environment records in `config/benchmark_environments/`.

Minimal Python dependencies for figure/table assembly include `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `openpyxl` and `python-docx`. Some targeted method reruns require method-specific Python, R, Octave or legacy environments; these are documented in `config/benchmark_environments/`.

Install the minimal figure/table assembly dependencies with:

```bash
python -m pip install -r requirements.txt
```

Representative provenance scripts include:

```powershell
code/supplementary_table_code/make_final_supplementary_tables.py
code/supplementary_table_code/make_supplementary_table_review.py
code/supplementary_figure_code/make_supplementary_atlas_top_tier.py
```

For a standalone public rerun, first update the path constants in the relevant scripts so that they point to the local clone's `source_data/` and output folders. For a full rerun from raw or processed datasets, install the method-specific environments listed in `config/benchmark_environments/`.

## Data Availability Notes

This GitHub-ready folder contains figure source data, supplementary table CSVs, figure exports and analysis scripts. It does not contain the full preprocessed dataset archive or all high-dimensional intermediate embeddings.

The original project package reports that processed datasets and benchmark result archives are available through Figshare:

- `results.tar.gz`: dimensionality-reduction results, downsampling runtime/memory outputs and evaluation metrics.
- `datasets.tar.gz`: preprocessed datasets.
- DOI: `10.6084/m9.figshare.32064900`

Before public release, the authors should confirm the final Data Availability wording, repository URLs, accession numbers and license terms used in the manuscript.

## Citation

If you use this benchmark package, please cite the associated manuscript once the final citation is available. Until then, cite the repository and the Figshare dataset DOI where appropriate.

## License

The final software and data license should be selected by the authors before public release. Recommended choices are a permissive software license for code and a clear data license for reusable source-data tables, subject to the terms of the original public datasets.

## Contact

For questions about the benchmark design, method scope, source-data routing or manuscript figures, please contact the corresponding authors listed in the manuscript.
