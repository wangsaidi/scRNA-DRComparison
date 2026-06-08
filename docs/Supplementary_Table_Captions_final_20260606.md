# Supplementary Table Captions

One-sentence argument: The supplementary tables make the benchmark auditable by linking method scope, the 100-dataset atlas, metric construction, figure source data, targeted sensitivity experiments and data/code availability routes to explicit source evidence.

## Ready-To-Paste Captions

### Supplementary Table S1. Dimensionality-reduction method catalogue and benchmark scope

Catalogue of dimensionality-reduction methods included in the benchmark. The table lists the 26 full-benchmark methods, their method families, implementation principles, implementation languages, software/source links and references, and separately marks result variants and the targeted scVI reference analysis that are not counted as part of the 26-method benchmark.

Linked evidence: Figure 1; Supplementary Figure S1.

### Supplementary Table S2. Manuscript 100-dataset atlas

Atlas of the 100 datasets used to frame the benchmark landscape. The table records 50 real datasets and 50 simulated datasets, including dataset labels, source metadata, cell counts, sequencing or simulation descriptors, and dataset-counting rules used in the manuscript.

Linked evidence: Figure 2; Supplementary Figure S2.

### Supplementary Table S3. Fifty-real-dataset metadata concordance table

Concordance table for the 50 real datasets in the manuscript atlas. Each row is anchored to the real-dataset backbone in Supplementary Table S2 and reports available metadata for source repository, species, tissue or cell type, condition, cell counts, sequencing technology, detailed gene/sparsity/cell-type fields where available, and metadata-status notes.

Linked evidence: Figure 2; Supplementary Figure S3.

### Supplementary Table S4. Simulated-dataset parameter atlas

Parameter atlas for the 50 simulated datasets used in the benchmark. The table reports simulation axes and parameter values for cell number, gene number, cell-type number, dropout, batch number, batch strength, differential-expression probability, differential-expression strength and outlier settings.

Linked evidence: Figure 2; Supplementary Figures S4 and S10.

### Supplementary Table S5. Benchmark metric inventory and profile-score components

Inventory of benchmark metrics and profile-score components. The table defines score domains, local and global structure-preservation metrics, clustering-concordance metrics, raw metric direction and score weights used to construct the profile-score summaries.

Linked evidence: Figures 3-5; Supplementary Figures S5-S9.

### Supplementary Table S6. Completed per-method profile-score matrix

Completed profile-score matrix for the 26 full-benchmark methods. The table reports local and global structure scores, clustering scores, efficiency scores, stability components and the overall mean profile score after completing the VASC and stability-score audit.

Linked evidence: Figure 3; Supplementary Figure S5.

### Supplementary Table S7. Structure-preservation metric coverage summary

Coverage summary for local and global structure-preservation metrics. For each source collection and metric, the table reports the number of datasets, number of methods, record counts and quartile summaries of metric values used to support Figure 4 and the structure-preservation supplementary atlas.

Linked evidence: Figure 4; Supplementary Figures S6-S8.

### Supplementary Table S8. Clustering-concordance metric coverage summary

Coverage summary for clustering-concordance metrics. The table summarizes adjusted Rand index, normalized mutual information, homogeneity, completeness and silhouette scores across k-means, Louvain and spectral clustering for each source collection.

Linked evidence: Figure 5; Supplementary Figure S9.

### Supplementary Table S9. Simulated robustness and stability score audit

Per-method robustness and stability audit across simulated perturbation axes. The table reports method-level scores for perturbations in cell number, gene number, cell-type number, dropout, batch number, batch strength, differential-expression probability, differential-expression strength and outlier settings, together with the stability median.

Linked evidence: Figure 7; Supplementary Figure S10.

### Supplementary Table S10. Scalability, completion and implementation audit

Audit of scalability, completion status and implementation verification for the 26 full-benchmark methods. The table records completion across cell-number scales, missing cell levels, largest completed scale, software environment, installation status, runtime summaries and peak-memory summaries.

Linked evidence: Figure 8; Supplementary Figure S15.

### Supplementary Table S11. Targeted sensitivity experiment manifest

Manifest and QA summary for targeted sensitivity experiments. The table documents the scVI reference analysis, latent-dimension sensitivity, visualization-workflow comparison and input-gene sensitivity analyses, including source-record counts, method scopes, dataset scopes, parameter scopes and Figure 6 QA fields.

Linked evidence: Figure 6; Supplementary Figures S11-S14.

### Supplementary Table S12. Supplementary figure panel map

Panel-level map of the Supplementary Figure atlas. The table assigns each panel in Supplementary Figures S1-S15 to its analytical role and source-data layer, allowing readers to trace each supplementary panel to its supporting evidence.

Linked evidence: Supplementary Figures S1-S15.

### Supplementary Table S13. Supplementary evidence-layer coverage map

Coverage map for the metric-level evidence layers represented in the Supplementary Figure atlas. The table records the supplementary figure, evidence layer, coverage scope, number of evidence units and reader-use note for each major structure, clustering and perturbation analysis layer.

Linked evidence: Supplementary Figures S6-S10.

### Supplementary Table S14. Source-data file manifest

Manifest of source-data files used to generate the final figures and supplementary tables. For each file, the table records its source role, relative package path, row count, column count and representative columns, providing an auditable route from source data to displayed results.

Linked evidence: All figures and supplementary tables.

### Supplementary Table S15. Data availability and source-data route audit

Audit of data and output classes, access routes, package locations and remaining availability actions. The table separates method metadata, dataset metadata, benchmark outputs, targeted sensitivity experiments, reproduction code and source-file manifests to support the final Data Availability and Code Availability statements.

Linked evidence: Data Availability; all source data.

## Chinese Author Notes

- S1 明确 26 个 full-benchmark methods，同时把 scVI 标为 targeted reference analysis，避免读者误解 26-method benchmark scope。
- S2/S3/S4 共同支撑 100-dataset atlas，其中 S3 已经按 50 个真实数据集重新对齐。
- S5/S6 是 Figure 3 分数计算和最终 score matrix 的表格证据。
- S7/S8/S9/S10 对应结构保持、聚类一致性、模拟稳定性和计算效率，支撑正文 Figure 4/5/7/8。
- S11 服务 targeted sensitivity experiments 和 scVI reference analysis；S12/S13 说明 Supplementary Figure atlas 的 panel-level 与 evidence-layer coverage；S14/S15 服务 source data 和 Data Availability。