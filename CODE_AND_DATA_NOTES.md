# Code and Data Notes

This file documents what is included in this GitHub-ready release and how it should be interpreted alongside the manuscript.

## Included Materials

- Final main figures in `figures/main/` as PNG, PDF and SVG.
- Final supplementary figures in `figures/supplementary/` as PNG, PDF and SVG.
- Source data for main and supplementary figures in `source_data/main_figures/` and `source_data/supplementary_figures/`.
- Supplementary table source CSVs in `source_data/supplementary_tables/`.
- Compiled supplementary table workbooks in `tables/`.
- Supplementary figure descriptions and table-caption files in `docs/`.
- Figure/table assembly scripts and targeted sensitivity scripts in `code/`.
- Environment records, package manifests and source-commit records in `config/`.

## Materials Not Included

- High-resolution TIFF files for journal submission. These are stored outside this GitHub-ready folder in the full materials package because several files are too large for routine GitHub storage.
- The full preprocessed dataset archive and all complete intermediate embeddings. The original project package reports these larger archives through Figshare DOI `10.6084/m9.figshare.32064900`.
- Controlled-access or private raw data. All reused real datasets should be cited through their original public accessions in the manuscript.

## Reproducibility Levels

### Level 1: Inspect the final evidence

Use the files in `figures/`, `source_data/`, `tables/` and `docs/` to inspect the plotted data and final visual materials.

### Level 2: Rebuild figure and table assemblies

Use:

- `code/main_figure_scripts/`
- `code/supplementary_figure_code/`
- `code/supplementary_table_code/`

These scripts use the included source-data files and regenerate the manuscript figure/table assemblies.

### Level 3: Rerun targeted sensitivity analyses

Use `code/targeted_sensitivity_scripts/` with the environment records in `config/`. Some methods require method-specific Python, R, Octave or legacy environments.

## Environment Notes

The package includes environment exports and package-freeze files for the analysis environments used during benchmark assembly. A single universal environment is not expected because the evaluated methods span multiple languages and software generations.

For figure/table assembly, start with:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl python-docx
```

For method reruns, use the environment records in `config/` and the method-specific notes in the relevant scripts.

## Scope Notes

- The 26-method benchmark refers to the prespecified full benchmark method set.
- scVI is included as a targeted reference analysis and sensitivity check, not as an additional member of the 26-method benchmark.
- Result variants, such as alternative display or parameter settings, are documented separately from the full benchmark method count.

## Public-Release Checklist

Before uploading this folder as a public GitHub repository, confirm:

1. The final software license and data license.
2. The final manuscript citation.
3. The public repository or archive DOI, if a citable software/data snapshot is minted.
4. The final Data Availability and Code Availability statements.
5. That no local-only raw data or controlled-access files have been copied into the release.
