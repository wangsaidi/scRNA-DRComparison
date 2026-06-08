# Code Directory

This directory contains scripts used to assemble the final manuscript figures, supplementary figures, supplementary tables and targeted sensitivity analyses.

## Subdirectories

- `main_figure_scripts/`: scripts used to assemble or polish the final main figures.
- `supplementary_figure_code/`: scripts used to assemble the supplementary figure atlas.
- `supplementary_table_code/`: scripts used to assemble supplementary tables, captions and table-review files.
- `targeted_sensitivity_scripts/`: scripts for targeted sensitivity analyses, including scVI reference analysis, latent-dimension sensitivity, visualization-workflow controls and input-gene sensitivity.

## Path Note

Some scripts preserve project-root paths from the manuscript assembly workspace. When running from a standalone GitHub clone, update the path constants near the top of the relevant script so that they point to the local clone's `source_data/`, `figures/`, `tables/` or desired output folders.

The scripts are retained to make the figure/table construction auditable. The final source data and rendered outputs are included in the repository for direct inspection.
