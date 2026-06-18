# Communications Biology Revision and Final-Submission Policy Compliance Audit

Date: 2026-06-18

Official guidance files used:
- Revision checklist PDF/text: `Publication/paper/submission_package_communications_biology_20260617/08_quality_control/journal_policy_checks_20260617/official_guidance/CommsBio-file-checklist-revision.pdf`
- Accepted-manuscript style guide PDF/text: `Publication/paper/submission_package_communications_biology_20260617/08_quality_control/journal_policy_checks_20260617/official_guidance/commsj-life-style-formatting-guide-accept.pdf`

Public availability links recorded in the manuscript and package:
- GitHub: https://github.com/wangsaidi/scRNA-DRComparison
- Figshare: https://doi.org/10.6084/m9.figshare.32064900

## Checklist

| Policy item | Status | Evidence | File or folder |
|---|---|---|---|
| Revision graph policy: mean/error-bar plots must show individual points or be converted to box/dot plots | PASS | Current main-figure scripts use heatmaps, scatter/dot plots, boxplots, violin plots and strip plots for quantitative panels. No errorbar/yerr/xerr/SEM pattern was detected in current main_figure_scripts. One fill_between hit is a non-SEM conceptual background curve in Fig. 3 score-construction schematic, not a single-point mean with error bars. | `Publication/paper/submission_package_communications_biology_20260617/07_source_data_and_code_availability/github_release/code/main_figure_scripts; distribution-style hits=110; fill_between hits=1` |
| Mandatory numerical source data for graphs and charts | PASS | Supplementary Data 1 workbook is present with 265 sheets and 263 manifest rows; external manifest has 263 files. Manuscript Data availability and Supplementary Information cite Supplementary Data 1. | `Publication/paper/submission_package_communications_biology_20260617/06_supplementary_tables/supplementary_data/Supplementary_Data_1_Source_Data.xlsx; Publication/paper/submission_package_communications_biology_20260617/06_supplementary_tables/supplementary_data/Supplementary_Data_1_Source_Data_manifest.csv` |
| Data deposition / public repository access | PASS | Data availability section cites the public GitHub repository, Figshare DOI, source accessions and Supplementary Table S15 data-route audit. GitHub and Figshare public URLs are also listed in the submission package README. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex; Publication/paper/submission_package_communications_biology_20260617/07_source_data_and_code_availability/PUBLIC_LINKS.md` |
| Human/animal research policy | PASS | Methods now include an Ethics and public-data use subsection stating that only public datasets and simulations were analyzed and no new human participants, human tissue, live animals or animal-derived samples were collected or generated. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex` |
| Final style guide: Abstract <=150 words | PASS | Current abstract word count is 145. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex` |
| Methods: separate Statistics and Reproducibility section | PASS | The clean manuscript now contains a dedicated Statistics and Reproducibility subsection defining n, analysis units, absence of wet-lab replicates, score/distribution summaries, missing-run handling and source-data location. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex` |
| Display-item count | PASS | The main submission folder contains Figures 1-9, within the Communications Biology revision checklist limit of up to 10 main display items. | `Publication/paper/submission_package_communications_biology_20260617/04_main_figures (TIFF=9, PNG=9, PDF=9, SVG=9)` |
| Supplementary Information and tables | PASS | Supplementary Information contains S1-S15 figure files, and supplementary tables/source data are provided as editable Excel/CSV files. Supplementary Data 1 is provided separately as a source-data workbook. | `Publication/paper/submission_package_communications_biology_20260617/05_supplementary_information (TIFF=15); Publication/paper/submission_package_communications_biology_20260617/06_supplementary_tables` |
| Word manuscript formatting for revision | PASS | Word formatting audit reports Times New Roman, 12 pt body text, justified body/caption/reference paragraphs, double spacing, 1-inch margins, continuous line numbering, footer page numbers and 9 embedded figures. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/word/manuscript_final_clean.docx; Publication/paper/submission_package_communications_biology_20260617/02_manuscript/word/manuscript_final_red.docx; Publication/paper/submission_package_communications_biology_20260617/08_quality_control/word_formatting_audit_20260617.md` |
| Citation/reference usability in Word | PASS | Word citation hyperlink audit reports numbered Nature-style bibliography items and preserves internal citation links generated by Pandoc/citeproc where present. | `Publication/paper/submission_package_communications_biology_20260617/08_quality_control/word_citation_hyperlink_audit_20260617.md` |
| Code availability and reproducibility | PASS | Code availability section cites the public GitHub repository; release package contains figure scripts, targeted sensitivity scripts, source-data tables, environment records and reproducibility guide. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex; Publication/paper/submission_package_communications_biology_20260617/07_source_data_and_code_availability/github_release` |
| Author contributions | PASS | The manuscript contains an Author contributions section using only initials that appear in the author list: S.W., D.J., M.C. and Y.F. | `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex` |

## File Integrity

- Clean LaTeX: `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_clean/manuscript_final_clean.tex`; SHA256 `8cdab80a5b70afd514331a72983287fbfe672ccaf2b3ff026eef1cbcba4ed69f`
- Redline LaTeX: `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/latex_red/manuscript_final_red.tex`; SHA256 `474b9c3db4132d3d4ec4550347efaa18c886dcf050a8f8172d451c3e03bfab4f`
- Clean Word: `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/word/manuscript_final_clean.docx`; SHA256 `f861d6e0790124a203991f5220ee7ae30588312d2296110ae1545d9c1e6ed3b5`
- Redline Word: `Publication/paper/submission_package_communications_biology_20260617/02_manuscript/word/manuscript_final_red.docx`; SHA256 `26281742e162da88e566fc54384ddc4e009f4190cf585dec90773731761db885`
- Supplementary Data 1: `Publication/paper/submission_package_communications_biology_20260617/06_supplementary_tables/supplementary_data/Supplementary_Data_1_Source_Data.xlsx`; SHA256 `9f6ed3ec5b50ebe5a801d24e048f884a71a7912e612d0d8324422e26b1c562d6`

## Notes

- The current revision manuscript uses numbered Nature-style citations and a numbered reference list in the LaTeX/PDF and Word outputs.
- Author contributions are now included in the manuscript and use only listed-author initials.
