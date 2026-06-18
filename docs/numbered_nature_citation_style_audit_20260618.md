# Numbered Nature Citation Style Audit

Date: 2026-06-18

## Scope

The manuscript citation system was converted from the previous `authoryear`/`elsarticle-harv` setup to a numbered Nature-style setup for the current Communications Biology revision package.

## LaTeX/PDF Changes

- Clean LaTeX source no longer uses the `authoryear` class option.
- In-text citations are configured with `\setcitestyle{numbers,super,sort&compress,open={},close={},citesep={,}}`.
- The bibliography style is now `\bibliographystyle{naturemag}`.
- The Nature bibliography style file `naturemag.bst` is included in the clean and redline LaTeX folders for reproducibility.
- Clean and redline PDFs were rebuilt successfully.

## Word Changes

- Word conversion now uses the CSL Nature style file `nature.csl`.
- The CSL file is stored in `02_manuscript/word/citation_styles/nature.csl`.
- Clean and redline Word manuscripts were regenerated and reformatted after citation conversion.
- Numbered reference lists are present in both Word files.

## Verification Summary

- Clean PDF compiled successfully: 59 pages.
- Redline PDF compiled successfully: 59 pages.
- Clean `.bbl` contains 74 numbered `\bibitem{...}` entries and 0 author-year optional `\bibitem[...]` entries.
- Redline `.bbl` contains 74 numbered `\bibitem{...}` entries and 0 author-year optional `\bibitem[...]` entries.
- Clean Word contains 9 embedded figures and the numbered reference list starts at paragraph 165.
- Redline Word contains 9 embedded figures and the numbered reference list starts at paragraph 166.
- Word formatting audit reports Times New Roman, 12 pt body text, justified text, double spacing, 1-inch margins, line numbering, page numbering and stable figure embedding.
- Online reference audit reports 82 BibTeX entries, 74 cited entries and no cited entries requiring action.

## Current Output Files

- Clean LaTeX: `02_manuscript/latex_clean/manuscript_final_clean.tex`
- Clean PDF: `02_manuscript/latex_clean/manuscript_final_clean.pdf`
- Redline LaTeX: `02_manuscript/latex_red/manuscript_final_red.tex`
- Redline PDF: `02_manuscript/latex_red/manuscript_final_red.pdf`
- Clean Word: `02_manuscript/word/manuscript_final_clean.docx`
- Redline Word: `02_manuscript/word/manuscript_final_red.docx`
