from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from update_manuscript_from_current_tex_20260616 import make_word_friendly_tex, prefix_word_captions


ROOT = Path.cwd()
PACKAGE = ROOT / "Publication" / "paper" / "submission_package_communications_biology_20260617"
FINAL = ROOT / "Publication" / "paper" / "final_materials_package_20260608_clean_figlegends"
CLEAN_DIR = PACKAGE / "02_manuscript" / "latex_clean"
RED_DIR = PACKAGE / "02_manuscript" / "latex_red"
WORD_DIR = PACKAGE / "02_manuscript" / "word"
QA_DIR = PACKAGE / "08_quality_control" / "journal_policy_checks_20260617"
TRUE_REDLINE_DIR = ROOT / "Publication" / "paper" / "manuscript_revision_true_redline_20260617"

PANDOC = Path(r"C:\Users\tjwan\AppData\Local\Pandoc\pandoc.exe")
PDFLATEX = Path(r"C:\Users\tjwan\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe")
BIBTEX = Path(r"C:\Users\tjwan\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe")


def run(cmd: list[str], cwd: Path, log_name: str) -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (QA_DIR / log_name).write_text(proc.stdout, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}; see {QA_DIR / log_name}")


def compile_latex(tex_dir: Path, tex_name: str, prefix: str) -> None:
    aux = Path(tex_name).with_suffix(".aux").name
    steps = [
        ([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", tex_name], f"{prefix}_pdflatex_1.log"),
        ([str(BIBTEX), aux], f"{prefix}_bibtex.log"),
        ([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", tex_name], f"{prefix}_pdflatex_2.log"),
        ([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", tex_name], f"{prefix}_pdflatex_3.log"),
    ]
    for cmd, log_name in steps:
        run(cmd, tex_dir, log_name)


def convert_clean_word() -> None:
    tex = (CLEAN_DIR / "manuscript_final_clean.tex").read_text(encoding="utf-8")
    word_tex = make_word_friendly_tex(tex)
    word_tex_path = CLEAN_DIR / "manuscript_final_clean_for_word.tex"
    word_tex_path.write_text(word_tex, encoding="utf-8")
    out_docx = WORD_DIR / "manuscript_final_clean.docx"
    run(
        [
            str(PANDOC),
            word_tex_path.name,
            "--bibliography=references.bib",
            "--citeproc",
            "--resource-path=.;figures",
            "-o",
            str(out_docx),
        ],
        CLEAN_DIR,
        "policy_update_clean_pandoc_docx.log",
    )
    prefix_word_captions(out_docx)


def refresh_true_redline() -> None:
    run(["python", str(ROOT / "Publication" / "paper" / "manuscript_revision_scripts" / "make_true_redline_20260617.py")], ROOT, "policy_update_true_redline.log")
    shutil.copy2(TRUE_REDLINE_DIR / "manuscript_final_red_true_changes.tex", RED_DIR / "manuscript_final_red.tex")
    shutil.copy2(TRUE_REDLINE_DIR / "manuscript_final_red_true_changes.pdf", RED_DIR / "manuscript_final_red.pdf")
    shutil.copy2(TRUE_REDLINE_DIR / "manuscript_final_red_true_changes.docx", WORD_DIR / "manuscript_final_red.docx")


def sync_release_documents() -> None:
    final_docs = FINAL / "submission_documents"
    final_docs.mkdir(parents=True, exist_ok=True)
    for src in [
        CLEAN_DIR / "manuscript_final_clean.tex",
        CLEAN_DIR / "manuscript_final_clean.pdf",
        RED_DIR / "manuscript_final_red.tex",
        RED_DIR / "manuscript_final_red.pdf",
        WORD_DIR / "manuscript_final_clean.docx",
        WORD_DIR / "manuscript_final_red.docx",
    ]:
        shutil.copy2(src, final_docs / src.name)


def main() -> None:
    WORD_DIR.mkdir(parents=True, exist_ok=True)
    RED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEAN_DIR / "references.bib", RED_DIR / "references.bib")
    for bst in ["elsarticle-harv.bst", "elsarticle-num.bst", "elsarticle-num-names.bst"]:
        if (CLEAN_DIR / bst).exists():
            shutil.copy2(CLEAN_DIR / bst, RED_DIR / bst)

    compile_latex(CLEAN_DIR, "manuscript_final_clean.tex", "policy_update_clean")
    convert_clean_word()
    refresh_true_redline()
    sync_release_documents()
    print("rebuilt policy-updated manuscript package")


if __name__ == "__main__":
    main()
