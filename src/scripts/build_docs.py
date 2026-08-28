"""Validate the first-read documentation and render its standalone PDFs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "latex"
REQUIRED = (
    ROOT / "README.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "experiment_catalog.md",
    ROOT / "docs" / "results_recap.md",
    LATEX / "method_overview.tex",
    LATEX / "method_overview.pdf",
    LATEX / "experiment_guideline.tex",
    LATEX / "executive_summary.tex",
)
README_LINKS = (
    "docs/architecture.md",
    "docs/experiment_catalog.md",
    "docs/results_recap.md",
    "latex/method_overview.pdf",
    "latex/experiment_guideline.pdf",
    "latex/executive_summary.pdf",
)
README_HEADINGS = (
    "## Documentation map",
    "## Setup",
    "## Main executions",
    "## Outputs and cluster operations",
    "## Documentation maintenance",
)
VIEW_LINKS = {
    ROOT / "docs" / "architecture.md": "../latex/method_overview.pdf",
    ROOT / "docs" / "experiment_catalog.md": "../latex/experiment_guideline.pdf",
    ROOT / "docs" / "results_recap.md": "../latex/executive_summary.pdf",
}


def dgx_fronts() -> list[Path]:
    nested = ROOT / "slurm" / "dgx"
    if nested.is_dir():
        return sorted(nested.rglob("*.slurm"))
    return sorted(
        path
        for path in ROOT.glob("*.slurm")
        if not path.stem.endswith("_selena")
    )


def validate() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing documentation files: {missing}")

    for path in (path for path in REQUIRED if path.suffix != ".pdf"):
        if path.read_text(encoding="utf-8").endswith("\n\n"):
            raise ValueError(f"{path.relative_to(ROOT)} has a blank line at EOF")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing_links = [link for link in README_LINKS if link not in readme]
    if missing_links:
        raise ValueError(f"README is missing first-read links: {missing_links}")
    readme_lines = readme.splitlines()
    headings = tuple(line for line in readme_lines if line.startswith("## "))
    if headings != README_HEADINGS:
        raise ValueError(f"README headings must be exactly {README_HEADINGS}")
    if len(readme_lines) > 160:
        raise ValueError("README exceeds the 160-line public quickstart limit")

    for path, link in VIEW_LINKS.items():
        if link not in path.read_text(encoding="utf-8"):
            raise ValueError(f"{path.relative_to(ROOT)} is missing owner link {link}")

    catalog = (ROOT / "docs" / "experiment_catalog.md").read_text(encoding="utf-8")
    undocumented = [
        path.relative_to(ROOT).as_posix()
        for path in dgx_fronts()
        if path.relative_to(ROOT).as_posix() not in catalog
    ]
    if undocumented:
        raise ValueError(f"Slurm fronts missing from experiment catalog: {undocumented}")

    stale = [
        path.relative_to(ROOT).as_posix()
        for suffix in (".aux", ".log", ".out", ".toc")
        for path in LATEX.glob(f"*{suffix}")
    ]
    if (ROOT / "$outDir").exists():
        stale.append("$outDir")
    if stale:
        raise ValueError(f"stale documentation artifacts must be removed: {stale}")


def render(names: tuple[str, ...]) -> None:
    executable = os.environ.get("PDFLATEX") or shutil.which("pdflatex")
    if executable is None:
        raise FileNotFoundError("pdflatex is required to render documentation PDFs")
    for name in names:
        source = LATEX / f"{name}.tex"
        for _ in range(2):
            subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", source.name],
                cwd=LATEX,
                check=True,
            )
        for suffix in (".aux", ".log", ".out", ".toc"):
            (LATEX / f"{name}{suffix}").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        choices=("method", "all"),
        help="render the method overview only, or every maintained PDF",
    )
    args = parser.parse_args()

    validate()
    if args.render == "method":
        render(("method_overview",))
    elif args.render == "all":
        render(("method_overview", "experiment_guideline", "executive_summary"))
    validate()
    print("Documentation contract is current.")


if __name__ == "__main__":
    main()
