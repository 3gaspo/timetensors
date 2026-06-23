"""Lightweight checks for experiment-table discovery and rendering."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from timetensors.results_table import discover_results, generate_results_table


def _save(root: Path, dataset: str, setting: str, method: str, mse: float) -> None:
    directory = root / dataset / setting / method
    directory.mkdir(parents=True)
    torch.save({"test1": {"mse": torch.tensor([mse, mse])}}, directory / "all_losses.pt")


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset in ("electricity", "traffic"):
            _save(root, dataset, "168_24", "reference", 0.0012)
            _save(root, dataset, "168_24", "candidate", 0.0009)
        _save(root, "electricity", "672_168", "reference", 120.0)
        _save(root, "electricity", "672_168", "candidate", 100.0)

        records = discover_results(root)
        assert len(records) == 6
        output = generate_results_table(
            root,
            methods=["reference", "candidate"],
            reference="reference",
            dataset_settings={"electricity": {"168_24"}},
        )
        latex = output.read_text(encoding="utf-8")
        assert output.name == "results_mse.tex"
        assert r"$\times 10^{-3}$" in latex
        assert r"\textbf{0.90}" in latex
        assert "25.00\\%" in latex
        assert "672--168" not in latex
        assert "traffic" in latex

    print("results table checks passed")


if __name__ == "__main__":
    main()
