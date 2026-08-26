"""Lightweight checks for experiment-table discovery and rendering."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from pipeline.runs import allocate_run, mark_status
from results.reporting import discover_results, generate_results_table


def _save_run(
    root: Path, dataset: str, setting: str, method: str, values: dict[int, float]
) -> None:
    lookback, horizon = map(int, setting.split("_"))
    identity = root / dataset / setting / "numpy_linear_proxy" / method
    allocation = allocate_run(
        identity,
        project="timetensors",
        workflow="test",
        dataset=dataset,
        lookback=lookback,
        horizon=horizon,
        backbone="numpy_linear_proxy",
        model_config_order=["method"],
        model_config={"method": method},
        pipeline_config={},
        seeds=list(values),
        display_name=method,
    )
    artifacts = []
    for seed, mse in values.items():
        relative = f"seed_{seed}/all_losses.pt"
        path = allocation.run_dir / relative
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "test1": {
                    "losses": {"mse": torch.tensor([mse, mse])},
                    "metadata": {},
                    "summaries": {},
                }
            },
            path,
        )
        mark_status(allocation.run_dir, "completed", seed=seed, required_artifacts=[relative])
        artifacts.append(relative)
    mark_status(allocation.run_dir, "completed", required_artifacts=artifacts)


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset in ("electricity", "traffic"):
            _save_run(root, dataset, "168_24", "reference", {1: 0.0012})
            candidate = {1: 0.0011, 2: 0.0007} if dataset == "electricity" else {1: 0.0009}
            _save_run(root, dataset, "168_24", "candidate", candidate)
        _save_run(root, "electricity", "672_168", "reference", {1: 120.0})
        _save_run(root, "electricity", "672_168", "candidate", {1: 100.0})

        records = discover_results(root)
        assert len(records) == 7
        output = generate_results_table(
            root,
            methods=["reference", "candidate"],
            reference="reference",
            dataset_settings={"electricity": {"168_24"}},
            show_std=True,
        )
        latex = output.read_text(encoding="utf-8")
        assert output.name == "results_mse.tex"
        assert r"$\times 10^{-3}$" in latex
        assert r"\textbf{0.90 " in latex
        assert r"\pm" in latex
        assert "25.00\\%" in latex
        assert "672--168" not in latex
        assert "traffic" in latex

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _save_run(root, "electricity", "504_168", "reference", {1: 0.0012})
        _save_run(root, "electricity", "504_168", "candidate", {1: 0.0009})
        output = generate_results_table(
            root,
            methods=["reference", "candidate"],
            reference="reference",
            show_std=True,
        )
        latex = output.read_text(encoding="utf-8")
        assert r"\pm" not in latex

    print("results table checks passed")


if __name__ == "__main__":
    main()
