"""Smoke-test scikit-learn linear training on TimeTensor loaders."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.dataset import TimeSeriesData, fetch_training_data, save_data
from training.sklearn import train_sklearn_stage


def main() -> None:
    lags = 6
    horizon = 2
    dates = 50
    values = torch.arange(2 * 1 * dates, dtype=torch.float32).reshape(2, 1, dates)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "tiny"
        save_data(TimeSeriesData(values, list(range(dates))), data_path)
        loaders, stats = fetch_training_data(
            data_path,
            splits={"date_splits": [0.6, 0.2, 0.2], "indiv_split": 1.0},
            sampling={
                "train_idx_mode": "all",
                "eval_idx_mode": "all",
                "train_stride": 2,
                "eval_stride": 3,
                "shuffle_train": False,
                "shuffle_eval": False,
            },
            subsets={},
            batch_size=8,
            lags=lags,
            horizon=horizon,
            seed=3,
            compute_stats=True,
        )
        result = train_sklearn_stage(
            {
                "data": {"path": str(data_path)},
                "model": {"name": "sklinear"},
                "normalization": {"name": "instance"},
                "training": {"loss": "mse", "complete_evaluation": True},
                "evaluation": {"splits": ["valid1"]},
                "output": {"dir": str(root / "outputs"), "name": "sklinear"},
                "experiment": {"plot_weights": True},
                "sklearn": {"unroll_mode": "accessible"},
            },
            loaders=loaders,
            stats=stats,
        )

        assert result["state_path"].exists()
        assert result["weight_plot_paths"]["image"].exists()
        assert result["weight_plot_paths"]["image"].suffix == ".pdf"
        assert "valid1" in result["all_losses"]
        assert result["all_losses"]["valid1"]["mse"].numel() > 0

    print("test_sklearn: ok")


if __name__ == "__main__":
    main()
