"""Smoke-test dataset saving, loader construction, metadata, and stats."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (
    SamplerConfig,
    TimeSeriesData,
    TimeSeriesDataset,
    fetch_training_data,
    save_data,
)


def main() -> None:
    lags = 4
    horizon = 2
    dates = 60
    values = torch.arange(3 * 1 * dates, dtype=torch.float32).reshape(3, 1, dates)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "tiny"
        artifacts = root / "artifacts"
        save_data(TimeSeriesData(values, list(range(dates))), data_path)

        loaders, stats = fetch_training_data(
            data_path,
            splits={"date_splits": [0.6, 0.2, 0.2], "indiv_split": 1.0},
            sampling={
                "train_idx_mode": "random",
                "eval_idx_mode": "all",
                "train_stride": 1,
                "eval_stride": 2,
                "shuffle_train": False,
                "shuffle_eval": False,
            },
            subsets={},
            batch_size=256,
            lags=lags,
            horizon=horizon,
            seed=7,
            stats_save_path=artifacts,
            compute_stats=True,
            stats_max_windows=10,
            stats_seed=7,
        )

        assert {"train", "valid1", "test1"} <= set(loaders)
        assert set(stats) == set(loaders)

        train_loader = loaders["train"]
        assert len(train_loader) == 1
        batch = next(iter(train_loader))
        assert batch["inputs"].shape[-1] == lags
        assert batch["targets"].shape[-1] == horizon

        valid_loader = loaders["valid1"]
        valid_sampler = valid_loader.dataset.index_sampler
        assert valid_sampler.config.idx_mode == "all"
        assert valid_sampler.config.stride == 2
        assert valid_sampler.date_candidates == [35, 37, 39, 41, 43, 45]
        valid_sample = valid_loader.dataset[0]
        torch.testing.assert_close(
            valid_sample["inputs"],
            torch.tensor([[[32.0, 33.0, 34.0, 35.0]]]),
        )
        torch.testing.assert_close(
            valid_sample["targets"],
            torch.tensor([[[36.0, 37.0]]]),
        )
        assert valid_sample["metadata"]["query_indices"].tolist() == [35]
        assert valid_sample["metadata"]["target_date_ids"].tolist() == [[36, 37]]
        shorter = TimeSeriesDataset(
            valid_loader.dataset.data,
            lags=2,
            horizon=horizon,
            sampler_config=SamplerConfig(idx_mode="all", stride=2),
        )
        assert shorter.index_sampler.date_candidates == valid_sampler.date_candidates
        torch.testing.assert_close(shorter[0]["targets"], valid_sample["targets"])
        assert len(valid_loader) == math.ceil(len(valid_loader.dataset) / 256)

        metadata_path = artifacts / "metadata.json"
        stats_path = artifacts / "stats.json"
        assert metadata_path.exists()
        assert stats_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["task"] == {"lags": lags, "horizon": horizon}
        assert metadata["loaders"]["valid1"]["mode"] == "all"
        assert metadata["loaders"]["valid1"]["stride"] == 2
        assert metadata["loaders"]["valid1"]["batches"] == len(valid_loader)

        constant_values = values.clone()
        constant_values[0] = 0
        constant_path = root / "constant"
        save_data(TimeSeriesData(constant_values, list(range(dates))), constant_path)
        filtered, filtered_stats = fetch_training_data(
            constant_path,
            splits={"date_splits": [0.6, 0.2, 0.2], "indiv_split": 1.0},
            sampling={
                "train_idx_mode": "random",
                "eval_idx_mode": "all",
                "drop_train_constant_individuals": True,
                "drop_eval_constant_individuals": False,
            },
            subsets={},
            batch_size=256,
            lags=lags,
            horizon=horizon,
            compute_stats=True,
        )
        assert filtered["train"].dataset.data.individuals == 2
        assert filtered["valid1"].dataset.data.individuals == 3
        assert filtered_stats["train"]["lookback_value_min"] <= filtered_stats["train"]["lookback_value_max"]

    print("test_dataloaders: ok")


if __name__ == "__main__":
    main()
