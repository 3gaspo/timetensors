"""Focused checks for single-pass aligned evaluation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import TimeSeriesData, fetch_training_data, save_data
from training.evaluate import build_loss_payload, eval_stage, merge_loss_payloads
from training.losses import get_losses
from training.pipeline import TorchLearner


class CountingPersistence(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        self.calls += 1
        return x[..., -1:].repeat(1, 1, self.horizon) + 0.0 * self.offset


def main() -> None:
    lags = 4
    horizon = 2
    dates = 40
    values = torch.arange(2 * dates, dtype=torch.float32).reshape(2, 1, dates)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "tiny"
        save_data(
            TimeSeriesData(
                values,
                list(range(dates)),
                individual_ids=[11, 29],
                individual_names=["alpha", "beta"],
            ),
            data_path,
        )
        loaders, _ = fetch_training_data(
            data_path,
            splits={"date_splits": [0.6, 0.2, 0.2], "indiv_split": 1.0},
            sampling={
                "train_idx_mode": "all",
                "eval_idx_mode": "all",
                "train_stride": 1,
                "eval_stride": 2,
                "shuffle_train": False,
                "shuffle_eval": False,
            },
            subsets={},
            batch_size=3,
            lags=lags,
            horizon=horizon,
            seed=7,
        )
        model = CountingPersistence(horizon)
        criterion, eval_losses = get_losses("mse", complete_evaluation=True)
        learner = TorchLearner(
            model,
            criterion,
            eval_losses=eval_losses,
            device="cpu",
        )
        valid_loader = loaders["valid1"]
        result = eval_stage(
            {
                "task": {"lags": lags, "horizon": horizon},
                "training": {"loss": "mse", "complete_evaluation": True},
                "evaluation": {"splits": ["valid1"], "runs": 1, "plot_example": True},
                "output": {"dir": str(root / "outputs"), "name": "single_pass"},
                "experiment": {"seed": 7},
            },
            learner=learner,
            loaders=loaders,
        )

        assert model.calls == len(valid_loader)
        payload = result["all_losses"]["valid1"]
        losses = payload["losses"]["mse"]
        metadata = payload["metadata"]
        assert losses.shape[0] == metadata["individual_ids"].numel()
        assert losses.shape[0] == metadata["query_ids"].numel()
        assert set(metadata["individual_ids"].tolist()) == {11, 29}
        assert metadata["individual_names"] == {"11": "alpha", "29": "beta"}
        assert torch.equal(metadata["run_ids"], torch.zeros(losses.shape[0], dtype=torch.int32))
        for metric in payload["losses"]:
            assert {
                metric,
                f"std_{metric}",
                f"user_{metric}",
                f"std_user_{metric}",
                f"w10_{metric}",
            } <= set(payload["summaries"])
        assert result["all_losses_path"].is_file()
        assert result["example_prediction_path"].is_file()
        assert not (result["all_losses_path"].parent / "per_user_all_losses.pt").exists()

        first_user = build_loss_payload(
            {
                "losses": {"mse": torch.tensor([1.0, 1.0, 1.0])},
                "metadata": {"individual_ids": torch.tensor([11, 11, 11])},
            }
        )
        second_user = build_loss_payload(
            {
                "losses": {"mse": torch.tensor([9.0])},
                "metadata": {"individual_ids": torch.tensor([29])},
            }
        )
        merged = merge_loss_payloads([first_user, second_user])
        summaries = merged["summaries"]
        assert torch.isclose(summaries["mse"], torch.tensor(3.0))
        assert torch.isclose(summaries["std_mse"], torch.sqrt(torch.tensor(12.0)))
        assert torch.isclose(summaries["user_mse"], torch.tensor(5.0))
        assert torch.isclose(summaries["std_user_mse"], torch.tensor(4.0))
        assert torch.isclose(summaries["w10_mse"], torch.tensor(9.0))

    print("test_evaluation: ok")


if __name__ == "__main__":
    main()
