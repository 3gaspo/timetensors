"""Smoke-test canonical defaults used by the TimeTensor pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.runtime import batch_size, default_sampling, model_specs
from data.io import fetch_csv
from data.load import _merge_dataset_config
from training.losses import get_losses
from training.pipeline import LearnerConfig


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tiny.csv").write_text("date,a\n2026-01-01,1\n2026-01-02,\n")
        values, _, _ = fetch_csv(root, "tiny", date_col="date")
        assert float(values.iloc[1, 0]) == 0.0
        try:
            fetch_csv(root, "tiny", date_col="date", missing_values="error")
        except ValueError as error:
            assert "missing values" in str(error)
        else:
            raise AssertionError("strict missing-value policy must reject NaNs")
        (root / "tiny.csv").write_text("date,a\n2026-01-01,1\n2026-01-02,inf\n")
        try:
            fetch_csv(root, "tiny", date_col="date")
        except ValueError as error:
            assert "infinite values" in str(error)
        else:
            raise AssertionError("CSV infinities must be rejected")

    criterion, eval_losses = get_losses()
    assert criterion.name == "nmse"
    assert {"mse", "nmse"} <= set(eval_losses)

    learner = LearnerConfig.from_dict({})
    assert learner.lr == 1e-5
    assert learner.epochs == 200

    assert batch_size({}) == 256

    sampling = default_sampling({})
    assert sampling["train_idx_mode"] == "random"
    assert sampling["eval_idx_mode"] == "all"
    assert sampling["train_stride"] == 1
    assert sampling["eval_stride"] == 1
    assert "reshuffle" not in sampling
    assert sampling["drop_train_constant_individuals"] is False

    _, complete_losses = get_losses("relative_mse", complete_evaluation=True)
    assert "relative_mse" in complete_losses

    specs = model_specs({"model": {"path": "linear"}}, (8, 2, 3))
    assert specs.name == "linear"
    assert specs.path == "linear"
    assert specs.kwargs["lags"] == 8
    assert specs.kwargs["dim"] == 2
    assert specs.kwargs["horizon"] == 3

    specs = model_specs({"model": {"path": "periodic_linear"}}, (672, 2, 168))
    assert specs.kwargs["period"] == 168

    specs = model_specs({"model": {"path": "periodic_linear"}}, (24, 2, 24))
    assert specs.kwargs["period"] == 24

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "drop_users": [1],
                    "date_col": "shared_date",
                    "timetensors": {"drop_users": [2], "date_col": "project_date"},
                }
            ),
            encoding="utf-8",
        )
        merged, provenance = _merge_dataset_config(
            {"raw_path": str(root / "dataset.csv"), "drop_users": [3], "date_col": "run_date"}
        )
        assert merged["drop_users"] == [3]
        assert merged["date_col"] == "run_date"
        assert merged["config_path"] == str(root / "config.json")
        assert provenance["effective_drop_users"] == [3]

    print("test_config_defaults: ok")


if __name__ == "__main__":
    main()
