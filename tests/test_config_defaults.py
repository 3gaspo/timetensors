"""Smoke-test canonical defaults used by the TimeTensor pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timetensors.models.losses import get_losses
from timetensors.models.pipeline import LearnerConfig
from timetensors.runtime import batch_size, default_sampling, model_specs


def main() -> None:
    criterion, eval_losses = get_losses()
    assert criterion.name == "nmse"
    assert {"mse", "nmse"} <= set(eval_losses)

    learner = LearnerConfig.from_dict({})
    assert learner.lr == 1e-5
    assert learner.epochs == 1

    assert batch_size({}) == 256

    sampling = default_sampling({})
    assert sampling["train_idx_mode"] == "random"
    assert sampling["eval_idx_mode"] == "all"
    assert sampling["train_stride"] == 1
    assert sampling["eval_stride"] == 1
    assert "reshuffle" not in sampling

    specs = model_specs({"model": {"path": "linear"}}, (8, 2, 3))
    assert specs.name == "linear"
    assert specs.path == "linear"
    assert specs.kwargs["lags"] == 8
    assert specs.kwargs["dim"] == 2
    assert specs.kwargs["horizon"] == 3

    print("test_config_defaults: ok")


if __name__ == "__main__":
    main()
