"""Smoke-test canonical model loading and wrapper behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from timetensors.models.models import ModelConfig, load_model


def _config(name: str, **kwargs) -> ModelConfig:
    base_kwargs = {"lags": 8, "dim": 2, "horizon": 3}
    base_kwargs.update(kwargs)
    return ModelConfig(
        name=name,
        path=name,
        kwargs=base_kwargs,
        normalization={"name": "identity", "kwargs": {}},
    )


def _assert_forward(name: str, **kwargs) -> None:
    model = load_model(_config(name, **kwargs))
    x = torch.randn(4, 2, 8)
    y = model(x)
    assert tuple(y.shape) == (4, 2, 3)


def main() -> None:
    _assert_forward("linear")
    _assert_forward("dlinear", kernel_size=3)
    _assert_forward(
        "patchtst",
        patch_len=4,
        stride=2,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )

    covariate_model = load_model(
        ModelConfig(
            name="linear_covariates",
            path="linear",
            kwargs={"lags": 8, "dim": 2, "horizon": 3},
            normalization={"name": "identity", "kwargs": {}},
            covariate_augmentation={
                "modes": [
                    {"name": "noise", "count": 1, "value": 0.01, "target": "past_only"},
                    {
                        "name": "constant",
                        "count": 1,
                        "value": 1.0,
                        "target": "future_included",
                    },
                ]
            },
        )
    )
    assert tuple(covariate_model(torch.randn(2, 2, 8)).shape) == (2, 2, 3)

    try:
        load_model(_config("DLinear"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate/case alias 'DLinear' should not resolve")

    print("test_models: ok")


if __name__ == "__main__":
    main()
