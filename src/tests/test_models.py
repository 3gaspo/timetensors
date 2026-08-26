"""Smoke-test canonical model loading and wrapper behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_loading.forecast import (
    BASELINE_REGISTRY,
    FOUNDATION_MODEL_ALIASES,
    ModelConfig,
    load_model,
)
from model_loading.baselines import PeriodicLinearBaseline
from proposal import GRevIN, RevIN, build_grevin_normalization, build_normalization
from pipeline.runtime import model_specs


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


def _assert_generated_config_forward(name: str, shape: tuple[int, int, int]) -> None:
    lags, dim, horizon = shape
    specs = model_specs(
        {
            "model": {"name": name, "path": name},
            "normalization": {"name": "identity"},
        },
        shape,
    )
    model = load_model(specs)
    y = model(torch.randn(2, dim, lags))
    assert tuple(y.shape) == (2, dim, horizon)


def main() -> None:
    assert isinstance(
        build_normalization({"name": "revin", "kwargs": {}}, dim=2),
        RevIN,
    )
    assert isinstance(build_grevin_normalization("grevin", 2), GRevIN)
    _assert_forward("linear")
    _assert_forward("periodic_linear", period=4)
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

    periodic = PeriodicLinearBaseline(lags=336, dim=1, horizon=24, period=168)
    assert periodic.indices_by_horizon[0] == [0, 168]
    assert periodic.indices_by_horizon[23] == [23, 191]

    benchmark_models = [
        "persistence",
        "expected",
        "repeat",
        "lookback",
        "linear",
        "periodic_linear",
        "dlinear",
        "patchtst",
    ]
    for shape in [(168, 2, 24), (672, 2, 168), (24, 2, 24), (1344, 2, 336)]:
        for name in benchmark_models:
            _assert_generated_config_forward(name, shape)

    shared_foundation = FOUNDATION_MODEL_ALIASES
    for name in shared_foundation:
        specs = model_specs({"model": {"name": name, "path": name}}, (512, 1, 64))
        assert specs.kwargs is not None
        kwargs = dict(specs.kwargs)
        assert (kwargs["lags"], kwargs["dim"], kwargs["horizon"]) == (512, 1, 64)
        assert name in BASELINE_REGISTRY
    assert shared_foundation == (
        "chronos2",
        "chronos_bolt",
        "ts_icl",
        "tirex2",
        "tabpfn_ts",
    )
    for removed in (
        "chronos",
        "chronos-2",
        "chronos-bolt",
        "tsicl",
        "ts-icl",
        "tirex_2",
        "tirex-2",
        "tyrex2",
        "tabpfn",
        "tabpfn-ts",
    ):
        assert removed not in BASELINE_REGISTRY
        try:
            load_model(_config(removed))
        except ValueError:
            pass
        else:
            raise AssertionError(f"removed alias {removed!r} should not resolve")
    try:
        load_model(
            ModelConfig(
                name="custom_chronos",
                path="external_models.chronos2.Chronos2",
                kwargs={"lags": 8, "dim": 1, "horizon": 3},
            )
        )
    except ValueError as error:
        assert "canonical alias" in str(error)
    else:
        raise AssertionError("foundation import paths must not bypass canonical aliases")

    try:
        load_model(_config("DLinear"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate/case alias 'DLinear' should not resolve")

    print("test_models: ok")


if __name__ == "__main__":
    main()
