"""Archived tensor adapter around the retired TiREx-2 package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _import_tirex2():
    try:
        from tirex2 import TimeseriesType, load_model  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TiRex-2 inference requires the official `tirex-2` package."
        ) from exc
    return load_model, TimeseriesType


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    relative = Path("tirex2")
    candidates = [
       project / "weights" / relative,
        project.parent / "weights" / relative,
        project.parents[2] / "weights" / relative,
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), None)


class TiRex2Forecaster(nn.Module):
    """Official TiRex-2 median forecast with tensor translation only."""

    supports_context = False
    supports_covariates = True

    def __init__(
        self,
        lags: int,
        dim: int = 1,
        horizon: int | None = None,
        *,
        weights_path: str | Path | None = None,
        pretrained_path: str | Path | None = None,
        device: str = "cuda",
        quantile_level: float = 0.5,
        **_: Any,
    ) -> None:
        super().__init__()
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.dim = int(dim)
        self.device_name = str(device)
        self.quantile_level = float(quantile_level)
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is None:
            raise FileNotFoundError(
                "TiRex-2 weights were not found; set weights_path or place "
                "model-config.yaml and model.ckpt in weights/tirex2."
            )
        missing = [
            name
            for name in ("model-config.yaml", "model.ckpt")
            if not (model_path / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"TiRex-2 checkpoint directory {model_path} is missing {', '.join(missing)}"
            )
        load_model, timeseries_type = _import_tirex2()
        self.timeseries_type = timeseries_type
        self.pipeline = load_model(str(model_path), device=device)
        quantiles = torch.as_tensor(self.pipeline.quantiles, dtype=torch.float64)
        matches = torch.nonzero(
            torch.isclose(
                quantiles,
                torch.tensor(self.quantile_level, dtype=torch.float64),
            ),
            as_tuple=False,
        ).flatten()
        if not len(matches):
            raise ValueError(
                f"TiRex-2 checkpoint does not expose quantile {self.quantile_level}"
            )
        self.quantile_index = int(matches[0])

    def forward(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if context is not None:
            raise ValueError("TiRex-2 does not consume retrieval context")
        if x.ndim != 3 or x.shape[-1] != self.lags:
            raise ValueError(f"expected (batch, dim, {self.lags}), got {tuple(x.shape)}")
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape[1]}")
        series = self._prepare_series(
            x,
            covariates,
            past_covariates,
            future_covariates,
        )
        with torch.inference_mode():
            forecasts = self.pipeline.forecast(
                series,
                prediction_length=self.horizon,
                output_type="torch",
                batch_size=len(series),
            )
        medians = [forecast[:, self.quantile_index, :] for forecast in forecasts]
        return torch.stack(medians).to(device=x.device, dtype=x.dtype)

    def _prepare_series(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None,
        past: torch.Tensor | None,
        future: torch.Tensor | None,
    ) -> list[Any]:
        if covariates is not None:
            if past is not None or future is not None:
                raise ValueError("provide structured or named covariates, not both")
            past = covariates.get("past")
            future = covariates.get("future")
        if (past is None) != (future is None):
            raise ValueError("TiRex-2 known covariates require both past and future values")
        if past is not None and future is not None and past.shape[:2] != future.shape[:2]:
            raise ValueError("past and future covariate channels do not match")
        known = None if past is None else torch.cat([past, future], dim=-1)
        return [
            self.timeseries_type(
                target=x[index].detach().cpu(),
                past_covariates=None,
                future_covariates=None if known is None else known[index].detach().cpu(),
            )
            for index in range(x.shape[0])
        ]


__all__ = ["TiRex2Forecaster"]
