"""Thin adapter around the official Chronos-Bolt inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _import_pipeline():
    try:
        from chronos import BaseChronosPipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chronos-Bolt inference requires `chronos-forecasting`."
        ) from exc
    return BaseChronosPipeline


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    relative = Path("chronos-bolt-base")
    candidates = [
       project / "weights" / relative,
        project.parent / "weights" / relative,
        project.parents[2] / "weights" / relative,
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), None)


class ChronosBolt(nn.Module):
    """Official Chronos-Bolt median forecast with tensor translation only."""

    supports_context = False
    supports_covariates = False

    def __init__(
        self,
        lags: int,
        dim: int = 1,
        horizon: int | None = None,
        *,
        weights_path: str | Path | None = None,
        pretrained_path: str | Path | None = None,
        device: str | None = None,
        device_map: str = "cuda",
        local_files_only: bool = True,
        quantile_level: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.quantile_level = float(quantile_level)
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is None:
            raise FileNotFoundError(
                "Chronos-Bolt weights were not found; set weights_path or place "
                "the checkpoint in weights/chronos-bolt-base."
            )
        resolved_device = str(device or device_map)
        self.pipeline = _import_pipeline().from_pretrained(
            str(model_path),
            device_map=resolved_device,
            local_files_only=bool(local_files_only),
            **kwargs,
        )

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
        if covariates is not None and (
            past_covariates is not None or future_covariates is not None
        ):
            raise ValueError("provide structured or named covariates, not both")
        if covariates is not None:
            past_covariates = covariates.get("past")
            future_covariates = covariates.get("future")
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("Chronos-Bolt does not consume covariates")
        if context is not None:
            raise ValueError("Chronos-Bolt does not consume retrieval context")
        if x.ndim != 3 or x.shape[-1] != self.lags:
            raise ValueError(f"expected (batch, dim, {self.lags}), got {tuple(x.shape)}")
        batch, dim, _ = x.shape
        flat = x.reshape(batch * dim, self.lags)
        with torch.inference_mode():
            _, point = self.pipeline.predict_quantiles(
                inputs=flat.detach().cpu(),
                prediction_length=self.horizon,
                quantile_levels=[self.quantile_level],
            )
        prediction = torch.stack(point) if isinstance(point, list) else torch.as_tensor(point)
        return prediction.reshape(batch, dim, self.horizon).to(
            device=x.device,
            dtype=x.dtype,
        )


__all__ = ["ChronosBolt"]
