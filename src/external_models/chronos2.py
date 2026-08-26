"""Thin Chronos-2 adapter over the official standard inference pipeline."""

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
            "Chronos-2 evaluation requires `chronos-forecasting`."
        ) from exc
    return BaseChronosPipeline


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    candidates = [
       project / "weights" / "chronos2",
        project.parent / "weights" / "chronos2",
        project.parents[2] / "weights" / "chronos2",
    ]
    return next((path.resolve() for path in candidates if path.exists()), None)


class Chronos2(nn.Module):
    """Chronos-2 median forecaster with past and future-known covariates."""

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
        device: str | None = None,
        device_map: str = "cuda",
        local_files_only: bool = True,
        cross_learning: bool = False,
        quantile_index: int | None = None,
        **_: Any,
    ):
        super().__init__()
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.cross_learning = bool(cross_learning)
        self.quantile_index = quantile_index
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is None:
            raise FileNotFoundError(
                "Chronos-2 weights were not found; set model.weights_path or "
                "place them in weights/chronos2."
            )
        self.pipeline = _import_pipeline().from_pretrained(
            str(model_path),
            device_map=str(device or device_map),
            local_files_only=bool(local_files_only),
        )
        model = getattr(self.pipeline, "model", None)
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

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
            raise ValueError("Chronos-2 does not consume retrieval context")
        if x.ndim != 3 or x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}"
            )
        if covariates is not None:
            if past_covariates is not None or future_covariates is not None:
                raise ValueError("provide structured or named covariates, not both")
            past_covariates = covariates.get("past")
            future_covariates = covariates.get("future")
        structured = {"past": past_covariates, "future": future_covariates}
        predictions = self.pipeline.predict(
            inputs=self._prepare_inputs(x, structured),
            prediction_length=self.horizon,
            cross_learning=self.cross_learning,
        )
        medians = []
        for prediction in predictions:
            quantile = (
                prediction.shape[1] // 2
                if self.quantile_index is None
                else int(self.quantile_index)
            )
            medians.append(prediction[:, quantile, :])
        return torch.stack(medians).to(device=x.device, dtype=x.dtype)

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None,
    ) -> list[dict[str, Any]]:
        past = None if covariates is None else covariates.get("past")
        future = None if covariates is None else covariates.get("future")
        items: list[dict[str, Any]] = []
        for batch_index in range(x.shape[0]):
            item: dict[str, Any] = {"target": x[batch_index].detach().cpu()}
            past_values: dict[str, torch.Tensor] = {}
            future_values: dict[str, torch.Tensor] = {}
            if past is not None:
                past_values.update(
                    {
                        f"covariate_{channel}": past[batch_index, channel].detach().cpu()
                        for channel in range(past.shape[1])
                    }
                )
            if future is not None:
                future_values.update(
                    {
                        f"covariate_{channel}": future[batch_index, channel].detach().cpu()
                        for channel in range(future.shape[1])
                    }
                )
            if past_values:
                item["past_covariates"] = past_values
            if future_values:
                item["future_covariates"] = future_values
            items.append(item)
        return items
