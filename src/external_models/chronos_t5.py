"""Thin adapter around the official Chronos-T5 sampling pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _import_pipeline():
    try:
        from chronos import ChronosPipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chronos-T5 inference requires `chronos-forecasting`."
        ) from exc
    return ChronosPipeline


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    relative = Path("chronos-t5-base")
    candidates = [
        project / "weights" / relative,
        project.parent / "weights" / relative,
        project.parents[2] / "weights" / relative,
    ]
    return next((path.resolve() for path in candidates if path.is_dir()), None)


class ChronosT5(nn.Module):
    """Chronos-T5 median forecast with tensor translation only."""

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
        num_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is None:
            raise FileNotFoundError(
                "Chronos-T5 weights were not found; set weights_path or place "
                "the checkpoint in weights/chronos-t5-base."
            )
        resolved_device = str(device or device_map)
        self.pipeline = _import_pipeline().from_pretrained(
            str(model_path),
            device_map=resolved_device,
            torch_dtype=torch.bfloat16 if "cuda" in resolved_device else torch.float32,
            local_files_only=bool(local_files_only),
            **kwargs,
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
        if covariates is not None and (
            past_covariates is not None or future_covariates is not None
        ):
            raise ValueError("provide structured or named covariates, not both")
        if covariates is not None:
            past_covariates = covariates.get("past")
            future_covariates = covariates.get("future")
        if past_covariates is not None or future_covariates is not None:
            raise ValueError("Chronos-T5 does not consume covariates")
        if context is not None:
            raise ValueError("Chronos-T5 does not consume retrieval context")
        if x.ndim != 3 or x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}"
            )
        batch, dim, _ = x.shape
        flat = x.reshape(batch * dim, self.lags)
        with torch.inference_mode():
            samples = self.pipeline.predict(
                flat.detach().cpu(),
                prediction_length=self.horizon,
                num_samples=self.num_samples,
            )
        sample_tensor = (
            torch.stack(samples) if isinstance(samples, list) else torch.as_tensor(samples)
        )
        point = sample_tensor.median(dim=1).values
        return point.reshape(batch, dim, self.horizon).to(
            device=x.device,
            dtype=x.dtype,
        )


__all__ = ["ChronosT5"]
