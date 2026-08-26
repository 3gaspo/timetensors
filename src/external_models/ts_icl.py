"""Thin tensor adapter around the official TS-ICL package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _import_tsicl():
    try:
        from tsicl import TSICL  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TS-ICL inference requires the official `tsicl` package."
        ) from exc
    return TSICL


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    relative = Path("tsicl") / "tsicl-v1.ckpt"
    candidates = [
       project / "weights" / relative,
        project.parent / "weights" / relative,
        project.parents[2] / "weights" / relative,
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


class TSICLForecaster(nn.Module):
    """Official TS-ICL median forecast with tensor translation only."""

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
        local_files_only: bool = True,
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
        if self.quantile_level != 0.5:
            raise ValueError("TS-ICL point evaluation requires quantile_level=0.5")
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is not None and not model_path.is_file():
            raise FileNotFoundError(f"TS-ICL checkpoint is not a file: {model_path}")
        if model_path is None and local_files_only:
            raise FileNotFoundError(
                "TS-ICL weights were not found; set weights_path or place "
                "tsicl-v1.ckpt in weights/tsicl."
            )
        self.pipeline = _import_tsicl()(
            model_path=None if model_path is None else str(model_path),
            allow_auto_download=not bool(local_files_only),
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
        if context is not None:
            raise ValueError("TS-ICL does not consume retrieval context")
        if x.ndim != 3 or x.shape[-1] != self.lags:
            raise ValueError(f"expected (batch, dim, {self.lags}), got {tuple(x.shape)}")
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape[1]}")
        known_covariates = self._known_covariates(
            covariates,
            past_covariates,
            future_covariates,
        )
        with torch.inference_mode():
            point, _ = self.pipeline.forecast(
                inputs=x.transpose(1, 2),
                covars=known_covariates,
                prediction_length=self.horizon,
                batch_size=x.shape[0],
                quantile_levels=[self.quantile_level],
                context_length=self.lags,
                device=x.device,
                denormalize=True,
                point_estimator="median",
                allow_auto_complete=False,
                allow_covar_forecast=False,
                squeeze_output=False,
            )
        return point.squeeze(-1).to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _known_covariates(
        covariates: dict[str, torch.Tensor | None] | None,
        past: torch.Tensor | None,
        future: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if covariates is not None:
            if past is not None or future is not None:
                raise ValueError("provide structured or named covariates, not both")
            past = covariates.get("past")
            future = covariates.get("future")
        if past is None and future is None:
            return None
        if past is None or future is None:
            raise ValueError("TS-ICL known covariates require both past and future values")
        if past.shape[:2] != future.shape[:2]:
            raise ValueError("past and future covariate channels do not match")
        return torch.cat([past, future], dim=-1).transpose(1, 2)


__all__ = ["TSICLForecaster"]
