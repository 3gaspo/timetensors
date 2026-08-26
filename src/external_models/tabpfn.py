"""Thin in-context adapter around the official TabPFN regressor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


def _import_regressor():
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TabPFN-TS inference requires `tabpfn`.") from exc
    return TabPFNRegressor


def _default_weights_path() -> Path | None:
    project = Path(__file__).resolve().parents[2]
    relative = Path("tabpfnts") / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
    candidates = [
       project / "weights" / relative,
        project.parent / "weights" / relative,
        project.parents[2] / "weights" / relative,
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


class TabPFNTS(nn.Module):
    """Fit TabPFN in context on each lookback and predict its horizon."""

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
        seasonal_periods: Sequence[float] = (24, 168),
        use_time_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.dim = int(dim)
        if self.dim != 1:
            raise ValueError("TabPFN-TS evaluates one target variate per instance")
        self.seasonal_periods = [float(period) for period in seasonal_periods]
        self.use_time_features = bool(use_time_features)
        selected = weights_path or pretrained_path
        model_path = (
            Path(selected).expanduser().resolve()
            if selected is not None
            else _default_weights_path()
        )
        if model_path is None:
            raise FileNotFoundError(
                "TabPFN weights were not found; set weights_path or place "
                "the v2.5 checkpoint in weights/tabpfnts/."
            )
        self.model = _import_regressor()(
            device=device,
            model_path=str(model_path),
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
        if context is not None:
            raise ValueError("TabPFN-TS does not consume retrieval context")
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != self.lags:
            raise ValueError(f"expected (batch, 1, {self.lags}), got {tuple(x.shape)}")
        if covariates is not None:
            if past_covariates is not None or future_covariates is not None:
                raise ValueError("provide structured or named covariates, not both")
            past_covariates = covariates.get("past")
            future_covariates = covariates.get("future")
        if not self.use_time_features and (
            past_covariates is None or future_covariates is None
        ):
            raise ValueError("disabling TabPFN time features requires full known covariates")

        predictions = []
        for index in range(x.shape[0]):
            past = None if past_covariates is None else past_covariates[index]
            future = None if future_covariates is None else future_covariates[index]
            train_features = self._features(
                0,
                self.lags,
                past,
                device=x.device,
                dtype=x.dtype,
            )
            test_features = self._features(
                self.lags,
                self.horizon,
                future,
                device=x.device,
                dtype=x.dtype,
            )
            self.model.fit(train_features, x[index, 0].detach().cpu().numpy())
            prediction = self.model.predict(test_features)
            predictions.append(
                torch.as_tensor(
                    prediction,
                    device=x.device,
                    dtype=x.dtype,
                ).view(1, 1, self.horizon)
            )
        return torch.cat(predictions, dim=0)

    def _features(
        self,
        start: int,
        length: int,
        covariates: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> np.ndarray:
        time_index = torch.arange(start, start + length, device=device, dtype=dtype)
        parts: list[torch.Tensor] = []
        if self.use_time_features:
            parts.append((time_index / max(self.lags, 1)).unsqueeze(1))
            for period in self.seasonal_periods:
                omega = 2 * np.pi / period
                parts.append(torch.sin(omega * time_index).unsqueeze(1))
                parts.append(torch.cos(omega * time_index).unsqueeze(1))
        if covariates is not None:
            if covariates.shape[-1] != length:
                raise ValueError("covariate length does not match TabPFN feature block")
            parts.append(covariates.transpose(0, 1))
        if not parts:
            raise ValueError("TabPFN requires time features or known covariates")
        return torch.cat(parts, dim=1).detach().cpu().numpy()


__all__ = ["TabPFNTS"]
