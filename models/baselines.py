"""Small baseline forecasting models.

The modules accept ``covariates`` and arbitrary keyword arguments so they can
sit behind the generic wrapper without requiring special call paths.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceBaseline(nn.Module):
    """Repeat the last observed value over the horizon."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        return x[..., -1:].repeat_interleave(self.horizon, dim=-1)


class ExpectedBaseline(nn.Module):
    """Repeat the lookback mean over the horizon."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        return x.mean(dim=-1, keepdim=True).repeat_interleave(self.horizon, dim=-1)


class RepeatBaseline(nn.Module):
    """Repeat the last horizon-sized segment of the lookback."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        if x.shape[-1] < self.horizon:
            raise ValueError("lookback length must be at least horizon")
        return x[..., -self.horizon :]


class LookbackBaseline(nn.Module):
    """Return a fixed horizon-sized slice from the lookback."""

    def __init__(self, horizon: int, index: int = 0):
        super().__init__()
        self.horizon = int(horizon)
        self.index = int(index)

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        stop = self.index + self.horizon
        if self.index < 0 or stop > x.shape[-1]:
            raise ValueError("lookback slice is outside the input window")
        return x[..., self.index : stop]


class LinearBaseline(nn.Module):
    """Linear map from flattened lookback to flattened horizon."""

    def __init__(self, lags: int, dim: int, horizon: int):
        super().__init__()
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.linear = nn.Linear(self.lags * self.dim, self.horizon * self.dim)

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        if x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected input shape (batch, {self.dim}, {self.lags}), "
                f"got {tuple(x.shape)}"
            )
        y = self.linear(x.reshape(x.shape[0], self.dim * self.lags))
        return y.reshape(x.shape[0], self.dim, self.horizon)


class PeriodicLinearBaseline(nn.Module):
    """Linear map from explicit periodic history positions.

    For each forecast step, this model selects the same phase from previous
    periods. With ``period=24`` and ``horizon=6``, it learns from the values
    at the previous days' next six forecast phases. ``forecast_offset`` shifts
    the first forecast phase, and ``cycles`` limits the number of previous
    periods used. ``cycles=None`` uses every available matching position.
    """

    def __init__(
        self,
        lags: int,
        dim: int,
        horizon: int,
        period: int,
        forecast_offset: int = 0,
        cycles: int | None = None,
    ):
        super().__init__()
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.period = int(period)
        self.forecast_offset = int(forecast_offset)
        self.cycles = None if cycles is None else int(cycles)
        if self.period < 1:
            raise ValueError("period must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.cycles is not None and self.cycles < 1:
            raise ValueError("cycles must be positive or None")
        indices = self._build_indices()
        if not indices:
            raise ValueError("period selection produced no lookback positions")
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long))
        self.linear = nn.Linear(len(indices) * self.dim, self.horizon * self.dim)

    def _build_indices(self) -> list[int]:
        # Phase of the first forecast step just after the lookback window.
        first_future_phase = (self.lags + self.forecast_offset) % self.period
        phases = {
            (first_future_phase + step) % self.period
            for step in range(self.horizon)
        }
        candidates = [
            index for index in range(self.lags) if index % self.period in phases
        ]
        if self.cycles is None:
            return candidates
        max_points = self.cycles * min(self.horizon, self.period)
        return candidates[-max_points:]

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        selected = x.index_select(dim=-1, index=self.indices.to(x.device))
        y = self.linear(selected.reshape(x.shape[0], -1))
        return y.reshape(x.shape[0], self.dim, self.horizon)
