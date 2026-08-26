"""Source-adapted DLinear forecasting architecture and local tensor wrapper.

Adapted from `cure-lab/LTSF-Linear` revision
`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`. Only the project tensor
layout and constructor/forward wrapper differ from the released model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Moving-average block used by the released series decomposition."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        return self.avg(torch.cat([front, x, end], dim=-1))


class SeriesDecomposition(nn.Module):
    """Separate a series into residual and moving-average components."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class DLinear(nn.Module):
    """Released DLinear heads over the local `(batch, dim, time)` layout."""

    def __init__(
        self,
        lags: int,
        dim: int,
        horizon: int,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.individual = bool(individual)
        self.decomposition = SeriesDecomposition(kernel_size)
        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
            )
            layers = [*self.linear_seasonal, *self.linear_trend]
        else:
            self.linear_seasonal = nn.Linear(self.lags, self.horizon)
            self.linear_trend = nn.Linear(self.lags, self.horizon)
            layers = [self.linear_seasonal, self.linear_trend]
        for layer in layers:
            layer.weight = nn.Parameter(
                torch.ones(self.horizon, self.lags) / self.lags
            )

    def forward(
        self,
        x: torch.Tensor,
        covariates: object | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del covariates, kwargs
        seasonal_init, trend_init = self.decomposition(x)
        if not self.individual:
            return self.linear_seasonal(seasonal_init) + self.linear_trend(trend_init)
        seasonal = x.new_empty(x.shape[0], self.dim, self.horizon)
        trend = torch.empty_like(seasonal)
        for index in range(self.dim):
            seasonal[:, index] = self.linear_seasonal[index](seasonal_init[:, index])
            trend[:, index] = self.linear_trend[index](trend_init[:, index])
        return seasonal + trend


__all__ = ["DLinear", "MovingAverage", "SeriesDecomposition"]
