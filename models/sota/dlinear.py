"""DLinear forecasting model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        return self.avg(torch.cat([front, x, end], dim=-1))


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    """Decomposition-linear baseline from the DLinear family."""

    def __init__(self, lags: int, dim: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.decomposition = SeriesDecomposition(kernel_size)
        self.linear_seasonal = nn.ModuleList(
            [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
        )
        self.linear_trend = nn.ModuleList(
            [nn.Linear(self.lags, self.horizon) for _ in range(self.dim)]
        )

    def forward(self, x: torch.Tensor, covariates=None, **kwargs) -> torch.Tensor:
        del covariates, kwargs
        seasonal_init, trend_init = self.decomposition(x)
        seasonal = torch.empty(
            x.shape[0], self.dim, self.horizon, dtype=x.dtype, device=x.device
        )
        trend = torch.empty_like(seasonal)
        for index in range(self.dim):
            seasonal[:, index, :] = self.linear_seasonal[index](seasonal_init[:, index, :])
            trend[:, index, :] = self.linear_trend[index](trend_init[:, index, :])
        return seasonal + trend
