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

    def weight_matrix(self) -> torch.Tensor:
        """Return weights as ``(output_dim, horizon, input_dim, lag)``."""
        return self.linear.weight.detach().reshape(
            self.dim,
            self.horizon,
            self.dim,
            self.lags,
        )


class PeriodicLinearBaseline(nn.Module):
    """Horizon-specific linear maps from matching periodic history positions.

    Forecast step ``h`` sees only lookback positions with the same phase as
    absolute future time ``lags + forecast_offset + h`` modulo ``period``.
    ``cycles`` limits the number of previous matching positions per horizon.
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
        indices_by_horizon = self._build_indices()
        if any(not indices for indices in indices_by_horizon):
            raise ValueError("period selection produced an empty horizon input")
        self._indices_by_horizon = [tuple(indices) for indices in indices_by_horizon]
        for horizon_index, indices in enumerate(indices_by_horizon):
            self.register_buffer(
                f"indices_{horizon_index}",
                torch.as_tensor(indices, dtype=torch.long),
            )
        self.linears = nn.ModuleList(
            [nn.Linear(len(indices) * self.dim, self.dim) for indices in indices_by_horizon]
        )

    @property
    def indices_by_horizon(self) -> list[list[int]]:
        return [list(indices) for indices in self._indices_by_horizon]

    def _build_indices(self) -> list[list[int]]:
        indices_by_horizon = []
        for horizon_index in range(self.horizon):
            future_phase = (
                self.lags + self.forecast_offset + horizon_index
            ) % self.period
            indices = [
                index for index in range(self.lags) if index % self.period == future_phase
            ]
            if self.cycles is not None:
                indices = indices[-self.cycles :]
            indices_by_horizon.append(indices)
        return indices_by_horizon

    def forward(
        self,
        x: torch.Tensor,
        covariates: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del covariates, kwargs
        outputs = []
        for horizon_index, linear in enumerate(self.linears):
            indices = getattr(self, f"indices_{horizon_index}").to(x.device)
            selected = x.index_select(dim=-1, index=indices)
            outputs.append(linear(selected.reshape(x.shape[0], -1)).unsqueeze(-1))
        return torch.cat(outputs, dim=-1)

    def weight_matrix(self) -> torch.Tensor:
        """Return sparse weights as ``(output_dim, horizon, input_dim, lag)``."""
        weights = torch.zeros(
            self.dim,
            self.horizon,
            self.dim,
            self.lags,
            dtype=self.linears[0].weight.dtype,
            device=self.linears[0].weight.device,
        )
        for horizon_index, linear in enumerate(self.linears):
            indices = getattr(self, f"indices_{horizon_index}")
            head = linear.weight.detach().reshape(self.dim, self.dim, len(indices))
            weights[:, horizon_index, :, indices] = head
        return weights.detach()
