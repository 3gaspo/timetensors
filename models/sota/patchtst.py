"""Compact PatchTST-style forecasting model.

This keeps the TimeTensor package self-contained: one file, PyTorch only, and
the same ``(batch, dim, lags) -> (batch, dim, horizon)`` contract as the other
models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Small PatchTST-style transformer over per-channel temporal patches."""

    def __init__(
        self,
        lags: int | None = None,
        dim: int = 1,
        horizon: int | None = None,
        *,
        context_window: int | None = None,
        target_window: int | None = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lags = int(lags if lags is not None else context_window)
        self.dim = int(dim)
        self.horizon = int(horizon if horizon is not None else target_window)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        if self.patch_len < 1 or self.stride < 1:
            raise ValueError("patch_len and stride must be positive")
        if self.lags < self.patch_len:
            raise ValueError("lags must be at least patch_len")
        self.patch_count = 1 + (self.lags - self.patch_len) // self.stride
        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.patch_count * d_model, self.horizon)

    def forward(self, x: torch.Tensor, covariates=None, **kwargs) -> torch.Tensor:
        del covariates, kwargs
        if x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(
                f"expected input shape (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch * self.dim, self.patch_count, self.patch_len)
        hidden = self.patch_projection(patches) + self.position
        encoded = self.encoder(hidden)
        out = self.head(encoded.flatten(start_dim=1))
        return out.reshape(batch, self.dim, self.horizon)
