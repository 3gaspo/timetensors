"""TabPFN forecasting wrapper.

This adapts TabPFN's tabular regressor to the TimeTensor forecasting contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_tabpfn():
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TabPFN support requires the optional dependency `tabpfn`. "
            "Install the SOTA extras or add it to the environment before "
            "using model.name=tabpfn."
        ) from exc
    return TabPFNRegressor


def _existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _default_weights_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    return _existing_path(
        repo_root.parent
        / "weights"
        / "tabpfnts"
        / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    )


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return values - values.mean() if values.size else values
    index = np.arange(values.size, dtype=np.float64)
    coeffs = np.polyfit(index, values, 1, rcond=None)
    return values - np.polyval(coeffs, index)


def _detect_dominant_periods(
    values: np.ndarray,
    *,
    max_top_k: int,
    magnitude_threshold: float | None = 0.05,
    zero_padding_factor: int = 2,
) -> list[float]:
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size < 3 or max_top_k <= 0:
        return []

    original_length = series.size
    series = _linear_detrend(series)
    series = series * np.hanning(original_length)
    padded_length = max(original_length, int(original_length * max(1, zero_padding_factor)))
    if padded_length > original_length:
        padded = np.zeros(padded_length, dtype=np.float64)
        padded[:original_length] = series
        series = padded

    magnitudes = np.abs(np.fft.rfft(series))
    if magnitudes.size <= 1:
        return []
    magnitudes[0] = 0.0
    freqs = np.fft.rfftfreq(series.size, d=1.0)

    threshold = None
    if magnitude_threshold is not None:
        threshold = float(magnitude_threshold) * float(np.max(magnitudes))

    if magnitudes.size > 2:
        peak_indices = np.flatnonzero(
            (magnitudes[1:-1] >= magnitudes[:-2])
            & (magnitudes[1:-1] >= magnitudes[2:])
        ) + 1
    else:
        peak_indices = np.arange(1, magnitudes.size)
    if peak_indices.size == 0:
        peak_indices = np.arange(1, magnitudes.size)
    if threshold is not None:
        peak_indices = peak_indices[magnitudes[peak_indices] >= threshold]
    if peak_indices.size == 0:
        return []

    sorted_indices = peak_indices[np.argsort(magnitudes[peak_indices])[::-1]]
    periods: list[float] = []
    seen: set[float] = set()
    for index in sorted_indices:
        freq = freqs[index]
        if freq <= 0:
            continue
        period = float(round(1.0 / freq))
        if period <= 0 or period in seen:
            continue
        periods.append(period)
        seen.add(period)
        if len(periods) >= max_top_k:
            break
    return periods


def _default_seasonal_periods(lags: int) -> list[float]:
    periods = []
    if lags > 24:
        periods.append(24.0)
    if lags > 168:
        periods.append(168.0)
    return periods


def _unique_periods(periods: list[float] | tuple[float, ...]) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for period in periods:
        value = float(period)
        if value <= 0 or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


class TabPFN(nn.Module):
    """TabPFN regressor converted into a time-series forecaster."""

    def __init__(
        self,
        lags: int,
        dim: int = 1,
        horizon: int | None = None,
        *,
        context_mode: str = "structured",
        seasonal_periods: list[int] | tuple[int, ...] | None = None,
        cross_learning: bool = False,
        dimension_encoding: str = "ordinal",
        context_as_features: bool = True,
        use_time_features: bool = True,
        auto_detect_frequencies: bool = True,
        auto_frequency_top_k: int = 5,
        auto_frequency_threshold: float | None = 0.05,
        device: str = "cuda",
        weights_path: str | Path | None = None,
        shared_context: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        if horizon is None:
            raise ValueError("TabPFN requires horizon")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.context_mode = str(context_mode)
        self.device = str(device)
        self.cross_learning = bool(cross_learning)
        self.dimension_encoding = str(dimension_encoding)
        self.context_as_features = bool(context_as_features)
        self.shared_context = bool(shared_context)
        self.use_time_features = bool(use_time_features)
        self.auto_detect_frequencies = bool(auto_detect_frequencies)
        self.auto_frequency_top_k = int(auto_frequency_top_k)
        self.auto_frequency_threshold = auto_frequency_threshold

        if seasonal_periods is None:
            self.seasonal_periods = _default_seasonal_periods(self.lags)
        else:
            self.seasonal_periods = _unique_periods([float(period) for period in seasonal_periods])

        model_path = Path(weights_path).expanduser().resolve() if weights_path is not None else _default_weights_path()
        if model_path is None:
            raise FileNotFoundError(
                "TabPFN weights were not found. Pass model.kwargs.weights_path "
                "or place the checkpoint under ../weights/tabpfnts/."
            )

        regressor = _import_tabpfn()
        self.model = regressor(device=self.device, model_path=str(model_path), **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if x.ndim != 3:
            raise ValueError(f"expected x with shape (batch, dim, lags), got {tuple(x.shape)}")
        if x.shape[-1] != self.lags:
            raise ValueError(f"expected lags={self.lags}, got {x.shape[-1]}")

        batch_size, dim, lags = x.shape
        past_context, future_context = self._select_context(covariates)

        if self.cross_learning:
            x_train, y_train, x_test = self._prepare_matrix(
                x=x,
                past_context=past_context,
                future_context=future_context,
            )
            self.model.fit(x_train, y_train)
            flat = self.model.predict(x_test)
            return torch.as_tensor(flat, device=x.device, dtype=x.dtype).reshape(
                batch_size,
                dim,
                self.horizon,
            )

        predictions = []
        for index in range(batch_size):
            past_i = self._sample_context(past_context, index, batch_size)
            future_i = self._sample_context(future_context, index, batch_size)
            x_train, y_train, x_test = self._prepare_matrix(
                x=x[index].unsqueeze(0),
                past_context=past_i,
                future_context=future_i,
            )
            self.model.fit(x_train, y_train)
            flat = self.model.predict(x_test)
            predictions.append(
                torch.as_tensor(flat, device=x.device, dtype=x.dtype).reshape(
                    1,
                    dim,
                    self.horizon,
                )
            )
        return torch.cat(predictions, dim=0)

    def _generate_time_features(
        self,
        values: torch.Tensor,
        window_length: int,
    ) -> torch.Tensor | None:
        if not self.use_time_features:
            return None
        if values.ndim != 3:
            raise ValueError(f"expected values with shape (batch, dim, time), got {tuple(values.shape)}")
        batch_size, dim, lookback = values.shape
        device = values.device
        dtype = values.dtype
        time_index = torch.arange(window_length, device=device, dtype=dtype)
        base_parts = [(time_index / max(lookback, 1)).view(1, 1, window_length, 1)]
        for period in self.seasonal_periods:
            omega = 2 * np.pi / period
            base_parts.append(torch.sin(omega * time_index).view(1, 1, window_length, 1))
            base_parts.append(torch.cos(omega * time_index).view(1, 1, window_length, 1))
        base = torch.cat(base_parts, dim=-1).expand(batch_size, dim, window_length, -1)
        if not self.auto_detect_frequencies or self.auto_frequency_top_k <= 0:
            return base

        auto = torch.zeros(
            batch_size,
            dim,
            window_length,
            2 * self.auto_frequency_top_k,
            device=device,
            dtype=dtype,
        )
        values_np = values.detach().cpu().numpy()
        for batch_index in range(batch_size):
            for dim_index in range(dim):
                periods = _detect_dominant_periods(
                    values_np[batch_index, dim_index],
                    max_top_k=self.auto_frequency_top_k,
                    magnitude_threshold=self.auto_frequency_threshold,
                )
                for period_index, period in enumerate(periods[: self.auto_frequency_top_k]):
                    omega = 2 * np.pi / period
                    auto[batch_index, dim_index, :, 2 * period_index] = torch.sin(omega * time_index)
                    auto[batch_index, dim_index, :, 2 * period_index + 1] = torch.cos(omega * time_index)
        return torch.cat([base, auto], dim=-1)

    def _select_context(
        self,
        covariates: dict[str, torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if covariates is None:
            return None, None
        past = covariates.get("past")
        future = covariates.get("future")
        static = covariates.get("static")
        if static is not None:
            static_past = static.expand(*static.shape[:-1], self.lags)
            static_future = static.expand(*static.shape[:-1], self.horizon)
            past = static_past if past is None else torch.cat([past, static_past], dim=1)
            future = static_future if future is None else torch.cat([future, static_future], dim=1)
        if self.context_mode == "past_only":
            return past, None
        if self.context_mode == "structured":
            return past, self._join_future_context(past, future)
        if self.context_mode == "future_included":
            return None, self._join_future_context(past, future)
        raise ValueError(f"unknown context_mode={self.context_mode!r}")

    def _join_future_context(
        self,
        past: torch.Tensor | None,
        future: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if future is None:
            return None
        if past is not None and past.shape[:-1] == future.shape[:-1]:
            prefix = past
        else:
            prefix = torch.zeros(
                *future.shape[:-1],
                self.lags,
                device=future.device,
                dtype=future.dtype,
            )
        return torch.cat([prefix, future], dim=-1)

    def _sample_context(
        self,
        context: torch.Tensor | None,
        index: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        if context is None or self.shared_context or context.shape[0] != batch_size:
            return context
        return context[index].unsqueeze(0)

    def _append_identity_features(
        self,
        features: torch.Tensor,
        *,
        batch_size: int,
        dim: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_offset: int = 0,
        dim_offset: int = 0,
        total_batch_classes: int = 1,
        total_dim_classes: int = 1,
    ) -> torch.Tensor:
        batch_ids = (torch.arange(batch_size, device=device) + batch_offset).view(
            batch_size,
            1,
            1,
        )
        dim_ids = (torch.arange(dim, device=device) + dim_offset).view(1, dim, 1)

        if self.dimension_encoding == "ordinal":
            batch_feature = batch_ids.to(dtype).expand(batch_size, dim, length).unsqueeze(-1)
            dim_feature = dim_ids.to(dtype).expand(batch_size, dim, length).unsqueeze(-1)
            return torch.cat([features, batch_feature, dim_feature], dim=-1)

        if self.dimension_encoding == "one-hot":
            series_id = batch_ids * total_dim_classes + dim_ids
            classes = total_batch_classes * total_dim_classes
            one_hot = F.one_hot(
                series_id.expand(batch_size, dim, length),
                num_classes=classes,
            ).to(dtype)
            return torch.cat([features, one_hot], dim=-1)

        if self.dimension_encoding == "categorical":
            raise NotImplementedError("dimension_encoding='categorical' is not implemented")
        raise ValueError(f"unknown dimension_encoding={self.dimension_encoding!r}")

    def _create_tabular_block(
        self,
        values: torch.Tensor,
        time_features: torch.Tensor | None,
        *,
        context_values: torch.Tensor | None = None,
        start_index: int = 0,
        batch_offset: int = 0,
        dim_offset: int = 0,
        total_batch_classes: int = 1,
        total_dim_classes: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim, length = values.shape
        feature_parts = []
        if time_features is not None:
            if time_features.ndim == 4:
                subset = time_features[:, :, start_index : start_index + length, :]
            else:
                subset = time_features[start_index : start_index + length]
            if subset.ndim == 2:
                subset = subset.view(1, 1, length, -1).expand(batch_size, dim, length, -1)
            feature_parts.append(subset)
        if self.context_as_features and context_values is not None:
            context_batch, context_dim, context_length = context_values.shape
            if context_length != length:
                raise ValueError(f"context length mismatch: expected {length}, got {context_length}")
            context_features = context_values.permute(2, 0, 1).reshape(
                length,
                context_batch * context_dim,
            )
            feature_parts.append(
                context_features.view(1, 1, length, -1).expand(batch_size, dim, length, -1)
            )
        if not feature_parts:
            raise ValueError("TabPFN needs time features or context_as_features inputs")
        features = torch.cat(feature_parts, dim=-1)
        features = self._append_identity_features(
            features,
            batch_size=batch_size,
            dim=dim,
            length=length,
            device=values.device,
            dtype=values.dtype,
            batch_offset=batch_offset,
            dim_offset=dim_offset,
            total_batch_classes=total_batch_classes,
            total_dim_classes=total_dim_classes,
        )
        return features.reshape(-1, features.shape[-1]), values.reshape(-1)

    def _prepare_matrix(
        self,
        x: torch.Tensor,
        past_context: torch.Tensor | None,
        future_context: torch.Tensor | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size, dim, lags = x.shape
        time_features = self._generate_time_features(
            x,
            window_length=lags + self.horizon,
        )

        if self.context_as_features:
            train_context = None
            test_context = None
            if future_context is not None:
                train_context = future_context[..., :lags]
                test_context = future_context[..., lags : lags + self.horizon]
            x_train, y_train = self._create_tabular_block(
                x,
                time_features,
                context_values=train_context,
                start_index=0,
                total_batch_classes=max(1, batch_size),
                total_dim_classes=max(1, dim),
            )
            dummy = torch.zeros(
                batch_size,
                dim,
                self.horizon,
                device=x.device,
                dtype=x.dtype,
            )
            x_test, _ = self._create_tabular_block(
                dummy,
                time_features,
                context_values=test_context,
                start_index=lags,
                total_batch_classes=max(1, batch_size),
                total_dim_classes=max(1, dim),
            )
            return (
                x_train.detach().cpu().numpy(),
                y_train.detach().cpu().numpy(),
                x_test.detach().cpu().numpy(),
            )

        past_batch = 0 if past_context is None else past_context.shape[0]
        past_dim = 0 if past_context is None else past_context.shape[1]
        future_batch = 0 if future_context is None else future_context.shape[0]
        future_dim = 0 if future_context is None else future_context.shape[1]
        total_batch_classes = max(1, batch_size + past_batch + future_batch)
        total_dim_classes = max(1, dim + past_dim + future_dim)

        def block(
            values: torch.Tensor,
            start_index: int,
            batch_offset: int,
            dim_offset: int,
            *,
            detection_values: torch.Tensor | None = None,
        ):
            block_features = self._generate_time_features(
                detection_values if detection_values is not None else values,
                window_length=start_index + values.shape[-1],
            )
            return self._create_tabular_block(
                values,
                block_features,
                context_values=None,
                start_index=start_index,
                batch_offset=batch_offset,
                dim_offset=dim_offset,
                total_batch_classes=total_batch_classes,
                total_dim_classes=total_dim_classes,
            )

        train_features = []
        train_targets = []
        features, targets = block(x, 0, 0, 0)
        train_features.append(features)
        train_targets.append(targets)

        if past_context is not None:
            features, targets = block(past_context, 0, batch_size, dim)
            train_features.append(features)
            train_targets.append(targets)

        if future_context is not None:
            features, targets = block(
                future_context[..., :lags],
                0,
                batch_size + past_batch,
                dim + past_dim,
                detection_values=future_context,
            )
            train_features.append(features)
            train_targets.append(targets)
            features, targets = block(
                future_context[..., lags : lags + self.horizon],
                lags,
                batch_size + past_batch,
                dim + past_dim,
                detection_values=future_context,
            )
            train_features.append(features)
            train_targets.append(targets)

        dummy = torch.zeros(
            batch_size,
            dim,
            self.horizon,
            device=x.device,
            dtype=x.dtype,
        )
        x_test, _ = block(dummy, lags, 0, 0, detection_values=x)
        return (
            torch.cat(train_features, dim=0).detach().cpu().numpy(),
            torch.cat(train_targets, dim=0).detach().cpu().numpy(),
            x_test.detach().cpu().numpy(),
        )
