"""Scikit-learn linear forecasting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch

from proposal.normalizations import (
    IdentityNormalization,
    InstanceMinMaxNormalization,
    MinMaxNormalization,
    RMSNormalization,
    RelativeMeanNormalization,
    RevIN,
    SigmoidNormalization,
    StandardNormalization,
    TanhNormalization,
    build_normalization,
)


def _import_linear_regression():
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SkLinear requires scikit-learn. Install scikit-learn before using "
            "`python -m scripts.train_sklearn`."
        ) from exc
    return LinearRegression


def _dataset_components(dataset: Any) -> Iterable[Any]:
    if hasattr(dataset, "datasets"):
        for component in dataset.datasets:
            yield from _dataset_components(component)
    else:
        yield dataset


def _slice_temporal_context(context: torch.Tensor, date: int, length: int) -> torch.Tensor:
    if context.shape[-1] == 1:
        return context
    return context[..., date : date + length]


def _accessible_xy_from_dataset(
    dataset: Any,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    individual_ids: list[int] = []
    individual_names: list[str] = []
    query_ids: list[int] = []
    for individual, query_t in dataset.index_sampler.iter_accessible_pairs():
        start = int(query_t) - dataset.lags + 1
        stop = int(query_t) + dataset.horizon + 1
        window = dataset.data.values[[individual], :, start:stop]
        inputs.append(window[:, :, : dataset.lags])
        targets.append(window[:, :, dataset.lags :])
        individual_ids.append(int(dataset.data.individual_ids[individual]))
        individual_names.append(str(dataset.data.individual_names[individual]))
        query_ids.append(int(dataset.data.date_ids[query_t]))
        if len(inputs) >= batch_size:
            yield torch.cat(inputs, dim=0), torch.cat(targets, dim=0), {
                "individual_ids": torch.tensor(individual_ids, dtype=torch.long),
                "individual_names": list(individual_names),
                "query_ids": torch.tensor(query_ids, dtype=torch.long),
            }
            inputs.clear()
            targets.clear()
            individual_ids.clear()
            individual_names.clear()
            query_ids.clear()
    if inputs:
        yield torch.cat(inputs, dim=0), torch.cat(targets, dim=0), {
            "individual_ids": torch.tensor(individual_ids, dtype=torch.long),
            "individual_names": list(individual_names),
            "query_ids": torch.tensor(query_ids, dtype=torch.long),
        }


def iter_loader_xy(
    loader: Any,
    *,
    mode: str = "accessible",
    max_windows: int | None = None,
    include_metadata: bool = False,
) -> Iterator[Any]:
    """Yield ``(inputs, targets)`` batches from a loader.

    ``mode="accessible"`` deterministically unrolls all sampler-accessible
    windows from ``loader.dataset``. This respects split/subset/stride and
    constant-window filtering while avoiding stochastic draw order. ``mode``
    may be set to ``"loader"`` to consume the dataloader exactly as-is.
    """
    mode = str(mode)
    seen = 0

    def emit(x: torch.Tensor, y: torch.Tensor, metadata: Mapping[str, Any] | None = None):
        nonlocal seen
        metadata = dict(metadata or {})
        if max_windows is not None:
            remaining = int(max_windows) - seen
            if remaining <= 0:
                return None
            x = x[:remaining]
            y = y[:remaining]
            for key in ("individual_ids", "individual_names", "query_ids"):
                if metadata.get(key) is not None:
                    metadata[key] = metadata[key][:remaining]
        seen += x.shape[0]
        return (x, y, metadata) if include_metadata else (x, y)

    if mode == "loader":
        for batch in loader:
            item = emit(batch["inputs"], batch["targets"], batch.get("metadata"))
            if item is None:
                break
            yield item
        return

    if mode != "accessible":
        raise ValueError("sklearn unroll mode must be 'accessible' or 'loader'")

    batch_size = int(getattr(loader, "batch_size", None) or 256)
    for dataset in _dataset_components(loader.dataset):
        for x, y, metadata in _accessible_xy_from_dataset(dataset, batch_size):
            item = emit(x, y, metadata)
            if item is None:
                return
            yield item


class SkLinearForecaster:
    """Multi-output scikit-learn linear regressor for TimeTensor windows."""

    def __init__(
        self,
        lags: int,
        dim: int,
        horizon: int,
        *,
        normalization: Mapping[str, Any] | None = None,
        normalization_stats: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ):
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.normalization_config = None if normalization is None else dict(normalization)
        self.normalization = build_normalization(
            None if normalization is None else dict(normalization),
            dim=self.dim,
            stats=normalization_stats,
        )
        regressor = _import_linear_regression()
        self.regressor = regressor(**dict(model_kwargs or {}))
        self.fitted = False

    def fit_loader(
        self,
        loader: Any,
        *,
        unroll_mode: str = "accessible",
        max_windows: int | None = None,
    ) -> dict[str, Any]:
        features = []
        targets = []
        windows = 0
        for x, y in iter_loader_xy(loader, mode=unroll_mode, max_windows=max_windows):
            x_norm, y_norm = self._normalize_pair(x, y)
            features.append(self._features(x_norm))
            targets.append(self._targets(y_norm))
            windows += x.shape[0]
        if not features:
            raise ValueError("cannot fit SkLinear on an empty loader")
        self.regressor.fit(np.concatenate(features, axis=0), np.concatenate(targets, axis=0))
        self.fitted = True
        return {"windows": windows, "features": features[0].shape[1], "targets": targets[0].shape[1]}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("SkLinearForecaster must be fitted before prediction")
        x_norm = self.normalization(x)
        pred = self.regressor.predict(self._features(x_norm))
        pred_norm = torch.as_tensor(pred, device=x.device, dtype=x.dtype).reshape(
            x.shape[0],
            self.dim,
            self.horizon,
        )
        return self.normalization.inverse(pred_norm)

    def _features(self, x: torch.Tensor) -> np.ndarray:
        if x.shape[1] != self.dim or x.shape[-1] != self.lags:
            raise ValueError(f"expected x with shape (batch, {self.dim}, {self.lags}), got {tuple(x.shape)}")
        return x.detach().cpu().reshape(x.shape[0], self.dim * self.lags).numpy()

    def _targets(self, y: torch.Tensor) -> np.ndarray:
        return y.detach().cpu().reshape(y.shape[0], self.dim * self.horizon).numpy()

    def _normalize_pair(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.normalization(x)
        return x_norm, self._normalize_target_from_cached_stats(y)

    def _normalize_target_from_cached_stats(self, y: torch.Tensor) -> torch.Tensor:
        norm = self.normalization
        if isinstance(norm, IdentityNormalization):
            return y
        if isinstance(norm, StandardNormalization):
            mean = norm.mean.to(device=y.device, dtype=y.dtype)
            std = norm.std.to(device=y.device, dtype=y.dtype)
            return (y - mean) / (std + norm.eps)
        if isinstance(norm, MinMaxNormalization):
            min_value = norm.min_value.to(device=y.device, dtype=y.dtype)
            max_value = norm.max_value.to(device=y.device, dtype=y.dtype)
            return (y - min_value) / (max_value - min_value + norm.eps)
        if isinstance(norm, InstanceMinMaxNormalization):
            if norm._min is None or norm._max is None:
                raise RuntimeError("instance min-max statistics are not available")
            return (y - norm._min) / (norm._max - norm._min + norm.eps)
        if isinstance(norm, RevIN):
            if norm._center is None or norm._std is None:
                raise RuntimeError("RevIN statistics are not available")
            out = (y - norm._center) / (norm._std + norm.eps)
            if norm.affine:
                out = out * norm.gamma + norm.beta
            if norm.transform == "arcsinh":
                out = torch.asinh(out)
            return out
        if isinstance(norm, RelativeMeanNormalization):
            if norm._scale is None:
                raise RuntimeError("relative-mean scale is not available")
            return y / norm._scale
        if isinstance(norm, RMSNormalization):
            if norm._scale is None:
                raise RuntimeError("RMS scale is not available")
            return y / norm._scale
        if isinstance(norm, (SigmoidNormalization, TanhNormalization)):
            return norm(y)
        raise ValueError(
            f"SkLinear does not support target normalization for {norm.__class__.__name__}"
        )

    def weight_matrix(self) -> torch.Tensor:
        coef = torch.as_tensor(self.regressor.coef_, dtype=torch.float32)
        if coef.ndim == 1:
            coef = coef.unsqueeze(0)
        return coef.reshape(self.dim, self.horizon, self.dim, self.lags)

    def bias(self) -> torch.Tensor:
        return torch.as_tensor(self.regressor.intercept_, dtype=torch.float32).reshape(
            self.dim,
            self.horizon,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lags": self.lags,
                "dim": self.dim,
                "horizon": self.horizon,
                "normalization": self.normalization_config,
                "normalization_state": self.normalization.state_dict(),
                "regressor": self.regressor,
                "fitted": self.fitted,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SkLinearForecaster":
        payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
        model = cls(
            payload["lags"],
            payload["dim"],
            payload["horizon"],
            normalization=payload.get("normalization"),
        )
        model.normalization.load_state_dict(payload.get("normalization_state", {}), strict=False)
        model.regressor = payload["regressor"]
        model.fitted = bool(payload.get("fitted", True))
        return model
