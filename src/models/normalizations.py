"""Normalization modules for wrapped forecasting models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


def _eps_like(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.as_tensor(eps, device=x.device, dtype=x.dtype)


def _as_like(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _clamp_unit_interval(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.clamp(min=eps, max=1.0 - eps)


def _clamp_tanh_domain(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.clamp(min=-1.0 + eps, max=1.0 - eps)


def get_normal_stats(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = True,
    detach: bool = True,
    unbiased: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and std along ``dim`` using population std by default."""
    mean = x.mean(dim=dim, keepdim=keepdim)
    std = x.std(dim=dim, keepdim=keepdim, unbiased=unbiased)
    if detach:
        mean = mean.detach()
        std = std.detach()
    return mean, std


def get_minmax_stats(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = True,
    detach: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return min and max along ``dim``."""
    min_value = x.amin(dim=dim, keepdim=keepdim)
    max_value = x.amax(dim=dim, keepdim=keepdim)
    if detach:
        min_value = min_value.detach()
        max_value = max_value.detach()
    return min_value, max_value


def get_rms_stats(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = True,
    detach: bool = True,
) -> torch.Tensor:
    """Return root-mean-square scale along ``dim``."""
    rms = x.pow(2).mean(dim=dim, keepdim=keepdim).sqrt()
    return rms.detach() if detach else rms


def normalize_standard(
    x: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean_t = _as_like(mean, x)
    std_t = _as_like(std, x)
    return (x - mean_t) / (std_t + _eps_like(x, eps))


def denormalize_standard(
    x: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    mean_t = _as_like(mean, x)
    std_t = _as_like(std, x)
    return x * (std_t + _eps_like(x, eps)) + mean_t


class IdentityNormalization(nn.Module):
    """Leave inputs and outputs unchanged."""

    name = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y


class StandardNormalization(nn.Module):
    """Normalize with provided global mean and standard deviation."""

    name = "standard"

    def __init__(self, mean: float | torch.Tensor, std: float | torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clone())
        if torch.any(self.std < 0):
            raise ValueError("std must be non-negative")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_standard(x, self.mean, self.std, eps=self.eps)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return denormalize_standard(y, self.mean, self.std, eps=self.eps)


class MinMaxNormalization(nn.Module):
    """Normalize with provided global min and max values."""

    name = "min-max"

    def __init__(self, min_value: float | torch.Tensor, max_value: float | torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("min_value", torch.as_tensor(min_value, dtype=torch.float32).clone())
        self.register_buffer("max_value", torch.as_tensor(max_value, dtype=torch.float32).clone())
        if torch.any(self.max_value < self.min_value):
            raise ValueError("max_value must be greater than or equal to min_value")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_value = _as_like(self.min_value, x)
        max_value = _as_like(self.max_value, x)
        return (x - min_value) / (max_value - min_value + self.eps)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        min_value = _as_like(self.min_value, y)
        max_value = _as_like(self.max_value, y)
        return y * (max_value - min_value + self.eps) + min_value


class InstanceMinMaxNormalization(nn.Module):
    """Normalize each sample with its own min and max."""

    name = "in-min-max"

    def __init__(self, eps: float = 1e-8, detach_stats: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_value, max_value = get_minmax_stats(
            x,
            dim=-1,
            keepdim=True,
            detach=self.detach_stats,
        )
        self._min = min_value
        self._max = max_value
        return (x - min_value) / (max_value - min_value + self.eps)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self._min is None or self._max is None:
            raise RuntimeError("min-max statistics are not available")
        return y * (self._max - self._min + self.eps) + self._min


class RevINNormalization(nn.Module):
    """Reversible instance normalization with optional affine and transform."""

    name = "revin"

    def __init__(
        self,
        dim: int | None = None,
        eps: float = 1e-8,
        affine: bool = True,
        center: str = "mean",
        transform: str | None = None,
        detach_stats: bool = True,
    ):
        super().__init__()
        if center not in {"mean", "last"}:
            raise ValueError("center must be 'mean' or 'last'")
        transform = (
            None
            if transform is None
            else str(transform).lower().replace("-", "_")
        )
        if transform not in {None, "arcsinh"}:
            raise ValueError("transform must be None or 'arcsinh'")
        self.dim = None if dim is None else int(dim)
        self.eps = float(eps)
        if isinstance(affine, str):
            affine = affine.lower() in {"1", "true", "yes", "on"}
        self.affine = bool(affine)
        self.center = center
        self.transform = transform
        self.detach_stats = bool(detach_stats)
        if self.affine:
            if self.dim is None:
                raise ValueError("affine RevIN requires dim")
            self.gamma = nn.Parameter(torch.ones(1, self.dim, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.dim, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._center: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center, std = self._stats(x)
        self._center = center
        self._std = std
        y = normalize_standard(x, center, std, eps=self.eps)
        if self.affine:
            y = y * self.gamma + self.beta
        if self.transform == "arcsinh":
            y = torch.asinh(y)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self._center is None or self._std is None:
            raise RuntimeError("RevIN statistics are not available")
        if self.transform == "arcsinh":
            y = torch.sinh(y)
        if self.affine:
            y = (y - self.beta) / (self.gamma + self.eps)
        return denormalize_standard(y, self._center, self._std, eps=self.eps)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.center == "last":
            center = x[..., -1:]
            if self.detach_stats:
                center = center.detach()
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            if self.detach_stats:
                std = std.detach()
            return center, std
        return get_normal_stats(
            x,
            dim=-1,
            keepdim=True,
            detach=self.detach_stats,
            unbiased=False,
        )


class SigmoidNormalization(nn.Module):
    """Apply sigmoid as a reversible normalization on bounded outputs."""

    name = "sigmoid"

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = _clamp_unit_interval(y, self.eps)
        return torch.logit(y)


class TanhNormalization(nn.Module):
    """Apply tanh as a reversible normalization on bounded outputs."""

    name = "tanh"

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y = _clamp_tanh_domain(y, self.eps)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))


class RelativeMeanNormalization(nn.Module):
    """Scale values by the absolute instance mean."""

    name = "relative_mean"

    def __init__(self, eps: float = 1e-8, detach_stats: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)
        self._scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        if self.detach_stats:
            mean = mean.detach()
        self._scale = mean.abs() + self.eps
        return x / self._scale

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self._scale is None:
            raise RuntimeError("relative mean scale is not available")
        return y * self._scale


class RMSNormalization(nn.Module):
    """Scale values by the root-mean-square of each instance."""

    name = "rms"

    def __init__(self, eps: float = 1e-8, detach_stats: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)
        self._scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = get_rms_stats(
            x,
            dim=-1,
            keepdim=True,
            detach=self.detach_stats,
        )
        self._scale = scale + self.eps
        return x / self._scale

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self._scale is None:
            raise RuntimeError("RMS scale is not available")
        return y * self._scale


def _select_stats(
    stats: Mapping[str, object] | None,
    *,
    split: str = "train",
) -> Mapping[str, object] | None:
    if not isinstance(stats, Mapping):
        return None
    payload = stats.get("stats") if isinstance(stats.get("stats"), Mapping) else stats
    if not isinstance(payload, Mapping):
        return None
    if split in payload and isinstance(payload[split], Mapping):
        return payload[split]
    if "alpha" in payload or "beta" in payload:
        return payload
    return None


def build_normalization(
    config: dict | None,
    *,
    dim: int | None = None,
    stats: Mapping[str, object] | None = None,
) -> nn.Module:
    """Build a normalization module from a small config dictionary."""
    if config is None:
        return IdentityNormalization()
    name = str(config.get("name", "identity"))
    kwargs = dict(config.get("kwargs") or {})
    train_stats = (
        stats.get("train")
        if isinstance(stats, Mapping) and isinstance(stats.get("train"), Mapping)
        else stats
    )
    if name == "identity":
        return IdentityNormalization()
    if name == "standard":
        if isinstance(train_stats, Mapping):
            kwargs.setdefault("mean", train_stats.get("lookback_value_mean"))
            kwargs.setdefault("std", train_stats.get("lookback_value_std"))
        return StandardNormalization(**kwargs)
    if name == "min-max":
        if isinstance(train_stats, Mapping):
            kwargs.setdefault("min_value", train_stats.get("lookback_value_min"))
            kwargs.setdefault("max_value", train_stats.get("lookback_value_max"))
        return MinMaxNormalization(**kwargs)
    if name == "in-min-max":
        return InstanceMinMaxNormalization(**kwargs)
    if name == "instance":
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("affine", False)
        return RevINNormalization(**kwargs)
    if name == "revin":
        kwargs.setdefault("dim", dim)
        return RevINNormalization(**kwargs)
    if name in {"grevin", "cmin", "previn"}:
        if dim is None:
            raise ValueError(f"{name!r} normalization requires dim")
        init_from_stats = bool(kwargs.pop("init_from_stats", False))
        stats_split = str(kwargs.pop("stats_split", "train"))
        stats_payload = kwargs.pop("stats", None)
        if stats_payload is None and init_from_stats:
            stats_payload = _select_stats(stats, split=stats_split)
        from .grevin import build_grevin_normalization

        return build_grevin_normalization(
            name,
            dim,
            stats=stats_payload if isinstance(stats_payload, Mapping) else None,
            **kwargs,
        )
    if name == "sigmoid":
        return SigmoidNormalization(**kwargs)
    if name == "tanh":
        return TanhNormalization(**kwargs)
    if name == "relative_mean":
        return RelativeMeanNormalization(**kwargs)
    if name == "rms":
        return RMSNormalization(**kwargs)
    raise ValueError(f"unknown normalization {name!r}")
