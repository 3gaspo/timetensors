"""Normalization modules for wrapped forecasting models."""

from __future__ import annotations

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

    name = "minmax"

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

    name = "imm"

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
    """Reversible instance normalization with optional learnable affine layer."""

    name = "revin"

    def __init__(
        self,
        dim: int | None = None,
        eps: float = 1e-8,
        affine: bool = True,
        center: str = "mean",
        detach_stats: bool = True,
    ):
        super().__init__()
        if center not in {"mean", "last"}:
            raise ValueError("center must be 'mean' or 'last'")
        self.dim = None if dim is None else int(dim)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.center = center
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
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self._center is None or self._std is None:
            raise RuntimeError("RevIN statistics are not available")
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


class InstanceNormalization(RevINNormalization):
    """RevIN without learnable affine parameters."""

    name = "instance"

    def __init__(
        self,
        eps: float = 1e-8,
        center: str = "mean",
        detach_stats: bool = True,
    ):
        super().__init__(
            dim=None,
            eps=eps,
            affine=False,
            center=center,
            detach_stats=detach_stats,
        )


class RevINArcsinhNormalization(RevINNormalization):
    """RevIN followed by an arcsinh transform."""

    name = "revin_arcsinh"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(super().forward(x))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return super().inverse(torch.sinh(y))


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


def build_normalization(config: dict | None, *, dim: int | None = None) -> nn.Module:
    """Build a normalization module from a small config dictionary."""
    if config is None:
        return IdentityNormalization()
    name = str(config.get("name", "identity")).lower().replace("-", "_").replace("+", "_")
    kwargs = dict(config.get("kwargs") or {})
    if name in {"none", "identity", "default"}:
        return IdentityNormalization()
    if name in {"standard", "standardnorm", "standard_norm", "zscore"}:
        return StandardNormalization(**kwargs)
    if name in {"minmax", "min_max"}:
        return MinMaxNormalization(**kwargs)
    if name in {"imm", "instance_minmax", "instance_min_max"}:
        return InstanceMinMaxNormalization(**kwargs)
    if name in {"instance", "instancenorm", "instance_norm"}:
        return InstanceNormalization(**kwargs)
    if name in {"revin", "reversible_instance_norm"}:
        kwargs.setdefault("dim", dim)
        return RevINNormalization(**kwargs)
    if name in {"revin_last", "last_revin"}:
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("center", "last")
        return RevINNormalization(**kwargs)
    if name in {"revin_arcsinh", "revin_asinh", "revin_arcsinsh", "revin_arcinsh"}:
        kwargs.setdefault("dim", dim)
        return RevINArcsinhNormalization(**kwargs)
    if name in {"sigmoid", "logistic"}:
        return SigmoidNormalization(**kwargs)
    if name in {"tanh", "hyperbolic_tangent"}:
        return TanhNormalization(**kwargs)
    if name in {"relative_mean", "rmean"}:
        return RelativeMeanNormalization(**kwargs)
    if name in {"rms", "rms_norm", "rmsnorm"}:
        return RMSNormalization(**kwargs)
    raise ValueError(f"unknown normalization {name!r}")
