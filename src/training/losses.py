"""PyTorch forecasting losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from proposal.normalizations import get_normal_stats


def _criterion(name: str, *, reduction: str = "mean", kwargs: Mapping[str, Any] | None = None) -> nn.Module:
    kwargs = dict(kwargs or {})
    name = name.lower()
    if name == "mse":
        return nn.MSELoss(reduction=reduction, **kwargs)
    if name == "mae":
        return nn.L1Loss(reduction=reduction, **kwargs)
    raise ValueError(f"unknown base loss {name!r}")


@dataclass(frozen=True)
class LossConfig:
    """Configuration for a forecasting loss."""

    name: str = "nmse"
    base: str = "mse"
    scaling: str | None = "normal"
    reduction: str = "mean"
    eps: float = 1e-8
    kwargs: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: Mapping[str, Any] | str | None) -> "LossConfig":
        if config is None:
            return cls()
        if isinstance(config, str):
            return config_to_loss_config(config)
        data = dict(config)
        return cls(
            name=str(data.get("name", data.get("base", "nmse"))),
            base=str(data.get("base", "mse")),
            scaling=data.get("scaling"),
            reduction=str(data.get("reduction", "mean")),
            eps=float(data.get("eps", 1e-8)),
            kwargs=data.get("kwargs"),
        )


class LossWrapper(nn.Module):
    """Apply the requested forecasting loss with optional loss-specific scaling."""

    def __init__(
        self,
        base_loss: str | nn.Module = "mse",
        *,
        scaling: str | None = None,
        name: str | None = None,
        reduction: str = "mean",
        eps: float = 1e-8,
        kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.loss = (
            base_loss
            if isinstance(base_loss, nn.Module)
            else _criterion(str(base_loss), reduction=reduction, kwargs=kwargs)
        )
        self.scaling = None if scaling is None else str(scaling).lower()
        self.name = name or self._default_name(base_loss, self.scaling)
        self.eps = float(eps)

    @staticmethod
    def _default_name(base_loss: str | nn.Module, scaling: str | None) -> str:
        base = base_loss if isinstance(base_loss, str) else base_loss.__class__.__name__
        return str(base) if scaling is None else f"{scaling}_{base}"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred, target = self.scale(pred, target, context=context, mean=mean, std=std)
        return self.loss(pred, target)

    def scale(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scaling = self.scaling
        if scaling is None:
            return pred, target
        if scaling == "normal":
            _, scale = self._context_stats(context, mean, std, require_std=True)
            assert scale is not None
            scale = scale.abs() + self.eps
            return pred / scale, target / scale
        if scaling == "relative_mean":
            center, _ = self._context_stats(context, mean, std, require_std=False)
            scale = center.abs() + self.eps
            return pred / scale, target / scale
        raise ValueError(f"unknown loss scaling {self.scaling!r}")

    @staticmethod
    def _context_stats(
        context: torch.Tensor | None,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        *,
        require_std: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mean is not None and (std is not None or not require_std):
            return mean, std
        if context is None:
            raise ValueError("normalized losses require context or explicit statistics")
        ctx_mean, ctx_std = get_normal_stats(context, dim=-1, keepdim=True, detach=True)
        return ctx_mean if mean is None else mean, ctx_std if std is None else std


def config_to_loss_config(name: str) -> LossConfig:
    key = str(name)
    configs = {
        "mse": LossConfig(name="mse", base="mse"),
        "mae": LossConfig(name="mae", base="mae"),
        "nmse": LossConfig(name="nmse", base="mse", scaling="normal"),
        "nmae": LossConfig(name="nmae", base="mae", scaling="normal"),
        "rmse": LossConfig(name="rmse", base="mse", scaling="relative_mean"),
        "relative_mse": LossConfig(
            name="relative_mse", base="mse", scaling="relative_mean"
        ),
    }
    if key not in configs:
        raise ValueError(f"unknown loss config {name!r}")
    return configs[key]


def build_loss(config: Mapping[str, Any] | str | LossConfig | LossWrapper | None) -> LossWrapper:
    if isinstance(config, LossWrapper):
        return config
    loss_config = config if isinstance(config, LossConfig) else LossConfig.from_dict(config)
    return LossWrapper(
        loss_config.base,
        scaling=loss_config.scaling,
        name=loss_config.name,
        reduction=loss_config.reduction,
        eps=loss_config.eps,
        kwargs=loss_config.kwargs,
    )


def get_losses(
    criterion_name: str | Mapping[str, Any] = "nmse",
    *,
    complete_evaluation: bool = False,
) -> tuple[LossWrapper, dict[str, LossWrapper]]:
    """Return a training criterion and common elementwise evaluation losses."""
    criterion = build_loss(LossConfig.from_dict(criterion_name))
    eval_names = ["mse", "nmse"]
    if complete_evaluation:
        eval_names.extend(["mae", "nmae", "relative_mse"])
    if criterion.name not in eval_names:
        eval_names.append(criterion.name)
    eval_losses = {}
    for name in eval_names:
        config = LossConfig.from_dict(name)
        config = LossConfig(
            name=config.name,
            base=config.base,
            scaling=config.scaling,
            reduction="none",
            eps=config.eps,
            kwargs=config.kwargs,
        )
        loss = build_loss(config)
        eval_losses[loss.name] = loss
    return criterion, eval_losses
