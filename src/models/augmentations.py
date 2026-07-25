"""Covariate and output augmentation modules."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_list(value) -> list:
    """Return a fresh list for config values that may be scalar or absent."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _expand_mode_specs(raw_modes: list) -> list:
    specs: list = []
    for item in raw_modes:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            specs.extend(part for part in text.split("-") if part)
        else:
            specs.append(item)
    return specs


def _parse_mode_spec(spec) -> dict[str, object]:
    if isinstance(spec, Mapping):
        mode = spec.get("name")
        if mode is None:
            raise ValueError("covariate augmentation spec requires 'name'")
        mode = str(mode)
        value = spec.get("value")
        if value is None and mode == "noise":
            value = spec.get("noise_scale", spec.get("scale"))
        if value is None and mode == "constant":
            value = spec.get("constant_value")
        count = int(spec.get("count", 1))
        if count < 1:
            raise ValueError("covariate augmentation count must be positive")
        target = spec.get("target")
        return {
            "mode": mode,
            "count": count,
            "value": None if value is None else float(value),
            "target": None if target is None else str(target),
        }
    return {"mode": str(spec), "count": 1, "value": None, "target": None}


def _cat_optional(parts: list[torch.Tensor]) -> torch.Tensor | None:
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def _expand_static(value: torch.Tensor, length: int) -> torch.Tensor:
    if value.shape[-1] != 1:
        raise ValueError(f"static covariates must have length 1, got {value.shape[-1]}")
    return value.expand(*value.shape[:-1], length)


def _coerce_past(value: torch.Tensor, lags: int, horizon: int) -> torch.Tensor:
    if value.shape[-1] == lags:
        return value
    if value.shape[-1] == 1:
        return _expand_static(value, lags)
    if value.shape[-1] == lags + horizon:
        return value[..., :lags]
    raise ValueError(
        f"past covariates must have length 1, {lags}, or {lags + horizon}; "
        f"got {value.shape[-1]}"
    )


def _coerce_future(value: torch.Tensor, lags: int, horizon: int) -> torch.Tensor:
    if value.shape[-1] == horizon:
        return value
    if value.shape[-1] == 1:
        return _expand_static(value, horizon)
    if value.shape[-1] == lags + horizon:
        return value[..., lags:]
    raise ValueError(
        f"future covariates must have length 1, {horizon}, or {lags + horizon}; "
        f"got {value.shape[-1]}"
    )


def _coerce_static(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 1:
        raise ValueError(f"static covariates must have length 1, got {value.shape[-1]}")
    return value


def empty_covariates() -> dict[str, torch.Tensor | None]:
    return {"past": None, "future": None, "static": None}


def merge_covariates(*items: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    """Merge structured covariate dictionaries along their channel dimension."""
    return {
        key: _cat_optional([item.get(key) for item in items])
        for key in ("past", "future", "static")
    }


def split_covariate_tensor(
    value: torch.Tensor,
    *,
    lags: int,
    horizon: int,
) -> dict[str, torch.Tensor | None]:
    """Classify a covariate tensor by its final time dimension."""
    if value.shape[-1] == 1:
        return {"past": None, "future": None, "static": value}
    if value.shape[-1] == lags:
        return {"past": value, "future": None, "static": None}
    if value.shape[-1] == horizon:
        return {"past": None, "future": value, "static": None}
    if value.shape[-1] == lags + horizon:
        return {
            "past": value[..., :lags],
            "future": value[..., lags:],
            "static": None,
        }
    raise ValueError(
        f"covariates must have length 1, {lags}, {horizon}, or "
        f"{lags + horizon}; got {value.shape[-1]}"
    )


def normalize_covariates(
    covariates=None,
    *,
    lags: int,
    horizon: int,
    past: torch.Tensor | None = None,
    future: torch.Tensor | None = None,
    static: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """Normalize tensor/tuple/dict covariates into past/future/static entries."""
    pieces: list[dict[str, torch.Tensor | None]] = []

    if covariates is None:
        pass
    elif torch.is_tensor(covariates):
        pieces.append(split_covariate_tensor(covariates, lags=lags, horizon=horizon))
    elif isinstance(covariates, tuple):
        if len(covariates) != 2:
            raise ValueError("tuple covariates must be (past, future)")
        cov_past, cov_future = covariates
        pieces.append(
            {
                "past": None if cov_past is None else _coerce_past(cov_past, lags, horizon),
                "future": None if cov_future is None else _coerce_future(cov_future, lags, horizon),
                "static": None,
            }
        )
    elif isinstance(covariates, dict):
        explicit = {}
        if covariates.get("past") is not None:
            explicit["past"] = _coerce_past(covariates["past"], lags, horizon)
        if covariates.get("future") is not None:
            explicit["future"] = _coerce_future(covariates["future"], lags, horizon)
        if covariates.get("static") is not None:
            explicit["static"] = _coerce_static(covariates["static"])
        if explicit:
            pieces.append(
                {
                    "past": explicit.get("past"),
                    "future": explicit.get("future"),
                    "static": explicit.get("static"),
                }
            )
        for key in ("individual_context", "global_context"):
            if covariates.get(key) is not None:
                pieces.append(
                    split_covariate_tensor(covariates[key], lags=lags, horizon=horizon)
                )
    else:
        raise TypeError("covariates must be None, a tensor, tuple, or dict")

    if past is not None:
        pieces.append({"past": _coerce_past(past, lags, horizon), "future": None, "static": None})
    if future is not None:
        pieces.append({"past": None, "future": _coerce_future(future, lags, horizon), "static": None})
    if static is not None:
        pieces.append({"past": None, "future": None, "static": _coerce_static(static)})

    if not pieces:
        return empty_covariates()
    return merge_covariates(*pieces)


class CovariateAugmentation(nn.Module):
    """Append generated covariates to past and optionally future covariates."""

    def __init__(
        self,
        modes: str | list[str] | tuple[str, ...] | None = None,
        *,
        noise_scale: float = 1.0,
        constant_value: float = 1.0,
        kernel_size: int = 5,
        eps: float = 1e-8,
        target: str = "past_only",
    ):
        super().__init__()
        self.modes = [_parse_mode_spec(spec) for spec in _expand_mode_specs(as_list(modes))]
        self.noise_scale = float(noise_scale)
        self.constant_value = float(constant_value)
        self.eps = float(eps)
        self.target = str(target)
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-0.5 * t.pow(2))
        self.register_buffer("smooth_kernel", kernel / kernel.sum())

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.smooth_kernel.view(1, 1, -1).repeat(x.shape[1], 1, 1)
        return F.conv1d(
            x,
            weight.to(device=x.device, dtype=x.dtype),
            padding=self.smooth_kernel.numel() // 2,
            groups=x.shape[1],
        )

    def forward(
        self,
        x: torch.Tensor,
        covariates=None,
        *,
        horizon: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        del kwargs
        if horizon is None:
            raise ValueError("CovariateAugmentation requires horizon")
        structured = normalize_covariates(
            covariates,
            lags=x.shape[-1],
            horizon=int(horizon),
        )
        pieces = []
        if structured["past"] is not None:
            pieces.append(structured["past"])
        future_pieces = []
        if structured["future"] is not None:
            future_pieces.append(structured["future"])
        for spec in self.modes:
            mode = str(spec["mode"])
            count = int(spec["count"])
            value = spec["value"]
            target = self.target if spec["target"] is None else str(spec["target"])
            if mode == "identity":
                pieces.append(x)
            elif mode == "square":
                pieces.append(x * x.abs())
            elif mode == "root":
                pieces.append(torch.sign(x) * torch.sqrt(x.abs() + self.eps))
            elif mode == "sign":
                pieces.append(torch.sign(x))
            elif mode == "mirror":
                pieces.append(-x)
            elif mode == "kernel":
                pieces.append(self._kernel(x))
            elif mode == "noise":
                scale = self.noise_scale if value is None else float(value)
                for _ in range(count):
                    self._append_generated(
                        pieces,
                        future_pieces,
                        scale * self._randn(x, horizon=int(horizon), target=target),
                        lags=x.shape[-1],
                    )
            elif mode == "constant":
                constant = self.constant_value if value is None else float(value)
                for _ in range(count):
                    self._append_generated(
                        pieces,
                        future_pieces,
                        self._full(x, constant, horizon=int(horizon), target=target),
                        lags=x.shape[-1],
                    )
            else:
                raise ValueError(f"unknown covariate augmentation {mode!r}")
        structured["past"] = _cat_optional(pieces)
        structured["future"] = _cat_optional(future_pieces)
        return structured

    def _generated_length(self, x: torch.Tensor, horizon: int, target: str) -> int:
        if target == "future_included":
            return x.shape[-1] + horizon
        if target == "past_only":
            return x.shape[-1]
        raise ValueError("target must be 'past_only' or 'future_included'")

    def _randn(self, x: torch.Tensor, *, horizon: int, target: str) -> torch.Tensor:
        return torch.randn(
            x.shape[0],
            1,
            self._generated_length(x, horizon, target),
            device=x.device,
            dtype=x.dtype,
        )

    def _full(self, x: torch.Tensor, value: float, *, horizon: int, target: str) -> torch.Tensor:
        return torch.full(
            (x.shape[0], 1, self._generated_length(x, horizon, target)),
            value,
            device=x.device,
            dtype=x.dtype,
        )

    def _append_generated(
        self,
        past_pieces: list[torch.Tensor],
        future_pieces: list[torch.Tensor],
        value: torch.Tensor,
        *,
        lags: int,
    ) -> None:
        if value.shape[-1] == lags:
            past_pieces.append(value)
            return
        past_pieces.append(value[..., :lags])
        future_pieces.append(value[..., lags:])


class RepeatConstantOutput(nn.Module):
    """Replace predictions for constant lookbacks by repeated last values."""

    def __init__(self, horizon: int, eps: float = 1e-8):
        super().__init__()
        self.horizon = int(horizon)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        is_constant = (x.std(dim=-1, unbiased=False) <= self.eps).all(dim=1)
        if not torch.any(is_constant):
            return prediction
        output = prediction.clone()
        output[is_constant] = x[is_constant, :, -1:].repeat_interleave(
            self.horizon, dim=-1
        )
        return output


def build_covariate_augmentation(config: dict | None) -> CovariateAugmentation:
    if config is None:
        return CovariateAugmentation()
    kwargs = dict(config.get("kwargs") or {})
    modes = config.get("modes")
    return CovariateAugmentation(modes=modes, **kwargs)
