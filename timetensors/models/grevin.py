"""Generalized RevIN normalization for the new TimeTensor pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .normalizations import get_normal_stats


class GRevINNormalization(nn.Module):
    """Generalized reversible instance normalization.

    The module follows the same contract as the other new-pipeline
    normalizations: ``forward(x)`` normalizes and caches lookback statistics,
    then ``inverse(y)`` maps model outputs back to the original value space.

    Formula:

    ``x_hat = (x - a * mean(x)) * (1 + b * (1 / (std(x) + eps) - 1))``
    ``x_mod = gamma * x_hat + nu``

    ``y_aff = alpha * y + beta``
    ``out = y_aff / (1 + d * (1 / (std(x) + eps) - 1)) + c * mean(x)``

    Setting ``a=b=c=d=1`` gives instance normalization with an exact inverse.
    Setting ``tie_revin=True`` ties ``c,d`` to ``a,b`` and makes the output
    affine the inverse of the input affine, reproducing classical RevIN when
    ``a=b=1``.
    """

    name = "grevin"

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-8,
        detach_stats: bool = True,
        center: str = "mean",
        start_in: bool = True,
        tie_revin: bool = False,
        personalize: str = "none",
        n_clusters: int | None = None,
        unknown_cluster_id: int | None = None,
        clamp_gamma_eps: float = 1e-6,
    ):
        super().__init__()
        if center not in {"mean", "last"}:
            raise ValueError("center must be 'mean' or 'last'")
        if personalize not in {"none", "affine", "all"}:
            raise ValueError("personalize must be 'none', 'affine', or 'all'")
        self.dim = int(dim)
        self.eps = float(eps)
        self.detach_stats = bool(detach_stats)
        self.center = center
        self.tie_revin = bool(tie_revin)
        self.personalize = personalize
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        self.unknown_cluster_id = unknown_cluster_id
        self.clamp_gamma_eps = float(clamp_gamma_eps)

        self.a = self._gate_parameter(start_in, self.dim)
        self.b = self._gate_parameter(start_in, self.dim)
        if self.tie_revin:
            self.register_parameter("c", None)
            self.register_parameter("d", None)
        else:
            self.c = self._gate_parameter(start_in, self.dim)
            self.d = self._gate_parameter(start_in, self.dim)

        self.gamma = nn.Parameter(torch.ones(1, self.dim, 1))
        self.nu = nn.Parameter(torch.zeros(1, self.dim, 1))
        self.alpha = nn.Parameter(torch.ones(1, self.dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.dim, 1))

        has_clusters = self.n_clusters is not None and self.n_clusters > 0
        self._has_clusters = has_clusters and personalize != "none"
        if self._has_clusters:
            if personalize in {"affine", "all"}:
                self.gamma_k = nn.Parameter(torch.ones(self.n_clusters, self.dim, 1))
                self.nu_k = nn.Parameter(torch.zeros(self.n_clusters, self.dim, 1))
                self.alpha_k = nn.Parameter(torch.ones(self.n_clusters, self.dim, 1))
                self.beta_k = nn.Parameter(torch.zeros(self.n_clusters, self.dim, 1))
            if personalize == "all":
                gate_init = 1.0 if start_in else 0.0
                self.a_k = nn.Parameter(torch.full((self.n_clusters, self.dim, 1), gate_init))
                self.b_k = nn.Parameter(torch.full((self.n_clusters, self.dim, 1), gate_init))
                if not self.tie_revin:
                    self.c_k = nn.Parameter(torch.full((self.n_clusters, self.dim, 1), gate_init))
                    self.d_k = nn.Parameter(torch.full((self.n_clusters, self.dim, 1), gate_init))

        self._center: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._cluster_ids: torch.Tensor | None = None
        self._pending_cluster_ids: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        cluster_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        center, std = self._stats(x)
        self._center = center
        self._std = std
        self._cluster_ids = self._resolve_cluster_ids(cluster_ids, x)

        a, b, _, _, gamma, nu, _, _ = self._select_params(x, self._cluster_ids)
        inv_sigma = self._partial_inverse_sigma(std, b)
        x_hat = (x - a * center) * inv_sigma
        return gamma * x_hat + nu

    def inverse(
        self,
        y: torch.Tensor,
        *,
        cluster_ids: torch.Tensor | None = None,
        affine_only: bool = False,
    ) -> torch.Tensor:
        if self._center is None or self._std is None:
            raise RuntimeError("GRevIN statistics are not available")
        resolved_cluster_ids = (
            self._resolve_cluster_ids(cluster_ids, y)
            if cluster_ids is not None or self._cluster_ids is None
            else self._cluster_ids
        )
        a, b, c, d, gamma, nu, alpha, beta = self._select_params(y, resolved_cluster_ids)
        if self.tie_revin:
            gamma_safe = gamma.clamp_min(self.clamp_gamma_eps)
            alpha = 1.0 / gamma_safe
            beta = -nu / gamma_safe
            c, d = a, b

        y_aff = alpha * y + beta
        if affine_only:
            return y_aff
        inv_sigma = self._partial_inverse_sigma(self._std, d)
        return y_aff / inv_sigma + c * self._center

    def set_cluster_ids(self, cluster_ids: torch.Tensor | None) -> None:
        """Set cluster ids for the next ``forward`` call.

        This keeps the current pipeline usable without changing the public
        normalization contract. A later pipeline patch can call this before
        ``normalization(x)`` when batch metadata contains ``cluster_ids``.
        """

        self._pending_cluster_ids = None if cluster_ids is None else torch.as_tensor(cluster_ids).detach()

    def clear_context(self) -> None:
        self._pending_cluster_ids = None
        self._cluster_ids = None

    def freeze(self, groups: str | list[str], freeze: bool = True) -> "GRevINNormalization":
        """Freeze parameter groups by name.

        Groups: ``ab``, ``cd``, ``gamma_nu``, ``alpha_beta``.
        Personalized variants are frozen with their shared counterparts.
        """

        if isinstance(groups, str):
            groups = [groups]
        requires_grad = not freeze
        group_names = {
            "ab": ("a", "b", "a_k", "b_k"),
            "cd": ("c", "d", "c_k", "d_k"),
            "gamma_nu": ("gamma", "nu", "gamma_k", "nu_k"),
            "alpha_beta": ("alpha", "beta", "alpha_k", "beta_k"),
        }
        for group in groups:
            if group not in group_names:
                raise ValueError(f"unknown freeze group {group!r}")
            for name in group_names[group]:
                param = getattr(self, name, None)
                if param is not None:
                    param.requires_grad_(requires_grad)
        return self

    def get_params(
        self,
        cluster_ids: int | torch.Tensor | None = None,
        *,
        clamp: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return effective parameters for inspection or logging."""

        if isinstance(cluster_ids, int):
            cluster_tensor = torch.tensor([cluster_ids], device=self.gamma.device)
        else:
            cluster_tensor = cluster_ids
        probe = self.gamma[:1]
        if cluster_tensor is not None:
            probe = probe.expand(int(cluster_tensor.numel()), -1, -1)
        a, b, c, d, gamma, nu, alpha, beta = self._select_params(
            probe,
            cluster_tensor,
            clamp=clamp,
        )
        if self.tie_revin:
            gamma_safe = gamma.clamp_min(self.clamp_gamma_eps)
            alpha = 1.0 / gamma_safe
            beta = -nu / gamma_safe
            c, d = a, b
        return {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def init_from_stats(
        self,
        stats: Mapping[str, Any],
        *,
        alpha_key: str = "alpha",
        beta_key: str = "beta",
    ) -> "GRevINNormalization":
        """Initialize output affine parameters from loader statistics.

        New dataset stats expose scalar ``alpha`` and ``beta`` where
        ``alpha = std(future) / std(lookback)`` and
        ``beta = (mean(future) - mean(lookback)) / std(lookback)``. Those map
        naturally to Grevin's post-model affine ``alpha`` and ``beta``.
        Non-finite values are ignored.
        """

        alpha = _finite_float(stats.get(alpha_key))
        beta = _finite_float(stats.get(beta_key))
        with torch.no_grad():
            if alpha is not None:
                self.alpha.fill_(alpha)
                if hasattr(self, "alpha_k"):
                    self.alpha_k.fill_(alpha)
            if beta is not None:
                self.beta.fill_(beta)
                if hasattr(self, "beta_k"):
                    self.beta_k.fill_(beta)
        return self

    @classmethod
    def build_in(cls, dim: int, **kwargs: Any) -> "GRevINNormalization":
        module = cls(dim, start_in=True, tie_revin=False, personalize="none", **kwargs)
        module.freeze(["ab", "cd", "gamma_nu", "alpha_beta"], freeze=True)
        return module

    @classmethod
    def build_revin(cls, dim: int, **kwargs: Any) -> "GRevINNormalization":
        module = cls(dim, start_in=True, tie_revin=True, personalize="none", **kwargs)
        module.freeze(["ab", "cd", "alpha_beta"], freeze=True)
        module.freeze("gamma_nu", freeze=False)
        return module

    @classmethod
    def build_personalized_revin(
        cls,
        dim: int,
        *,
        n_clusters: int,
        **kwargs: Any,
    ) -> "GRevINNormalization":
        module = cls(
            dim,
            start_in=True,
            tie_revin=True,
            personalize="affine",
            n_clusters=n_clusters,
            **kwargs,
        )
        module.freeze(["ab", "cd", "alpha_beta"], freeze=True)
        module.freeze("gamma_nu", freeze=False)
        return module

    @classmethod
    def build_cmin(
        cls,
        dim: int,
        *,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> "GRevINNormalization":
        personalize = "affine" if n_clusters is not None and int(n_clusters) > 0 else "none"
        module = cls(
            dim,
            start_in=True,
            tie_revin=False,
            personalize=personalize,
            n_clusters=n_clusters,
            **kwargs,
        )
        module.freeze(["ab", "cd", "gamma_nu"], freeze=True)
        module.freeze("alpha_beta", freeze=False)
        return module

    @staticmethod
    def _gate_parameter(start_in: bool, dim: int) -> nn.Parameter:
        value = 1.0 if start_in else 0.0
        return nn.Parameter(torch.full((1, dim, 1), value))

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

    def _partial_inverse_sigma(self, std: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return 1.0 + gate * (1.0 / (std + self.eps) - 1.0)

    def _resolve_cluster_ids(
        self,
        cluster_ids: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if cluster_ids is None:
            cluster_ids = self._pending_cluster_ids
        if cluster_ids is None:
            return None
        cluster_ids = cluster_ids.to(device=x.device, dtype=torch.long).flatten()
        if cluster_ids.numel() != x.shape[0]:
            raise ValueError(
                f"cluster_ids has {cluster_ids.numel()} entries, expected batch size {x.shape[0]}"
            )
        return cluster_ids

    def _select_params(
        self,
        x: torch.Tensor,
        cluster_ids: torch.Tensor | None,
        *,
        clamp: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        batch = x.shape[0]
        shared = self._shared_params(x, batch, clamp=clamp)
        if not self._has_clusters or cluster_ids is None:
            return shared

        cluster_ids = cluster_ids.to(device=x.device, dtype=torch.long).flatten()
        a, b, c, d, gamma, nu, alpha, beta = shared
        valid = self._valid_cluster_mask(cluster_ids)
        safe_ids = cluster_ids.clamp(0, int(self.n_clusters) - 1)

        if self.personalize in {"affine", "all"}:
            gamma = torch.where(valid, self._cluster_param(self.gamma_k, safe_ids, x), gamma)
            nu = torch.where(valid, self._cluster_param(self.nu_k, safe_ids, x), nu)
            alpha = torch.where(valid, self._cluster_param(self.alpha_k, safe_ids, x), alpha)
            beta = torch.where(valid, self._cluster_param(self.beta_k, safe_ids, x), beta)

        if self.personalize == "all":
            a = torch.where(valid, self._maybe_clamp(self._cluster_param(self.a_k, safe_ids, x), clamp), a)
            b = torch.where(valid, self._maybe_clamp(self._cluster_param(self.b_k, safe_ids, x), clamp), b)
            if self.tie_revin:
                c, d = a, b
            else:
                c = torch.where(valid, self._maybe_clamp(self._cluster_param(self.c_k, safe_ids, x), clamp), c)
                d = torch.where(valid, self._maybe_clamp(self._cluster_param(self.d_k, safe_ids, x), clamp), d)
        return a, b, c, d, gamma, nu, alpha, beta

    def _shared_params(
        self,
        x: torch.Tensor,
        batch: int,
        *,
        clamp: bool,
    ) -> tuple[torch.Tensor, ...]:
        def expand(param: torch.Tensor) -> torch.Tensor:
            return param.to(device=x.device, dtype=x.dtype).expand(batch, self.dim, 1)

        a = expand(self._maybe_clamp(self.a, clamp))
        b = expand(self._maybe_clamp(self.b, clamp))
        if self.tie_revin:
            c, d = a, b
        else:
            assert self.c is not None and self.d is not None
            c = expand(self._maybe_clamp(self.c, clamp))
            d = expand(self._maybe_clamp(self.d, clamp))
        gamma = expand(self.gamma)
        nu = expand(self.nu)
        alpha = expand(self.alpha)
        beta = expand(self.beta)
        return a, b, c, d, gamma, nu, alpha, beta

    @staticmethod
    def _maybe_clamp(param: torch.Tensor, clamp: bool) -> torch.Tensor:
        return param.clamp(0.0, 1.0) if clamp else param

    @staticmethod
    def _cluster_param(param: torch.Tensor, cluster_ids: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return param[cluster_ids].to(device=like.device, dtype=like.dtype)

    def _valid_cluster_mask(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        assert self.n_clusters is not None
        valid = (cluster_ids >= 0) & (cluster_ids < self.n_clusters)
        if self.unknown_cluster_id is not None:
            valid &= cluster_ids != int(self.unknown_cluster_id)
        return valid.view(-1, 1, 1)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach().cpu())
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_grevin_normalization(
    mode: str,
    dim: int,
    *,
    stats: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> GRevINNormalization:
    """Build a Grevin-family normalization by preset name."""

    key = mode.lower().replace("-", "_")
    if key in {"grevin", "grevin_normalization", "generalized_revin"}:
        module = GRevINNormalization(dim, **kwargs)
    elif key in {"in", "instance"}:
        module = GRevINNormalization.build_in(dim, **kwargs)
    elif key == "revin":
        module = GRevINNormalization.build_revin(dim, **kwargs)
    elif key in {"previn", "personalized_revin"}:
        module = GRevINNormalization.build_personalized_revin(dim, **kwargs)
    elif key == "cmin":
        module = GRevINNormalization.build_cmin(dim, **kwargs)
    else:
        raise ValueError(f"unknown Grevin mode {mode!r}")
    if stats is not None:
        module.init_from_stats(stats)
    return module


__all__ = ["GRevINNormalization", "build_grevin_normalization"]
