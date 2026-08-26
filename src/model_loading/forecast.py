"""Generic TimeTensor model wrapper and config loading."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .augmentations import (
    RepeatConstantOutput,
    build_covariate_augmentation,
    normalize_covariates,
)
from .baselines import (
    ExpectedBaseline,
    LinearBaseline,
    LookbackBaseline,
    PeriodicLinearBaseline,
    PersistenceBaseline,
    RepeatBaseline,
)
from proposal.normalizations import build_normalization
from external_models import (
    Chronos2,
    ChronosBolt,
    DLinear,
    PatchTST,
    TabPFNTS,
    TiRex2Forecaster,
    TSICLForecaster,
)


FOUNDATION_MODEL_ALIASES = (
    "chronos2",
    "chronos_bolt",
    "ts_icl",
    "tirex2",
    "tabpfn_ts",
)
REMOVED_FOUNDATION_ALIASES = {
    "chronos",
    "chronos-2",
    "chronos-bolt",
    "tsicl",
    "ts-icl",
    "tirex_2",
    "tirex-2",
    "tyrex2",
    "tabpfn",
    "tabpfn-ts",
}


BASELINE_REGISTRY = {
    "persistence": PersistenceBaseline,
    "expected": ExpectedBaseline,
    "repeat": RepeatBaseline,
    "lookback": LookbackBaseline,
    "linear": LinearBaseline,
    "periodic_linear": PeriodicLinearBaseline,
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "chronos2": Chronos2,
    "chronos_bolt": ChronosBolt,
    "tabpfn_ts": TabPFNTS,
    "ts_icl": TSICLForecaster,
    "tirex2": TiRex2Forecaster,
}


def clone_state_dict(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Deep-copy tensors in a state dict so callers cannot mutate modules."""
    copied = OrderedDict()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            copied[key] = value.detach().clone()
        else:
            copied[key] = copy.deepcopy(value)
    return copied


class TimeTensorModel(nn.Module):
    """Compose normalization, covariate augmentation, a base model, and output rules."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        name: str | None = None,
        normalization: nn.Module | None = None,
        covariate_augmentation: nn.Module | None = None,
        output_augmentation: nn.Module | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.name = name or base_model.__class__.__name__
        self.normalization = normalization or build_normalization(None)
        self.covariate_augmentation = covariate_augmentation
        self.output_augmentation = output_augmentation

    def forward(
        self,
        x: torch.Tensor,
        covariates=None,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._set_normalization_cluster_ids(cluster_ids)
        x_norm = self.normalization(x)
        horizon = self._horizon()
        covariates = normalize_covariates(
            covariates,
            lags=x_norm.shape[-1],
            horizon=horizon,
            past=past_covariates,
            future=future_covariates,
            static=static_covariates,
        )
        covariates = (
            self.covariate_augmentation(
                x_norm,
                covariates,
                horizon=horizon,
                **kwargs,
            )
            if self.covariate_augmentation is not None
            else covariates
        )
        prediction = self.base_model(x_norm, covariates=covariates, **kwargs)
        prediction = self.normalization.inverse(prediction)
        if self.output_augmentation is not None:
            prediction = self.output_augmentation(x, prediction)
        return prediction

    def state_dict(self, *args, **kwargs):  # noqa: D401
        """Return a cloned state dict for safety."""
        return clone_state_dict(super().state_dict(*args, **kwargs))

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        safe_state = clone_state_dict(state_dict)
        return super().load_state_dict(safe_state, strict=strict, assign=assign)

    def safe_state_dict(self) -> OrderedDict[str, Any]:
        return self.state_dict()

    def _horizon(self) -> int:
        horizon = getattr(self.base_model, "horizon", None)
        if horizon is None:
            raise ValueError("base model must expose horizon")
        return int(horizon)

    def save_state_dict(self, path: str | Path) -> Path:
        """Save a cloned state dict and return the resolved path."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def architecture(self) -> str:
        """Return a compact, readable architecture summary."""
        lines = [
            f"TimeTensorModel(name={self.name!r})",
            f"  normalization: {self.normalization.__class__.__name__}",
            f"  covariate_augmentation: {self.covariate_augmentation.__class__.__name__ if self.covariate_augmentation is not None else 'None'}",
            f"  base_model: {self.base_model.__class__.__name__}",
            f"  output_augmentation: {self.output_augmentation.__class__.__name__ if self.output_augmentation is not None else 'None'}",
        ]
        return "\n".join(lines)

    def display_architecture(self, print_fn=print) -> str:
        """Print and return the compact architecture summary."""
        text = self.architecture()
        print_fn(text)
        return text

    def representation(
        self,
        x: torch.Tensor,
        covariates=None,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        static_covariates: torch.Tensor | None = None,
        cluster_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return base-model representations through the same wrapper inputs."""
        if not hasattr(self.base_model, "representation"):
            raise AttributeError(f"{self.base_model.__class__.__name__} has no representation()")
        self._set_normalization_cluster_ids(cluster_ids)
        x_norm = self.normalization(x)
        horizon = self._horizon()
        structured = normalize_covariates(
            covariates,
            lags=x_norm.shape[-1],
            horizon=horizon,
            past=past_covariates,
            future=future_covariates,
            static=static_covariates,
        )
        if self.covariate_augmentation is not None:
            structured = self.covariate_augmentation(
                x_norm,
                structured,
                horizon=horizon,
                **kwargs,
            )
        return self.base_model.representation(x_norm, covariates=structured, **kwargs)

    def _set_normalization_cluster_ids(self, cluster_ids: torch.Tensor | None) -> None:
        set_cluster_ids = getattr(self.normalization, "set_cluster_ids", None)
        if callable(set_cluster_ids):
            set_cluster_ids(cluster_ids)


@dataclass(frozen=True)
class ModelConfig:
    """Config loaded from a model YAML file."""

    name: str
    path: Path | str
    config_path: Path | None = None
    class_name: str | None = None
    kwargs: Mapping[str, Any] | None = None
    normalization: Mapping[str, Any] | None = None
    covariate_augmentation: Mapping[str, Any] | None = None
    repeat_constant: bool = False
    state_dict_path: Path | str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        config_path = Path(path).expanduser().resolve()
        data = load_config_dict(config_path)
        if "name" not in data or "path" not in data:
            raise ValueError("model config requires 'name' and 'path'")
        return cls(
            name=str(data["name"]),
            path=data["path"],
            config_path=config_path,
            class_name=data.get("class"),
            kwargs=data.get("kwargs") or {},
            normalization=data.get("normalization"),
            covariate_augmentation=data.get("covariate_augmentation"),
            repeat_constant=bool(data.get("repeat_constant", False)),
            state_dict_path=data.get("state_dict_path"),
        )


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load a small YAML config, using PyYAML when available."""
    config_path = Path(path).expanduser()
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError("model config must contain a mapping")
    return data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used for model configs.

    Supports top-level scalar keys and one-level nested dictionaries with
    two-space indentation. This keeps the package independent from PyYAML for
    simple local configs.
    """
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"invalid YAML line: {raw_line!r}")
        key, value = line.strip().split(":", 1)
        value = value.strip()
        if indent == 0:
            if value == "":
                current = {}
                root[key] = current
            else:
                root[key] = parse_scalar(value)
                current = None
        elif indent == 2 and current is not None:
            current[key] = parse_scalar(value)
        else:
            raise ValueError("only one nested mapping level is supported")
    return root


def parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        return json.loads(value.replace("'", '"'))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def resolve_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def load_object(path: str | Path, *, base_dir: Path | None = None, class_name: str | None = None):
    """Load a class/function from import path or Python file path."""
    path_text = str(path)
    if ":" in path_text:
        module_name, attr = path_text.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    if path_text in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[path_text]
    maybe_file = resolve_path(path_text, base_dir=base_dir)
    if maybe_file.suffix == ".py" and maybe_file.exists():
        spec = importlib.util.spec_from_file_location(maybe_file.stem, maybe_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {maybe_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if class_name is not None:
            return getattr(module, class_name)
        if hasattr(module, "build_model"):
            return getattr(module, "build_model")
        raise ValueError("Python model file must expose build_model or class")
    if "." in path_text:
        module_name, attr = path_text.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise ValueError(f"cannot resolve model path {path!r}")


def build_base_model(config: ModelConfig) -> nn.Module:
    name = str(config.name)
    path = str(config.path)
    name_key = name.lower()
    path_key = path.lower()
    if name_key in REMOVED_FOUNDATION_ALIASES or path_key in REMOVED_FOUNDATION_ALIASES:
        raise ValueError("removed foundation-model alias; use the canonical snake_case name")
    if name_key in FOUNDATION_MODEL_ALIASES or path_key in FOUNDATION_MODEL_ALIASES:
        if name != path or name != name_key:
            raise ValueError(
                "foundation model name and path must be the same canonical alias"
            )
    base_dir = None if config.config_path is None else config.config_path.parent
    target = load_object(config.path, base_dir=base_dir, class_name=config.class_name)
    foundation_targets = {
        BASELINE_REGISTRY[alias] for alias in FOUNDATION_MODEL_ALIASES
    }
    if target in foundation_targets and path not in FOUNDATION_MODEL_ALIASES:
        raise ValueError(
            "foundation adapters must be selected by their canonical alias, not an import path"
        )
    kwargs = dict(config.kwargs or {})
    if isinstance(target, type):
        model = target(**kwargs)
    else:
        model = target(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    return model


def build_model_from_config(
    path_or_config: str | Path | ModelConfig,
    *,
    state_dict: Mapping[str, Any] | None = None,
    state_dict_path: str | Path | None = None,
    normalization_stats: Mapping[str, Any] | None = None,
) -> TimeTensorModel:
    config = (
        path_or_config
        if isinstance(path_or_config, ModelConfig)
        else ModelConfig.from_yaml(path_or_config)
    )
    base = build_base_model(config)
    dim = getattr(base, "dim", None)
    normalization = build_normalization(
        dict(config.normalization) if config.normalization is not None else None,
        dim=dim,
        stats=normalization_stats,
    )
    covariate_augmentation = (
        build_covariate_augmentation(dict(config.covariate_augmentation))
        if config.covariate_augmentation is not None
        else None
    )
    output_augmentation = None
    if config.repeat_constant:
        horizon = getattr(base, "horizon", None)
        if horizon is None:
            raise ValueError("repeat_constant requires base model to expose horizon")
        output_augmentation = RepeatConstantOutput(horizon)
    model = TimeTensorModel(
        base,
        name=config.name,
        normalization=normalization,
        covariate_augmentation=covariate_augmentation,
        output_augmentation=output_augmentation,
    )
    if state_dict is not None and state_dict_path is not None:
        raise ValueError("provide state_dict or state_dict_path, not both")
    path_from_config = config.state_dict_path
    load_path = state_dict_path if state_dict_path is not None else path_from_config
    if state_dict is not None:
        model.load_state_dict(state_dict)
    elif load_path is not None:
        base_dir = None if config.config_path is None else config.config_path.parent
        state_path = resolve_path(load_path, base_dir=base_dir)
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    return model


def load_model(
    specs: str | Path | ModelConfig,
    *,
    state_dict: Mapping[str, Any] | None = None,
    state_dict_path: str | Path | None = None,
    normalization_stats: Mapping[str, Any] | None = None,
) -> TimeTensorModel:
    """Load a wrapped model from specs and optional state dict."""
    return build_model_from_config(
        specs,
        state_dict=state_dict,
        state_dict_path=state_dict_path,
        normalization_stats=normalization_stats,
    )
