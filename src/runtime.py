"""Shared runtime helpers for TimeTensor scripts."""

from __future__ import annotations

import logging
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

from models.models import ModelConfig


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def to_plain_config(config: Any) -> dict[str, Any]:
    """Convert OmegaConf/dicts/namespaces into plain dictionaries."""
    try:
        from omegaconf import OmegaConf  # type: ignore

        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return {str(key): to_plain_config(value) for key, value in config.items()}
    if isinstance(config, list):
        return [to_plain_config(value) for value in config]  # type: ignore[return-value]
    if hasattr(config, "__dict__"):
        return {
            key: to_plain_config(value)
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    return config  # type: ignore[return-value]


def section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        return {"name": value}
    return dict(value)


def config_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def rebuild_dataset(config: Mapping[str, Any]) -> bool:
    """Return whether raw data should be rebuilt into tensor artifacts."""
    experiment = section(config, "experiment")
    return config_bool(experiment.get("rebuild_dataset", False))


def recompute_stats(config: Mapping[str, Any]) -> bool:
    """Return whether L/H-dependent dataset statistics should be computed."""
    experiment = section(config, "experiment")
    return config_bool(experiment.get("recompute_stats", True)) or normalization_needs_stats(config)


def normalization_needs_stats(config: Mapping[str, Any]) -> bool:
    """Return whether model construction needs loader statistics."""
    normalization = section(config, "normalization")
    name = str(normalization.get("name", "identity"))
    if name in {"standard", "min-max"}:
        return True
    if name not in {"grevin", "cmin", "previn"}:
        return False
    kwargs = dict(normalization.get("kwargs") or {})
    if kwargs.get("stats") is not None:
        return False
    return config_bool(kwargs.get("init_from_stats", False))


def stats_max_windows(config: Mapping[str, Any]) -> int | None:
    experiment = section(config, "experiment")
    value = experiment.get("stats_max_windows")
    return None if value in {None, "None", "none", ""} else int(value)


def stats_seed(config: Mapping[str, Any]) -> int | None:
    return seed(config)


def stats_eps(config: Mapping[str, Any]) -> float:
    experiment = section(config, "experiment")
    return float(experiment.get("stats_eps", 1e-8))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(config: Mapping[str, Any]) -> Path:
    data = section(config, "data")
    return Path(data.get("path", "run/dataset")).expanduser()


def output_dir(config: Mapping[str, Any]) -> Path:
    output = section(config, "output")
    return ensure_dir(output.get("dir", "outputs/manual_debug"))


def save_name(config: Mapping[str, Any]) -> str:
    output = section(config, "output")
    if output.get("name") is not None:
        return str(output["name"])
    model = section(config, "model")
    return str(model.get("name", "model"))


def run_dir(config: Mapping[str, Any]) -> Path:
    return ensure_dir(output_dir(config) / save_name(config))


def seeded_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand ``experiment.seeds`` and isolate every run under ``seed_N``."""
    config = to_plain_config(config)
    experiment = section(config, "experiment")
    values = experiment.get("seeds")
    if values is None or (isinstance(values, str) and values in {"None", "none", ""}):
        return [config]
    if isinstance(values, str):
        values = [value for value in re.split(r"[,;]", values) if value.strip()]
    base_name = save_name(config)
    expanded = []
    for index, value in enumerate(values):
        seeded = deepcopy(config)
        seeded.setdefault("experiment", {}).pop("seeds", None)
        seeded["experiment"]["seed"] = int(value)
        if index:
            seeded["experiment"]["rebuild_dataset"] = False
        seeded.setdefault("output", {})["name"] = (
            f"{base_name}/seed_{int(value)}" if base_name else f"seed_{int(value)}"
        )
        expanded.append(seeded)
    return expanded


def default_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    data = section(config, "data")
    splits = dict(data.get("splits") or section(config, "splits"))
    splits.setdefault("date_splits", [0.6, 0.2, 0.2])
    splits.setdefault("indiv_split", 1.0)
    return splits


def default_sampling(config: Mapping[str, Any]) -> dict[str, Any]:
    data = section(config, "data")
    sampling = dict(data.get("sampling") or section(config, "sampling"))
    sampling.setdefault("train_idx_mode", "random")
    sampling.setdefault("eval_idx_mode", "all")
    sampling.setdefault("train_stride", 1)
    sampling.setdefault("eval_stride", 1)
    sampling.setdefault("shuffle_train", True)
    sampling.setdefault("shuffle_eval", False)
    sampling.setdefault("remove_train_cte", False)
    sampling.setdefault("remove_eval_cte", False)
    sampling.setdefault("drop_train_constant_individuals", False)
    sampling.setdefault("drop_eval_constant_individuals", False)
    sampling.setdefault("train_block_individuals", 1)
    sampling.setdefault("eval_block_individuals", 1)
    sampling.setdefault("use_individual_context", True)
    sampling.setdefault("use_global_context", True)
    return sampling


def default_subsets(config: Mapping[str, Any]) -> dict[str, Any]:
    data = section(config, "data")
    return dict(data.get("subsets") or section(config, "subsets"))


def task_shape(config: Mapping[str, Any]) -> tuple[int, int]:
    task = section(config, "task")
    return int(task.get("lags", 168)), int(task.get("horizon", 24))


def batch_size(config: Mapping[str, Any]) -> int:
    training = section(config, "training")
    return int(training.get("batch_size", 256))


def seed(config: Mapping[str, Any]) -> int | None:
    value = section(config, "experiment").get("seed")
    return None if value in {None, "None"} else int(value)


def device(config: Mapping[str, Any]) -> str:
    training = section(config, "training")
    return str(training.get("device", "auto"))


def save_torch(value: Any, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return path


def model_specs(config: Mapping[str, Any], shape: tuple[int, int, int]) -> str | Path | ModelConfig:
    model = section(config, "model")
    if model.get("specs") is not None:
        return model["specs"]
    lags, dim, horizon = shape
    name = str(model.get("name", model.get("path", "linear")))
    path = str(model.get("path", name))
    kwargs = dict(model.get("kwargs") or {})
    key = path.lower()
    if key in {"persistence", "expected", "repeat"}:
        kwargs.setdefault("horizon", horizon)
    elif key == "lookback":
        kwargs.setdefault("horizon", horizon)
    elif key == "periodic_linear":
        kwargs.setdefault("lags", lags)
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("horizon", horizon)
        kwargs.setdefault("period", max(1, min(lags, 168)))
    else:
        kwargs.setdefault("lags", lags)
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("horizon", horizon)
    normalization = section(config, "normalization")
    if normalization.get("name") is None and not normalization.get("kwargs"):
        normalization_config = None
    else:
        normalization_config = {
            "name": normalization.get("name", "identity"),
            "kwargs": normalization.get("kwargs", {}) or {},
        }
    return ModelConfig(
        name=name,
        path=path,
        kwargs=kwargs,
        normalization=normalization_config,
        covariate_augmentation=model.get("covariate_augmentation"),
        repeat_constant=config_bool(model.get("repeat_constant", False)),
        state_dict_path=model.get("state_dict_path"),
    )


def pretrained_path(config: Mapping[str, Any]) -> str | Path | None:
    experiment = section(config, "experiment")
    training = section(config, "training")
    model = section(config, "model")
    value = (
        experiment.get("pretrained_path")
        or training.get("pretrained_path")
        or training.get("init")
        or model.get("state_dict_path")
    )
    return None if value in {None, "None", "none", ""} else value
