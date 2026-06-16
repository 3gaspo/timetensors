"""Shared runtime helpers for TimeTensor scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .models import ModelConfig


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
    """Return whether raw data should be rebuilt into tensor artifacts.

    Prefer ``experiment.rebuild_dataset``. Older keys are kept as aliases so
    existing commands keep working while new scripts use one spelling.
    """
    experiment = section(config, "experiment")
    data = section(config, "data")
    for key in ("rebuild_dataset",):
        if key in experiment:
            return config_bool(experiment[key])
        if key in data:
            return config_bool(data[key])
    if config_bool(experiment.get("skip_dataset_build", False)):
        return False
    return config_bool(
        experiment.get(
            "build_dataset",
            data.get("rebuild", data.get("build", False)),
        )
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(config: Mapping[str, Any]) -> Path:
    data = section(config, "data")
    return Path(data.get("built_path", data.get("path", "run/dataset"))).expanduser()


def output_dir(config: Mapping[str, Any]) -> Path:
    output = section(config, "output")
    misc = section(config, "misc")
    return ensure_dir(output.get("dir", misc.get("output_dir", "outputs")))


def save_name(config: Mapping[str, Any]) -> str:
    output = section(config, "output")
    misc = section(config, "misc")
    if output.get("name") is not None:
        return str(output["name"])
    if misc.get("save_name") is not None:
        return str(misc["save_name"])
    model = section(config, "model")
    return str(model.get("name", "model"))


def run_dir(config: Mapping[str, Any]) -> Path:
    return ensure_dir(output_dir(config) / save_name(config))


def default_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    data = section(config, "data")
    splits = dict(data.get("splits") or section(config, "splits"))
    splits.setdefault("date_splits", [0.6, 0.2, 0.2])
    splits.setdefault("indiv_split", 1.0)
    splits.setdefault("reshuffle", True)
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
    return int(training.get("batch_size", training.get("bs", 32)))


def seed(config: Mapping[str, Any]) -> int | None:
    misc = section(config, "misc")
    value = section(config, "experiment").get("seed", misc.get("seed"))
    return None if value in {None, "None"} else int(value)


def device(config: Mapping[str, Any]) -> str:
    training = section(config, "training")
    misc = section(config, "misc")
    raw = str(training.get("device", misc.get("device", "auto")))
    return "gpu" if raw == "cuda" else raw


def save_json(value: Any, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    return path


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
    kwargs = dict(model.get("kwargs") or model.get("configs") or {})
    key = path.lower()
    if key in {"persistence", "expected", "repeat"}:
        kwargs.setdefault("horizon", horizon)
    elif key == "lookback":
        kwargs.setdefault("horizon", horizon)
    else:
        kwargs.setdefault("lags", lags)
        kwargs.setdefault("dim", dim)
        kwargs.setdefault("horizon", horizon)
    normalization = section(config, "normalization")
    if normalization.get("name") in {None, "None", "none"} and not normalization.get("kwargs"):
        normalization_config = None
    else:
        normalization_config = {
            "name": normalization.get("name", "identity"),
            "kwargs": normalization.get("kwargs", normalization.get("configs", {})) or {},
        }
    augmentation = model.get("covariate_augmentation", model.get("augmentation"))
    return ModelConfig(
        name=name,
        path=path,
        kwargs=kwargs,
        normalization=normalization_config,
        covariate_augmentation=augmentation,
        repeat_constant=bool(model.get("repeat_constant", False)),
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
