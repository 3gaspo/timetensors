"""Dataset-building stage for TimeTensor experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .io import build_dataset, load_data, save_data
from .loaders import fetch_training_data, get_sizes
from pipeline.runtime import (
    batch_size,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    recompute_stats,
    rebuild_dataset,
    run_dir,
    section,
    seed,
    setup_logging,
    stats_eps,
    stats_max_windows,
    stats_seed,
    task_shape,
    to_plain_config,
)


LOGGER = logging.getLogger(__name__)


DATASET_CONFIG_KEYS = {
    "global_context_cols",
    "drop_users",
    "build_individual_ids_context",
    "rename_cols",
    "aggr",
    "aggr_period",
    "users_dim",
    "date_col",
    "dates",
    "drop",
    "prefix",
}


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _merge_drop_users(*values: Any) -> list[Any]:
    merged: list[Any] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_list(value):
            key = str(item)
            if key not in seen:
                merged.append(item)
                seen.add(key)
    return merged


def _dataset_config_path(data_cfg: Mapping[str, Any]) -> tuple[Path | None, bool]:
    config_path = data_cfg.get("config_path")
    if config_path not in {None, ""}:
        path = Path(str(config_path)).expanduser()
        return (path / "config.json" if path.is_dir() else path), True
    base = data_cfg.get("raw_path") or data_cfg.get("path")
    if base in {None, ""}:
        return None, False
    base_path = Path(str(base)).expanduser()
    directory = base_path.parent if base_path.suffix.lower() == ".csv" else base_path
    return directory / "config.json", False


def _dataset_config_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: raw[key] for key in DATASET_CONFIG_KEYS if key in raw}
    scoped = raw.get("timetensors")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'timetensors' must be an object")
        if "drop_users" in scoped:
            options["drop_users"] = _merge_drop_users(
                options.get("drop_users"), scoped["drop_users"]
            )
        options.update(
            {
                key: value
                for key, value in scoped.items()
                if key in DATASET_CONFIG_KEYS and key != "drop_users"
            }
        )
    return options


def _merge_dataset_config(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    path, explicit = _dataset_config_path(data_cfg)
    if path is None or not path.exists():
        if explicit:
            raise FileNotFoundError(path)
        return dict(data_cfg)
    if path.suffix.lower() != ".json":
        raise ValueError(f"dataset config must be JSON, got {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset config must contain a JSON object: {path}")
    loaded = _dataset_config_options(raw)
    merged = dict(loaded)
    explicit = {key: value for key, value in data_cfg.items() if value is not None}
    merged.update({key: value for key, value in explicit.items() if key != "drop_users"})
    merged["drop_users"] = _merge_drop_users(
        loaded.get("drop_users"), explicit.get("drop_users")
    )
    merged["config_path"] = str(path)
    LOGGER.info("loaded dataset config path=%s keys=%s", path, sorted(loaded))
    return merged


def build_dataset_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build or load tensors, then optionally construct loaders and stats."""
    config = to_plain_config(config)
    data_cfg = _merge_dataset_config(section(config, "data"))
    config = {**config, "data": data_cfg}
    experiment = section(config, "experiment")
    out_path = dataset_path(config)
    if rebuild_dataset(config):
        raw_path = Path(data_cfg.get("raw_path", "."))
        data_name = (
            raw_path.stem
            if raw_path.suffix.lower() == ".csv"
            else data_cfg.get("name")
        )
        if data_name is None:
            raise ValueError("data.name is required to rebuild tensors")
        if raw_path.suffix.lower() == ".csv":
            raw_path = raw_path.parent
        LOGGER.debug("building tensors name=%s raw_path=%s output_path=%s", data_name, raw_path, out_path)
        build_dataset(
            raw_path,
            str(data_name),
            global_context_cols=data_cfg.get("global_context_cols"),
            drop_users=data_cfg.get("drop_users"),
            build_individual_ids_context=bool(data_cfg.get("build_individual_ids_context", False)),
            rename_cols=data_cfg.get("rename_cols"),
            aggr=data_cfg.get("aggr"),
            aggr_period=data_cfg.get("aggr_period", "h"),
            users_dim=int(data_cfg.get("users_dim", 1)),
            date_col=data_cfg.get("date_col"),
            dates=data_cfg.get("dates"),
            drop=data_cfg.get("drop"),
            prefix=data_cfg.get("prefix", ""),
            output_path=out_path,
        )
    data = load_data(
        out_path,
        prefix=data_cfg.get("prefix", ""),
        legacy_context_kind=data_cfg.get("legacy_context_kind"),
    )
    result: dict[str, Any] = {"data": data, "dataset_path": out_path}
    if bool(data_cfg.get("save_loaded_copy", False)):
        save_data(data, out_path, prefix=data_cfg.get("prefix", ""))
        LOGGER.debug("saved loaded tensor copy: %s", out_path)
    if bool(experiment.get("prepare_loaders", data_cfg.get("prepare_loaders", True))):
        lags, horizon = task_shape(config)
        loaders, stats = fetch_training_data(
            out_path,
            default_splits(config),
            default_sampling(config),
            default_subsets(config),
            batch_size(config),
            lags,
            horizon,
            seed=seed(config),
            stats_save_path=run_dir(config) / "dataset_artifacts",
            compute_stats=recompute_stats(config),
            stats_max_windows=stats_max_windows(config),
            stats_seed=stats_seed(config),
            stats_eps=stats_eps(config),
            legacy_context_kind=data_cfg.get("legacy_context_kind"),
        )
        result["loaders"] = loaders
        result["stats"] = stats
        result["shape"] = get_sizes(loaders)
    return result


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return build_dataset_stage(config or {})
