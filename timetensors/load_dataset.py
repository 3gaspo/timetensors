"""Dataset-building stage for TimeTensor experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .dataset import (
    build_dataset,
    fetch_training_data,
    get_sizes,
    load_data,
    save_data,
)
from .runtime import (
    batch_size,
    dataset_path,
    default_sampling,
    default_splits,
    default_subsets,
    rebuild_dataset,
    run_dir,
    section,
    seed,
    setup_logging,
    task_shape,
    to_plain_config,
)


LOGGER = logging.getLogger(__name__)


def _data_shape_summary(data: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in ("values", "individual_context", "global_context", "individual_ids"):
        value = getattr(data, name, None)
        if value is not None and hasattr(value, "shape"):
            summary[name] = list(value.shape)
    if not summary and isinstance(data, Mapping):
        for key, value in data.items():
            if hasattr(value, "shape"):
                summary[str(key)] = list(value.shape)
    return summary


def build_dataset_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build or load tensors, then optionally construct loaders and stats."""
    config = to_plain_config(config)
    LOGGER.info("===== Running dataset script =====")
    data_cfg = section(config, "data")
    experiment = section(config, "experiment")
    out_path = dataset_path(config)
    LOGGER.info(
        "Dataset configuration: path=%s rebuild_dataset=%s prepare_loaders=%s",
        out_path,
        rebuild_dataset(config),
        experiment.get("prepare_loaders", data_cfg.get("prepare_loaders", True)),
    )
    if rebuild_dataset(config):
        data_name = data_cfg.get("name")
        if data_name is None:
            raise ValueError("data.name is required to rebuild tensors")
        raw_path = Path(data_cfg.get("raw_path", "."))
        LOGGER.info("Building tensor dataset: name=%s raw_path=%s output_path=%s", data_name, raw_path, out_path)
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
        LOGGER.info("Tensor dataset built: %s", out_path)
    else:
        LOGGER.info("Dataset rebuild disabled; loading existing tensors from %s", out_path)
    data = load_data(
        out_path,
        prefix=data_cfg.get("prefix", ""),
        legacy_context_kind=data_cfg.get("legacy_context_kind"),
    )
    LOGGER.info("Loaded tensor dataset: %s", _data_shape_summary(data))
    result: dict[str, Any] = {"data": data, "dataset_path": out_path}
    if bool(data_cfg.get("save_loaded_copy", False)):
        save_data(data, out_path, prefix=data_cfg.get("prefix", ""))
        LOGGER.info("Saved loaded tensor copy: %s", out_path)
    if bool(experiment.get("prepare_loaders", data_cfg.get("prepare_loaders", True))):
        lags, horizon = task_shape(config)
        LOGGER.info("Preparing loaders: lags=%s horizon=%s batch_size=%s", lags, horizon, batch_size(config))
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
            legacy_context_kind=data_cfg.get("legacy_context_kind"),
        )
        result["loaders"] = loaders
        result["stats"] = stats
        result["shape"] = get_sizes(loaders)
        try:
            _, split_info, batch_info = get_sizes(loaders, str_info=True)
            LOGGER.info("Prepared loaders:\n%s", split_info)
            LOGGER.info("Example batch:\n%s", batch_info)
        except Exception as exc:
            LOGGER.debug("Could not log detailed loader sizes: %s", exc)
        LOGGER.info("Loader preparation finished: shape=%s", result["shape"])
    else:
        LOGGER.info("Loader preparation skipped")
    LOGGER.info("End of dataset script")
    return result


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    setup_logging(section(to_plain_config(config or {}), "misc").get("log_level", "INFO"))
    return build_dataset_stage(config or {})


try:
    import hydra  # type: ignore
except Exception:  # pragma: no cover - import-time fallback for minimal envs
    hydra = None


if hydra is not None:

    @hydra.main(version_base=None, config_path=None, config_name=None)
    def _hydra_main(cfg):
        main(cfg)


if __name__ == "__main__":
    if hydra is None:
        main({})
    else:
        _hydra_main()
