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


def build_dataset_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build or load tensors, then optionally construct loaders and stats."""
    config = to_plain_config(config)
    data_cfg = section(config, "data")
    experiment = section(config, "experiment")
    out_path = dataset_path(config)
    if rebuild_dataset(config):
        data_name = data_cfg.get("name")
        if data_name is None:
            raise ValueError("data.name is required to rebuild tensors")
        raw_path = Path(data_cfg.get("raw_path", "."))
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


try:
    import hydra  # type: ignore
except Exception:  # pragma: no cover - import-time fallback for minimal envs
    hydra = None


if hydra is not None:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def _hydra_main(cfg):
        main(cfg)


if __name__ == "__main__":
    if hydra is None:
        main({})
    else:
        _hydra_main()
