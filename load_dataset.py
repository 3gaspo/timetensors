"""Dataset-building stage for TimeTensor experiments."""

from __future__ import annotations

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
    run_dir,
    section,
    seed,
    task_shape,
    to_plain_config,
)


def build_dataset_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build or load tensors, then optionally construct loaders and stats."""
    config = to_plain_config(config)
    data_cfg = section(config, "data")
    experiment = section(config, "experiment")
    out_path = dataset_path(config)
    rebuild = bool(
        data_cfg.get(
            "rebuild",
            data_cfg.get("build", experiment.get("build_dataset", False)),
        )
    )
    if rebuild:
        data_name = data_cfg.get("name", data_cfg.get("dataset"))
        if data_name is None:
            raise ValueError("data.name or data.dataset is required to rebuild tensors")
        raw_path = Path(data_cfg.get("raw_path", data_cfg.get("source_path", data_cfg.get("path", "."))))
        build_dataset(
            raw_path,
            str(data_name),
            global_context_cols=data_cfg.get("global_context_cols", data_cfg.get("context_cols")),
            drop_users=data_cfg.get("drop_users"),
            build_individual_ids_context=bool(data_cfg.get("build_individual_ids_context", False)),
            rename_cols=data_cfg.get("rename_cols"),
            aggr=data_cfg.get("aggr", data_cfg.get("aggregation")),
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
            legacy_context_kind=data_cfg.get("legacy_context_kind"),
        )
        result["loaders"] = loaders
        result["stats"] = stats
        result["shape"] = get_sizes(loaders)
    return result


def main(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
