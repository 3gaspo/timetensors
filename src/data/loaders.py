"""Training-loader construction and experiment data assembly."""

from __future__ import annotations

import json
import math
import warnings
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .core import *  # noqa: F401,F403
from .core import _cuda_available, _path, _ensure_dir, _json_file, _read_json, _write_json, _as_values_3d, _as_individual_context, _as_global_context, _normalise_indices, _valid_query_dates
from .sampling import *  # noqa: F401,F403
from .sampling import _slice_temporal_context
from .frames import *  # noqa: F401,F403
from .frames import _selected_variates, _value_columns, _repeat_static_context, _unroll_batches
from .io import *  # noqa: F401,F403
from .splits import *  # noqa: F401,F403
from .splits import _coerce_ratios, _date_blocks, _individual_blocks, _coerce_split_specs, _reject_removed_group_alias, _precomputed_split_specs, _load_split_metadata, _load_cluster_metadata
from .statistics import *  # noqa: F401,F403
from .statistics import _series_gamma_stats, _data_collection_stats, _sampler_potential_windows, _sampler_window_metadata, _leaf_datasets, _finite_stats, _accumulate_window_stats, _empty_loader_stats_accumulator, _finalize_loader_stats, _single_loader_stats, _stats_file, _artifact_file, _save_loader_artifacts, _sampling_for_key, _group_options, _split_creation_summary, _subset_file

def get_train_loaders(
    data_dict: Mapping[str, TimeSeriesData],
    batch_size: int,
    lags: int,
    horizon: int,
    sampling: Mapping[str, Any],
    subsets: Optional[Mapping[str, Any]] = None,
    subset_save_path: Optional[str | Path] = None,
    subset_load_path: Optional[str | Path] = None,
    subset_group: str = "default",
    standard_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
    seed: Optional[int | str] = None,
) -> Dict[str, DataLoader]:
    subsets = subsets or {}
    _reject_removed_group_alias(subsets)
    subset_modes = subsets.get("modes") or {}
    subset_sizes = {
        key: float(value) for key, value in (subsets.get("sizes") or {}).items()
    }
    loaders = {}
    for key, data in data_dict.items():
        settings = _sampling_for_key(sampling, key)
        if settings["drop_constant_individuals"]:
            data = drop_constant_individuals(
                data, lags, horizon, stride=settings["stride"]
            )
        subset_mode = subset_modes.get(key, subsets.get("mode"))
        effective_mode = subset_mode or settings["idx_mode"]
        ratio = subset_sizes.get(key, 1.0)
        subset_indices = None
        stride = settings["stride"]
        precomputed_raw = (subsets.get("specs") or {}).get(key)
        if precomputed_raw is not None:
            if isinstance(precomputed_raw, Mapping):
                effective_mode = str(
                    precomputed_raw.get("mode", effective_mode)
                )
                subset_indices = [
                    int(index) for index in precomputed_raw["indices"]
                ]
                stride = int(precomputed_raw.get("stride", stride))
            else:
                subset_indices = [int(index) for index in precomputed_raw]
        base_query_dates = _valid_query_dates(
            data.dates,
            lags,
            horizon,
            data.target_date_indices,
        )
        valid_date_count = len(base_query_dates)
        load_file = _subset_file(subset_load_path)
        save_file = _subset_file(subset_save_path)
        loaded_subset = subset_indices is not None
        subset_metadata = {
            "creation_mode": "precomputed",
            "access_mode": "provided",
            "randomized": False,
        } if loaded_subset else {}
        if load_file is not None and load_file.exists():
            try:
                saved = load_subset_spec(
                    load_file,
                    key,
                    individuals=data.individuals,
                    valid_dates=valid_date_count,
                    group=subset_group,
                )
            except KeyError:
                if ratio != 1:
                    raise
            else:
                effective_mode = saved["mode"]
                subset_indices = saved["indices"]
                stride = saved["stride"]
                loaded_subset = True
                subset_metadata = {
                    **saved.get("metadata", {}),
                    "creation_mode": saved.get("metadata", {}).get(
                        "creation_mode",
                        "unknown",
                    ),
                    "access_mode": "loaded",
                    "source": str(load_file),
                }
        if not loaded_subset and ratio != 1:
            if effective_mode in {"dates", "all"} and stride > 1:
                strided_dates = base_query_dates[::stride]
                candidates = (
                    strided_dates
                    if effective_mode == "dates"
                    else [
                        date * data.individuals + individual
                        for date in strided_dates
                        for individual in range(data.individuals)
                    ]
                )
                rng = np.random.default_rng(
                    None if seed in {None, "None"} else int(seed)
                )
                count = max(1, int(len(candidates) * ratio))
                subset_indices = (
                    rng.choice(candidates, count, replace=False).astype(int).tolist()
                )
                stride = 1
            else:
                subset_indices = get_subset_indices(
                    data.dates,
                    data.individuals,
                    lags,
                    horizon,
                    ratio,
                    effective_mode,
                    seed,
                    data.target_date_indices,
                )
            subset_metadata = {
                "creation_mode": "generated",
                "access_mode": "created",
                "randomized": True,
                "ratio": ratio,
                "seed": None if seed in {None, "None"} else int(seed),
            }
        if subset_indices is not None and save_file is not None:
            save_subset_spec(
                save_file,
                key,
                mode=effective_mode,
                indices=subset_indices,
                stride=stride,
                individuals=data.individuals,
                valid_dates=valid_date_count,
                group=subset_group,
                metadata=subset_metadata,
            )
        config = SamplerConfig(
            idx_mode=settings["idx_mode"],
            block_individuals=settings["block_individuals"],
            use_individual_context=settings["use_individual_context"],
            use_global_context=settings["use_global_context"],
            remove_cte=settings["remove_cte"],
            weight=settings["weight"],
            subset_indices=subset_indices,
            subset_mode=effective_mode if subset_indices is not None else None,
            stride=stride,
        )
        dataset = TimeSeriesDataset(
            data,
            lags=lags,
            horizon=horizon,
            sampler_config=config,
        )
        dataset.selection_metadata = subset_metadata or {
            "creation_mode": "none",
            "access_mode": "none",
            "randomized": False,
            "ratio": 1.0,
        }
        if standard_stats is not None and key in standard_stats:
            dataset.normalize(standard_stats[key])
        loaders[key] = TimeSeriesDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=settings["shuffle"],
            collate_fn=collate_fn,
            num_workers=0,
        )
    return loaders


def _combine_data(items: Sequence[TimeSeriesData]) -> TimeSeriesData:
    if not items:
        raise ValueError("cannot combine an empty data list")
    first = items[0]
    if any(not np.array_equal(item.datetimes, first.datetimes) for item in items[1:]):
        raise ValueError("cluster splits must share datetimes")
    if any(not torch.equal(item.date_ids, first.date_ids) for item in items[1:]):
        raise ValueError("cluster splits must share source date ids")
    if any(
        not torch.equal(item.target_date_ids, first.target_date_ids)
        for item in items[1:]
    ):
        raise ValueError("cluster splits must share target date ids")
    if any(
        (item.global_context is None) != (first.global_context is None)
        or (
            item.global_context is not None
            and not torch.equal(item.global_context, first.global_context)
        )
        for item in items[1:]
    ):
        raise ValueError("cluster splits must share global context")
    individual_context = None
    if all(item.individual_context is not None for item in items):
        individual_context = torch.cat(
            [item.individual_context for item in items], dim=0
        )
    elif any(item.individual_context is not None for item in items):
        raise ValueError("individual context must be present for every cluster or none")
    cluster_ids = None
    if any(item.cluster_ids is not None for item in items):
        cluster_ids = torch.cat(
            [
                item.cluster_ids
                if item.cluster_ids is not None
                else torch.full((item.individuals,), -1, dtype=torch.long)
                for item in items
            ]
        )
    return TimeSeriesData(
        values=torch.cat([item.values for item in items], dim=0),
        datetimes=first.datetimes,
        individual_context=individual_context,
        global_context=first.global_context,
        individual_ids=torch.cat([item.individual_ids for item in items]),
        cluster_ids=cluster_ids,
        date_ids=first.date_ids,
        target_date_ids=first.target_date_ids,
        individual_names={
            key: value
            for item in items
            for key, value in item.individual_names.items()
        },
    )


def aggregate_loaders_dict(
    loaders_dicts: Sequence[Mapping[str, DataLoader]],
    lags: int,
    horizon: int,
    sampling: Mapping[str, Any],
    batch_size: int,
) -> Dict[str, DataLoader]:
    del lags, horizon
    return _aggregate_configured_loaders(
        {
            f"group{index}": loaders
            for index, loaders in enumerate(loaders_dicts)
        },
        sampling,
        batch_size,
    )


def _aggregate_configured_loaders(
    loaders_by_cluster: Mapping[str, Mapping[str, DataLoader]],
    sampling: Mapping[str, Any],
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Aggregate loaders while preserving each cluster's configured sampler."""
    names = list(loaders_by_cluster)
    split_names = list(loaders_by_cluster[names[0]])
    result = {}
    for split in split_names:
        dataset = AggregatedTimeSeriesDataset(
            [loaders_by_cluster[name][split].dataset for name in names]
        )
        settings = _sampling_for_key(sampling, split)
        result[split] = TimeSeriesDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=settings["shuffle"],
            collate_fn=collate_fn,
            num_workers=0,
        )
    return result


def get_sizes(
    loaders_dict: Mapping[str, DataLoader],
    str_info: bool = False,
) -> Any:
    loader = next(iter(loaders_dict.values()))
    batch = next(iter(loader))
    shape = [
        batch["inputs"].shape[-1],
        batch["inputs"].shape[1],
        batch["targets"].shape[-1],
    ]
    if not str_info:
        return shape

    def index_mode(dataset: Dataset) -> str:
        if isinstance(dataset, AggregatedTimeSeriesDataset):
            modes = [
                component.index_sampler.config.idx_mode
                for component in dataset.datasets
            ]
            unique_modes = list(dict.fromkeys(modes))
            return (
                unique_modes[0]
                if len(unique_modes) == 1
                else "aggregate[" + ",".join(unique_modes) + "]"
            )
        if isinstance(dataset, TimeSeriesDataset):
            return dataset.index_sampler.config.idx_mode
        return "unknown"

    shape_str = "Splits:\n" + "\n".join(
        f"{key}\tidx_mode={index_mode(loader.dataset)}\t{loader.dataset.shape}"
        for key, loader in loaders_dict.items()
    )
    batch_str = "\n".join(
        [
            "Batch:",
            " batches="
            + ", ".join(
                f"{key}={len(loader)}" for key, loader in loaders_dict.items()
            ),
            f" inputs={list(batch['inputs'].shape)}",
            f" targets={list(batch['targets'].shape)}",
            " individual_context="
            + (
                "None"
                if batch["individual_context"] is None
                else str(list(batch["individual_context"].shape))
            ),
            " global_context="
            + (
                "None"
                if batch["global_context"] is None
                else str(list(batch["global_context"].shape))
            ),
        ]
    )
    return shape, shape_str, batch_str


def _data_metadata(data: TimeSeriesData, *, source: str | Path | None = None) -> Dict[str, Any]:
    metadata = {
        "values": list(data.values.shape),
        "individual_context": (
            None
            if data.individual_context is None
            else list(data.individual_context.shape)
        ),
        "global_context": (
            None
            if data.global_context is None
            else list(data.global_context.shape)
        ),
        "individuals": data.individuals,
        "variates": data.variates,
        "dates": data.dates,
        "target_dates": len(data.target_date_ids),
        "first_target_date_id": (
            None
            if len(data.target_date_ids) == 0
            else int(data.target_date_ids[0])
        ),
        "last_target_date_id": (
            None
            if len(data.target_date_ids) == 0
            else int(data.target_date_ids[-1])
        ),
    }
    if source is not None:
        metadata["source"] = str(_path(source))
    return metadata


def _split_metadata(split_data: Mapping[str, TimeSeriesData]) -> Dict[str, Any]:
    return {key: _data_metadata(data) for key, data in split_data.items()}


def _fetch_training_data_legacy(
    data_path: str | Path,
    splits: Mapping[str, Any],
    sampling: Mapping[str, Any],
    subsets: Optional[Mapping[str, Any]],
    batch_size: int,
    lags: int,
    horizon: int,
    aggregate: bool = True,
    seed: Optional[int | str] = None,
    cluster_ids: Optional[Sequence[int]] = None,
    cluster_path: Optional[str | Path] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_config: Optional[Mapping[str, Any]] = None,
    cluster_save_path: Optional[str | Path] = None,
    split_save_path: Optional[str | Path] = None,
    split_load_path: Optional[str | Path] = None,
    subset_save_path: Optional[str | Path] = None,
    subset_load_path: Optional[str | Path] = None,
    stats_save_path: Optional[str | Path] = None,
    compute_stats: bool = True,
    selected_splits: Optional[Sequence[str]] = None,
    legacy_context_kind: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    set_seed(seed)
    if isinstance(selected_splits, str):
        selected_splits = [selected_splits]
    data = load_data(data_path, legacy_context_kind=legacy_context_kind)
    if cluster_ids is not None:
        data = data.subset(cluster_ids)
    overall_stats = (
        _data_collection_stats(
            [data],
            lags,
            horizon,
            bool(sampling.get("remove_eval_cte", False)),
        )
        if compute_stats
        else None
    )
    cluster_creation: Dict[str, Any] = {
        "creation_mode": "none",
        "access_mode": "none",
        "randomized": False,
    }
    provided_cluster_sources = sum(
        value is not None for value in (cluster_path, clusters, cluster_config)
    )
    if provided_cluster_sources > 1:
        raise ValueError(
            "provide only one of cluster_path, clusters, or cluster_config"
        )
    if cluster_path is not None:
        clusters = load_clusters(cluster_path, data.individuals)
        loaded_cluster_metadata = _load_cluster_metadata(cluster_path)
        cluster_creation = {
            **loaded_cluster_metadata,
            "creation_mode": loaded_cluster_metadata.get(
                "creation_mode",
                "unknown",
            ),
            "access_mode": "loaded",
            "source": str(_json_file(cluster_path, "clusters")),
        }
    elif cluster_config is not None:
        clusters = create_iid_clusters(
            data.individuals,
            ratios=cluster_config.get("ratios"),
            sizes=cluster_config.get("sizes"),
            n_clusters=cluster_config.get("n_clusters"),
            seed=seed,
        )
        cluster_creation = {
            "creation_mode": "generated",
            "access_mode": "created",
            "randomized": True,
            "method": next(
                key
                for key in ("ratios", "sizes", "n_clusters")
                if cluster_config.get(key) is not None
            ),
            "ratios": cluster_config.get("ratios"),
            "sizes": cluster_config.get("sizes"),
            "n_clusters": cluster_config.get("n_clusters"),
            "seed": None if seed in {None, "None"} else int(seed),
        }
    elif clusters is not None:
        cluster_creation = {
            "creation_mode": "precomputed",
            "access_mode": "provided",
            "randomized": False,
        }
    if clusters is not None and cluster_save_path is not None:
        save_clusters(
            clusters,
            cluster_save_path,
            metadata=cluster_creation,
        )
    selected_cluster = (subsets or {}).get("cluster")
    if clusters is not None:
        data = assign_cluster_ids(data, clusters)
        items = (
            [(selected_cluster, clusters[selected_cluster])]
            if selected_cluster is not None
            else list(clusters.items())
        )
        loaders_by_cluster = {}
        split_stats_by_cluster = {}
        subset_stats_by_cluster = {}
        cluster_stats = {}
        split_creation_by_cluster = {}
        for name, ids in items:
            cluster_splits = _group_options(splits, name)
            cluster_subsets = _group_options(subsets, name)
            cluster_data = data.subset(ids)
            if compute_stats:
                cluster_stats[name] = {
                    "stage": "before_splitting",
                    "stats": get_dataset_stats(
                        {name: cluster_data},
                        lags,
                        horizon,
                        sampling,
                    )[name],
                }
            split_data = get_dataset_splits(
                cluster_splits,
                data=cluster_data,
                split_save_path=split_save_path,
                split_load_path=split_load_path,
                split_group=name,
                seed=seed,
            )
            split_creation_by_cluster[name] = _split_creation_summary(
                cluster_splits,
                split_load_path,
                seed,
                group=name,
            )
            if selected_splits is not None:
                missing = set(selected_splits) - set(split_data)
                if missing:
                    raise KeyError(
                        f"cluster {name!r} has no splits {sorted(missing)}"
                    )
                split_data = {
                    split: split_data[split] for split in selected_splits
                }
            loaders = get_train_loaders(
                split_data,
                batch_size,
                lags,
                horizon,
                sampling,
                cluster_subsets,
                subset_save_path,
                subset_load_path,
                subset_group=name,
                seed=seed,
            )
            loaders_by_cluster[name] = loaders
            if compute_stats:
                split_stats_by_cluster[name] = get_dataset_stats(
                    split_data, lags, horizon, sampling
                )
                subset_stats_by_cluster[name] = get_subset_stats(loaders)
        if not aggregate:
            if compute_stats:
                _save_stats(
                    stats_save_path,
                    dataset={
                        "version": 1,
                        "kind": "dataset_stats",
                        "stage": "before_clustering",
                        "creation": {
                            "creation_mode": "loaded",
                            "source": str(_path(data_path)),
                        },
                        "stats": overall_stats,
                    },
                    splits={
                        "version": 1,
                        "kind": "split_stats",
                        "scope": "by_cluster",
                        "groups": {
                            name: {
                                "creation": split_creation_by_cluster[name],
                                "splits": split_stats_by_cluster[name],
                            }
                            for name, _ in items
                        },
                    },
                    subsets={
                        "version": 1,
                        "kind": "subset_stats",
                        "scope": "by_cluster",
                        "groups": subset_stats_by_cluster,
                    },
                    clusters={
                        "version": 1,
                        "kind": "cluster_stats",
                        "scope": "selected_clusters",
                        "creation": cluster_creation,
                        "clusters": cluster_stats,
                    },
                )
            return loaders_by_cluster, split_stats_by_cluster if compute_stats else {}
        split_names = next(iter(loaders_by_cluster.values())).keys()
        if any(
            set(loaders_by_cluster[name]) != set(split_names)
            for name, _ in items
        ):
            raise ValueError(
                "clusters must expose the same split names to be re-aggregated"
            )
        loaders = _aggregate_configured_loaders(
            loaders_by_cluster,
            sampling,
            batch_size,
        )
        stats = {}
        if compute_stats:
            stats = {
                split: _data_collection_stats(
                    [
                        loaders_by_cluster[name][split].dataset.data
                        for name, _ in items
                    ],
                    lags,
                    horizon,
                    bool(
                        sampling.get(
                            "remove_train_cte"
                            if split == "train"
                            else "remove_eval_cte",
                            False,
                        )
                    ),
                )
                for split in split_names
            }
            _save_stats(
                stats_save_path,
                dataset={
                    "version": 1,
                    "kind": "dataset_stats",
                    "stage": "before_clustering",
                    "creation": {
                        "creation_mode": "loaded",
                        "source": str(_path(data_path)),
                    },
                    "stats": overall_stats,
                },
                splits={
                    "version": 1,
                    "kind": "split_stats",
                    "scope": "aggregate",
                    "creation": {
                        "creation_mode": "aggregate",
                        "components": list(loaders_by_cluster),
                    },
                    "per_cluster": {
                        name: {
                            "creation": split_creation_by_cluster[name],
                            "splits": split_stats_by_cluster[name],
                        }
                        for name, _ in items
                    },
                    "aggregate_splits": stats,
                },
                subsets={
                    "version": 1,
                    "kind": "subset_stats",
                    "scope": "aggregate",
                    "per_cluster": subset_stats_by_cluster,
                    "subsets": get_subset_stats(loaders),
                },
                clusters={
                    "version": 1,
                    "kind": "cluster_stats",
                    "scope": "selected_clusters",
                    "creation": cluster_creation,
                    "clusters": cluster_stats,
                },
            )
        return loaders, stats
    split_data = get_dataset_splits(
        splits,
        data=data,
        split_save_path=split_save_path,
        split_load_path=split_load_path,
        seed=seed,
    )
    split_creation = _split_creation_summary(splits, split_load_path, seed)
    if selected_splits is not None:
        missing = set(selected_splits) - set(split_data)
        if missing:
            raise KeyError(f"dataset has no splits {sorted(missing)}")
        split_data = {
            split: split_data[split] for split in selected_splits
        }
    loaders = get_train_loaders(
        split_data,
        batch_size,
        lags,
        horizon,
        sampling,
        subsets,
        subset_save_path,
        subset_load_path,
        subset_group="default",
        seed=seed,
    )
    stats = {}
    if compute_stats:
        stats = get_dataset_stats(split_data, lags, horizon, sampling)
        _save_stats(
            stats_save_path,
            dataset={
                "version": 1,
                "kind": "dataset_stats",
                "stage": "before_clustering",
                "creation": {
                    "creation_mode": "loaded",
                    "source": str(_path(data_path)),
                },
                "stats": overall_stats,
            },
            splits={
                "version": 1,
                "kind": "split_stats",
                "scope": "dataset",
                "creation": split_creation,
                "splits": stats,
            },
            subsets={
                "version": 1,
                "kind": "subset_stats",
                "scope": "dataset",
                "subsets": get_subset_stats(loaders),
            },
        )
    return loaders, stats


def _compute_loader_stats_if_requested(
    loaders: Mapping[str, DataLoader],
    *,
    compute_stats: bool,
    stats_max_windows: Optional[int],
    stats_seed: Optional[int | str],
    stats_eps: float,
) -> Dict[str, Dict[str, Any]]:
    if not compute_stats:
        return {}
    return get_loader_stats(
        loaders,
        max_windows=stats_max_windows,
        seed=stats_seed,
        eps=stats_eps,
    )


def fetch_training_data(
    data_path: str | Path,
    splits: Mapping[str, Any],
    sampling: Mapping[str, Any],
    subsets: Optional[Mapping[str, Any]],
    batch_size: int,
    lags: int,
    horizon: int,
    aggregate: bool = True,
    seed: Optional[int | str] = None,
    cluster_ids: Optional[Sequence[int]] = None,
    cluster_path: Optional[str | Path] = None,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_config: Optional[Mapping[str, Any]] = None,
    cluster_save_path: Optional[str | Path] = None,
    split_save_path: Optional[str | Path] = None,
    split_load_path: Optional[str | Path] = None,
    subset_save_path: Optional[str | Path] = None,
    subset_load_path: Optional[str | Path] = None,
    stats_save_path: Optional[str | Path] = None,
    compute_stats: bool = True,
    stats_max_windows: Optional[int] = None,
    stats_seed: Optional[int | str] = None,
    stats_eps: float = 1e-8,
    selected_splits: Optional[Sequence[str]] = None,
    legacy_context_kind: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Build final loaders and compute stats only on their accessible windows."""
    set_seed(seed)
    stats_seed = seed if stats_seed is None else stats_seed
    if isinstance(selected_splits, str):
        selected_splits = [selected_splits]
    data = load_data(data_path, legacy_context_kind=legacy_context_kind)
    if cluster_ids is not None:
        data = data.subset(cluster_ids)
    metadata: Dict[str, Any] = {
        "version": 1,
        "kind": "loader_metadata",
        "dataset": _data_metadata(data, source=data_path),
        "task": {"lags": int(lags), "horizon": int(horizon)},
        "window_anchor": "query_t",
        "sampling": dict(sampling),
        "subsets": dict(subsets or {}),
        "stats": {
            "computed": bool(compute_stats),
            "max_windows": stats_max_windows,
            "seed": None if stats_seed in {None, "None"} else int(stats_seed),
            "eps": float(stats_eps),
        },
    }

    cluster_creation: Dict[str, Any] = {
        "creation_mode": "none",
        "access_mode": "none",
        "randomized": False,
    }
    provided_cluster_sources = sum(
        value is not None for value in (cluster_path, clusters, cluster_config)
    )
    if provided_cluster_sources > 1:
        raise ValueError(
            "provide only one of cluster_path, clusters, or cluster_config"
        )
    if cluster_path is not None:
        clusters = load_clusters(cluster_path, data.individuals)
        loaded_cluster_metadata = _load_cluster_metadata(cluster_path)
        cluster_creation = {
            **loaded_cluster_metadata,
            "creation_mode": loaded_cluster_metadata.get(
                "creation_mode",
                "unknown",
            ),
            "access_mode": "loaded",
            "source": str(_json_file(cluster_path, "clusters")),
        }
    elif cluster_config is not None:
        clusters = create_iid_clusters(
            data.individuals,
            ratios=cluster_config.get("ratios"),
            sizes=cluster_config.get("sizes"),
            n_clusters=cluster_config.get("n_clusters"),
            seed=seed,
        )
        cluster_creation = {
            "creation_mode": "generated",
            "access_mode": "created",
            "randomized": True,
            "method": next(
                key
                for key in ("ratios", "sizes", "n_clusters")
                if cluster_config.get(key) is not None
            ),
            "ratios": cluster_config.get("ratios"),
            "sizes": cluster_config.get("sizes"),
            "n_clusters": cluster_config.get("n_clusters"),
            "seed": None if seed in {None, "None"} else int(seed),
        }
    elif clusters is not None:
        cluster_creation = {
            "creation_mode": "precomputed",
            "access_mode": "provided",
            "randomized": False,
        }
    if clusters is not None and cluster_save_path is not None:
        save_clusters(clusters, cluster_save_path, metadata=cluster_creation)

    selected_cluster = (subsets or {}).get("cluster")
    if clusters is not None:
        data = assign_cluster_ids(data, clusters)
        items = (
            [(selected_cluster, clusters[selected_cluster])]
            if selected_cluster is not None
            else list(clusters.items())
        )
        loaders_by_cluster: Dict[str, Mapping[str, DataLoader]] = {}
        stats_by_cluster: Dict[str, Any] = {}
        split_metadata_by_cluster: Dict[str, Any] = {}
        loader_metadata_by_cluster: Dict[str, Any] = {}
        cluster_metadata: Dict[str, Any] = {}
        for name, ids in items:
            cluster_splits = _group_options(splits, name)
            cluster_subsets = _group_options(subsets, name)
            cluster_data = data.subset(ids)
            cluster_metadata[name] = _data_metadata(cluster_data)
            split_data = get_dataset_splits(
                cluster_splits,
                data=cluster_data,
                split_save_path=split_save_path,
                split_load_path=split_load_path,
                split_group=name,
                seed=seed,
            )
            if selected_splits is not None:
                missing = set(selected_splits) - set(split_data)
                if missing:
                    raise KeyError(
                        f"cluster {name!r} has no splits {sorted(missing)}"
                    )
                split_data = {
                    split: split_data[split] for split in selected_splits
                }
            split_metadata_by_cluster[name] = {
                "creation": _split_creation_summary(
                    cluster_splits,
                    split_load_path,
                    seed,
                    group=name,
                ),
                "splits": _split_metadata(split_data),
            }
            loaders = get_train_loaders(
                split_data,
                batch_size,
                lags,
                horizon,
                sampling,
                cluster_subsets,
                subset_save_path,
                subset_load_path,
                subset_group=name,
                seed=seed,
            )
            loaders_by_cluster[name] = loaders
            loader_metadata_by_cluster[name] = get_loader_metadata(loaders)
            stats_by_cluster[name] = _compute_loader_stats_if_requested(
                loaders,
                compute_stats=compute_stats,
                stats_max_windows=stats_max_windows,
                stats_seed=stats_seed,
                stats_eps=stats_eps,
            )
        metadata.update(
            {
                "cluster_creation": cluster_creation,
                "clusters": cluster_metadata,
                "splits": split_metadata_by_cluster,
            }
        )
        if not aggregate:
            metadata.update(
                {
                    "scope": "by_cluster",
                    "loaders": loader_metadata_by_cluster,
                }
            )
            _save_loader_artifacts(
                stats_save_path,
                stats=stats_by_cluster if compute_stats else None,
                metadata=metadata,
            )
            return loaders_by_cluster, stats_by_cluster if compute_stats else {}

        split_names = next(iter(loaders_by_cluster.values())).keys()
        if any(
            set(loaders_by_cluster[name]) != set(split_names)
            for name, _ in items
        ):
            raise ValueError(
                "clusters must expose the same split names to be re-aggregated"
            )
        loaders = _aggregate_configured_loaders(
            loaders_by_cluster,
            sampling,
            batch_size,
        )
        stats = _compute_loader_stats_if_requested(
            loaders,
            compute_stats=compute_stats,
            stats_max_windows=stats_max_windows,
            stats_seed=stats_seed,
            stats_eps=stats_eps,
        )
        metadata.update(
            {
                "scope": "aggregate",
                "per_cluster_loaders": loader_metadata_by_cluster,
                "loaders": get_loader_metadata(loaders),
            }
        )
        _save_loader_artifacts(
            stats_save_path,
            stats=stats if compute_stats else None,
            metadata=metadata,
        )
        return loaders, stats

    split_data = get_dataset_splits(
        splits,
        data=data,
        split_save_path=split_save_path,
        split_load_path=split_load_path,
        seed=seed,
    )
    if selected_splits is not None:
        missing = set(selected_splits) - set(split_data)
        if missing:
            raise KeyError(f"dataset has no splits {sorted(missing)}")
        split_data = {
            split: split_data[split] for split in selected_splits
        }
    loaders = get_train_loaders(
        split_data,
        batch_size,
        lags,
        horizon,
        sampling,
        subsets,
        subset_save_path,
        subset_load_path,
        subset_group="default",
        seed=seed,
    )
    stats = _compute_loader_stats_if_requested(
        loaders,
        compute_stats=compute_stats,
        stats_max_windows=stats_max_windows,
        stats_seed=stats_seed,
        stats_eps=stats_eps,
    )
    metadata.update(
        {
            "scope": "dataset",
            "splits": {
                "creation": _split_creation_summary(splits, split_load_path, seed),
                "splits": _split_metadata(split_data),
            },
            "loaders": get_loader_metadata(loaders),
        }
    )
    _save_loader_artifacts(
        stats_save_path,
        stats=stats if compute_stats else None,
        metadata=metadata,
    )
    return loaders, stats


def apply_standard_norm(
    loaders_dict: Mapping[str, DataLoader],
    stats_dict: Mapping[str, Mapping[str, float]],
) -> None:
    for key, loader in loaders_dict.items():
        if key in stats_dict:
            loader.dataset.normalize(stats_dict[key])
