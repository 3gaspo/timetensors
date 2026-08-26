"""Dataset, window, loader, and subset statistics."""

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

def _series_gamma_stats(
    series: np.ndarray,
    lags: int,
    horizon: int,
    remove_cte: bool,
    eps: float,
    query_dates: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if query_dates is None:
        query_dates = _valid_query_dates(len(series), lags, horizon)
    query_dates = [int(value) for value in query_dates]
    if not query_dates:
        return np.empty(0), np.empty(0)
    windows = np.stack(
        [
            series[query_t - lags + 1 : query_t + horizon + 1]
            for query_t in query_dates
        ]
    )
    lookbacks = windows[:, :lags]
    futures = windows[:, lags:]
    fully_finite = np.isfinite(lookbacks).all(axis=1) & np.isfinite(futures).all(axis=1)
    mean_x = np.mean(lookbacks, axis=1)
    std_x = np.std(lookbacks, axis=1, ddof=0)
    mean_y = np.mean(futures, axis=1)
    std_y = np.std(futures, axis=1, ddof=0)
    valid = fully_finite
    if remove_cte:
        valid &= std_x > 0
    alpha = std_y[valid] / (std_x[valid] + eps)
    beta = (mean_y[valid] - mean_x[valid]) / (std_x[valid] + eps)
    return alpha, beta


def get_dataset_stats(
    data_dict: Mapping[str, TimeSeriesData],
    lags: int,
    horizon: int,
    sampling: Mapping[str, Any],
    save_path: Optional[str | Path] = None,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Compute TimeTensor alpha/beta statistics with population std.

    For each forecasting window:

    ``alpha = std(future) / (std(lookback) + eps)``
    ``beta = (mean(future) - mean(lookback)) / (std(lookback) + eps)``
    """
    result: Dict[str, Dict[str, Any]] = {}
    for key, data in data_dict.items():
        remove_cte = bool(
            sampling.get(
                "remove_train_cte" if key == "train" else "remove_eval_cte",
                False,
            )
        )
        result[key] = _data_collection_stats(
            [data],
            lags,
            horizon,
            remove_cte,
            eps,
        )
    if save_path is not None:
        file = _json_file(save_path, "stats")
        _write_json(result, file)
    return result


def _data_collection_stats(
    items: Sequence[TimeSeriesData],
    lags: int,
    horizon: int,
    remove_cte: bool,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    if not items:
        raise ValueError("cannot compute statistics for an empty collection")
    full_arrays = [
        item.values.detach().cpu().numpy().astype(float)
        for item in items
    ]
    arrays = [
        values[..., item.target_date_indices]
        for item, values in zip(items, full_arrays)
    ]
    original_finite_count = sum(
        int(np.isfinite(values).sum()) for values in arrays
    )
    alphas: List[np.ndarray] = []
    betas: List[np.ndarray] = []
    per_series_stds: List[np.ndarray] = []
    finite_values: List[np.ndarray] = []
    for item, full_values, values in zip(items, full_arrays, arrays):
        query_dates = _valid_query_dates(
            item.dates,
            lags,
            horizon,
            item.target_date_indices,
        )
        target_indices = item.target_date_indices
        for individual in range(full_values.shape[0]):
            for variate in range(full_values.shape[1]):
                series = full_values[individual, variate]
                target_series = series[target_indices]
                alpha, beta = _series_gamma_stats(
                    series,
                    lags,
                    horizon,
                    remove_cte,
                    eps,
                    query_dates,
                )
                if alpha.size:
                    alphas.append(alpha)
                    betas.append(beta)
                if remove_cte and query_dates:
                    lookbacks = np.stack(
                        [
                            series[query_t - lags + 1 : query_t + 1]
                            for query_t in query_dates
                        ]
                    )
                    constant = (
                        np.isfinite(lookbacks).all(axis=1)
                        & (np.std(lookbacks, axis=1, ddof=0) == 0)
                    )
                    excluded = {
                        query_t
                        for query_t, is_constant in zip(query_dates, constant)
                        if is_constant
                    }
                    keep = np.asarray(
                        [index not in excluded for index in target_indices],
                        dtype=bool,
                    )
                    finite_values.append(
                        target_series[keep & np.isfinite(target_series)]
                    )
                else:
                    finite_values.append(
                        target_series[np.isfinite(target_series)]
                    )
        per_series_stds.append(np.nanstd(values, axis=-1).reshape(-1))
    alpha_values = np.concatenate(alphas) if alphas else np.empty(0)
    beta_values = np.concatenate(betas) if betas else np.empty(0)
    if remove_cte:
        std_values = np.asarray(
            [
                np.std(series)
                for series in finite_values
                if series.size
            ],
            dtype=float,
        )
    else:
        std_values = np.concatenate(per_series_stds)
    non_empty_values = [series for series in finite_values if series.size]
    flat_values = (
        np.concatenate(non_empty_values)
        if non_empty_values
        else np.empty(0)
    )
    shapes = [list(values.shape) for values in arrays]
    same_dates = len({shape[-1] for shape in shapes}) == 1
    same_variates = len({shape[1] for shape in shapes}) == 1
    shape: Any
    if len(shapes) == 1:
        shape = shapes[0]
    elif same_dates and same_variates:
        shape = [
            sum(item_shape[0] for item_shape in shapes),
            shapes[0][1],
            shapes[0][2],
        ]
    else:
        shape = {
            "component_shapes": shapes,
            "total_individuals": sum(item_shape[0] for item_shape in shapes),
            "variates": (
                shapes[0][1] if same_variates else [shape_[1] for shape_ in shapes]
            ),
        }
    return {
        "sampling": {
            "lags": int(lags),
            "horizon": int(horizon),
            "remove_cte": bool(remove_cte),
        },
        "shape": shape,
        "values": int(flat_values.size),
        "removed_values": int(original_finite_count - flat_values.size),
        "windows": int(alpha_values.size),
        "mean": float(np.mean(flat_values)) if flat_values.size else math.nan,
        "stds": float(np.mean(std_values)) if std_values.size else math.nan,
        "std": float(np.std(flat_values)) if flat_values.size else math.nan,
        "alpha": (
            float(np.mean(alpha_values)) if alpha_values.size else math.nan
        ),
        "beta": (
            float(np.mean(beta_values)) if beta_values.size else math.nan
        ),
    }


def _sampler_potential_windows(sampler: IndexSampler) -> int:
    config = sampler.config
    if config.subset_mode == "all" and config.subset_indices is not None:
        return len(config.subset_indices)
    if config.subset_mode == "dates" and config.subset_indices is not None:
        date_count = len(config.subset_indices)
    else:
        date_count = len(sampler.unfiltered_date_candidates)
    if config.subset_mode == "individuals" and config.subset_indices is not None:
        individual_count = len(config.subset_indices)
    else:
        individual_count = sampler.individuals
    return int(date_count * individual_count)


def _sampler_window_metadata(sampler: IndexSampler) -> Dict[str, int]:
    potential = _sampler_potential_windows(sampler)
    accessible = (
        sum(1 for _ in sampler.iter_accessible_pairs())
        if sampler.config.remove_cte
        else potential
    )
    return {
        "potential_windows": potential,
        "accessible_windows": accessible,
        "constant_removed_windows": max(potential - accessible, 0),
    }


def get_loader_metadata(
    loaders: Mapping[str, DataLoader],
) -> Dict[str, Dict[str, Any]]:
    """Describe the effective candidate space after subset and stride rules."""
    result: Dict[str, Dict[str, Any]] = {}
    for key, loader in loaders.items():
        dataset = loader.dataset
        if isinstance(dataset, AggregatedTimeSeriesDataset):
            components = []
            for component in dataset.datasets:
                sampler = component.index_sampler
                window_metadata = _sampler_window_metadata(sampler)
                components.append(
                    {
                        "creation": getattr(
                            component,
                            "selection_metadata",
                            {"creation_mode": "unknown"},
                        ),
                        "mode": sampler.config.idx_mode,
                        "subset_mode": sampler.config.subset_mode,
                        "subset_size": (
                            None
                            if sampler.config.subset_indices is None
                            else len(sampler.config.subset_indices)
                        ),
                        "stride": sampler.config.stride,
                        "true_length": sampler.true_len,
                        "dataset_length": len(component),
                        **window_metadata,
                    }
                )
            result[key] = {
                "aggregated": True,
                "components": components,
                "dataset_length": len(dataset),
                "batch_size": loader.batch_size,
                "batches": len(loader),
            }
            continue
        sampler = dataset.index_sampler
        config = sampler.config
        users_per_item = (
            len(sampler.individual_candidates)
            if config.block_individuals is None
            else min(config.block_individuals, len(sampler.individual_candidates))
        )
        full_item_batch = min(int(loader.batch_size), len(dataset))
        window_metadata = _sampler_window_metadata(sampler)
        stats: Dict[str, Any] = {
            "creation": getattr(
                dataset,
                "selection_metadata",
                {"creation_mode": "unknown"},
            ),
            "mode": config.idx_mode,
            "subset_mode": config.subset_mode,
            "subset_size": (
                None
                if config.subset_indices is None
                else len(config.subset_indices)
            ),
            "stride": config.stride,
            "block_individuals": config.block_individuals,
            "users_per_item": users_per_item,
            "remove_cte": config.remove_cte,
            "candidate_individuals": len(sampler.individual_candidates),
            "candidate_dates": len(sampler.date_candidates),
            "true_length": sampler.true_len,
            "dataset_length": len(dataset),
            "batch_size": loader.batch_size,
            "effective_first_batch_size": full_item_batch * users_per_item,
            "batches": len(loader),
            **window_metadata,
        }
        if config.idx_mode == "all":
            stats["candidate_pairs"] = len(sampler.pair_candidates)
        result[key] = stats
    return result


def get_subset_stats(loaders: Mapping[str, DataLoader]) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible name for loader metadata."""
    return get_loader_metadata(loaders)


def _leaf_datasets(dataset: Dataset) -> List[TimeSeriesDataset]:
    if isinstance(dataset, AggregatedTimeSeriesDataset):
        leaves: List[TimeSeriesDataset] = []
        for component in dataset.datasets:
            leaves.extend(_leaf_datasets(component))
        return leaves
    if isinstance(dataset, TimeSeriesDataset):
        return [dataset]
    raise TypeError(f"unsupported dataset type for stats: {type(dataset)!r}")


def _finite_stats(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.mean(finite)), float(np.std(finite))


def _accumulate_window_stats(
    acc: Dict[str, Any],
    dataset: TimeSeriesDataset,
    individual: int,
    date: int,
    eps: float,
) -> None:
    start = int(date) - dataset.lags + 1
    stop = int(date) + dataset.horizon + 1
    window = (
        dataset.data.values[individual, :, start:stop]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    lookback = window[:, : dataset.lags]
    future = window[:, dataset.lags :]
    acc["item_windows"] += 1
    for variate in range(window.shape[0]):
        x = lookback[variate]
        y = future[variate]
        x_finite = np.isfinite(x).all()
        y_finite = np.isfinite(y).all()
        acc["variate_windows"] += 1
        if not (x_finite and y_finite):
            acc["invalid_variate_windows"] += 1
            continue
        mean_x = float(np.mean(x))
        std_x = float(np.std(x))
        mean_y = float(np.mean(y))
        std_y = float(np.std(y))
        if std_x == 0:
            acc["constant_variate_windows"] += 1
        acc["finite_variate_windows"] += 1
        acc["lookback_mean_sum"] += mean_x
        acc["lookback_std_sum"] += std_x
        acc["future_mean_sum"] += mean_y
        acc["future_std_sum"] += std_y
        acc["alpha_sum"] += std_y / (std_x + eps)
        acc["beta_sum"] += (mean_y - mean_x) / (std_x + eps)
        x_value_mean, x_value_std = _finite_stats(x)
        y_value_mean, y_value_std = _finite_stats(y)
        if not math.isnan(x_value_mean):
            acc["lookback_value_mean_sum"] += x_value_mean
            acc["lookback_value_std_sum"] += x_value_std
            acc["lookback_value_sum"] += float(x.sum())
            acc["lookback_value_square_sum"] += float(np.square(x).sum())
            acc["lookback_value_count"] += int(x.size)
            acc["lookback_value_min"] = min(acc["lookback_value_min"], float(x.min()))
            acc["lookback_value_max"] = max(acc["lookback_value_max"], float(x.max()))
        if not math.isnan(y_value_mean):
            acc["future_value_mean_sum"] += y_value_mean
            acc["future_value_std_sum"] += y_value_std


def _empty_loader_stats_accumulator() -> Dict[str, Any]:
    return {
        "item_windows": 0,
        "variate_windows": 0,
        "finite_variate_windows": 0,
        "invalid_variate_windows": 0,
        "constant_variate_windows": 0,
        "lookback_mean_sum": 0.0,
        "lookback_std_sum": 0.0,
        "future_mean_sum": 0.0,
        "future_std_sum": 0.0,
        "lookback_value_mean_sum": 0.0,
        "lookback_value_std_sum": 0.0,
        "lookback_value_sum": 0.0,
        "lookback_value_square_sum": 0.0,
        "lookback_value_count": 0,
        "lookback_value_min": math.inf,
        "lookback_value_max": -math.inf,
        "future_value_mean_sum": 0.0,
        "future_value_std_sum": 0.0,
        "alpha_sum": 0.0,
        "beta_sum": 0.0,
        "accessible_windows": 0,
        "sampled_windows": 0,
    }


def _finalize_loader_stats(acc: Dict[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    finite = max(int(acc["finite_variate_windows"]), 1)

    def average(key: str) -> float:
        if acc["finite_variate_windows"] == 0:
            return math.nan
        return float(acc[key] / finite)

    value_count = int(acc["lookback_value_count"])
    value_mean = (
        float(acc["lookback_value_sum"] / value_count) if value_count else math.nan
    )
    value_variance = (
        max(float(acc["lookback_value_square_sum"] / value_count) - value_mean**2, 0.0)
        if value_count
        else math.nan
    )
    return {
        "metadata": dict(metadata),
        "accessible_windows": int(acc["accessible_windows"]),
        "sampled_windows": int(acc["sampled_windows"]),
        "item_windows": int(acc["item_windows"]),
        "variate_windows": int(acc["variate_windows"]),
        "finite_variate_windows": int(acc["finite_variate_windows"]),
        "invalid_variate_windows": int(acc["invalid_variate_windows"]),
        "constant_variate_windows": int(acc["constant_variate_windows"]),
        "lookback_mean": average("lookback_mean_sum"),
        "lookback_std": average("lookback_std_sum"),
        "future_mean": average("future_mean_sum"),
        "future_std": average("future_std_sum"),
        "lookback_value_mean": value_mean,
        "lookback_value_std": math.sqrt(value_variance),
        "lookback_value_min": (
            float(acc["lookback_value_min"]) if value_count else math.nan
        ),
        "lookback_value_max": (
            float(acc["lookback_value_max"]) if value_count else math.nan
        ),
        "future_value_mean": average("future_value_mean_sum"),
        "future_value_std": average("future_value_std_sum"),
        "alpha": average("alpha_sum"),
        "beta": average("beta_sum"),
    }


def _single_loader_stats(
    loader: DataLoader,
    *,
    max_windows: Optional[int],
    seed: Optional[int | str],
    eps: float,
) -> Dict[str, Any]:
    acc = _empty_loader_stats_accumulator()
    component_records: List[Dict[str, Any]] = []
    for dataset in _leaf_datasets(loader.dataset):
        component_records.append(
            {
                "dataset": dataset,
                "values": list(dataset.data.values.shape),
                "lags": dataset.lags,
                "horizon": dataset.horizon,
                "accessible_windows": 0,
                "sampled_windows": 0,
            }
        )
    limit = None if max_windows is None else int(max_windows)
    if limit is not None and limit < 1:
        raise ValueError("stats_max_windows must be positive when set")

    if limit is None:
        for record in component_records:
            dataset = record["dataset"]
            for individual, date in dataset.index_sampler.iter_accessible_pairs():
                record["accessible_windows"] += 1
                record["sampled_windows"] += 1
                acc["accessible_windows"] += 1
                acc["sampled_windows"] += 1
                _accumulate_window_stats(acc, dataset, individual, date, eps)
    else:
        rng = np.random.default_rng(None if seed in {None, "None"} else int(seed))
        reservoir: List[Tuple[int, int, int]] = []
        seen = 0
        for component_index, record in enumerate(component_records):
            dataset = record["dataset"]
            for individual, date in dataset.index_sampler.iter_accessible_pairs():
                seen += 1
                record["accessible_windows"] += 1
                acc["accessible_windows"] += 1
                if len(reservoir) < limit:
                    reservoir.append((component_index, individual, date))
                    continue
                replace_at = int(rng.integers(seen))
                if replace_at < limit:
                    reservoir[replace_at] = (component_index, individual, date)
        for component_index, individual, date in reservoir:
            record = component_records[component_index]
            dataset = record["dataset"]
            record["sampled_windows"] += 1
            acc["sampled_windows"] += 1
            _accumulate_window_stats(acc, dataset, individual, date, eps)

    components = [
        {key: value for key, value in record.items() if key != "dataset"}
        for record in component_records
    ]
    metadata = {
        "batch_size": loader.batch_size,
        "batches": len(loader),
        "dataset_length": len(loader.dataset),
        "max_windows": max_windows,
        "eps": float(eps),
        "components": components,
    }
    return _finalize_loader_stats(acc, metadata)


def get_loader_stats(
    loaders: Mapping[str, DataLoader],
    *,
    max_windows: Optional[int] = None,
    seed: Optional[int | str] = None,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, Any]]:
    """Compute numerical stats on accessible windows for each final loader."""
    return {
        key: _single_loader_stats(
            loader,
            max_windows=max_windows,
            seed=seed,
            eps=eps,
        )
        for key, loader in loaders.items()
    }


def _stats_file(path: str | Path, kind: str) -> Path:
    path = _path(path)
    folder = path.parent if path.suffix else path
    return folder / f"{kind}_stats.json"


def _artifact_file(path: str | Path, name: str) -> Path:
    path = _path(path)
    folder = path.parent if path.suffix else path
    return folder / f"{name}.json"


def _save_loader_artifacts(
    path: Optional[str | Path],
    *,
    stats: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if path is None:
        return
    if stats is not None:
        _write_json(stats, _artifact_file(path, "stats"))
    if metadata is not None:
        _write_json(metadata, _artifact_file(path, "metadata"))


def _sampling_for_key(sampling: Mapping[str, Any], key: str) -> Dict[str, Any]:
    prefix = "train" if key == "train" else "eval"
    legacy_use_context = bool(sampling.get("use_context", True))
    return {
        "idx_mode": sampling.get(f"{prefix}_idx_mode", "random"),
        "remove_cte": bool(sampling.get(f"remove_{prefix}_cte", False)),
        "drop_constant_individuals": bool(
            sampling.get(f"drop_{prefix}_constant_individuals", False)
        ),
        "shuffle": bool(sampling.get(f"shuffle_{prefix}", key == "train")),
        "stride": int(sampling.get(f"{prefix}_stride", 1)),
        "weight": int(sampling.get(f"{prefix}_len_multiplier", 1)),
        "block_individuals": int(
            sampling.get(f"{prefix}_block_individuals", 1)
        ) if sampling.get(f"{prefix}_block_individuals", 1) is not None else None,
        "use_individual_context": bool(
            sampling.get("use_individual_context", legacy_use_context)
        ),
        "use_global_context": bool(
            sampling.get("use_global_context", legacy_use_context)
        ),
    }


def drop_constant_individuals(
    data: TimeSeriesData,
    lags: int,
    horizon: int,
    stride: int = 1,
) -> TimeSeriesData:
    """Drop users with a constant or non-finite accessible lookback window."""
    query_dates = _valid_query_dates(
        data.dates,
        lags,
        horizon,
        data.target_date_indices,
    )[:: int(stride)]
    if not query_dates:
        raise ValueError("split has no query date with a complete target")
    windows = torch.stack(
        [
            data.values[..., query_t - int(lags) + 1 : query_t + 1]
            for query_t in query_dates
        ],
        dim=-2,
    )
    invalid = ~torch.isfinite(windows).all(dim=-1)
    constant = windows.std(dim=-1, unbiased=False) == 0
    drop = (invalid | constant).any(dim=(1, 2))
    keep = torch.nonzero(~drop, as_tuple=False).flatten().tolist()
    return data.subset(individual_indices=keep)


def _group_options(
    options: Optional[Mapping[str, Any]],
    group: str,
) -> Dict[str, Any]:
    """Merge common options with an optional per-cluster override."""
    _reject_removed_group_alias(options)
    options = dict(options or {})
    overrides = options.pop("by_cluster", None) or {}
    group_options = overrides.get(group, {})
    merged = dict(options)
    for key in ("date_splits", "indiv_split"):
        value = merged.get(key)
        if isinstance(value, Mapping):
            if group in value:
                merged[key] = value[group]
            elif "default" in value:
                merged[key] = value["default"]
            else:
                merged.pop(key)
    for key in ("sizes", "modes", "specs"):
        value = merged.get(key)
        if (
            isinstance(value, Mapping)
            and group in value
            and isinstance(value[group], Mapping)
        ):
            merged[key] = value[group]
    for key, value in group_options.items():
        if (
            key in {"sizes", "modes"}
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _split_creation_summary(
    options: Mapping[str, Any],
    load_path: Optional[str | Path],
    seed: Optional[int | str],
    group: str = "default",
) -> Dict[str, Any]:
    if load_path is not None:
        metadata = _load_split_metadata(load_path, group=group)
        return {
            **metadata,
            "creation_mode": metadata.get("creation_mode", "unknown"),
            "access_mode": "loaded",
            "source": str(_json_file(load_path, "splits")),
        }
    if _precomputed_split_specs(options) is not None:
        return {
            "creation_mode": "precomputed",
            "access_mode": "provided",
            "randomized": False,
        }
    return {
        "creation_mode": "generated",
        "access_mode": "created",
        "randomized": bool(
            options.get("shuffle_individuals", True)
            and options.get("indiv_split") not in {None, 1, 1.0}
        ),
        "date_splits": _coerce_ratios(options.get("date_splits")),
        "indiv_split": options.get("indiv_split"),
        "shuffle_individuals": bool(
            options.get("shuffle_individuals", True)
        ),
        "seed": None if seed in {None, "None"} else int(seed),
    }


def _subset_file(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    return _json_file(path, "subsets")


def save_subset_spec(
    path: str | Path,
    split: str,
    *,
    mode: str,
    indices: Sequence[int],
    stride: int,
    individuals: int,
    valid_dates: int,
    group: str = "default",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    file = _json_file(path, "subsets")
    payload = _read_json(
        file,
        {"version": 3, "kind": "subset_specs", "groups": {}},
    )
    if payload.get("kind") != "subset_specs":
        raise ValueError(f"{file} is not a subset specification file")
    payload["version"] = 3
    group_specs = payload.setdefault("groups", {}).setdefault(group, {})
    group_specs[split] = {
        "metadata": dict(
            metadata
            or {
                "creation_mode": "precomputed",
                "access_mode": "provided",
                "randomized": False,
            }
        ),
        "mode": mode,
        "indices": [int(index) for index in indices],
        "stride": int(stride),
        "individuals": int(individuals),
        "valid_dates": int(valid_dates),
        "date_anchor": "query_t",
    }
    return _write_json(payload, file)


def load_subset_spec(
    path: str | Path,
    split: str,
    *,
    individuals: int,
    valid_dates: int,
    group: str = "default",
) -> Dict[str, Any]:
    file = _json_file(path, "subsets")
    payload = _read_json(file)
    if payload is None:
        raise FileNotFoundError(file)
    if payload.get("kind") != "subset_specs":
        raise ValueError(f"{file} is not a subset specification file")
    try:
        spec = payload["groups"][group][split]
    except KeyError as error:
        raise KeyError(
            f"subset {group!r}/{split!r} not found in {file}"
        ) from error
    if spec.get("date_anchor") != "query_t":
        raise ValueError(
            f"subset {group!r}/{split!r} predates query-date window anchoring; "
            "regenerate the subset specification"
        )
    if int(spec["individuals"]) != individuals:
        raise ValueError(
            f"subset expects {spec['individuals']} individuals, got {individuals}"
        )
    if int(spec["valid_dates"]) != valid_dates:
        raise ValueError(
            f"subset expects {spec['valid_dates']} valid dates, got {valid_dates}"
        )
    return {
        "metadata": dict(spec.get("metadata", {})),
        "mode": str(spec["mode"]),
        "indices": [int(index) for index in spec["indices"]],
        "stride": int(spec["stride"]),
    }
