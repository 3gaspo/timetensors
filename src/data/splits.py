"""Dataset split, cluster, and subset specifications."""

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

def _coerce_ratios(ratios: Any) -> List[float]:
    if ratios is None:
        return [1.0]
    if isinstance(ratios, str):
        ratios = [float(value) for value in ratios.split(";")]
    elif isinstance(ratios, (int, float)):
        ratios = [float(ratios)]
    else:
        ratios = [float(value) for value in ratios]
    if not 1 <= len(ratios) <= 3:
        raise ValueError("date_splits must contain one, two, or three ratios")
    if any(value <= 0 for value in ratios):
        raise ValueError("date split ratios must be positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"date split ratios must sum to 1, got {ratios}")
    return ratios


def _date_blocks(dates: int, ratios: Sequence[float]) -> List[List[int]]:
    stops = [0]
    cumulative = 0.0
    for ratio in ratios[:-1]:
        cumulative += ratio
        stops.append(int(cumulative * dates))
    stops.append(dates)
    blocks = [list(range(stops[i], stops[i + 1])) for i in range(len(ratios))]
    if any(not block for block in blocks):
        raise ValueError("date split creates an empty block")
    return blocks


def _individual_blocks(
    individuals: int,
    indiv_split: Optional[float],
    seed: Optional[int | str],
    shuffle: bool,
) -> List[List[int]]:
    if indiv_split is None or float(indiv_split) == 1.0:
        return [list(range(individuals))]
    ratio = float(indiv_split)
    if not 0 < ratio < 1:
        raise ValueError("indiv_split must be in (0, 1]")
    rng = np.random.default_rng(None if seed in {None, "None"} else int(seed))
    indices = (
        rng.permutation(individuals).astype(int).tolist()
        if shuffle
        else list(range(individuals))
    )
    stop = int(ratio * individuals)
    if stop == 0 or stop == individuals:
        raise ValueError("individual split creates an empty group")
    return [indices[:stop], indices[stop:]]


def make_split_specs(
    individuals: int,
    dates: int,
    *,
    date_splits: Any = None,
    indiv_split: Optional[float] = None,
    seed: Optional[int | str] = None,
    shuffle_individuals: bool = True,
) -> Dict[str, SplitSpec]:
    date_blocks = _date_blocks(dates, _coerce_ratios(date_splits))
    if len(date_blocks) == 1:
        return {
            "test1": SplitSpec(
                "test1",
                list(range(individuals)),
                date_blocks[0],
            )
        }
    individual_blocks = _individual_blocks(
        individuals, indiv_split, seed, shuffle_individuals
    )
    if len(individual_blocks) == 1:
        names = {
            1: ["test1"],
            2: ["train", "test1"],
            3: ["train", "valid1", "test1"],
        }[len(date_blocks)]
        return {
            name: SplitSpec(name, individual_blocks[0], dates_)
            for name, dates_ in zip(names, date_blocks)
        }
    names = {
        1: [["test1"], ["test2"]],
        2: [["train", "test1"], ["test2", "test3"]],
        3: [
            ["train", "valid1", "test1"],
            ["valid2", "valid3", "test2"],
        ],
    }[len(date_blocks)]
    specs: Dict[str, SplitSpec] = {}
    for group, individuals_ in enumerate(individual_blocks):
        for name, dates_ in zip(names[group], date_blocks):
            specs[name] = SplitSpec(name, individuals_, dates_)
    return specs


def _coerce_split_specs(raw: Mapping[str, Any]) -> Dict[str, SplitSpec]:
    specs = {}
    for name, value in raw.items():
        if isinstance(value, SplitSpec):
            specs[name] = value
        else:
            specs[name] = SplitSpec(
                name=str(value.get("name", name)),
                individual_indices=[
                    int(index) for index in value["individual_indices"]
                ],
                date_indices=[int(index) for index in value["date_indices"]],
            )
    return specs


def _reject_removed_group_alias(options: Optional[Mapping[str, Any]]) -> None:
    if options is not None and "by_group" in options:
        raise ValueError("by_group was removed; use by_cluster")


def _precomputed_split_specs(options: Mapping[str, Any]) -> Optional[Dict[str, SplitSpec]]:
    _reject_removed_group_alias(options)
    raw = options.get("specs")
    if raw is None and options and all(
        isinstance(value, (SplitSpec, Mapping))
        and (
            isinstance(value, SplitSpec)
            or {"individual_indices", "date_indices"} <= set(value)
        )
        for key, value in options.items()
        if key != "by_cluster"
    ):
        raw = {
            key: value
            for key, value in options.items()
            if key != "by_cluster"
        }
    return None if raw is None else _coerce_split_specs(raw)


def save_split_specs(
    specs: Mapping[str, SplitSpec],
    path: str | Path,
    group: str = "default",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    file = _json_file(path, "splits")
    payload = _read_json(
        file,
        {"version": 2, "kind": "split_specs", "groups": {}},
    )
    if payload.get("kind") != "split_specs":
        raise ValueError(f"{file} is not a split specification file")
    payload.setdefault("groups", {})[group] = {
        "metadata": dict(
            metadata
            or {
                "creation_mode": "precomputed",
                "access_mode": "provided",
                "randomized": False,
            }
        ),
        "splits": {
            name: asdict(spec) for name, spec in specs.items()
        },
    }
    return _write_json(payload, file)


def load_split_specs(
    path: str | Path,
    individuals: Optional[int] = None,
    dates: Optional[int] = None,
    group: str = "default",
) -> Dict[str, SplitSpec]:
    file = _json_file(path, "splits")
    payload = _read_json(file)
    if payload is None:
        raise FileNotFoundError(file)
    if payload.get("kind") == "split_specs":
        try:
            group_payload = payload["groups"][group]
        except KeyError as error:
            raise KeyError(f"split group {group!r} not found in {file}") from error
        raw = group_payload.get("splits", group_payload)
    else:
        # Read legacy plain split JSON files.
        raw = payload
    specs = {
        name: SplitSpec(
            name=value.get("name", name),
            individual_indices=[int(i) for i in value["individual_indices"]],
            date_indices=[int(i) for i in value["date_indices"]],
        )
        for name, value in raw.items()
    }
    if individuals is not None:
        for spec in specs.values():
            _normalise_indices(spec.individual_indices, individuals)
    if dates is not None:
        for spec in specs.values():
            _normalise_indices(spec.date_indices, dates)
    return specs


def _load_split_metadata(
    path: str | Path,
    group: str,
) -> Dict[str, Any]:
    payload = _read_json(_json_file(path, "splits"), {})
    group_payload = payload.get("groups", {}).get(group, {})
    return dict(group_payload.get("metadata", {}))


def get_dataset_splits(
    splits: Mapping[str, Any],
    data_path: Optional[str | Path] = None,
    data: Optional[TimeSeriesData] = None,
    cluster_ids: Optional[Sequence[int]] = None,
    split_save_path: Optional[str | Path] = None,
    split_load_path: Optional[str | Path] = None,
    split_group: str = "default",
    seed: Optional[int | str] = None,
    legacy_context_kind: Optional[str] = None,
) -> Dict[str, TimeSeriesData]:
    _reject_removed_group_alias(splits)
    if data is None:
        if data_path is None:
            raise ValueError("data_path is required when data is not provided")
        data = load_data(data_path, legacy_context_kind=legacy_context_kind)
    if cluster_ids is not None:
        data = data.subset(cluster_ids)
    precomputed = _precomputed_split_specs(splits)
    if split_load_path is not None:
        specs = load_split_specs(
            split_load_path,
            data.individuals,
            data.dates,
            group=split_group,
        )
        metadata = {
            **_load_split_metadata(split_load_path, split_group),
            "creation_mode": _load_split_metadata(
                split_load_path,
                split_group,
            ).get("creation_mode", "unknown"),
            "access_mode": "loaded",
            "source": str(_json_file(split_load_path, "splits")),
        }
    elif precomputed is not None:
        specs = precomputed
        for spec in specs.values():
            _normalise_indices(spec.individual_indices, data.individuals)
            _normalise_indices(spec.date_indices, data.dates)
        metadata = {
            "creation_mode": "precomputed",
            "access_mode": "provided",
            "randomized": False,
        }
    else:
        specs = make_split_specs(
            data.individuals,
            data.dates,
            date_splits=splits.get("date_splits"),
            indiv_split=splits.get("indiv_split"),
            seed=seed,
            shuffle_individuals=bool(splits.get("shuffle_individuals", True)),
        )
        metadata = {
            "creation_mode": "generated",
            "access_mode": "created",
            "randomized": bool(
                splits.get("shuffle_individuals", True)
                and splits.get("indiv_split") not in {None, 1, 1.0}
            ),
            "date_splits": _coerce_ratios(splits.get("date_splits")),
            "indiv_split": splits.get("indiv_split"),
            "shuffle_individuals": bool(
                splits.get("shuffle_individuals", True)
            ),
            "seed": None if seed in {None, "None"} else int(seed),
        }
    if split_save_path is not None:
        save_split_specs(
            specs,
            split_save_path,
            group=split_group,
            metadata=metadata,
        )
    data_by_individuals: Dict[Tuple[int, ...], TimeSeriesData] = {}
    result: Dict[str, TimeSeriesData] = {}
    for name, spec in specs.items():
        user_key = tuple(spec.individual_indices)
        if user_key not in data_by_individuals:
            data_by_individuals[user_key] = data.subset(
                individual_indices=spec.individual_indices
            )
        result[name] = replace(
            data_by_individuals[user_key],
            target_date_ids=data.date_ids[spec.date_indices],
        )
    return result


def create_iid_clusters(
    individuals: int,
    *,
    ratios: Optional[Sequence[float]] = None,
    sizes: Optional[Sequence[int]] = None,
    n_clusters: Optional[int] = None,
    seed: Optional[int | str] = None,
) -> Dict[str, List[int]]:
    provided = sum(value is not None for value in (ratios, sizes, n_clusters))
    if provided != 1:
        raise ValueError("provide exactly one of ratios, sizes, or n_clusters")
    rng = np.random.default_rng(None if seed in {None, "None"} else int(seed))
    permutation = rng.permutation(individuals).astype(int).tolist()
    if n_clusters is not None:
        if int(n_clusters) < 1:
            raise ValueError("n_clusters must be positive")
        groups = np.array_split(permutation, int(n_clusters))
        return {f"cluster{i}": list(map(int, group)) for i, group in enumerate(groups)}
    if ratios is not None:
        ratios_array = np.asarray(ratios, dtype=float)
        if np.any(ratios_array <= 0):
            raise ValueError("cluster ratios must be positive")
        raw = ratios_array / ratios_array.sum() * individuals
        sizes_array = np.floor(raw).astype(int)
        remainder = individuals - int(sizes_array.sum())
        for index in np.argsort(raw - sizes_array)[::-1][:remainder]:
            sizes_array[index] += 1
        sizes = sizes_array.tolist()
    else:
        sizes = [int(size) for size in sizes or []]
        if any(size <= 0 for size in sizes) or sum(sizes) != individuals:
            raise ValueError("cluster sizes must be positive and sum to individuals")
    clusters = {}
    start = 0
    for index, size in enumerate(sizes):
        clusters[f"cluster{index}"] = permutation[start : start + size]
        start += size
    return clusters


def save_clusters(
    clusters: Mapping[str, Sequence[int]],
    path: str | Path,
    overwrite: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    file = _json_file(path, "clusters")
    if file.exists() and not overwrite:
        raise FileExistsError(file)
    payload = {
        "version": 2,
        "kind": "cluster_specs",
        "metadata": dict(
            metadata
            or {
                "creation_mode": "precomputed",
                "access_mode": "provided",
                "randomized": False,
            }
        ),
        "clusters": {
            name: [int(i) for i in indices]
            for name, indices in clusters.items()
        },
    }
    return _write_json(payload, file)


def create_and_save_iid_clusters(
    individuals: int,
    path: str | Path,
    *,
    ratios: Optional[Sequence[float]] = None,
    sizes: Optional[Sequence[int]] = None,
    n_clusters: Optional[int] = None,
    seed: Optional[int | str] = None,
) -> Dict[str, List[int]]:
    clusters = create_iid_clusters(
        individuals,
        ratios=ratios,
        sizes=sizes,
        n_clusters=n_clusters,
        seed=seed,
    )
    method = (
        "ratios"
        if ratios is not None
        else "sizes"
        if sizes is not None
        else "n_clusters"
    )
    save_clusters(
        clusters,
        path,
        metadata={
            "creation_mode": "generated",
            "access_mode": "created",
            "randomized": True,
            "method": method,
            "ratios": None if ratios is None else list(map(float, ratios)),
            "sizes": None if sizes is None else list(map(int, sizes)),
            "n_clusters": None if n_clusters is None else int(n_clusters),
            "seed": None if seed in {None, "None"} else int(seed),
        },
    )
    return clusters


def load_clusters(
    path: str | Path,
    individuals: Optional[int] = None,
) -> Dict[str, List[int]]:
    file = _json_file(path, "clusters")
    raw = _read_json(file)
    if raw is None:
        raise FileNotFoundError(file)
    values = raw["clusters"] if raw.get("kind") == "cluster_specs" else raw
    clusters = {name: [int(i) for i in indices] for name, indices in values.items()}
    if not clusters:
        raise ValueError(f"no clusters found at {file}")
    flat = [individual for values in clusters.values() for individual in values]
    if len(flat) != len(set(flat)):
        raise ValueError("cluster definitions overlap")
    if individuals is not None:
        _normalise_indices(flat, individuals)
    return clusters


def _load_cluster_metadata(path: str | Path) -> Dict[str, Any]:
    payload = _read_json(_json_file(path, "clusters"), {})
    return dict(payload.get("metadata", {}))


def assign_cluster_ids(
    data: TimeSeriesData,
    clusters: Mapping[str, Sequence[int]],
) -> TimeSeriesData:
    cluster_ids = torch.full((data.individuals,), -1, dtype=torch.long)
    for cluster_id, indices in enumerate(clusters.values()):
        valid = _normalise_indices(indices, data.individuals)
        if (cluster_ids[valid] >= 0).any():
            raise ValueError("cluster definitions overlap")
        cluster_ids[valid] = cluster_id
    return replace(data, cluster_ids=cluster_ids)


def get_subset_indices(
    dates: int,
    individuals: int,
    lags: int,
    horizon: int,
    ratio: float,
    subset_mode: str,
    seed: Optional[int | str] = None,
    target_date_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("subset ratio must be in (0, 1]")
    query_dates = _valid_query_dates(
        dates,
        lags,
        horizon,
        target_date_indices,
    )
    if subset_mode == "dates":
        candidates = query_dates
    elif subset_mode == "individuals":
        candidates = list(range(individuals))
    elif subset_mode == "all":
        candidates = [
            query_t * individuals + individual
            for query_t in query_dates
            for individual in range(individuals)
        ]
    else:
        raise ValueError(f"unknown subset_mode {subset_mode!r}")
    count = max(1, int(len(candidates) * ratio))
    rng = np.random.default_rng(None if seed in {None, "None"} else int(seed))
    return rng.choice(candidates, size=count, replace=False).astype(int).tolist()
