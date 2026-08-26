"""Core time-series data contracts and shared shape helpers."""

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

def normalize(
    x: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (x - mean) / (std + eps)


def set_seed(seed: Optional[int | str]) -> None:
    if seed in {None, "None"}:
        return
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_available() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA initialization:.*")
            return torch.cuda.is_available()
    except RuntimeError:
        return False


def _path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _ensure_dir(path: str | Path) -> Path:
    path = _path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_file(path: str | Path, default_stem: str) -> Path:
    """Resolve a JSON artifact without creating a dedicated subfolder."""
    path = _path(path)
    if path.suffix:
        if path.suffix.lower() != ".json":
            raise ValueError(f"expected a .json path, got {path}")
        return path
    if path.exists() and path.is_dir():
        return path / f"{default_stem}.json"
    return path.with_suffix(".json")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(value: Any, path: Path) -> Path:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    return path


def _as_values_3d(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim == 1:
        return values.view(1, 1, -1)
    if values.ndim == 2:
        return values.unsqueeze(1)
    if values.ndim == 3:
        return values.float()
    raise ValueError(f"values must have 1, 2, or 3 dimensions, got {tuple(values.shape)}")


def _as_individual_context(
    context: Optional[torch.Tensor | np.ndarray],
    individuals: int,
    dates: int,
) -> Optional[torch.Tensor]:
    if context is None:
        return None
    context = torch.as_tensor(context, dtype=torch.float32)
    if context.ndim == 1:
        if context.shape[0] != individuals:
            raise ValueError("1D individual context must contain one value per individual")
        context = context.view(individuals, 1, 1)
    elif context.ndim == 2:
        if context.shape[0] != individuals:
            raise ValueError("2D individual context must have individuals on axis 0")
        context = context.unsqueeze(1)
    elif context.ndim != 3:
        raise ValueError("individual context must have 1, 2, or 3 dimensions")
    if context.shape[0] != individuals:
        raise ValueError(
            f"individual context has {context.shape[0]} individuals, expected {individuals}"
        )
    if context.shape[-1] not in {1, dates}:
        raise ValueError(f"individual context dates must be 1 or {dates}")
    return context.float()


def _as_global_context(
    context: Optional[torch.Tensor | np.ndarray],
    dates: int,
) -> Optional[torch.Tensor]:
    if context is None:
        return None
    context = torch.as_tensor(context, dtype=torch.float32)
    if context.ndim == 1:
        context = context.view(1, -1)
    elif context.ndim == 3 and context.shape[0] == 1:
        context = context.squeeze(0)
    elif context.ndim != 2:
        raise ValueError("global context must have shape (variates, dates_or_1)")
    if context.shape[-1] not in {1, dates}:
        raise ValueError(f"global context dates must be 1 or {dates}")
    return context.float()


def _normalise_indices(indices: Optional[Sequence[int]], n: int) -> List[int]:
    if indices is None:
        return list(range(n))
    out = [int(i) for i in indices]
    if not out:
        raise ValueError("index list cannot be empty")
    bad = [i for i in out if i < 0 or i >= n]
    if bad:
        raise IndexError(f"indices out of bounds for size {n}: {bad[:5]}")
    return out


def _valid_query_dates(
    dates: int,
    lags: int,
    horizon: int,
    target_date_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return cutoffs ``t`` with ``X=(t-L,t]`` and an owned full horizon."""
    if target_date_indices is None:
        target_date_indices = list(range(dates))
    targets = np.zeros(int(dates), dtype=bool)
    target_indices = np.asarray(
        [int(index) for index in target_date_indices],
        dtype=int,
    )
    targets[target_indices] = True
    cumulative = np.concatenate(([0], np.cumsum(targets, dtype=np.int64)))
    return [
        query_t
        for query_t in range(int(lags) - 1, int(dates) - int(horizon))
        if (
            cumulative[query_t + int(horizon) + 1]
            - cumulative[query_t + 1]
            == int(horizon)
        )
    ]


@dataclass
class TimeSeriesData:
    """Full timeline plus the date IDs owned as forecast targets."""

    values: torch.Tensor | np.ndarray
    datetimes: Sequence[Any]
    individual_context: Optional[torch.Tensor | np.ndarray] = None
    global_context: Optional[torch.Tensor | np.ndarray] = None
    individual_ids: Optional[Sequence[int] | torch.Tensor] = None
    cluster_ids: Optional[Sequence[int] | torch.Tensor] = None
    date_ids: Optional[Sequence[int] | torch.Tensor] = None
    individual_names: Optional[Mapping[int, str] | Sequence[str]] = None
    target_date_ids: Optional[Sequence[int] | torch.Tensor] = None

    def __post_init__(self) -> None:
        self.values = _as_values_3d(self.values)
        individuals, _, dates = self.values.shape
        self.datetimes = np.asarray(self.datetimes)
        if len(self.datetimes) != dates:
            raise ValueError(f"datetimes length {len(self.datetimes)} does not match {dates}")
        self.individual_context = _as_individual_context(
            self.individual_context, individuals, dates
        )
        self.global_context = _as_global_context(self.global_context, dates)
        if self.individual_ids is None:
            self.individual_ids = torch.arange(individuals, dtype=torch.long)
        else:
            self.individual_ids = torch.as_tensor(self.individual_ids, dtype=torch.long)
        if len(self.individual_ids) != individuals:
            raise ValueError("individual_ids must contain one id per individual")
        if torch.unique(self.individual_ids).numel() != individuals:
            raise ValueError("individual_ids must be unique")
        ids = [int(value) for value in self.individual_ids.tolist()]
        if self.individual_names is None:
            self.individual_names = {
                individual_id: f"serie_{individual_id}" for individual_id in ids
            }
        elif isinstance(self.individual_names, Mapping):
            names = {int(key): str(value) for key, value in self.individual_names.items()}
            missing = [individual_id for individual_id in ids if individual_id not in names]
            if missing:
                raise ValueError(
                    f"individual_names is missing source ids: {missing[:5]}"
                )
            self.individual_names = {
                individual_id: names[individual_id] for individual_id in ids
            }
        else:
            names = [str(value) for value in self.individual_names]
            if len(names) != individuals:
                raise ValueError(
                    "individual_names must contain one name per individual"
                )
            self.individual_names = dict(zip(ids, names))
        if self.cluster_ids is not None:
            self.cluster_ids = torch.as_tensor(self.cluster_ids, dtype=torch.long)
            if len(self.cluster_ids) != individuals:
                raise ValueError("cluster_ids must contain one id per individual")
        if self.date_ids is None:
            self.date_ids = torch.arange(dates, dtype=torch.long)
        else:
            self.date_ids = torch.as_tensor(self.date_ids, dtype=torch.long)
        if len(self.date_ids) != dates:
            raise ValueError("date_ids must contain one id per date")
        if torch.unique(self.date_ids).numel() != dates:
            raise ValueError("date_ids must be unique")
        if self.target_date_ids is None:
            self.target_date_ids = self.date_ids.clone()
        else:
            self.target_date_ids = torch.as_tensor(
                self.target_date_ids,
                dtype=torch.long,
            )
        if torch.unique(self.target_date_ids).numel() != len(self.target_date_ids):
            raise ValueError("target_date_ids must be unique")
        unknown_targets = set(self.target_date_ids.tolist()) - set(self.date_ids.tolist())
        if unknown_targets:
            raise ValueError(
                f"target_date_ids contains dates absent from date_ids: "
                f"{sorted(unknown_targets)[:5]}"
            )

    @property
    def individuals(self) -> int:
        return self.values.shape[0]

    @property
    def variates(self) -> int:
        return self.values.shape[1]

    @property
    def dates(self) -> int:
        return self.values.shape[2]

    @property
    def target_date_indices(self) -> List[int]:
        targets = set(int(value) for value in self.target_date_ids.tolist())
        return [
            index
            for index, date_id in enumerate(self.date_ids.tolist())
            if int(date_id) in targets
        ]

    def to_dataframes(
        self, variate: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Return values and each context variate as separate date-indexed frames."""
        from .frames import dataframes_from_data

        return dataframes_from_data(self, variate=variate)

    def get_df(self, variate: Optional[int] = None) -> pd.DataFrame:
        """Return the values frame; use ``to_dataframes`` to include contexts."""
        return self.to_dataframes(variate=variate)["values"]

    def subset(
        self,
        individual_indices: Optional[Sequence[int]] = None,
        date_indices: Optional[Sequence[int]] = None,
    ) -> "TimeSeriesData":
        indiv = _normalise_indices(individual_indices, self.individuals)
        dates = _normalise_indices(date_indices, self.dates)
        all_individuals = indiv == list(range(self.individuals))
        all_dates = dates == list(range(self.dates))
        values = self.values if all_individuals else self.values[indiv]
        if not all_dates:
            values = values[:, :, dates]
        individual_context = self.individual_context
        if individual_context is not None:
            if not all_individuals:
                individual_context = individual_context[indiv]
            if individual_context.shape[-1] > 1 and not all_dates:
                individual_context = individual_context[:, :, dates]
        global_context = self.global_context
        if (
            global_context is not None
            and global_context.shape[-1] > 1
            and not all_dates
        ):
            global_context = global_context[:, dates]
        individual_ids = (
            self.individual_ids
            if all_individuals
            else self.individual_ids[indiv]
        )
        cluster_ids = (
            self.cluster_ids
            if self.cluster_ids is None or all_individuals
            else self.cluster_ids[indiv]
        )
        selected_date_ids = self.date_ids[dates]
        targets = set(int(value) for value in self.target_date_ids.tolist())
        target_date_ids = [
            int(value)
            for value in selected_date_ids.tolist()
            if int(value) in targets
        ]
        return TimeSeriesData(
            values=values,
            datetimes=self.datetimes[dates],
            individual_context=individual_context,
            global_context=global_context,
            individual_ids=individual_ids,
            cluster_ids=cluster_ids,
            date_ids=self.date_ids[dates],
            individual_names=self.individual_names,
            target_date_ids=target_date_ids,
        )


@dataclass(frozen=True)
class SplitSpec:
    name: str
    individual_indices: List[int]
    date_indices: List[int]  # target dates; lookbacks may use earlier dates


@dataclass(frozen=True)
class SamplerConfig:
    idx_mode: str = "random"
    # Number of users returned by one dataset item. ``None`` means all users
    # and is supported by random/date indexing.
    block_individuals: Optional[int] = 1
    use_individual_context: bool = True
    use_global_context: bool = True
    remove_cte: bool = False
    weight: int = 1
    subset_indices: Optional[List[int]] = None
    subset_mode: Optional[str] = None
    stride: int = 1


def apply_split(data: TimeSeriesData, spec: SplitSpec) -> TimeSeriesData:
    split = data.subset(individual_indices=spec.individual_indices)
    return replace(
        split,
        target_date_ids=data.date_ids[spec.date_indices],
    )
