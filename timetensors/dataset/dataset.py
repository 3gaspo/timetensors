"""Standalone time-series dataset utilities for notebook experiments.

Values always use shape ``(individuals, variates, dates)``. A two-dimensional
input is interpreted as ``(individuals, dates)``; callers representing one
multivariate series must provide ``(1, variates, dates)`` explicitly.

Individual and global covariates are separate:

- individual context: ``(individuals, context_variates, dates_or_1)``
- global context: ``(context_variates, dates_or_1)``

Cluster identifiers are metadata, not repeated covariates.
Human-readable individual names are keyed by stable source individual IDs.
"""

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


def _valid_start_count(dates: int, lags: int, horizon: int) -> int:
    return dates - lags - horizon + 1


@dataclass
class TimeSeriesData:
    values: torch.Tensor | np.ndarray
    datetimes: Sequence[Any]
    individual_context: Optional[torch.Tensor | np.ndarray] = None
    global_context: Optional[torch.Tensor | np.ndarray] = None
    individual_ids: Optional[Sequence[int] | torch.Tensor] = None
    cluster_ids: Optional[Sequence[int] | torch.Tensor] = None
    date_ids: Optional[Sequence[int] | torch.Tensor] = None
    individual_names: Optional[Mapping[int, str] | Sequence[str]] = None

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

    @property
    def individuals(self) -> int:
        return self.values.shape[0]

    @property
    def variates(self) -> int:
        return self.values.shape[1]

    @property
    def dates(self) -> int:
        return self.values.shape[2]

    def to_dataframes(
        self, variate: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Return values and each context variate as separate date-indexed frames."""
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
        individual_context = self.individual_context
        if individual_context is not None:
            individual_context = individual_context[indiv]
            if individual_context.shape[-1] > 1:
                individual_context = individual_context[:, :, dates]
        global_context = self.global_context
        if global_context is not None and global_context.shape[-1] > 1:
            global_context = global_context[:, dates]
        cluster_ids = None if self.cluster_ids is None else self.cluster_ids[indiv]
        return TimeSeriesData(
            values=self.values[indiv][:, :, dates],
            datetimes=self.datetimes[dates],
            individual_context=individual_context,
            global_context=global_context,
            individual_ids=self.individual_ids[indiv],
            cluster_ids=cluster_ids,
            date_ids=self.date_ids[dates],
            individual_names=self.individual_names,
        )


@dataclass(frozen=True)
class SplitSpec:
    name: str
    individual_indices: List[int]
    date_indices: List[int]


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
    return data.subset(spec.individual_indices, spec.date_indices)


class IndexSampler:
    VALID_IDX_MODES = {"random", "dates", "individuals", "all"}
    VALID_SUBSET_MODES = {None, "dates", "individuals", "all"}

    def __init__(
        self,
        values: torch.Tensor,
        lags: int,
        horizon: int,
        config: Optional[SamplerConfig] = None,
    ):
        self.values = values
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.config = config or SamplerConfig()
        self._validate()
        if not self.date_candidates:
            raise ValueError("date candidate set cannot be empty")
        if not self.individual_candidates:
            raise ValueError("individual candidate set cannot be empty")
        if self.config.idx_mode == "all" and not self.pair_candidates:
            raise ValueError("no valid individual/date pairs")

    @property
    def individuals(self) -> int:
        return self.values.shape[0]

    @property
    def dates(self) -> int:
        return self.values.shape[-1]

    @property
    def max_dates(self) -> int:
        return _valid_start_count(self.dates, self.lags, self.horizon)

    @cached_property
    def date_candidates(self) -> List[int]:
        if self.config.subset_indices is not None and self.config.subset_mode == "dates":
            dates = [int(i) for i in self.config.subset_indices]
            bad = [i for i in dates if i < 0 or i >= self.max_dates]
            if bad:
                raise IndexError(f"date subset contains invalid starts: {bad[:5]}")
            if len(dates) != len(set(dates)):
                raise ValueError("date subset indices must be unique")
            candidates = dates
        else:
            candidates = list(range(0, self.max_dates, self.config.stride))
        if (
            self.config.remove_cte
            and self.config.idx_mode in {"random", "dates"}
        ):
            candidates = [
                date
                for date in candidates
                if len(self._valid_individuals(date))
                >= (
                    len(self._raw_individual_candidates())
                    if self.config.block_individuals is None
                    else min(
                        self.config.block_individuals,
                        len(self._raw_individual_candidates()),
                    )
                )
            ]
        return candidates

    @cached_property
    def individual_candidates(self) -> List[int]:
        candidates = self._raw_individual_candidates()
        if self.config.remove_cte and self.config.idx_mode == "individuals":
            candidates = [
                individual
                for individual in candidates
                if self._valid_dates(individual)
            ]
        return candidates

    def _raw_individual_candidates(self) -> List[int]:
        if (
            self.config.subset_indices is not None
            and self.config.subset_mode == "individuals"
        ):
            individuals = _normalise_indices(
                self.config.subset_indices, self.individuals
            )
            if len(individuals) != len(set(individuals)):
                raise ValueError("individual subset indices must be unique")
            return individuals
        return list(range(self.individuals))

    @cached_property
    def pair_candidates(self) -> List[Tuple[int, int]]:
        if self.config.subset_indices is not None and self.config.subset_mode == "all":
            pairs = []
            for raw in self.config.subset_indices:
                raw = int(raw)
                if raw < 0:
                    raise IndexError(f"flat pair index {raw} cannot be negative")
                indiv = raw % self.individuals
                date = raw // self.individuals
                if date >= self.max_dates:
                    raise IndexError(f"flat pair index {raw} maps to invalid date {date}")
                pairs.append((indiv, date))
            if len(pairs) != len(set(pairs)):
                raise ValueError("flat pair subset indices must be unique")
        else:
            pairs = [
                (individual, date)
                for date in self.date_candidates
                for individual in self.individual_candidates
            ]
        if self.config.remove_cte:
            pairs = [
                (individual, date)
                for individual, date in pairs
                if self._non_constant([individual], date)
            ]
        return pairs

    @property
    def true_len(self) -> int:
        if self.config.idx_mode == "random":
            return 1
        if self.config.idx_mode == "dates":
            return len(self.date_candidates)
        if self.config.idx_mode == "individuals":
            return len(self.individual_candidates)
        return len(self.pair_candidates)

    def __len__(self) -> int:
        return self.config.weight * self.true_len

    def _validate(self) -> None:
        cfg = self.config
        if cfg.idx_mode not in self.VALID_IDX_MODES:
            raise ValueError(f"unknown idx_mode {cfg.idx_mode!r}")
        if cfg.subset_mode not in self.VALID_SUBSET_MODES:
            raise ValueError(f"unknown subset_mode {cfg.subset_mode!r}")
        if cfg.stride < 1 or cfg.weight < 1:
            raise ValueError("stride and weight must be positive")
        if cfg.block_individuals is not None and cfg.block_individuals < 1:
            raise ValueError("block_individuals must be positive or None")
        if self.max_dates <= 0:
            raise ValueError("not enough dates for requested lags and horizon")
        if cfg.subset_mode == "all" and cfg.stride != 1:
            raise ValueError("encode stride in flat subset indices for subset_mode='all'")
        if cfg.subset_mode == "all" and cfg.idx_mode != "all":
            raise ValueError("subset_mode='all' requires idx_mode='all'")
        if cfg.idx_mode in {"individuals", "all"} and cfg.block_individuals != 1:
            raise ValueError(
                "block_individuals must be 1 for individuals and all index modes"
            )

    def _sample_individuals(self, fixed: Optional[int] = None) -> List[int]:
        if fixed is not None:
            return [fixed]
        candidates = self.individual_candidates
        count = (
            len(candidates)
            if self.config.block_individuals is None
            else min(self.config.block_individuals, len(candidates))
        )
        if count == len(candidates):
            return list(candidates)
        return np.random.choice(candidates, size=count, replace=False).astype(int).tolist()

    def _valid_individuals(self, date: int) -> List[int]:
        return [
            individual
            for individual in self._raw_individual_candidates()
            if self._non_constant([individual], date)
        ]

    def _sample_valid_individuals(self, date: int) -> List[int]:
        candidates = self._valid_individuals(date)
        count = (
            len(candidates)
            if self.config.block_individuals is None
            else min(self.config.block_individuals, len(candidates))
        )
        if count == 0:
            raise ValueError(f"date {date} has no non-constant individual window")
        if count == len(candidates):
            return list(candidates)
        return np.random.choice(
            candidates, size=count, replace=False
        ).astype(int).tolist()

    def _valid_dates(self, individual: int) -> List[int]:
        candidates = (
            [int(index) for index in self.config.subset_indices]
            if self.config.subset_mode == "dates"
            and self.config.subset_indices is not None
            else list(range(0, self.max_dates, self.config.stride))
        )
        return [
            date
            for date in candidates
            if self._non_constant([individual], date)
        ]

    def _sample_date(self, fixed_step: Optional[int] = None) -> int:
        if fixed_step is None:
            return int(np.random.choice(self.date_candidates))
        return int(self.date_candidates[fixed_step])

    def _draw(self, idx: int) -> Tuple[List[int], int]:
        mode = self.config.idx_mode
        if mode == "random":
            date = self._sample_date()
            individuals = (
                self._sample_valid_individuals(date)
                if self.config.remove_cte
                else self._sample_individuals()
            )
            return individuals, date
        if mode == "dates":
            date = self._sample_date(idx)
            individuals = (
                self._sample_valid_individuals(date)
                if self.config.remove_cte
                else self._sample_individuals()
            )
            return individuals, date
        if mode == "individuals":
            individual = self.individual_candidates[idx]
            if self.config.remove_cte:
                return [individual], int(np.random.choice(self._valid_dates(individual)))
            return [individual], self._sample_date()
        individual, date = self.pair_candidates[idx]
        return [individual], date

    def _non_constant(self, individuals: Sequence[int], date: int) -> bool:
        lookback = self.values[list(individuals), :, date : date + self.lags]
        finite = torch.isfinite(lookback).all(dim=-1)
        return bool(
            ((lookback.std(dim=-1, unbiased=False) > 0) & finite)
            .any(dim=1)
            .all()
        )

    def __call__(self, raw_idx: int) -> Tuple[List[int], int]:
        if self.true_len == 0:
            raise ValueError("no valid sampling candidates")
        idx = int(raw_idx) % self.true_len
        individuals, date = self._draw(idx)
        if not self.config.remove_cte:
            return individuals, date
        if not self._non_constant(individuals, date):
            raise RuntimeError("remove_cte candidate filtering produced a constant window")
        return individuals, date

    def iter_accessible_pairs(self) -> Any:
        """Yield deterministic individual/date pairs reachable by this sampler.

        This ignores random ordering and item length inflation, but respects
        split/subset membership, stride, and constant-window filtering.
        """
        if self.config.subset_mode == "all":
            yield from self.pair_candidates
            return
        for date in self.date_candidates:
            for individual in self.individual_candidates:
                if self.config.remove_cte and not self._non_constant([individual], date):
                    continue
                yield individual, date


def _slice_temporal_context(
    context: torch.Tensor,
    date: int,
    length: int,
) -> torch.Tensor:
    if context.shape[-1] == 1:
        return context
    return context[..., date : date + length]


class TimeSeriesDataset(Dataset):
    """Sample forecasting windows and return explicit context and metadata."""

    def __init__(
        self,
        data: TimeSeriesData | torch.Tensor | np.ndarray,
        datetimes: Optional[Sequence[Any]] = None,
        individual_context: Optional[torch.Tensor | np.ndarray] = None,
        global_context: Optional[torch.Tensor | np.ndarray] = None,
        lags: int = 168,
        horizon: int = 24,
        individual_ids: Optional[Sequence[int] | torch.Tensor] = None,
        cluster_ids: Optional[Sequence[int] | torch.Tensor] = None,
        sampler_config: Optional[SamplerConfig] = None,
    ):
        super().__init__()
        if isinstance(data, TimeSeriesData):
            self.data = data
        else:
            values = _as_values_3d(data)
            if datetimes is None:
                datetimes = np.arange(values.shape[-1])
            self.data = TimeSeriesData(
                values,
                datetimes,
                individual_context,
                global_context,
                individual_ids,
                cluster_ids,
            )
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.index_sampler = IndexSampler(
            self.data.values, self.lags, self.horizon, sampler_config
        )
        self.standard_stats: Optional[Mapping[str, float]] = None

    @property
    def values(self) -> torch.Tensor:
        return self.data.values

    @property
    def datetimes(self) -> np.ndarray:
        return self.data.datetimes

    @property
    def shape(self) -> Dict[str, Any]:
        return {
            "values": tuple(self.data.values.shape),
            "individual_context": (
                None
                if self.data.individual_context is None
                else tuple(self.data.individual_context.shape)
            ),
            "global_context": (
                None
                if self.data.global_context is None
                else tuple(self.data.global_context.shape)
            ),
            "sampled_individuals": len(self.index_sampler.individual_candidates),
            "sampled_dates": len(self.index_sampler.date_candidates),
            "samples": len(self),
        }

    def __len__(self) -> int:
        return len(self.index_sampler)

    def to_dataframes(
        self, variate: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and unroll every sampled window into DataFrames."""
        return dataframes_from_dataset(self, variate=variate)

    def get_df(self, variate: Optional[int] = None) -> pd.DataFrame:
        """Return the unrolled values frame."""
        return self.to_dataframes(variate=variate)["values"]

    def normalize(self, stats: Mapping[str, float]) -> None:
        self.standard_stats = stats
        values = normalize(self.data.values, float(stats["mean"]), float(stats["std"]))
        self.data = replace(self.data, values=values)
        self.index_sampler = IndexSampler(
            self.data.values,
            self.lags,
            self.horizon,
            self.index_sampler.config,
        )

    def set_sampler(self, **kwargs: Any) -> None:
        config = asdict(self.index_sampler.config)
        for key, value in kwargs.items():
            if key not in config:
                raise AttributeError(f"SamplerConfig has no field {key!r}")
            config[key] = value
        self.index_sampler = IndexSampler(
            self.data.values,
            self.lags,
            self.horizon,
            SamplerConfig(**config),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        individuals, date = self.index_sampler(idx)
        total = self.lags + self.horizon
        window = self.data.values[individuals, :, date : date + total]
        individual_context = None
        if (
            self.index_sampler.config.use_individual_context
            and self.data.individual_context is not None
        ):
            individual_context = _slice_temporal_context(
                self.data.individual_context[individuals], date, total
            )
        global_context = None
        if (
            self.index_sampler.config.use_global_context
            and self.data.global_context is not None
        ):
            global_context = _slice_temporal_context(
                self.data.global_context, date, total
            )
            global_context = global_context.unsqueeze(0).expand(
                len(individuals), -1, -1
            )
        metadata: Dict[str, Any] = {
            "individual_indices": torch.as_tensor(individuals, dtype=torch.long),
            "individual_ids": self.data.individual_ids[individuals].clone(),
            "individual_names": [
                self.data.individual_names[
                    int(self.data.individual_ids[individual].item())
                ]
                for individual in individuals
            ],
            "date_indices": torch.full((len(individuals),), date, dtype=torch.long),
            "date_ids": self.data.date_ids[date].repeat(len(individuals)),
            "window_date_ids": self.data.date_ids[
                date : date + total
            ].unsqueeze(0).expand(len(individuals), -1).clone(),
            "datetimes": [self.data.datetimes[date] for _ in individuals],
            "window_datetimes": [
                self.data.datetimes[date : date + total].copy()
                for _ in individuals
            ],
        }
        if self.data.cluster_ids is not None:
            metadata["cluster_ids"] = self.data.cluster_ids[individuals].clone()
        return {
            "inputs": window[:, :, : self.lags],
            "targets": window[:, :, self.lags :],
            "individual_context": individual_context,
            "global_context": global_context,
            "metadata": metadata,
        }


class AggregatedTimeSeriesDataset(Dataset):
    """Concatenate configured datasets without rebuilding their samplers."""

    def __init__(self, datasets: Sequence[TimeSeriesDataset]):
        if not datasets:
            raise ValueError("cannot aggregate an empty dataset list")
        self.datasets = list(datasets)
        self._ends = np.cumsum([len(dataset) for dataset in self.datasets]).tolist()
        try:
            self.data: Optional[TimeSeriesData] = _combine_data(
                [dataset.data for dataset in self.datasets]
            )
        except ValueError:
            self.data = None
        self.standard_stats: Optional[Mapping[str, float]] = None

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_index = bisect_right(self._ends, idx)
        start = 0 if dataset_index == 0 else self._ends[dataset_index - 1]
        return self.datasets[dataset_index][idx - start]

    @property
    def shape(self) -> Dict[str, Any]:
        return {
            "values": (
                None if self.data is None else tuple(self.data.values.shape)
            ),
            "component_shapes": [
                tuple(dataset.data.values.shape) for dataset in self.datasets
            ],
            "component_samples": [len(dataset) for dataset in self.datasets],
            "samples": len(self),
        }

    def normalize(self, stats: Mapping[str, float]) -> None:
        self.standard_stats = stats
        for dataset in self.datasets:
            dataset.normalize(stats)
        if self.data is not None:
            self.data = _combine_data(
                [dataset.data for dataset in self.datasets]
            )

    def to_dataframes(
        self, variate: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and unroll every sampled window from every component."""
        return dataframes_from_dataset(self, variate=variate)

    def get_df(self, variate: Optional[int] = None) -> pd.DataFrame:
        return self.to_dataframes(variate=variate)["values"]


def collate_fn(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def cat_optional(key: str) -> Optional[torch.Tensor]:
        values = [sample[key] for sample in samples]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"mixed None/non-None values for {key}")
        return torch.cat(values, dim=0)

    metadata = {
        "sample_groups": torch.cat(
            [
                torch.full(
                    (len(sample["metadata"]["individual_ids"]),),
                    sample_index,
                    dtype=torch.long,
                )
                for sample_index, sample in enumerate(samples)
            ]
        ),
        "individual_indices": torch.cat(
            [sample["metadata"]["individual_indices"] for sample in samples]
        ),
        "individual_ids": torch.cat(
            [sample["metadata"]["individual_ids"] for sample in samples]
        ),
        "individual_names": [
            value
            for sample in samples
            for value in sample["metadata"]["individual_names"]
        ],
        "date_indices": torch.cat(
            [sample["metadata"]["date_indices"] for sample in samples]
        ),
        "date_ids": torch.cat(
            [sample["metadata"]["date_ids"] for sample in samples]
        ),
        "window_date_ids": torch.cat(
            [sample["metadata"]["window_date_ids"] for sample in samples]
        ),
        "datetimes": [
            value
            for sample in samples
            for value in sample["metadata"]["datetimes"]
        ],
        "window_datetimes": [
            value
            for sample in samples
            for value in sample["metadata"]["window_datetimes"]
        ],
    }
    if all("cluster_ids" in sample["metadata"] for sample in samples):
        metadata["cluster_ids"] = torch.cat(
            [sample["metadata"]["cluster_ids"] for sample in samples]
        )
    return {
        "inputs": torch.cat([sample["inputs"] for sample in samples], dim=0),
        "targets": torch.cat([sample["targets"] for sample in samples], dim=0),
        "individual_context": cat_optional("individual_context"),
        "global_context": cat_optional("global_context"),
        "metadata": metadata,
    }


def _selected_variates(variates: int, variate: Optional[int]) -> List[int]:
    if variate is None:
        return list(range(variates))
    variate = int(variate)
    if variate < 0 or variate >= variates:
        raise IndexError(f"variate {variate} is out of bounds for {variates} variates")
    return [variate]


def _value_columns(
    individual_ids: Sequence[int],
    individual_names: Mapping[int, str],
    variates: int,
    selected_variates: Sequence[int],
) -> pd.Index:
    ids = [int(value) for value in individual_ids]
    names = [individual_names[individual_id] for individual_id in ids]
    if variates == 1 or len(selected_variates) == 1:
        return pd.Index(names, name="user")
    return pd.MultiIndex.from_tuples(
        [(name, variable) for name in names for variable in selected_variates],
        names=["user", "variable"],
    )


def _repeat_static_context(values: torch.Tensor, dates: int) -> torch.Tensor:
    if values.shape[-1] == 1:
        return values.expand(*values.shape[:-1], dates)
    return values


def dataframes_from_data(
    data: TimeSeriesData,
    variate: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Convert complete data to date-by-user values and separate context frames."""
    selected = _selected_variates(data.variates, variate)
    index = pd.Index(data.datetimes, name="datetime")
    values = data.values[:, selected, :].permute(2, 0, 1)
    columns = _value_columns(
        data.individual_ids.tolist(),
        data.individual_names,
        data.variates,
        selected,
    )
    frames = {
        "values": pd.DataFrame(
            values.reshape(data.dates, -1).detach().cpu().numpy(),
            index=index,
            columns=columns,
        )
    }
    if data.individual_context is not None:
        context = _repeat_static_context(data.individual_context, data.dates)
        for variable in range(context.shape[1]):
            frames[f"individual_context_{variable}"] = pd.DataFrame(
                context[:, variable, :].T.detach().cpu().numpy(),
                index=index,
                columns=pd.Index(
                    [
                        data.individual_names[int(value)]
                        for value in data.individual_ids.tolist()
                    ],
                    name="user",
                ),
            )
    if data.global_context is not None:
        context = _repeat_static_context(data.global_context, data.dates)
        for variable in range(context.shape[0]):
            frames[f"global_context_{variable}"] = pd.DataFrame(
                context[variable].detach().cpu().numpy(),
                index=index,
                columns=pd.Index(["value"], name="global"),
            )
    return frames


def _unroll_batches(
    batches: Sequence[Dict[str, Any]] | Any,
    variate: Optional[int],
) -> Dict[str, pd.DataFrame]:
    value_frames: List[pd.DataFrame] = []
    individual_frames: Dict[int, List[pd.DataFrame]] = {}
    global_frames: Dict[int, List[pd.DataFrame]] = {}
    window_number = 0

    for batch in batches:
        windows = torch.cat([batch["inputs"], batch["targets"]], dim=-1)
        selected = _selected_variates(windows.shape[1], variate)
        individual_context = batch["individual_context"]
        global_context = batch["global_context"]
        metadata = batch["metadata"]
        sample_groups = metadata.get(
            "sample_groups", torch.arange(windows.shape[0])
        )
        for sample_group in torch.unique_consecutive(sample_groups).tolist():
            rows = torch.nonzero(
                sample_groups == sample_group, as_tuple=False
            ).flatten()
            users = metadata["individual_ids"][rows].detach().cpu().tolist()
            user_names = [
                metadata["individual_names"][int(row)] for row in rows.tolist()
            ]
            name_map = {
                int(user): str(name) for user, name in zip(users, user_names)
            }
            first_row = int(rows[0])
            date_ids = (
                metadata["window_date_ids"][first_row].detach().cpu().tolist()
            )
            datetimes = list(metadata["window_datetimes"][first_row])
            total = windows.shape[-1]
            index = pd.MultiIndex.from_arrays(
                [
                    np.full(total, window_number),
                    np.arange(total),
                    date_ids,
                    datetimes,
                ],
                names=["window", "step", "date_id", "datetime"],
            )
            values = (
                windows[rows][:, selected, :]
                .permute(2, 0, 1)
                .reshape(total, -1)
                .detach()
                .cpu()
                .numpy()
            )
            value_frames.append(
                pd.DataFrame(
                    values,
                    index=index,
                    columns=_value_columns(
                        users,
                        name_map,
                        windows.shape[1],
                        selected,
                    ),
                )
            )
            if individual_context is not None:
                context = _repeat_static_context(individual_context[rows], total)
                for variable in range(context.shape[1]):
                    individual_frames.setdefault(variable, []).append(
                        pd.DataFrame(
                            context[:, variable, :].T.detach().cpu().numpy(),
                            index=index,
                            columns=pd.Index(user_names, name="user"),
                        )
                    )
            if global_context is not None:
                context = _repeat_static_context(
                    global_context[first_row], total
                )
                for variable in range(context.shape[0]):
                    global_frames.setdefault(variable, []).append(
                        pd.DataFrame(
                            context[variable].detach().cpu().numpy(),
                            index=index,
                            columns=pd.Index(["value"], name="global"),
                        )
                    )
            window_number += 1

    if not value_frames:
        raise ValueError("cannot build DataFrames from an empty dataset or loader")
    frames = {"values": pd.concat(value_frames, axis=0, sort=False)}
    for variable, parts in individual_frames.items():
        frames[f"individual_context_{variable}"] = pd.concat(
            parts, axis=0, sort=False
        )
    for variable, parts in global_frames.items():
        frames[f"global_context_{variable}"] = pd.concat(
            parts, axis=0, sort=False
        )
    return frames


def dataframes_from_dataset(
    dataset: Dataset,
    variate: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch exactly ``len(dataset)`` items and unroll all returned windows."""
    return _unroll_batches(
        (collate_fn([dataset[index]]) for index in range(len(dataset))),
        variate,
    )


def dataframes_from_dataloader(
    loader: DataLoader,
    variate: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Iterate every loader batch and unroll all returned windows."""
    return _unroll_batches(iter(loader), variate)


class TimeSeriesDataLoader(DataLoader):
    """DataLoader with DataFrame export methods for complete loader passes."""

    def to_dataframes(
        self, variate: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        return dataframes_from_dataloader(self, variate=variate)

    def get_df(self, variate: Optional[int] = None) -> pd.DataFrame:
        return self.to_dataframes(variate=variate)["values"]


def fetch_csv(
    data_path: str | Path,
    data_name: str,
    global_context_cols: Optional[Sequence[str]] = None,
    drop_users: Optional[str | Sequence[int]] = None,
    rename_cols: Optional[Mapping[str, str]] = None,
    aggr: Optional[str] = None,
    aggr_period: str = "h",
    users_dim: int = 1,
    date_col: Optional[str] = None,
    dates: Optional[Sequence[Any]] = None,
    drop: Optional[str | Sequence[Any]] = None,
    return_metadata: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[Any]] | Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    List[Any],
    Dict[str, Any],
]:
    """Load a CSV into date-by-series values and optional global context.

    ``users_dim=1`` expects series in columns. ``users_dim=0`` expects series
    in rows and transposes the frame. Source series positions are retained as
    stable individual IDs even after columns are dropped.
    """
    if users_dim not in {0, 1}:
        raise ValueError("users_dim must be 0 or 1")
    csv_path = _path(data_path) / f"{data_name}.csv"
    if date_col is not None:
        if users_dim == 0:
            raise ValueError("date_col is only supported when users_dim=1")
        df = pd.read_csv(csv_path)
        if date_col not in df.columns:
            raise KeyError(f"date column {date_col!r} not found")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df = pd.read_csv(
            csv_path,
            index_col=0,
            parse_dates=users_dim == 1,
        )
    if users_dim == 0:
        df = df.T
        try:
            df.index = pd.to_datetime(df.index)
        except (TypeError, ValueError):
            pass
    if dates is not None:
        if len(dates) != len(df):
            raise ValueError(
                f"dates length {len(dates)} does not match dataframe length {len(df)}"
            )
        df.index = pd.to_datetime(dates)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.RangeIndex(len(df))

    if global_context_cols is None:
        values_df, global_context_df = df.copy(), None
    else:
        columns = list(global_context_cols)
        global_context_df = df[columns].copy()
        values_df = df.drop(columns=columns).copy()

    source_columns = list(values_df.columns)
    source_ids = list(range(len(source_columns)))
    source_names = {
        source_id: str(
            rename_cols.get(column, column) if rename_cols is not None else column
        )
        for source_id, column in zip(source_ids, source_columns)
    }
    requested_drop = drop if drop is not None else drop_users
    if requested_drop is not None:
        items = (
            requested_drop.split(";")
            if isinstance(requested_drop, str)
            else list(requested_drop)
        )
        drop_ids = set()
        for item in items:
            if item in source_columns:
                drop_ids.add(source_columns.index(item))
            elif str(item) in source_columns:
                drop_ids.add(source_columns.index(str(item)))
            elif isinstance(item, (int, np.integer)) or str(item).isdigit():
                source_id = int(item)
                if source_id < 0 or source_id >= len(source_columns):
                    raise IndexError(f"series index {source_id} is out of bounds")
                drop_ids.add(source_id)
            else:
                raise KeyError(f"series {item!r} not found")
        keep_ids = [source_id for source_id in source_ids if source_id not in drop_ids]
        values_df = values_df.iloc[:, keep_ids]
        source_ids = keep_ids

    values_df.columns = [f"serie_{source_id}" for source_id in source_ids]

    if aggr is not None:
        if not isinstance(values_df.index, pd.DatetimeIndex):
            raise ValueError(
                "aggregation requires a DatetimeIndex; provide date_col or dates"
            )
        if aggr == "sum":
            reducer = "sum"
        elif aggr in {"mean", "last", "first"}:
            reducer = aggr
        elif aggr == "asfreq":
            reducer = None
        else:
            raise ValueError(
                "aggr must be one of: None, 'sum', 'mean', 'last', 'first', "
                "'asfreq'"
            )
        if reducer is None:
            values_df = values_df.asfreq(aggr_period)
            if global_context_df is not None:
                global_context_df = global_context_df.asfreq(aggr_period)
        else:
            values_df = getattr(values_df.resample(aggr_period), reducer)()
            if global_context_df is not None:
                global_context_df = getattr(
                    global_context_df.resample(aggr_period), reducer
                )()

    result = (values_df, global_context_df, list(values_df.index))
    if not return_metadata:
        return result
    metadata = {
        "individual_ids": source_ids,
        "individual_names": {
            source_id: source_names[source_id] for source_id in source_ids
        },
    }
    return (*result, metadata)


def save_data(
    data: TimeSeriesData,
    path: str | Path,
    prefix: str = "",
    overwrite: bool = True,
) -> Path:
    path = _ensure_dir(path)
    file_prefix = f"{prefix}_" if prefix else ""
    legacy_context = path / f"{file_prefix}context.pt"
    if overwrite and legacy_context.exists():
        legacy_context.unlink()
    files = {
        "values": data.values,
        "datetimes": data.datetimes,
        "individual_context": data.individual_context,
        "global_context": data.global_context,
        "individual_ids": data.individual_ids,
        "cluster_ids": data.cluster_ids,
        "date_ids": data.date_ids,
    }
    for name, value in files.items():
        file = path / f"{file_prefix}{name}.pt"
        if value is None:
            if overwrite and file.exists():
                file.unlink()
        else:
            torch.save(value, file)
    metadata_file = path / f"{file_prefix}dataset_metadata.json"
    _write_json(
        {
            "version": 1,
            "individual_names": {
                str(key): value for key, value in data.individual_names.items()
            },
        },
        metadata_file,
    )
    return path


def build_dataset(
    data_path: str | Path,
    data_name: str,
    global_context_cols: Optional[Sequence[str]] = None,
    drop_users: Optional[str | Sequence[int]] = None,
    build_individual_ids_context: bool = False,
    rename_cols: Optional[Mapping[str, str]] = None,
    aggr: Optional[str] = None,
    aggr_period: str = "h",
    users_dim: int = 1,
    date_col: Optional[str] = None,
    dates: Optional[Sequence[Any]] = None,
    drop: Optional[str | Sequence[Any]] = None,
    prefix: str = "",
    output_path: Optional[str | Path] = None,
) -> TimeSeriesData:
    values_df, global_df, datetimes, metadata = fetch_csv(
        data_path,
        data_name,
        global_context_cols,
        drop_users,
        rename_cols,
        aggr,
        aggr_period,
        users_dim,
        date_col,
        dates,
        drop,
        True,
    )
    values = torch.tensor(values_df.values, dtype=torch.float32).T.unsqueeze(1)
    individual_context = None
    if build_individual_ids_context:
        individual_context = torch.tensor(
            metadata["individual_ids"], dtype=torch.float32
        ).view(values.shape[0], 1, 1)
    global_context = None
    if global_df is not None:
        global_context = torch.tensor(
            global_df.values, dtype=torch.float32
        ).T
    data = TimeSeriesData(
        values,
        datetimes,
        individual_context=individual_context,
        global_context=global_context,
        individual_ids=metadata["individual_ids"],
        individual_names=metadata["individual_names"],
    )
    save_data(data, output_path or data_path, prefix=prefix)
    return data


def load_data(
    path: str | Path,
    prefix: str = "",
    legacy_context_kind: Optional[str] = None,
) -> TimeSeriesData:
    path = _path(path)
    file_prefix = f"{prefix}_" if prefix else ""

    def load_optional(name: str) -> Any:
        file = path / f"{file_prefix}{name}.pt"
        return torch.load(file, weights_only=False) if file.exists() else None

    values = load_optional("values")
    if values is None:
        raise FileNotFoundError(path / f"{file_prefix}values.pt")
    values = _as_values_3d(values)
    datetimes = load_optional("datetimes")
    if datetimes is None:
        datetimes = np.arange(values.shape[-1])
    individual_context = load_optional("individual_context")
    global_context = load_optional("global_context")
    legacy = load_optional("context")
    if legacy is not None and individual_context is None and global_context is None:
        if legacy_context_kind == "individual":
            individual_context = legacy
        elif legacy_context_kind == "global":
            global_context = legacy
        else:
            raise ValueError(
                "legacy context.pt found; pass legacy_context_kind='individual' "
                "or 'global' explicitly"
            )
    metadata = _read_json(
        path / f"{file_prefix}dataset_metadata.json",
        default={},
    )
    return TimeSeriesData(
        values=values,
        datetimes=datetimes,
        individual_context=individual_context,
        global_context=global_context,
        individual_ids=load_optional("individual_ids"),
        cluster_ids=load_optional("cluster_ids"),
        date_ids=load_optional("date_ids"),
        individual_names=metadata.get("individual_names"),
    )


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
    return {name: apply_split(data, spec) for name, spec in specs.items()}


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
) -> List[int]:
    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("subset ratio must be in (0, 1]")
    if subset_mode == "dates":
        size = _valid_start_count(dates, lags, horizon)
    elif subset_mode == "individuals":
        size = individuals
    elif subset_mode == "all":
        size = individuals * _valid_start_count(dates, lags, horizon)
    else:
        raise ValueError(f"unknown subset_mode {subset_mode!r}")
    count = max(1, int(size * ratio))
    rng = np.random.default_rng(None if seed in {None, "None"} else int(seed))
    return rng.choice(size, size=count, replace=False).astype(int).tolist()


def _series_gamma_stats(
    series: np.ndarray,
    lags: int,
    horizon: int,
    remove_cte: bool,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    total = lags + horizon
    if len(series) < total:
        return np.empty(0), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(series, total)
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
    arrays = [
        item.values.detach().cpu().numpy().astype(float)
        for item in items
    ]
    original_finite_count = sum(
        int(np.isfinite(values).sum()) for values in arrays
    )
    alphas: List[np.ndarray] = []
    betas: List[np.ndarray] = []
    per_series_stds: List[np.ndarray] = []
    finite_values: List[np.ndarray] = []
    for values in arrays:
        for individual in range(values.shape[0]):
            for variate in range(values.shape[1]):
                series = values[individual, variate]
                alpha, beta = _series_gamma_stats(
                    series,
                    lags,
                    horizon,
                    remove_cte,
                    eps,
                )
                if alpha.size:
                    alphas.append(alpha)
                    betas.append(beta)
                if remove_cte and len(series) >= lags:
                    lookbacks = np.lib.stride_tricks.sliding_window_view(
                        series,
                        lags,
                    )
                    constant = (
                        np.isfinite(lookbacks).all(axis=1)
                        & (np.std(lookbacks, axis=1, ddof=0) == 0)
                    )
                    keep = np.ones(len(series), dtype=bool)
                    keep[np.arange(lags - 1, len(series))[constant]] = False
                    finite_values.append(series[keep & np.isfinite(series)])
                else:
                    finite_values.append(series[np.isfinite(series)])
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
        date_count = len(range(0, sampler.max_dates, config.stride))
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
    total = dataset.lags + dataset.horizon
    window = (
        dataset.data.values[individual, :, date : date + total]
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
        "lookback_value_mean": average("lookback_value_mean_sum"),
        "lookback_value_std": average("lookback_value_std_sum"),
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
        {"version": 2, "kind": "subset_specs", "groups": {}},
    )
    if payload.get("kind") != "subset_specs":
        raise ValueError(f"{file} is not a subset specification file")
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
        valid_date_count = _valid_start_count(data.dates, lags, horizon)
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
                strided_dates = list(
                    range(
                        0,
                        valid_date_count,
                        stride,
                    )
                )
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
    shape_str = "Splits:\n" + "\n".join(
        f"{key}\t{loader.dataset.shape}" for key, loader in loaders_dict.items()
    )
    batch_str = "\n".join(
        [
            "Batch:",
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
            f" metadata={list(batch['metadata'])}",
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
            seed=cluster_config.get("seed", seed),
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
            "seed": (
                None
                if cluster_config.get("seed", seed) in {None, "None"}
                else int(cluster_config.get("seed", seed))
            ),
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
            seed=cluster_config.get("seed", seed),
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
            "seed": (
                None
                if cluster_config.get("seed", seed) in {None, "None"}
                else int(cluster_config.get("seed", seed))
            ),
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
