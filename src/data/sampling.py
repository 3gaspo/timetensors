"""Window sampling, dataset, collation, and loader primitives."""

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

class IndexSampler:
    """Sample users and cutoff dates ``t`` for ``X=(t-L,t]``, ``Y=(t,t+H]``."""

    VALID_IDX_MODES = {"random", "dates", "individuals", "all"}
    VALID_SUBSET_MODES = {None, "dates", "individuals", "all"}

    def __init__(
        self,
        values: torch.Tensor,
        lags: int,
        horizon: int,
        config: Optional[SamplerConfig] = None,
        target_date_indices: Optional[Sequence[int]] = None,
    ):
        self.values = values
        self.lags = int(lags)
        self.horizon = int(horizon)
        self.config = config or SamplerConfig()
        self.target_date_indices = (
            list(range(self.dates))
            if target_date_indices is None
            else _normalise_indices(target_date_indices, self.dates)
        )
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
        """Backward-compatible count of valid cutoff dates before stride/subsets."""
        return len(self.base_date_candidates)

    @cached_property
    def base_date_candidates(self) -> List[int]:
        return _valid_query_dates(
            self.dates,
            self.lags,
            self.horizon,
            self.target_date_indices,
        )

    @cached_property
    def unfiltered_date_candidates(self) -> List[int]:
        if self.config.subset_indices is not None and self.config.subset_mode == "dates":
            dates = [int(i) for i in self.config.subset_indices]
            valid = set(self.base_date_candidates)
            bad = [i for i in dates if i not in valid]
            if bad:
                raise IndexError(f"date subset contains invalid query dates: {bad[:5]}")
            if len(dates) != len(set(dates)):
                raise ValueError("date subset indices must be unique")
            return dates
        return self.base_date_candidates[:: self.config.stride]

    @cached_property
    def date_candidates(self) -> List[int]:
        candidates = self.unfiltered_date_candidates
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
                if date not in set(self.base_date_candidates):
                    raise IndexError(
                        f"flat pair index {raw} maps to invalid query date {date}"
                    )
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
        if self.lags < 1 or self.horizon < 1:
            raise ValueError("lags and horizon must be positive")
        if cfg.idx_mode not in self.VALID_IDX_MODES:
            raise ValueError(f"unknown idx_mode {cfg.idx_mode!r}")
        if cfg.subset_mode not in self.VALID_SUBSET_MODES:
            raise ValueError(f"unknown subset_mode {cfg.subset_mode!r}")
        if cfg.stride < 1 or cfg.weight < 1:
            raise ValueError("stride and weight must be positive")
        if cfg.block_individuals is not None and cfg.block_individuals < 1:
            raise ValueError("block_individuals must be positive or None")
        if not self.base_date_candidates:
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
            else self.unfiltered_date_candidates
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
        start = int(date) - self.lags + 1
        lookback = self.values[list(individuals), :, start : int(date) + 1]
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
    start: int,
    length: int,
) -> torch.Tensor:
    if context.shape[-1] == 1:
        return context
    return context[..., start : start + length]


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
            self.data.values,
            self.lags,
            self.horizon,
            sampler_config,
            target_date_indices=self.data.target_date_indices,
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
        from .frames import dataframes_from_dataset

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
            target_date_indices=self.data.target_date_indices,
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
            target_date_indices=self.data.target_date_indices,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        individuals, query_t = self.index_sampler(idx)
        total = self.lags + self.horizon
        start = int(query_t) - self.lags + 1
        stop = int(query_t) + self.horizon + 1
        window = self.data.values[individuals, :, start:stop]
        individual_context = None
        if (
            self.index_sampler.config.use_individual_context
            and self.data.individual_context is not None
        ):
            individual_context = _slice_temporal_context(
                self.data.individual_context[individuals], start, total
            )
        global_context = None
        if (
            self.index_sampler.config.use_global_context
            and self.data.global_context is not None
        ):
            global_context = _slice_temporal_context(
                self.data.global_context, start, total
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
            "query_indices": torch.full(
                (len(individuals),),
                query_t,
                dtype=torch.long,
            ),
            "query_ids": self.data.date_ids[query_t].repeat(len(individuals)),
            "date_indices": torch.full(
                (len(individuals),),
                query_t,
                dtype=torch.long,
            ),
            "date_ids": self.data.date_ids[query_t].repeat(len(individuals)),
            "window_date_ids": self.data.date_ids[
                start:stop
            ].unsqueeze(0).expand(len(individuals), -1).clone(),
            "target_date_ids": self.data.date_ids[
                query_t + 1 : stop
            ].unsqueeze(0).expand(len(individuals), -1).clone(),
            "query_datetimes": [
                self.data.datetimes[query_t] for _ in individuals
            ],
            "datetimes": [self.data.datetimes[query_t] for _ in individuals],
            "window_datetimes": [
                self.data.datetimes[start:stop].copy()
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
        from .frames import dataframes_from_dataset

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
        "query_indices": torch.cat(
            [sample["metadata"]["query_indices"] for sample in samples]
        ),
        "query_ids": torch.cat(
            [sample["metadata"]["query_ids"] for sample in samples]
        ),
        "date_indices": torch.cat(
            [sample["metadata"]["date_indices"] for sample in samples]
        ),
        "date_ids": torch.cat(
            [sample["metadata"]["date_ids"] for sample in samples]
        ),
        "window_date_ids": torch.cat(
            [sample["metadata"]["window_date_ids"] for sample in samples]
        ),
        "target_date_ids": torch.cat(
            [sample["metadata"]["target_date_ids"] for sample in samples]
        ),
        "query_datetimes": [
            value
            for sample in samples
            for value in sample["metadata"]["query_datetimes"]
        ],
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
