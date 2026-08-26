"""DataFrame exports for time-series data, datasets, and loaders."""

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
    """Convert owned target dates to values and separate context frames."""
    selected = _selected_variates(data.variates, variate)
    target_dates = data.target_date_indices
    index = pd.Index(data.datetimes[target_dates], name="datetime")
    values = data.values[:, selected, :][:, :, target_dates].permute(2, 0, 1)
    columns = _value_columns(
        data.individual_ids.tolist(),
        data.individual_names,
        data.variates,
        selected,
    )
    frames = {
        "values": pd.DataFrame(
            values.reshape(len(target_dates), -1).detach().cpu().numpy(),
            index=index,
            columns=columns,
        )
    }
    if data.individual_context is not None:
        context = _repeat_static_context(
            data.individual_context,
            data.dates,
        )[..., target_dates]
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
        context = _repeat_static_context(
            data.global_context,
            data.dates,
        )[..., target_dates]
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
