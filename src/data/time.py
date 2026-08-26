"""Prepare portable wide-CSV panels from the TIME dataset collection."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPO_ID = "Real-TSF/TIME-ProcessedCSV"
DEFAULT_SETTINGS = ((336, 48), (504, 168))
DEFAULT_FREQUENCIES = ("15T", "H", "D")
DEFAULT_STRIDE = 512
DEFAULT_MAX_SERIES = 500
DEFAULT_MAX_DATES_PER_SERIES = 10_000
TIMESTAMP_NAMES = {"date", "datetime", "timestamp", "time"}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    dataset: str
    frequency: str


@dataclass
class ParsedFile:
    source: SourceFile
    index: pd.DatetimeIndex
    targets: pd.DataFrame
    dropped_series: list[str]


def parse_setting(value: str) -> tuple[int, int]:
    try:
        lags, horizon = (int(part) for part in value.split(":", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("settings must use L:H, for example 336:48") from error
    if lags <= 0 or horizon <= 0:
        raise argparse.ArgumentTypeError("L and H must be positive")
    return lags, horizon


def _source_parts(relative_path: str) -> tuple[str, str] | None:
    parts = Path(relative_path).parts
    if len(parts) < 3 or Path(relative_path).suffix.lower() != ".csv":
        return None
    return parts[0], parts[1]


def _matches(
    relative_path: str,
    datasets: set[str] | None,
    frequencies: set[str] | None,
) -> bool:
    source = _source_parts(relative_path)
    if source is None:
        return False
    dataset, frequency = source
    return (datasets is None or dataset in datasets) and (
        frequencies is None or frequency.casefold() in frequencies
    )


def _local_sources(
    source_root: Path,
    datasets: set[str] | None,
    frequencies: set[str] | None,
) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in sorted(source_root.rglob("*.csv")):
        relative = path.relative_to(source_root).as_posix()
        if not _matches(relative, datasets, frequencies):
            continue
        dataset, frequency = _source_parts(relative)  # type: ignore[misc]
        sources.append(SourceFile(path, relative, dataset, frequency))
    return sources


def _download_sources(
    repo_id: str,
    revision: str,
    cache_dir: Path | None,
    datasets: set[str] | None,
    frequencies: set[str] | None,
) -> tuple[list[SourceFile], str, int | None]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face download requires the project's huggingface-hub dependency"
        ) from error

    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    selected = [
        sibling
        for sibling in info.siblings
        if _matches(sibling.rfilename, datasets, frequencies)
    ]
    if not selected:
        raise ValueError("no TIME CSV files match the requested dataset/frequency filters")
    selected_paths = sorted(sibling.rfilename for sibling in selected)
    selected_bytes = (
        sum(int(sibling.size) for sibling in selected)
        if all(sibling.size is not None for sibling in selected)
        else None
    )
    resolved_revision = info.sha
    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            allow_patterns=selected_paths,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    sources = []
    for relative in selected_paths:
        dataset, frequency = _source_parts(relative)  # type: ignore[misc]
        sources.append(
            SourceFile(snapshot_root / Path(relative), relative, dataset, frequency)
        )
    return sources, resolved_revision, selected_bytes


def _timestamp_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if str(column).strip().casefold() in TIMESTAMP_NAMES:
            return str(column)
    return str(frame.columns[0])


def _read_source(source: SourceFile) -> ParsedFile:
    frame = pd.read_csv(source.path)
    if frame.shape[1] < 2:
        raise ValueError("requires a timestamp and at least one target column")
    timestamp_col = _timestamp_column(frame)
    timestamps = pd.to_datetime(frame.pop(timestamp_col), errors="raise")
    if timestamps.isna().any():
        raise ValueError("contains missing timestamps")
    frame.index = pd.DatetimeIndex(timestamps)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("contains duplicate timestamps")

    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    finite = np.isfinite(numeric.to_numpy()).all(axis=0)
    dropped = [str(column) for column, keep in zip(numeric.columns, finite) if not keep]
    numeric = numeric.loc[:, finite]
    numeric.columns = [str(column) for column in numeric.columns]
    if numeric.shape[1] == 0:
        raise ValueError("contains no complete finite target series")
    return ParsedFile(source, numeric.index, numeric, dropped)


def _alignment_groups(files: Sequence[ParsedFile]) -> list[list[ParsedFile]]:
    groups: list[list[ParsedFile]] = []
    for parsed in files:
        for group in groups:
            if parsed.index.equals(group[0].index):
                group.append(parsed)
                break
        else:
            groups.append([parsed])
    return groups


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug or "dataset"


def _merge_targets(group: Sequence[ParsedFile]) -> tuple[pd.DataFrame, list[str]]:
    prefix = len(group) > 1
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    dropped: list[str] = []
    for parsed in group:
        renamed: dict[str, str] = {}
        for column in parsed.targets.columns:
            candidate = (
                f"{_safe_name(Path(parsed.source.relative_path).stem)}__{column}"
                if prefix or column in seen
                else column
            )
            base = candidate
            suffix = 2
            while candidate in seen:
                candidate = f"{base}_{suffix}"
                suffix += 1
            renamed[column] = candidate
            seen.add(candidate)
        frames.append(parsed.targets.rename(columns=renamed))
        dropped.extend(
            f"{parsed.source.relative_path}:{column}" for column in parsed.dropped_series
        )
    return pd.concat(frames, axis=1), dropped


def _frequency_metadata(index: pd.DatetimeIndex) -> tuple[str | None, bool]:
    inferred = pd.infer_freq(index) if len(index) >= 3 else None
    differences = np.diff(index.asi8)
    regular = bool(len(differences) == 0 or ((differences > 0).all() and np.all(differences == differences[0])))
    return inferred, regular


def _window_counts(
    timestamps: int,
    series: int,
    settings: Sequence[tuple[int, int]],
    stride: int,
) -> dict[str, dict[str, int | bool]]:
    counts: dict[str, dict[str, int | bool]] = {}
    for lags, horizon in settings:
        available = timestamps - lags - horizon
        eligible = available >= 0
        query_stride_1 = available + 1 if eligible else 0
        query_configured = 1 + available // stride if eligible else 0
        counts[f"{lags}:{horizon}"] = {
            "eligible": eligible,
            "query_dates_stride_1": query_stride_1,
            "series_windows_stride_1": query_stride_1 * series,
            f"query_dates_stride_{stride}": query_configured,
            f"series_windows_stride_{stride}": query_configured * series,
        }
    return counts


def _write_dataset(
    output_root: Path,
    folder_name: str,
    targets: pd.DataFrame,
    metadata: dict[str, object],
    overwrite: bool,
) -> tuple[Path, Path]:
    folder = output_root / folder_name
    csv_path = folder / f"{folder_name}.csv"
    config_path = folder / "config.json"
    if not overwrite and (csv_path.exists() or config_path.exists()):
        raise FileExistsError(f"prepared dataset already exists: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    output = targets.copy()
    output.insert(0, "timestamp", targets.index)
    output.to_csv(csv_path, index=False)
    config = {
        "date_col": "timestamp",
        "target_cols": list(targets.columns),
        "time": metadata,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return csv_path, config_path


def prepare_time_csv(
    output_root: Path,
    settings: Sequence[tuple[int, int]] = DEFAULT_SETTINGS,
    stride: int = DEFAULT_STRIDE,
    datasets: Sequence[str] | None = None,
    frequencies: Sequence[str] | None = DEFAULT_FREQUENCIES,
    max_series: int | None = DEFAULT_MAX_SERIES,
    max_dates_per_series: int | None = DEFAULT_MAX_DATES_PER_SERIES,
    source_root: Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    cache_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not settings:
        raise ValueError("at least one L:H setting is required")
    if max_series is not None and max_series <= 0:
        raise ValueError("max_series must be positive or omitted")
    if max_dates_per_series is not None and max_dates_per_series <= 0:
        raise ValueError("max_dates_per_series must be positive or omitted")
    catalog_path = output_root / "catalog.json"
    if catalog_path.exists() and not overwrite:
        raise FileExistsError(f"TIME catalog already exists: {catalog_path}")
    dataset_filter = set(datasets) if datasets else None
    frequency_filter = (
        None
        if not frequencies or any(value.casefold() == "all" for value in frequencies)
        else {value.casefold() for value in frequencies}
    )

    if source_root is not None:
        sources = _local_sources(source_root, dataset_filter, frequency_filter)
        resolved_revision = f"local:{source_root.resolve()}"
        selected_bytes = sum(source.path.stat().st_size for source in sources)
    else:
        sources, resolved_revision, selected_bytes = _download_sources(
            repo_id, revision, cache_dir, dataset_filter, frequency_filter
        )
    if not sources:
        raise ValueError("no TIME CSV files match the requested dataset/frequency filters")

    parsed_by_key: dict[tuple[str, str], list[ParsedFile]] = {}
    skipped: list[dict[str, object]] = []
    for source in sources:
        try:
            parsed = _read_source(source)
        except (OSError, ValueError, TypeError) as error:
            skipped.append({"source_files": [source.relative_path], "reason": str(error)})
            continue
        parsed_by_key.setdefault((source.dataset, source.frequency), []).append(parsed)

    entries: list[dict[str, object]] = []
    used_folder_names: set[str] = set()
    minimum_length = max(lags + horizon for lags, horizon in settings)
    for (dataset, frequency), parsed_files in sorted(parsed_by_key.items()):
        task_source_files = [parsed.source.relative_path for parsed in parsed_files]
        task_series = sum(parsed.targets.shape[1] for parsed in parsed_files)
        task_max_dates = max(len(parsed.index) for parsed in parsed_files)
        if max_series is not None and task_series > max_series:
            skipped.append(
                {
                    "source_dataset": dataset,
                    "source_frequency": frequency,
                    "source_files": task_source_files,
                    "reason": f"too many series: {task_series} > {max_series}",
                }
            )
            continue
        if max_dates_per_series is not None and task_max_dates > max_dates_per_series:
            skipped.append(
                {
                    "source_dataset": dataset,
                    "source_frequency": frequency,
                    "source_files": task_source_files,
                    "reason": (
                        f"series too long: {task_max_dates} > {max_dates_per_series} dates"
                    ),
                }
            )
            continue
        groups = _alignment_groups(parsed_files)
        base_name = f"{_safe_name(dataset)}_{_safe_name(frequency)}"
        for group_number, group in enumerate(groups, start=1):
            source_files = [parsed.source.relative_path for parsed in group]
            targets, dropped = _merge_targets(group)
            inferred_frequency, regular = _frequency_metadata(targets.index)
            if not regular:
                skipped.append({"source_files": source_files, "reason": "irregular timestamps"})
                continue
            if len(targets) < minimum_length:
                skipped.append(
                    {
                        "source_files": source_files,
                        "reason": f"too short: {len(targets)} < {minimum_length}",
                    }
                )
                continue

            folder_name = base_name if len(groups) == 1 else f"{base_name}_part{group_number:02d}"
            if folder_name in used_folder_names:
                raise ValueError(f"TIME dataset names map to the same output folder: {folder_name}")
            used_folder_names.add(folder_name)
            counts = _window_counts(len(targets), targets.shape[1], settings, stride)
            metadata: dict[str, object] = {
                "repository": repo_id,
                "revision": resolved_revision,
                "source_dataset": dataset,
                "source_frequency": frequency,
                "source_files": source_files,
                "alignment_group": group_number,
                "configured_frequency": frequency,
                "inferred_frequency": inferred_frequency,
                "regular": regular,
                "start": targets.index[0].isoformat(),
                "end": targets.index[-1].isoformat(),
                "num_samples": len(targets),
                "num_timestamps": len(targets),
                "num_series": targets.shape[1],
                "num_values": int(targets.size),
                "dropped_series": dropped,
                "evaluation_samples": counts,
            }
            csv_path, config_path = _write_dataset(
                output_root, folder_name, targets, metadata, overwrite
            )
            entries.append(
                {
                    "name": folder_name,
                    "csv": csv_path.relative_to(output_root).as_posix(),
                    "config": config_path.relative_to(output_root).as_posix(),
                    **metadata,
                }
            )

    catalog: dict[str, object] = {
        "repository": repo_id,
        "revision": resolved_revision,
        "selected_datasets": sorted(dataset_filter) if dataset_filter else "all",
        "selected_frequencies": sorted(frequency_filter) if frequency_filter else "all",
        "selected_source_files": len(sources),
        "selected_download_bytes": selected_bytes,
        "max_series": max_series,
        "max_dates_per_series": max_dates_per_series,
        "settings": [f"{lags}:{horizon}" for lags, horizon in settings],
        "stride": stride,
        "num_datasets": len(entries),
        "num_series": sum(int(entry["num_series"]) for entry in entries),
        "num_timestamps": sum(int(entry["num_timestamps"]) for entry in entries),
        "num_values": sum(int(entry["num_values"]) for entry in entries),
        "datasets": entries,
        "skipped": skipped,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare aligned TIME-ProcessedCSV panels."
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/time"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--frequencies", nargs="+", default=list(DEFAULT_FREQUENCIES))
    parser.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES)
    parser.add_argument(
        "--max-dates-per-series", type=int, default=DEFAULT_MAX_DATES_PER_SERIES
    )
    parser.add_argument(
        "--settings", nargs="+", type=parse_setting, default=list(DEFAULT_SETTINGS)
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    catalog = prepare_time_csv(
        output_root=args.output_root,
        settings=args.settings,
        stride=args.stride,
        datasets=args.datasets,
        frequencies=args.frequencies,
        max_series=args.max_series,
        max_dates_per_series=args.max_dates_per_series,
        source_root=args.source_root,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )
    selected_bytes = catalog["selected_download_bytes"]
    size = "unknown" if selected_bytes is None else f"{int(selected_bytes) / 2**20:.2f} MiB"
    print(
        f"Prepared {catalog['num_datasets']} datasets / {catalog['num_series']} series "
        f"from {catalog['selected_source_files']} CSV files ({size} selected)."
    )


if __name__ == "__main__":
    main()
