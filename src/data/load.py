"""CSV preparation and project-local tensor-cache loading for TimeTensors."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from .io import build_dataset, load_data
from .loaders import fetch_training_data, get_sizes
from pipeline.runtime import (
    batch_size,
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_NAME = "dataset_manifest.json"
MANIFEST_VERSION = 1
SERIALIZED_PT_FIELDS = (
    "values",
    "datetimes",
    "individual_context",
    "global_context",
    "individual_ids",
    "cluster_ids",
    "date_ids",
    "target_date_ids",
)

DATASET_CONFIG_KEYS = {
    "global_context_cols",
    "target_cols",
    "drop_users",
    "build_individual_ids_context",
    "rename_cols",
    "aggr",
    "aggr_period",
    "users_dim",
    "date_col",
    "dates",
    "prefix",
}
PREPARATION_KEYS = (
    "name",
    "global_context_cols",
    "target_cols",
    "drop_users",
    "build_individual_ids_context",
    "rename_cols",
    "aggr",
    "aggr_period",
    "users_dim",
    "date_col",
    "dates",
    "prefix",
)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _dataset_config_path(data_cfg: Mapping[str, Any]) -> tuple[Path | None, bool]:
    config_path = data_cfg.get("config_path")
    if config_path not in {None, ""}:
        path = Path(str(config_path)).expanduser()
        return (path / "config.json" if path.is_dir() else path), True
    raw = data_cfg.get("raw_path")
    if raw in {None, ""}:
        return None, False
    raw_path = Path(str(raw)).expanduser()
    directory = raw_path.parent if raw_path.suffix.lower() == ".csv" else raw_path
    return directory / "config.json", False


def _dataset_config_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: raw[key] for key in DATASET_CONFIG_KEYS if key in raw}
    scoped = raw.get("timetensors")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'timetensors' must be an object")
        if scoped.get("drop_users") is not None:
            options["drop_users"] = _as_list(scoped["drop_users"])
        options.update(
            {
                key: value
                for key, value in scoped.items()
                if key in DATASET_CONFIG_KEYS and key != "drop_users"
            }
        )
    return options


def _merge_dataset_config(
    data_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    explicit_keys = sorted(key for key, value in data_cfg.items() if value is not None)
    path, required = _dataset_config_path(data_cfg)
    loaded: dict[str, Any] = {}
    if path is not None and path.exists():
        if path.suffix.lower() != ".json":
            raise ValueError(f"dataset config must be JSON, got {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"dataset config must contain a JSON object: {path}")
        loaded = _dataset_config_options(raw)
    elif required:
        raise FileNotFoundError(path)

    explicit = {key: value for key, value in data_cfg.items() if value is not None}
    merged = dict(loaded)
    merged.update({key: value for key, value in explicit.items() if key != "drop_users"})
    merged["drop_users"] = (
        _as_list(explicit["drop_users"])
        if "drop_users" in explicit
        else _as_list(loaded.get("drop_users"))
    )
    if path is not None and path.exists():
        merged["config_path"] = str(path.resolve())
    provenance = {
        "selected_path": None if path is None or not path.exists() else str(path.resolve()),
        "applied_keys": sorted(loaded),
        "explicit_keys": explicit_keys,
        "effective_drop_users": merged["drop_users"],
        "effective_target_cols": merged.get("target_cols"),
    }
    LOGGER.info(
        "dataset config path=%s applied_keys=%s explicit_keys=%s",
        provenance["selected_path"],
        provenance["applied_keys"],
        provenance["explicit_keys"],
    )
    return merged, provenance


def _resolve_csv(data_cfg: Mapping[str, Any]) -> tuple[Path, str]:
    raw_value = data_cfg.get("raw_path")
    if raw_value in {None, ""}:
        raise ValueError("data.raw_path must select a CSV file or its directory")
    raw_path = Path(str(raw_value)).expanduser()
    logical_name = str(data_cfg.get("name") or raw_path.stem)
    if raw_path.suffix.lower() == ".csv":
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        return raw_path.resolve(), logical_name
    if not raw_path.is_dir():
        raise FileNotFoundError(raw_path)
    candidates = sorted(raw_path.glob("*.csv"), key=lambda item: item.name.casefold())
    expected = f"{logical_name}.csv".casefold()
    named = [item for item in candidates if item.name.casefold() == expected]
    if len(named) == 1:
        return named[0].resolve(), logical_name
    if len(candidates) == 1:
        LOGGER.info("dataset CSV name differs label=%s path=%s", logical_name, candidates[0])
        return candidates[0].resolve(), logical_name
    raise FileNotFoundError(
        f"cannot select CSV for dataset {logical_name!r} in {raw_path}; "
        f"found {[item.name for item in candidates]}"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def _preparation_config(data_cfg: Mapping[str, Any], logical_name: str) -> dict[str, Any]:
    defaults = {
        "name": logical_name,
        "global_context_cols": None,
        "target_cols": None,
        "drop_users": [],
        "build_individual_ids_context": False,
        "rename_cols": None,
        "aggr": None,
        "aggr_period": "h",
        "users_dim": 1,
        "date_col": None,
        "dates": None,
        "prefix": "",
    }
    return {
        key: _json_value(data_cfg.get(key, defaults[key]))
        for key in PREPARATION_KEYS
    }


def _signature(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepared_directory(data_cfg: Mapping[str, Any], logical_name: str, signature: str) -> Path:
    root_value = data_cfg.get("prepared_root")
    root = Path(str(root_value)).expanduser() if root_value not in {None, ""} else PROJECT_ROOT / "datasets" / "prepared"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", logical_name).strip("_") or "dataset"
    return root.resolve() / slug / signature[:16]


def _source_record(csv_path: Path) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "csv_path": str(csv_path),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"path": item.name, "size_bytes": int(item.stat().st_size)}
        for item in sorted(directory.iterdir())
        if item.is_file() and item.name != MANIFEST_NAME and not item.name.endswith(".tmp")
    ]


def _serialized_pt_shapes(data: Any, prefix: str) -> dict[str, list[int]]:
    file_prefix = f"{prefix}_" if prefix else ""
    shapes: dict[str, list[int]] = {}
    for name in SERIALIZED_PT_FIELDS:
        value = getattr(data, name, None)
        if name == "target_date_ids" and torch.equal(value, data.date_ids):
            value = None
        if value is not None:
            shapes[f"{file_prefix}{name}.pt"] = [int(size) for size in value.shape]
    return shapes


def _pt_files_match_manifest(
    directory: Path,
    manifest: Mapping[str, Any],
) -> bool:
    shapes = manifest.get("pt_shapes")
    artifacts = manifest.get("artifacts")
    if not isinstance(shapes, Mapping) or not isinstance(artifacts, list):
        return False
    records = {
        item.get("path"): item.get("size_bytes")
        for item in artifacts
        if isinstance(item, Mapping)
    }
    for relative in shapes:
        path = directory / str(relative)
        if (
            not path.is_file()
            or path.stat().st_size <= 0
            or records.get(relative) != path.stat().st_size
        ):
            return False
    return True


def _manifest_reusable(
    manifest: Mapping[str, Any] | None,
    *,
    signature: str,
    source: Mapping[str, Any],
    directory: Path,
    values_name: str,
) -> bool:
    return bool(
        manifest
        and manifest.get("version") == MANIFEST_VERSION
        and manifest.get("kind") == "timetensors_prepared_dataset"
        and manifest.get("status") == "complete"
        and manifest.get("signature") == signature
        and manifest.get("source") == source
        and isinstance(manifest.get("pt_shapes"), Mapping)
        and values_name in manifest.get("pt_shapes", {})
        and _pt_files_match_manifest(directory, manifest)
    )


def build_dataset_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build or reuse the exact project-local tensor variant, then make loaders."""
    config = to_plain_config(config)
    data_cfg, dataset_config = _merge_dataset_config(section(config, "data"))
    config = {**config, "data": data_cfg}
    experiment = section(config, "experiment")
    csv_path, logical_name = _resolve_csv(data_cfg)
    preparation = _preparation_config(data_cfg, logical_name)
    signature = _signature(preparation)
    out_path = _prepared_directory(data_cfg, logical_name, signature)
    manifest_path = out_path / MANIFEST_NAME
    prefix = str(preparation["prefix"] or "")
    file_prefix = f"{prefix}_" if prefix else ""
    values_name = f"{file_prefix}values.pt"
    source = _source_record(csv_path)
    manifest = _read_manifest(manifest_path)
    reusable = _manifest_reusable(
        manifest,
        signature=signature,
        source=source,
        directory=out_path,
        values_name=values_name,
    )
    data = None
    force_rebuild = rebuild_dataset(config)
    if reusable and not force_rebuild:
        try:
            data = load_data(
                out_path,
                prefix=prefix,
                legacy_context_kind=data_cfg.get("legacy_context_kind"),
            )
            reusable = _serialized_pt_shapes(data, prefix) == manifest["pt_shapes"]
        except Exception as exc:
            LOGGER.warning("prepared tensor validation failed path=%s error=%s", out_path, exc)
            reusable = False

    if force_rebuild or not reusable:
        LOGGER.info(
            "preparing CSV tensors dataset=%s source=%s output=%s force=%s",
            logical_name,
            csv_path,
            out_path,
            force_rebuild,
        )
        built = build_dataset(
            csv_path.parent,
            csv_path.stem,
            global_context_cols=preparation["global_context_cols"],
            target_cols=preparation["target_cols"],
            drop_users=preparation["drop_users"],
            build_individual_ids_context=bool(preparation["build_individual_ids_context"]),
            rename_cols=preparation["rename_cols"],
            aggr=preparation["aggr"],
            aggr_period=str(preparation["aggr_period"]),
            users_dim=int(preparation["users_dim"]),
            date_col=preparation["date_col"],
            dates=preparation["dates"],
            prefix=prefix,
            output_path=out_path,
        )
        expected_shapes = _serialized_pt_shapes(built, prefix)
        prepared = load_data(
            out_path,
            prefix=prefix,
            legacy_context_kind=data_cfg.get("legacy_context_kind"),
        )
        actual_shapes = _serialized_pt_shapes(prepared, prefix)
        if actual_shapes != expected_shapes or any(
            not (out_path / relative).is_file()
            or (out_path / relative).stat().st_size <= 0
            for relative in expected_shapes
        ):
            raise RuntimeError(
                f"prepared tensor artifacts are incomplete or have unexpected shapes: {out_path}"
            )
        _write_manifest(
            manifest_path,
            {
                "version": MANIFEST_VERSION,
                "kind": "timetensors_prepared_dataset",
                "status": "complete",
                "signature": signature,
                "dataset": logical_name,
                "source": source,
                "preparation": preparation,
                "dataset_config": dataset_config,
                "pt_shapes": expected_shapes,
                "artifacts": _artifact_records(out_path),
            },
        )
        data = prepared
    else:
        LOGGER.info("reusing prepared tensors dataset=%s path=%s", logical_name, out_path)
        assert data is not None

    provenance_path = run_dir(config) / "dataset_config.json"
    provenance_path.write_text(
        json.dumps(dataset_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "data": data,
        "dataset_path": out_path,
        "dataset_manifest_path": manifest_path,
        "dataset_config": dataset_config,
        "dataset_config_path": provenance_path,
    }
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
