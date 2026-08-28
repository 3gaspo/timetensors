"""Smoke-test signed, project-local prepared dataset variants."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load import build_dataset_stage


def _config(root: Path, raw: Path, prepared: Path, drop_users: list[int]) -> dict:
    return {
        "data": {
            "raw_path": str(raw),
            "prepared_root": str(prepared),
            "name": "tiny",
            "drop_users": drop_users,
            "global_context_cols": ["weather"],
            "build_individual_ids_context": True,
        },
        "experiment": {
            "prepare_loaders": False,
            "rebuild_dataset": False,
        },
        "output": {"dir": str(root / "outputs"), "name": "cache_test"},
    }


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "shared" / "tiny"
        prepared = root / "project" / "datasets" / "prepared"
        raw.mkdir(parents=True)
        pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
                "weather": [10.0, 11.0, 12.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        ).to_csv(raw / "tiny.csv")
        (raw / "config.json").write_text(
            json.dumps(
                {
                    "drop_users": [0],
                    "timetensors": {"drop_users": [1]},
                }
            ),
            encoding="utf-8",
        )

        all_users = build_dataset_stage(_config(root, raw, prepared, []))
        reused = build_dataset_stage(_config(root, raw, prepared, []))
        drop_last = build_dataset_stage(_config(root, raw, prepared, [2]))

        all_path = Path(all_users["dataset_path"])
        dropped_path = Path(drop_last["dataset_path"])
        manifest = json.loads(
            Path(all_users["dataset_manifest_path"]).read_text(encoding="utf-8")
        )
        assert all_path == Path(reused["dataset_path"])
        assert all_path != dropped_path
        assert all_users["data"].values.shape[0] == 3
        assert drop_last["data"].values.shape[0] == 2
        assert manifest["status"] == "complete"
        assert manifest["preparation"]["drop_users"] == []
        assert manifest["source"]["csv_path"] == str((raw / "tiny.csv").resolve())
        assert manifest["pt_shapes"] == {
            "date_ids.pt": [3],
            "datetimes.pt": [3],
            "global_context.pt": [1, 3],
            "individual_context.pt": [3, 1, 1],
            "individual_ids.pt": [3],
            "values.pt": [3, 1, 3],
        }

        (all_path / "datetimes.pt").unlink()
        restored_missing = build_dataset_stage(_config(root, raw, prepared, []))
        assert (all_path / "datetimes.pt").is_file()
        assert str(restored_missing["data"].datetimes[0]).startswith("2024-01-01")

        context_path = all_path / "individual_context.pt"
        expected_bytes = context_path.stat().st_size
        torch.save(torch.zeros(1, 1, 3), context_path)
        assert context_path.stat().st_size == expected_bytes
        restored_shape = build_dataset_stage(_config(root, raw, prepared, []))
        assert restored_shape["data"].individual_context.shape == (3, 1, 1)
        assert not (raw / "values.pt").exists()

    print("test_prepared_dataset_cache: ok")


if __name__ == "__main__":
    main()
