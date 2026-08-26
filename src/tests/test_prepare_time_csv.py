from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data.time import prepare_time_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_local_time_preparation_contract() -> None:
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as directory:
        root = Path(directory)
        source = root / "source" / "Toy" / "H"
        source.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
                "a": np.arange(12, dtype=np.float32),
                "b": np.arange(12, dtype=np.float32) + 1,
            }
        ).to_csv(source / "panel.csv", index=False)
        output = root / "prepared"
        catalog = prepare_time_csv(
            output,
            settings=[(4, 2)],
            stride=2,
            source_root=root / "source",
        )
        assert catalog["num_datasets"] == 1
        entry = catalog["datasets"][0]
        assert entry["configured_frequency"] == "H"
        config = json.loads((output / entry["config"]).read_text(encoding="utf-8"))
        assert config["target_cols"] == ["a", "b"]
        assert config["time"]["evaluation_samples"]["4:2"]["eligible"] is True
