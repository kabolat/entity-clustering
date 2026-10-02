import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from entity_clustering.cli import app
from entity_clustering.data import PROFILE_COLUMNS, load_daily_profiles, prepare_daily_profiles


def _metadata(path: Path) -> None:
    pd.DataFrame(
        {
            "ID": ["ID001", "ID002"],
            "estimated_ac_capacity": [1000, 2000],
            "tilt": [30, 40],
            "azimuth": [150, 180],
        }
    ).to_csv(path, sep=";", index=False)


def test_load_example_with_declared_id_template():
    root = Path(__file__).parents[1]
    data = load_daily_profiles(
        root / "data/example/X_daily_15min_example.csv",
        root / "data/example/metadata.csv",
        metadata_id_template="ID{:03d}",
    )
    assert data.values.shape == (2, 1461, 96)
    assert data.system_ids == ("ID001", "ID002")


def test_prepare_daily_profiles_requires_complete_fifteen_minute_bins(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    _metadata(metadata)
    timestamps = pd.date_range("2014-01-01", periods=30, freq="min", tz="UTC")
    raw = pd.DataFrame({"timestamp": timestamps, "ID001": 1.0, "ID002": 2.0})
    raw.loc[17, "ID001"] = np.nan
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    output = tmp_path / "daily.csv"

    prepare_daily_profiles(raw_path, metadata, output)
    prepared = pd.read_csv(output)
    assert list(prepared.columns) == [*PROFILE_COLUMNS, "DATE", "ID"]
    assert prepared.shape == (2, 98)
    first = prepared.loc[prepared["ID"] == "ID001"].iloc[0]
    assert first["X_0"] == 1.0
    assert np.isnan(first["X_17"])
    assert output.with_suffix(".provenance.json").is_file()


def test_prepare_data_cli_verifies_the_declared_source(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    _metadata(metadata)
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2014-01-01", periods=15, freq="min", tz="UTC"),
            "ID001": 1.0,
            "ID002": 2.0,
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    md5 = hashlib.md5(raw_path.read_bytes(), usedforsecurity=False).hexdigest()
    output = tmp_path / "prepared.csv"
    config = tmp_path / "base.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "kind": "base",
                "id": "raw-test",
                "input": {"profiles_csv": str(output), "metadata_csv": str(metadata)},
                "source": {
                    "record_url": "https://example.invalid/record",
                    "expected_md5": md5,
                    "raw_csv": str(raw_path),
                    "output_profiles_csv": str(output),
                },
            }
        )
    )
    result = CliRunner().invoke(app, ["prepare-data", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert output.is_file()
