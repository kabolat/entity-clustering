"""Input validation and deterministic preparation of daily PV profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from entity_clustering.provenance import sha256_file

PROFILE_COLUMNS = tuple(f"X_{index}" for index in range(96))


@dataclass(frozen=True)
class DailyProfiles:
    """Capacity-normalised daily profiles for an ordered set of PV systems."""

    values: np.ndarray
    system_ids: tuple[str, ...]
    dates: tuple[str, ...]
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if self.values.ndim != 3 or self.values.shape[-1] != len(PROFILE_COLUMNS):
            raise ValueError("daily profiles must have shape (systems, days, 96)")
        if self.values.shape[:2] != (len(self.system_ids), len(self.dates)):
            raise ValueError("profile dimensions must match system_ids and dates")


def _format_ids(values: pd.Series, template: str | None) -> pd.Series:
    if template is None:
        return values.astype(str)
    try:
        return values.map(lambda value: template.format(int(value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"could not apply metadata_id_template {template!r} to profile IDs") from error


def _validate_profile_frame(frame: pd.DataFrame) -> None:
    required = {"ID", "DATE", *PROFILE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    unexpected_profiles = sorted(
        column for column in frame.columns if column.startswith("X_") and column not in PROFILE_COLUMNS
    )
    if missing or unexpected_profiles:
        pieces = []
        if missing:
            pieces.append(f"missing columns: {', '.join(missing)}")
        if unexpected_profiles:
            pieces.append(f"unexpected profile columns: {', '.join(unexpected_profiles)}")
        raise ValueError("invalid daily-profile CSV; " + "; ".join(pieces))
    if frame.duplicated(["ID", "DATE"]).any():
        raise ValueError("daily-profile CSV contains duplicate (ID, DATE) rows")


def load_daily_profiles(
    profiles_csv: Path,
    metadata_csv: Path,
    *,
    metadata_id_template: str | None = None,
    capacity_column: str = "estimated_ac_capacity",
) -> DailyProfiles:
    """Load the documented wide daily-profile CSV and align it to metadata by ID."""

    if not profiles_csv.is_file() or not metadata_csv.is_file():
        raise ValueError("profiles_csv and metadata_csv must both exist")
    profiles_frame = pd.read_csv(profiles_csv)
    if "DATE" not in profiles_frame and "Date" in profiles_frame:
        profiles_frame = profiles_frame.rename(columns={"Date": "DATE"})
    metadata = pd.read_csv(metadata_csv, sep=";")
    _validate_profile_frame(profiles_frame)
    if "ID" not in metadata or capacity_column not in metadata:
        raise ValueError(f"metadata CSV must contain ID and {capacity_column!r}")
    if metadata["ID"].duplicated().any():
        raise ValueError("metadata CSV contains duplicate IDs")

    profiles_frame = profiles_frame.copy()
    profiles_frame["ID"] = _format_ids(profiles_frame["ID"], metadata_id_template)
    profiles_frame["DATE"] = pd.to_datetime(profiles_frame["DATE"], format="%Y-%m-%d", errors="raise")
    metadata = metadata.copy().set_index("ID")
    system_ids = tuple(sorted(profiles_frame["ID"].unique()))
    unknown_ids = sorted(set(system_ids).difference(metadata.index))
    if unknown_ids:
        raise ValueError(f"profile IDs absent from metadata: {', '.join(unknown_ids)}")

    date_sets = {
        system_id: tuple(sorted(profiles_frame.loc[profiles_frame["ID"] == system_id, "DATE"].unique()))
        for system_id in system_ids
    }
    first_dates = next(iter(date_sets.values()), ())
    if not first_dates or any(dates != first_dates for dates in date_sets.values()):
        raise ValueError("every PV system must contain the same ordered set of daily dates")
    dates = tuple(pd.Timestamp(date).date().isoformat() for date in first_dates)

    capacity = pd.to_numeric(metadata.loc[list(system_ids), capacity_column], errors="raise")
    if capacity.isna().any() or (capacity <= 0).any():
        raise ValueError(f"metadata {capacity_column!r} must be finite and strictly positive")

    ordered = profiles_frame.set_index(["ID", "DATE"]).loc[
        pd.MultiIndex.from_product([system_ids, first_dates], names=["ID", "DATE"])
    ]
    raw_values = (
        ordered.loc[:, PROFILE_COLUMNS]
        .to_numpy(dtype=float)
        .copy()
        .reshape(len(system_ids), len(dates), 96)
    )
    raw_values[raw_values < 0] = 0.0
    raw_values[raw_values > 0] = np.maximum(raw_values[raw_values > 0], 1e-3)
    scaled = raw_values * (1000.0 / capacity.to_numpy(dtype=float))[:, None, None]
    return DailyProfiles(scaled, system_ids, dates, metadata.loc[list(system_ids)].reset_index())


def prepare_daily_profiles(
    raw_csv: Path,
    metadata_csv: Path,
    output_profiles_csv: Path,
    *,
    timestamp_column: str | None = None,
    chunksize: int = 100_000,
    expected_md5: str | None = None,
    record_url: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Aggregate a wide one-minute source CSV into the repository's daily CSV schema.

    Values are mean-aggregated in UTC 15-minute bins. A bin is missing for a
    system unless all 15 one-minute measurements are present.
    """

    if output_profiles_csv.exists() and not overwrite:
        raise ValueError(f"output already exists: {output_profiles_csv}; pass --overwrite to replace it")
    source_md5 = _md5_file(raw_csv)
    if expected_md5 is not None and source_md5 != expected_md5:
        raise ValueError("raw data MD5 does not match the source declared by the base configuration")
    metadata = pd.read_csv(metadata_csv, sep=";")
    if "ID" not in metadata or metadata["ID"].duplicated().any():
        raise ValueError("metadata CSV must contain unique ID values")
    system_ids = list(metadata["ID"].astype(str))
    sums: pd.DataFrame | None = None
    counts: pd.DataFrame | None = None

    for chunk in pd.read_csv(raw_csv, chunksize=chunksize):
        timestamp = timestamp_column
        if timestamp is None:
            candidates = [name for name in ("timestamp", "time", "datetime", "date") if name in chunk]
            timestamp = candidates[0] if candidates else str(chunk.columns[0])
        if timestamp not in chunk:
            raise ValueError(f"timestamp column {timestamp!r} is not present in {raw_csv}")
        missing_ids = sorted(set(system_ids).difference(chunk.columns))
        if missing_ids:
            raise ValueError(f"raw data is missing metadata ID columns, e.g. {missing_ids[0]!r}")
        index = pd.to_datetime(chunk[timestamp], utc=True, errors="raise").dt.floor("15min")
        values = chunk.loc[:, system_ids].apply(pd.to_numeric, errors="coerce")
        grouped_sums = values.groupby(index, sort=True).sum(min_count=1)
        grouped_counts = values.groupby(index, sort=True).count()
        sums = grouped_sums if sums is None else sums.add(grouped_sums, fill_value=0.0)
        counts = grouped_counts if counts is None else counts.add(grouped_counts, fill_value=0)

    if sums is None or counts is None:
        raise ValueError(f"raw data CSV is empty: {raw_csv}")
    means = sums.divide(counts).where(counts == 15)
    first_day = means.index.min().normalize()
    last_day = means.index.max().normalize()
    full_index = pd.date_range(first_day, last_day + pd.Timedelta(days=1), freq="15min", tz="UTC")[:-1]
    means = means.reindex(full_index)
    days = pd.date_range(full_index.min().normalize(), full_index.max().normalize(), freq="D", tz="UTC")
    rows: list[pd.DataFrame] = []
    for system_id in system_ids:
        series = means[system_id].to_numpy().reshape(len(days), 96)
        frame = pd.DataFrame(series, columns=PROFILE_COLUMNS)
        frame["DATE"] = days.date.astype(str)
        frame["ID"] = system_id
        rows.append(frame)
    output_profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output_profiles_csv, index=False)
    provenance_path = output_profiles_csv.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(
            {
                "raw_csv": str(raw_csv),
                "raw_sha256": sha256_file(raw_csv),
                "raw_md5": source_md5,
                "record_url": record_url,
                "metadata_csv": str(metadata_csv),
                "metadata_sha256": sha256_file(metadata_csv),
                "aggregation": "UTC 15-minute mean; require all 15 measurements per bin",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_profiles_csv


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
