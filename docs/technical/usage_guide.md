# Usage guide

Use `uv sync --group dev` once, then run all commands through `uv run`.

`prepare-data` first verifies the declared Zenodo-file MD5, then reads a
timestamp-indexed, wide one-minute CSV whose PV column names match metadata
`ID` values. It detects a timestamp column named
`timestamp`, `time`, `datetime`, or `date`; otherwise it uses the first column.
Set `source.timestamp_column` when automatic detection is unsuitable.

`run` loads a study, resolves its base, method, and evaluation files, validates
the resulting configuration, and writes a new run directory. Supplying an
existing run ID fails unless `--resume` is explicit and the configuration hash
matches. A completed identical run is returned without being overwritten.

`report` only consumes a completed `results.csv`; it writes a median score table
and a compact score-by-cluster figure. It never refits the model.

The daily input schema has one row per `(ID, DATE)`, the 96 columns `X_0` through
`X_95`, and ISO dates. Metadata must provide unique `ID`,
`estimated_ac_capacity`, `tilt`, and `azimuth` fields. The example maps numeric
daily IDs to metadata IDs with `metadata_id_template: "ID{:03d}"`; other inputs
must use identical IDs unless such a template is explicitly configured.
