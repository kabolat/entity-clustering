# Configuration reference

Every YAML file declares a `kind`, has a stable human-readable `id`, and
rejects unknown fields. A file may use `extends: parent.yaml`; maps merge
recursively while lists are always replaced, so a study's complete sweep stays
visible in its own file.

| Kind | Role | Required fields |
| --- | --- | --- |
| `base` | data location and optional raw-source preparation | `input.profiles_csv`, `input.metadata_csv` |
| `method` | LDA fitting controls | `encoder` (defaults supplied) |
| `evaluation` | quantile levels and vanilla/leave-one-out scoring | `quantiles`, `modes`, `min_cluster_size` |
| `study` | complete comparison declaration | `base`, `method`, `evaluation`, `sweep` |

`study.sweep` declares lists for clusters, topics, wording granularity, lower
dimensions, distance measures, and linkage criteria. Their Cartesian product
is intentional and saved in the resolved configuration. Invalid settings are
recorded in `skipped_trials.csv` with the cluster-size reason.

For a raw source, `base.source` declares `record_url`, `expected_md5`,
`raw_csv`, optional `timestamp_column`, `chunksize`, and
`output_profiles_csv`. The reader verifies the source file before processing
and requires a wide table whose system columns exactly match metadata IDs. This
makes the input mapping explicit rather than silently relying on CSV column
position.
