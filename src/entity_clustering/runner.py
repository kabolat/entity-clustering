"""Configuration-driven study execution and lightweight reporting."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from entity_clustering.config import (
    load_base,
    load_evaluation,
    load_method,
    load_study,
    resolved_study,
)
from entity_clustering.data import load_daily_profiles
from entity_clustering.pipeline import (
    EntityClusterer,
    cluster_gamma,
    evaluate_labels,
    labels_are_evaluable,
    physics_labels,
    pooled_labels,
)
from entity_clustering.provenance import (
    canonical_json_hash,
    environment_metadata,
    sha256_file,
    utc_run_id,
)

DISPLAY_DISTANCE = {"symmetric_kl": "Symmetric KL Div", "bhattacharyya": "Bhattacharyya"}


def _trial(
    *,
    method: str,
    number_clusters: int,
    number_topics: int = -1,
    wording_granularity: int = -1,
    number_lower_dims: int = -1,
    distance_measure: str = "N/A",
    linkage: str = "N/A",
) -> dict[str, object]:
    return {
        "Number of Clusters": number_clusters,
        "Number of Topics": number_topics,
        "Wording Granularity": wording_granularity,
        "Number of Lower Dims": number_lower_dims,
        "Distance Measure": DISPLAY_DISTANCE.get(distance_measure, distance_measure),
        "Linkage": linkage,
        "Method": method,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(UTC).isoformat()} {message}\n")


def _prepare_run_root(
    study_path: Path,
    run_id: str | None,
    resume: bool,
) -> tuple[Path, dict[str, Any], str]:
    study = load_study(study_path)
    resolved = resolved_study(study_path)
    resolved_hash = canonical_json_hash(resolved)
    chosen_run_id = run_id or utc_run_id()
    root = study.runs_dir / study.id / chosen_run_id
    metadata_path = root / "run_metadata.json"
    if root.exists():
        if not resume:
            raise ValueError(f"run directory already exists: {root}; choose a new --run-id or pass --resume")
        if not metadata_path.is_file():
            raise ValueError(f"cannot resume unrecognised run directory: {root}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("resolved_config_sha256") != resolved_hash:
            raise ValueError("refusing to resume: the resolved configuration hash does not match")
        if metadata.get("status") == "completed":
            return root, resolved, resolved_hash
    else:
        root.mkdir(parents=True)
        (root / "models").mkdir()
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return root, resolved, resolved_hash


def run_study(study_path: Path, *, run_id: str | None = None, resume: bool = False) -> Path:
    """Run every explicitly declared trial and save provenance-complete artifacts."""

    root, resolved, resolved_hash = _prepare_run_root(study_path.resolve(), run_id, resume)
    metadata_path = root / "run_metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return root
    study = load_study(study_path)
    base = load_base(study.base)
    method = load_method(study.method)
    evaluation = load_evaluation(study.evaluation)
    log_path = root / "run.log"
    _log(log_path, f"starting study {study.id}")
    metadata = {
        "status": "running",
        "study_id": study.id,
        "command": list(sys.argv),
        "resolved_config_sha256": resolved_hash,
        "environment": environment_metadata(),
        "input_sha256": {
            "profiles_csv": sha256_file(base.input.profiles_csv),
            "metadata_csv": sha256_file(base.input.metadata_csv),
        },
        "resolved_config": resolved,
    }
    _write_json(metadata_path, metadata)
    data = load_daily_profiles(
        base.input.profiles_csv,
        base.input.metadata_csv,
        metadata_id_template=base.input.metadata_id_template,
        capacity_column=base.input.capacity_column,
    )
    result_frames: list[pd.DataFrame] = []
    assignment_frames: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []

    def record(trial: dict[str, object], labels: np.ndarray) -> None:
        valid, counts = labels_are_evaluable(labels, evaluation)
        if not valid:
            skipped.append({**trial, "Reason": f"cluster sizes {counts} violate minimum {evaluation.min_cluster_size}"})
            return
        assignment_frames.append(pd.DataFrame({**trial, "ID": data.system_ids, "Label": labels}))
        for mode in evaluation.modes:
            result_frames.append(evaluate_labels(data, labels, evaluation=evaluation, mode=mode, trial=trial))

    try:
        if "pooled" in study.baselines:
            record(_trial(method="Pooled", number_clusters=1), pooled_labels(data))
        if "physics" in study.baselines:
            for number_clusters in study.sweep.number_clusters:
                if number_clusters > len(data.system_ids):
                    skipped.append(
                        {
                            **_trial(method="Physics-based", number_clusters=number_clusters),
                            "Reason": "more clusters than systems",
                        }
                    )
                    continue
                record(
                    _trial(method="Physics-based", number_clusters=number_clusters),
                    physics_labels(data, number_clusters, study.seed),
                )

        embedding_settings = product(
            study.sweep.number_topics,
            study.sweep.wording_granularity,
            study.sweep.number_lower_dims,
        )
        for number_topics, wording_granularity, number_lower_dims in embedding_settings:
            model_key = f"topics_{number_topics}_words_{wording_granularity}_dims_{number_lower_dims}"
            _log(log_path, f"fitting {model_key}")
            encoder = EntityClusterer(
                number_topics=number_topics,
                wording_granularity=wording_granularity,
                number_clusters=1,
                number_lower_dims=number_lower_dims,
                distance_measure="bhattacharyya",
                linkage="average",
                random_state=study.seed,
                encoder=method.encoder,
            ).fit(data.values)
            encoder.save(root / "models" / model_key)
            gamma = encoder.result_.gamma  # type: ignore[union-attr]
            for distance_measure, linkage, number_clusters in product(
                study.sweep.distance_measures, study.sweep.linkages, study.sweep.number_clusters
            ):
                trial = _trial(
                    method="Entity",
                    number_clusters=number_clusters,
                    number_topics=number_topics,
                    wording_granularity=wording_granularity,
                    number_lower_dims=number_lower_dims,
                    distance_measure=distance_measure,
                    linkage=linkage,
                )
                if number_clusters > len(data.system_ids):
                    skipped.append({**trial, "Reason": "more clusters than systems"})
                    continue
                labels, _ = cluster_gamma(gamma, number_clusters, distance_measure, linkage)
                record(trial, labels)
        if not result_frames:
            raise ValueError("no evaluable trials were produced; check cluster counts and minimum cluster size")
        pd.concat(result_frames, ignore_index=True).to_csv(root / "results.csv", index=False)
        pd.concat(assignment_frames, ignore_index=True).to_csv(root / "assignments.csv", index=False)
        pd.DataFrame(skipped).to_csv(root / "skipped_trials.csv", index=False)
        metadata["status"] = "completed"
        metadata["output_files"] = ["results.csv", "assignments.csv", "skipped_trials.csv"]
        _write_json(metadata_path, metadata)
        _log(log_path, "completed")
        return root
    except Exception:
        metadata["status"] = "failed"
        _write_json(metadata_path, metadata)
        _log(log_path, "failed")
        raise


def report_run(run_root: Path) -> Path:
    """Create a compact score summary and figure from a completed run."""

    results_path = run_root / "results.csv"
    if not results_path.is_file():
        raise ValueError(f"completed results are required: {results_path}")
    results = pd.read_csv(results_path)
    summary = (
        results.groupby(["Evaluation", "Method", "Number of Clusters"], as_index=False)["Value"]
        .median()
        .rename(columns={"Value": "Median Quantile Loss"})
    )
    report_dir = run_root / "report"
    report_dir.mkdir(exist_ok=True)
    summary.to_csv(report_dir / "score_summary.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    for (evaluation, method), subset in summary.groupby(["Evaluation", "Method"]):
        axis.plot(
            subset["Number of Clusters"],
            subset["Median Quantile Loss"],
            marker="o",
            label=f"{method} ({evaluation})",
        )
    axis.set(xlabel="Number of clusters", ylabel="Median quantile loss")
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(report_dir / "score_summary.png", dpi=160)
    plt.close(figure)
    return report_dir
