from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from entity_clustering.cli import app
from entity_clustering.data import PROFILE_COLUMNS
from entity_clustering.runner import report_run, run_study


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for system_id, level in (("ID001", 1.0), ("ID002", 2.0)):
        for date in ("2020-01-01", "2020-01-02", "2020-01-03"):
            rows.append({**dict.fromkeys(PROFILE_COLUMNS, level), "DATE": date, "ID": system_id})
    profiles = tmp_path / "profiles.csv"
    pd.DataFrame(rows).to_csv(profiles, index=False)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "ID": ["ID001", "ID002"],
            "estimated_ac_capacity": [1000, 1000],
            "tilt": [30, 40],
            "azimuth": [150, 180],
        }
    ).to_csv(metadata, sep=";", index=False)
    return profiles, metadata


def _write_study(tmp_path: Path, profiles: Path, metadata: Path, seed: int = 2112) -> Path:
    base = tmp_path / "base.yaml"
    method = tmp_path / "method.yaml"
    evaluation = tmp_path / "evaluation.yaml"
    study = tmp_path / "study.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "kind": "base",
                "id": "synthetic",
                "input": {"profiles_csv": str(profiles), "metadata_csv": str(metadata)},
            }
        )
    )
    method.write_text(
        yaml.safe_dump(
            {
                "kind": "method",
                "id": "small",
                "encoder": {"max_iter": 5, "evaluate_every": 0, "batch_size": 2},
            }
        )
    )
    evaluation.write_text(
        yaml.safe_dump(
            {
                "kind": "evaluation",
                "id": "standard",
                "quantiles": [0.5],
                "modes": ["vanilla", "leave_one_out"],
                "min_cluster_size": 2,
            }
        )
    )
    study.write_text(
        yaml.safe_dump(
            {
                "kind": "study",
                "id": "synthetic",
                "base": str(base),
                "method": str(method),
                "evaluation": str(evaluation),
                "seed": seed,
                "runs_dir": str(tmp_path / "runs"),
                "sweep": {
                    "number_clusters": [1],
                    "number_topics": [2],
                    "wording_granularity": [2],
                    "number_lower_dims": [96],
                    "distance_measures": ["bhattacharyya"],
                    "linkages": ["average"],
                },
            }
        )
    )
    return study


def test_run_report_and_resume_protection(tmp_path: Path):
    profiles, metadata = _write_inputs(tmp_path)
    study = _write_study(tmp_path, profiles, metadata)
    invocation = CliRunner().invoke(app, ["run", "--config", str(study), "--run-id", "case"])
    assert invocation.exit_code == 0, invocation.output
    root = tmp_path / "runs/synthetic/case"
    assert (root / "results.csv").is_file()
    assert (root / "assignments.csv").is_file()
    assert report_run(root).joinpath("score_summary.png").is_file()
    assert run_study(study, run_id="case", resume=True) == root

    changed_study = _write_study(tmp_path, profiles, metadata, seed=7)
    with pytest.raises(ValueError, match="configuration hash"):
        run_study(changed_study, run_id="case", resume=True)
