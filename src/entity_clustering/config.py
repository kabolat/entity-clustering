"""Strict YAML configuration for reproducible entity-clustering studies."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class ConfigModel(BaseModel):
    """Configuration base that rejects accidental fields and invalid defaults."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class InputConfig(ConfigModel):
    profiles_csv: Path
    metadata_csv: Path
    metadata_id_template: str | None = None
    capacity_column: str = "estimated_ac_capacity"


class SourceConfig(ConfigModel):
    raw_csv: Path
    record_url: str
    expected_md5: str
    timestamp_column: str | None = None
    chunksize: PositiveInt = 100_000
    output_profiles_csv: Path


class BaseConfig(ConfigModel):
    kind: Literal["base"]
    id: str
    input: InputConfig
    source: SourceConfig | None = None


class EncoderDefaults(ConfigModel):
    max_iter: PositiveInt = 1000
    learning_method: Literal["online", "batch"] = "online"
    batch_size: PositiveInt = 64
    evaluate_every: int = 5
    perplexity_tolerance: float = 0.1


class MethodConfig(ConfigModel):
    kind: Literal["method"]
    id: str
    encoder: EncoderDefaults = Field(default_factory=EncoderDefaults)


class EvaluationConfig(ConfigModel):
    kind: Literal["evaluation"]
    id: str
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
    modes: list[Literal["vanilla", "leave_one_out"]] = Field(default_factory=lambda: ["vanilla", "leave_one_out"])
    min_cluster_size: PositiveInt = 2

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[float]) -> list[float]:
        if not value or value != sorted(set(value)) or any(q <= 0.0 or q >= 1.0 for q in value):
            raise ValueError("quantiles must be sorted, unique values strictly between zero and one")
        return value


class SweepConfig(ConfigModel):
    number_clusters: list[PositiveInt]
    number_topics: list[PositiveInt]
    wording_granularity: list[PositiveInt]
    number_lower_dims: list[PositiveInt]
    distance_measures: list[Literal["symmetric_kl", "bhattacharyya"]]
    linkages: list[Literal["average", "complete"]]

    @field_validator(
        "number_clusters",
        "number_topics",
        "wording_granularity",
        "number_lower_dims",
        "distance_measures",
        "linkages",
    )
    @classmethod
    def nonempty_unique(cls, value: list[Any]) -> list[Any]:
        if not value or len(value) != len(set(value)):
            raise ValueError("every sweep field must be a non-empty list without duplicates")
        return value


class StudyConfig(ConfigModel):
    kind: Literal["study"]
    id: str
    base: Path
    method: Path
    evaluation: Path
    seed: int = 2112
    baselines: list[Literal["pooled", "physics"]] = Field(default_factory=lambda: ["pooled", "physics"])
    sweep: SweepConfig
    runs_dir: Path = Path("runs")


def _deep_merge(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(parent))
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read configuration {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"configuration {path} must contain a YAML mapping")
    parent = raw.get("extends")
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    if parent_path == path.resolve():
        raise ValueError(f"configuration {path} cannot extend itself")
    return _deep_merge(_load_mapping(parent_path), raw)


def _resolve_paths(value: Any, source: Path) -> Any:
    if isinstance(value, Path):
        return value if value.is_absolute() else (source.parent / value).resolve()
    if isinstance(value, dict):
        return {key: _resolve_paths(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, source) for item in value]
    return value


def _load(path: Path, model: type[BaseModel]) -> BaseModel:
    raw = _load_mapping(path.resolve())
    parsed = model.model_validate(raw)
    return model.model_validate(_resolve_paths(parsed.model_dump(mode="python"), path.resolve()))


def load_base(path: Path) -> BaseConfig:
    return _load(path, BaseConfig)  # type: ignore[return-value]


def load_method(path: Path) -> MethodConfig:
    return _load(path, MethodConfig)  # type: ignore[return-value]


def load_evaluation(path: Path) -> EvaluationConfig:
    return _load(path, EvaluationConfig)  # type: ignore[return-value]


def load_study(path: Path) -> StudyConfig:
    return _load(path, StudyConfig)  # type: ignore[return-value]


def resolved_study(path: Path) -> dict[str, Any]:
    """Resolve a study and its referenced YAML documents into one serialisable mapping."""

    study = load_study(path)
    base = load_base(study.base)
    method = load_method(study.method)
    evaluation = load_evaluation(study.evaluation)
    return {
        "study": study.model_dump(mode="json"),
        "base": base.model_dump(mode="json"),
        "method": method.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }
