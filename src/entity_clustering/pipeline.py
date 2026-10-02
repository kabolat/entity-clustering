"""Core probabilistic entity-embedding clustering workflow."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from entity_clustering.config import EncoderDefaults, EvaluationConfig, SweepConfig
from entity_clustering.data import DailyProfiles
from entity_clustering.metrics import (
    bhattacharyya_matrix,
    cluster_quantile_predictions,
    quantile_loss,
    symmetric_kl_matrix,
    valid_cluster_assignment,
)

DistanceMeasure = Literal["symmetric_kl", "bhattacharyya"]
Linkage = Literal["average", "complete"]


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    gamma: np.ndarray
    distances: np.ndarray
    model_config: dict[str, object]


class EntityClusterer:
    """Fit the paper's word--LDA--Dirichlet-distance clustering pipeline."""

    def __init__(
        self,
        *,
        number_topics: int,
        wording_granularity: int,
        number_clusters: int,
        number_lower_dims: int,
        distance_measure: DistanceMeasure,
        linkage: Linkage,
        random_state: int,
        encoder: EncoderDefaults | None = None,
    ) -> None:
        self.number_topics = number_topics
        self.wording_granularity = wording_granularity
        self.number_clusters = number_clusters
        self.number_lower_dims = number_lower_dims
        self.distance_measure = distance_measure
        self.linkage = linkage
        self.random_state = random_state
        self.encoder_config = encoder or EncoderDefaults()
        self.result_: ClusteringResult | None = None

    def _corpus(self, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        system_count, day_count, feature_count = profiles.shape
        flattened = profiles.reshape(-1, feature_count)
        complete = ~np.isnan(flattened).any(axis=1)
        entity_index = np.repeat(np.arange(system_count), day_count)
        available_per_entity = np.bincount(entity_index[complete], minlength=system_count)
        if not complete.any() or (available_per_entity == 0).any():
            raise ValueError("every system must have at least one complete daily profile")
        if self.wording_granularity > int(complete.sum()):
            raise ValueError("wording_granularity cannot exceed the number of complete daily profiles")
        valid = flattened[complete]
        if self.number_lower_dims > feature_count:
            raise ValueError("number_lower_dims cannot exceed the 96 daily profile features")
        self.reducer_: TruncatedSVD | None = None
        if self.number_lower_dims < feature_count:
            max_components = min(valid.shape) - 1
            if self.number_lower_dims > max_components:
                raise ValueError("number_lower_dims is too high for the available complete daily profiles")
            self.reducer_ = TruncatedSVD(n_components=self.number_lower_dims, random_state=self.random_state).fit(valid)
            valid = self.reducer_.transform(valid)
        self.word_model_ = MiniBatchKMeans(
            n_clusters=self.wording_granularity,
            random_state=self.random_state,
            batch_size=min(8192, max(256, len(valid))),
            n_init="auto",
            tol=1e-5,
        ).fit(valid)
        words = self.word_model_.predict(valid)
        corpus = np.zeros((system_count, self.wording_granularity), dtype=float)
        np.add.at(corpus, (entity_index[complete], words), 1.0)
        return corpus, available_per_entity

    def fit(self, profiles: np.ndarray) -> EntityClusterer:
        values = np.asarray(profiles, dtype=float)
        if values.ndim != 3 or values.shape[-1] != 96:
            raise ValueError("profiles must have shape (systems, days, 96)")
        if not 1 <= self.number_clusters <= values.shape[0]:
            raise ValueError("number_clusters must be between 1 and the number of systems")
        corpus, doc_lengths = self._corpus(values)
        self.doc_lengths_ = doc_lengths
        self.lda_ = LatentDirichletAllocation(
            n_components=self.number_topics,
            random_state=self.random_state,
            learning_method=self.encoder_config.learning_method,
            max_iter=self.encoder_config.max_iter,
            batch_size=self.encoder_config.batch_size,
            evaluate_every=self.encoder_config.evaluate_every,
            perp_tol=self.encoder_config.perplexity_tolerance,
            doc_topic_prior=1.0 / self.number_topics,
            topic_word_prior=1.0 / self.wording_granularity,
        ).fit(corpus)
        gamma = self.lda_._unnormalized_transform(corpus)
        labels, distances = cluster_gamma(gamma, self.number_clusters, self.distance_measure, self.linkage)
        self.result_ = ClusteringResult(labels, gamma, distances, self.settings)
        return self

    @property
    def settings(self) -> dict[str, object]:
        return {
            "number_topics": self.number_topics,
            "wording_granularity": self.wording_granularity,
            "number_clusters": self.number_clusters,
            "number_lower_dims": self.number_lower_dims,
            "distance_measure": self.distance_measure,
            "linkage": self.linkage,
            "random_state": self.random_state,
            "encoder": self.encoder_config.model_dump(mode="json"),
        }

    def fit_predict(self, profiles: np.ndarray) -> ClusteringResult:
        return self.fit(profiles).result_  # type: ignore[return-value]

    def save(self, folder: Path) -> None:
        """Persist a fitted local model and its readable configuration."""

        if self.result_ is None:
            raise ValueError("fit the model before saving it")
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "model.pkl").open("wb") as handle:
            pickle.dump(self, handle)
        (folder / "model_config.json").write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(folder: Path) -> EntityClusterer:
        with (folder / "model.pkl").open("rb") as handle:
            return pickle.load(handle)


def trial_settings(sweep: SweepConfig):
    """Yield every explicitly declared hyperparameter combination."""

    return product(
        sweep.number_clusters,
        sweep.number_topics,
        sweep.wording_granularity,
        sweep.number_lower_dims,
        sweep.distance_measures,
        sweep.linkages,
    )


def cluster_gamma(
    gamma: np.ndarray,
    number_clusters: int,
    distance_measure: DistanceMeasure,
    linkage: Linkage,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster already-fitted entity distributions without repeating word/LDA fitting."""

    values = np.asarray(gamma, dtype=float)
    if not 1 <= number_clusters <= len(values):
        raise ValueError("number_clusters must be between 1 and the number of systems")
    distances = symmetric_kl_matrix(values) if distance_measure == "symmetric_kl" else bhattacharyya_matrix(values)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    labels = AgglomerativeClustering(n_clusters=number_clusters, metric="precomputed", linkage=linkage).fit_predict(
        distances
    )
    return labels, distances


def evaluate_labels(
    data: DailyProfiles,
    labels: np.ndarray,
    *,
    evaluation: EvaluationConfig,
    mode: Literal["vanilla", "leave_one_out"],
    trial: dict[str, object],
) -> pd.DataFrame:
    """Evaluate one clustering using the paper's quantile representation."""

    predictions = cluster_quantile_predictions(
        data.values, labels, evaluation.quantiles, leave_one_out=mode == "leave_one_out"
    )
    _, losses = quantile_loss(data.values, predictions, evaluation.quantiles, nonzero=True)
    rows = []
    for quantile_index, quantile in enumerate(evaluation.quantiles):
        for system_index, system_id in enumerate(data.system_ids):
            rows.append(
                {
                    **trial,
                    "Evaluation": mode,
                    "Metric": "Quantile Loss",
                    "Quantile": quantile,
                    "ID": system_id,
                    "Value": float(np.nanmean(losses[quantile_index, system_index])),
                }
            )
    return pd.DataFrame(rows)


def physics_labels(data: DailyProfiles, number_clusters: int, seed: int) -> np.ndarray:
    features = data.metadata.loc[:, ["tilt", "azimuth"]].to_numpy(dtype=float)
    return KMeans(n_clusters=number_clusters, random_state=seed, n_init="auto").fit_predict(features)


def pooled_labels(data: DailyProfiles) -> np.ndarray:
    return np.zeros(len(data.system_ids), dtype=int)


def labels_are_evaluable(labels: np.ndarray, evaluation: EvaluationConfig) -> tuple[bool, list[int]]:
    required_size = evaluation.min_cluster_size
    if "leave_one_out" not in evaluation.modes:
        required_size = 1
    return valid_cluster_assignment(labels, required_size)
