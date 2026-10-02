import numpy as np
import pytest

from entity_clustering.metrics import (
    bhattacharyya_matrix,
    cluster_quantile_predictions,
    quantile_loss,
    symmetric_kl_matrix,
)


def test_dirichlet_distances_are_symmetric_and_zero_on_the_diagonal():
    gamma = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    for distance in (symmetric_kl_matrix(gamma), bhattacharyya_matrix(gamma)):
        np.testing.assert_allclose(distance, distance.T)
        np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-12)
        assert np.all(distance >= 0.0)


def test_quantile_loss_and_leave_one_out_predictions():
    profiles = np.array([[[1.0] * 96], [[3.0] * 96]])
    labels = np.array([0, 0])
    predictions = cluster_quantile_predictions(profiles, labels, [0.5], leave_one_out=True)
    np.testing.assert_allclose(predictions[0, 0], profiles[1])
    np.testing.assert_allclose(predictions[0, 1], profiles[0])
    loss, per_quantile = quantile_loss(profiles, predictions, [0.5])
    assert loss == pytest.approx(1.0)
    assert per_quantile.shape == (1, 2, 1)
