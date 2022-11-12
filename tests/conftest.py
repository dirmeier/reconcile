import pytest
from jax import numpy as jnp
from jax import random

from examples.data import sample_hierarchical_timeseries
from examples.reconciliation import GPForecaster
from reconcile import ProbabilisticReconciliation
from reconcile.grouping import Grouping


@pytest.fixture()
def grouping():
    _, groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    return grouping


@pytest.fixture()
def reconciliator():
    (b, x), groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    all_timeseries = grouping.all_timeseries(b)
    all_features = jnp.tile(x, [1, all_timeseries.shape[1], 1])

    forecaster = GPForecaster()
    forecaster.fit(
        random.PRNGKey(1),
        all_timeseries[:, :90, :],
        all_features[:, :90, :],
        100,
    )

    recon = ProbabilisticReconciliation(grouping, forecaster)
    return (all_timeseries, all_features), recon
