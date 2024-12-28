# pylint: skip-file

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp
from jax import random

from examples.reconciliation import NeuralProcessForecaster
from reconcile import ProbabilisticReconciliation
from reconcile.grouping import Grouping


def _sample_hierarchical_timeseries():
    def _group_names():
        hierarchy = ["A:10", "A:20", "B:10", "B:20", "B:30"]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    def _sample_timeseries(N, D):
        xs = np.arange(N) / N
        xs = xs.reshape((1, 1, N))
        ys = np.random.normal(size=(1, D, N))
        return ys, xs

    return _sample_timeseries(100, 5), _group_names()


@pytest.fixture()
def grouping():
    _, groups = _sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    return grouping


@pytest.fixture()
def reconciliator():
    (b, x), groups = _sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    all_timeseries = grouping.all_timeseries(b)
    all_features = jnp.tile(x, [1, all_timeseries.shape[1], 1])

    forecaster = NeuralProcessForecaster()
    forecaster.fit(
        random.PRNGKey(1),
        all_timeseries[:, :, :90],
        all_features[:, :, :90],
        100,
    )

    recon = ProbabilisticReconciliation(grouping, forecaster)
    return (all_timeseries, all_features), recon
