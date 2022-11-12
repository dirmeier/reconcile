import pytest

from examples.data import sample_hierarchical_timeseries
from reconcile.grouping import Grouping


@pytest.fixture()
def grouping():
    _, groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    return grouping
