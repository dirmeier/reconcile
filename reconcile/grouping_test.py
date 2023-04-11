# pylint: skip-file

import chex


def test_grouping_size(grouping):
    chex.assert_equal(grouping.n_groups, 1)


def test_grouping_colnames(grouping):
    for e, f in zip(
        grouping.all_timeseries_column_names(),
        ["Total", "A", "B", "A:10", "A:20", "B:10", "B:20", "B:30"],
    ):
        chex.assert_equal(e, f)


def test_grouping_summing_matrix(grouping):
    chex.assert_axis_dimension(grouping.summing_matrix().toarray(), 0, 8)
    chex.assert_axis_dimension(grouping.summing_matrix().toarray(), 1, 5)
