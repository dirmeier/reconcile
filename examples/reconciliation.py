from reconcile import GroupedTimeseries
from reconcile.data import sample_grouped_timeseries, \
    sample_hierarchical_timeseries


def run():
    timeseries, groups = sample_grouped_timeseries()
    gts = GroupedTimeseries(timeseries, groups)
    print(gts.all_timeseries().shape)

    timeseries, groups = sample_hierarchical_timeseries()
    gts = GroupedTimeseries(timeseries, groups)
    print(gts.all_timeseries())


if __name__ == "__main__":
    run()
