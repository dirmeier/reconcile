import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from reconcile import GroupedTimeseries


def sample_grouped_timeseries():
    N, D = 100, 16

    def _sample_timeseries():
        np.random.seed(23)
        arparams = np.array([1.0, -0.1])
        maparams = np.array([2.0, 0.35])
        ar = np.r_[1, -arparams]
        ma = np.r_[1, maparams]
        ys = np.vstack(
            [arma_generate_sample(ar, ma, N) for _ in np.arange(D)]
        ).T
        return ys

    def _group_names():
        group_one = [
            "VIC:Mel",
            "VIC:Mel",
            "VIC:Gel",
            "VIC:Gel",
            "VIC:Mel",
            "VIC:Mel",
            "VIC:Gel",
            "VIC:Gel",
            "NSW:Syd",
            "NSW:Syd",
            "NSW:Woll",
            "NSW:Woll",
            "NSW:Syd",
            "NSW:Syd",
            "NSW:Woll",
            "NSW:Woll",
        ]
        group_two = [
            "A:A",
            "A:B",
            "A:A",
            "A:B",
            "B:A",
            "B:B",
            "B:A",
            "B:B",
            "A:A",
            "A:B",
            "A:A",
            "A:B",
            "B:A",
            "B:B",
            "B:A",
            "B:B",
        ]
        return pd.DataFrame.from_dict({"g1": group_one, "g2": group_two})

    return _sample_timeseries(), _group_names()


def run():
    timeseries, groups = sample_grouped_timeseries()
    GroupedTimeseries(timeseries, groups)


if __name__ == "__main__":
    run()
