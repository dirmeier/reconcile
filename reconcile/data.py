import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample


def _sample_timeseries(N, D):
    np.random.seed(23)
    arparams = np.array([1.0, -0.1])
    maparams = np.array([2.0, 0.35])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]

    xs = np.arange(N) / N
    xs = xs.reshape((1, 1, N))
    ys = np.vstack([arma_generate_sample(ar, ma, N) for _ in np.arange(D)]).T
    ys = ys.reshape((1, D, N))

    return ys, xs


def sample_grouped_timeseries():
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

    return _sample_timeseries(100, 16), _group_names()


def sample_hierarchical_timeseries():
    def _group_names():
        hierarchy = [
            "A:10:A",
            "A:10:B",
            "A:10:C",
            "A:20:A",
            "A:20:B",
            "B:30:A",
            "B:30:B",
            "B:30:C",
            "B:40:A",
            "B:40:B",
        ]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    return _sample_timeseries(100, 10), _group_names()
