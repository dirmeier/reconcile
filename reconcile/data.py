import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample


def _sample_timeseries(N, D):
    np.random.seed(23)
    arparams = np.array([1.0, -0.1])
    maparams = np.array([2.0, 0.35])
    ar = np.r_[1, -arparams]
    ma = np.r_[1, maparams]
    ys = np.vstack([arma_generate_sample(ar, ma, N) for _ in np.arange(D)]).T
    return ys


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
            "VIC:Mel:A",
            "VIC:Mel:B",
            "VIC:Gel:C",
            "VIC:Gel:D",
            "VIC:Mel:E",
            "VIC:Mel:F",
            "VIC:Gel:G",
            "VIC:Gel:H",
            "NSW:Syd:I",
            "NSW:Syd:J",
            "NSW:Woll:K",
            "NSW:Woll:L",
            "NSW:Syd:M",
            "NSW:Syd:N",
            "NSW:Woll:O",
            "NSW:Woll:P",
        ]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    return _sample_timeseries(100, 16), _group_names()
