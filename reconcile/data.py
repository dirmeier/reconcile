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
    """
    Sample a grouped timeseries from an ARMA

    Returns
    -------
    Tuple
        a tuple where the first element is a matrix of time series measurements
        and the second one is a pd.DataFrame of groups
    """

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
    """
    Sample a hierarchical timeseries from an ARMA

    Returns
    -------
    Tuple
        a tuple where the first element is a matrix of time series measurements
        and the second one is a pd.DataFrame of groups
    """

    def _group_names():
        hierarchy = ["A:10", "A:20", "B:10", "B:20", "B:30"]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    return _sample_timeseries(100, 5), _group_names()
