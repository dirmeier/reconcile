"""Forecasting module."""

import abc

from jax import Array
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfp


class Forecaster(metaclass=abc.ABCMeta):
    """Forecast base class.

    Needs to be inherited for using a custom forecaster
    """

    def __init__(self):
        """Construct a forecaster."""

    @property
    @abc.abstractmethod
    def data(self) -> tuple[Array, Array]:
        """Returns the data set used for training.

        Returns:
            returns a tuple consisting of two chex.Arrays where the first
            element are the time series (Y), and the second element are the
            features (X)
        """

    @abc.abstractmethod
    def fit(self, rng_key: jr.PRNGKey, ys: Array, xs: Array) -> None:
        """Fit the forecaster to data.

        Fit a forecaster for each base and upper time series. Can be implemented
        as global model or by fitting one model per time series.

        Args:
            rng_key: a key for random number generation
            ys: a (1 x P x N)-dimensional array of time series measurements
                where the second axis (P) corresponds to the different time
                series and the last axis (N) are measurements at different time
                points
            xs: a (1 x P x N)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (N) are the time points for which measurements
                are taken
        """

    @abc.abstractmethod
    def posterior_predictive(
        self, rng_key: jr.PRNGKey, xs_test: Array
    ) -> tfp.Distribution:
        """Computes the posterior predictive distribution at some input points.

        Args:
            rng_key: a key for random number generation
            xs_test: a (1 x P x M)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (M) are the time points for which measurements
                are to be predicted. The second axis, P, needs to have as many
                elements as the original training data

        Return:
            returns a TFP Distribution with batch shape (,P) and event
            shape (,M), such that a single sample has shape (P, M) and
            multiple samples have shape (S, P, M)
        """

    @abc.abstractmethod
    def predictive_posterior_probability(
        self, rng_key: jr.PRNGKey, ys_test: Array, xs_test: Array
    ) -> Array:
        """Evaluates the probability of an observation.

        Args:
            rng_key: a key for random number generation
            ys_test: a (1 x P x M)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (M) are the time points for which measurements
                are to be predicted. The second axis, P, needs to have as many
                elements as the original training data
            xs_test: a (1 x P x M)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (M) are the time points for which measurements
                are to be predicted. The second axis, P, needs to have as many
                elements as the original training data

        Returns:
            returns a chex Array of size P with the log predictive probability
            of the data given a fit
        """
