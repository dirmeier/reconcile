import abc
from typing import Tuple

import distrax
from chex import Array, PRNGKey


class Forecaster(metaclass=abc.ABCMeta):
    """
    Forecast base class

    Needs to be inherited for using a custom forecaster
    """

    @property
    @abc.abstractmethod
    def data(self) -> Tuple[Array, Array]:
        """
        Returns the data set used for training

        Returns
        -------
        Tuple
            returns a tuple consisting of two chex.Arrays where the first
            element are the time series (Y), and the second element are the
            features (X)
        """
        pass

    @abc.abstractmethod
    def fit(self, rng_key: PRNGKey, ys: Array, xs: Array) -> None:
        """
        Fit the forecaster to data

        Fit a forecaster for each base and upper time series. Can be implemented
        as global model or by fitting one model per time series

        Parameters
        ----------
        rng_key: chex.PRNGKey
            a key for random number generation
        ys: chex.Array
            a (1 x P x N)-dimensional array of time series measurements where
            the second axis (P) corresponds to the different time series
            and the last axis (N) are measurements at different time points
        xs: chex.Array
            a (1 x P x N)-dimensional array of time points where
            the second axis (P) corresponds to the different time series
            and the last axis (N) are the time points for which measurements
            are taken
        """
        pass

    @abc.abstractmethod
    def posterior_predictive(
        self, rng_key: PRNGKey, xs_test: Array
    ) -> distrax.Distribution:
        """
        Computes the posterior predictive distribution at some input points

        Parameters
        ----------
        rng_key: chex.PRNGKey
            a key for random number generation
        xs_test: chex.Array
            a (1 x P x M)-dimensional array of time points where
            the second axis (P) corresponds to the different time series
            and the last axis (M) are the time points for which measurements
            are to be predicted. The second axis, P, needs to have as many
            elements as the original training data

        Returns
        -------
        distrax.Distribution
            returns a distrax Distribution with batch shape (,P) and event
            shape (,M), such that a single sample has shape (P, M) and
            multiple samples have shape (S, P, M)
        """
        pass

    @abc.abstractmethod
    def predictive_posterior_probability(
        self, rng_key: PRNGKey, ys_test: Array, xs_test: Array
    ) -> Array:
        """
        Evaluates the probability of an observation

        Parameters
        ----------
        rng_key: chex.PRNGKey
            a key for random number generation
        ys_test: chex.Array
            a (1 x P x M)-dimensional array of time points where
            the second axis (P) corresponds to the different time series
            and the last axis (M) are the time points for which measurements
            are to be predicted. The second axis, P, needs to have as many
            elements as the original training data
        xs_test: chex.Array
            a (1 x P x M)-dimensional array of time points where
            the second axis (P) corresponds to the different time series
            and the last axis (M) are the time points for which measurements
            are to be predicted. The second axis, P, needs to have as many
            elements as the original training data

        Returns
        -------
        chex.Array
            returns a chex Array of size P with the log predictive probability
            of the data given a fit
        """
        pass
