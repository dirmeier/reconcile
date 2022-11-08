import abc

from chex import Array, PRNGKey


class Forecaster(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, rng_key: PRNGKey, ys: Array, xs: Array) -> Array:
        pass

    @abc.abstractmethod
    def posterior_predictive(self, rng_key: PRNGKey, xs_test: Array) -> Array:
        pass

    @abc.abstractmethod
    def predictive_posterior_probability(
        self, rng_key: PRNGKey, ys_test: Array, txs_test: Array
    ) -> Array:
        pass
