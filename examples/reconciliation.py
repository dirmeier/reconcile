import chex
import jax
import numpy as np
import pandas as pd
from einops import rearrange
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from ramsey import NP, train_neural_process
from ramsey.nn import MLP
from statsmodels.tsa.arima_process import arma_generate_sample
from tensorflow_probability.substrates.jax import distributions as tfd

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping
from reconcile.probabilistic_reconciliation import ProbabilisticReconciliation


class NeuralProcessForecaster(Forecaster):
    """Example implementation of a forecaster."""

    def __init__(self):
        super().__init__()
        self._models: list = []
        self._xs: jax.Array
        self._ys: jax.Array

    @property
    def data(self) -> tuple[Array, Array]:
        return self._xs, self._ys

    def fit(
        self, rng_key: jr.PRNGKey, ys: jax.Array, xs: jax.Array, niter=2000
    ):
        """Fit a model to each of the time series."""
        self._xs = xs
        self._ys = ys
        chex.assert_rank([ys, xs], [3, 3])
        chex.assert_equal_shape([ys, xs])

        p = xs.shape[1]
        self._models = [None] * p
        for i in np.arange(p):
            x, y = xs[..., i, :], ys[..., i, :]
            # fit a model for each time series
            model, params = self._fit_one(rng_key, x, y, niter)
            # save the learned parameters and the original data
            self._models[i] = model, params

    def _fit_one(self, rng_key, x, y, niter):
        # here we use neural processes to model the time series
        model = self._model()
        n_context, n_target = 10, 20
        params, _ = train_neural_process(
            rng_key,
            model,
            x=x.reshape(1, -1, 1),
            y=y.reshape(1, -1, 1),
            n_context=n_context,
            n_target=n_target,
            n_iter=1000,
            batch_size=1,
        )
        return model, params

    @staticmethod
    def _model():
        def get_neural_process():
            dim = 128
            np = NP(
                decoder=MLP([dim] * 3 + [2]),
                latent_encoder=(MLP([dim] * 3), MLP([dim, dim * 2])),
            )
            return np

        neural_process = get_neural_process()
        return neural_process

    def posterior_predictive(self, rng_key, xs_test: jax.Array):
        """Compute the joint posterior predictive distribution at xs_test."""
        chex.assert_rank(xs_test, 3)

        q = xs_test.shape[1]
        means = [None] * q
        scales = [None] * q
        for i in np.arange(q):
            x_context = self._xs[..., i, :]
            y_context = self._ys[..., i, :]
            x_test = xs_test[..., i, :]

            model, params = self._models[i]
            predictive_dist = model.apply(
                variables=params,
                rngs={"sample": rng_key},
                x_context=x_context.reshape(1, -1, 1),
                y_context=y_context.reshape(1, -1, 1),
                x_target=x_test.reshape(1, -1, 1),
            )
            means[i] = predictive_dist.mean
            scales[i] = predictive_dist.scale

        means = rearrange(jnp.vstack(means), "b t ... -> ... b t")
        scales = rearrange(jnp.vstack(scales), "b t ... -> ... b t")
        # posterior of _all_ models
        posterior_predictive = tfd.MultivariateNormalDiag(means, scales)
        return posterior_predictive

    def predictive_posterior_probability(
        self, rng_key: jr.PRNGKey, ys_test: jax.Array, xs_test: jax.Array
    ):
        """Compute the log predictive posterior probability of an observation"""
        chex.assert_rank([ys_test, xs_test], [3, 3])
        chex.assert_equal_shape([ys_test, xs_test])

        preds = self.posterior_predictive(rng_key, xs_test)
        lp = preds.log_prob(ys_test)
        return lp


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


def sample_hierarchical_timeseries():
    """
    Sample a hierarchical timeseries from an ARMA process

    Returns
    -------
    Tuple
        a tuple where the first element is a matrix of time series measurements
        and the second one is a pd.DataFrame of groups
    """

    def _hierarchy():
        hierarchy = ["A:10", "A:20", "B:10", "B:20", "B:30"]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    return _sample_timeseries(100, 5), _hierarchy()


def run():
    (b, x), groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    all_timeseries = grouping.all_timeseries(b)
    all_features = jnp.tile(x, [1, all_timeseries.shape[1], 1])

    forecaster = NeuralProcessForecaster()
    forecaster.fit(
        jr.PRNGKey(1),
        all_timeseries[:, :, :90],
        all_features[:, :, :90],
    )

    recon = ProbabilisticReconciliation(grouping, forecaster)
    # do reconciliation via sampling
    _ = recon.sample_reconciled_posterior_predictive(
        jr.PRNGKey(1), all_features, n_iter=100, n_warmup=50
    )
    # do reconciliation via optimization of the energy score
    _ = recon.fit_reconciled_posterior_predictive(
        jr.PRNGKey(1), all_features, n_samples=100
    )


if __name__ == "__main__":
    run()
