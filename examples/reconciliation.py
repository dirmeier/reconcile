from typing import List

import chex
import distrax
import gpjax as gpx
import jax
import numpy as np
import optax
import pandas as pd
from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random
from jax.config import config
from statsmodels.tsa.arima_process import arma_generate_sample

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping
from reconcile.probabilistic_reconciliation import ProbabilisticReconciliation

config.update("jax_enable_x64", True)


class GPForecaster(Forecaster):
    """Example implementation of a forecaster"""

    def __init__(self):
        self._models: List = []
        self._xs: Array = None
        self._ys: Array = None

    @property
    def data(self):
        """Returns the data"""
        return self._ys, self._xs

    def fit(self, rng_key: PRNGKey, ys: Array, xs: Array, niter=2000):
        """Fit a model to each of the time series"""

        self._xs = xs
        self._ys = ys
        chex.assert_rank([ys, xs], [3, 3])
        chex.assert_equal_shape([ys, xs])

        p = xs.shape[1]
        self._models = [None] * p
        for i in np.arange(p):
            x, y = xs[:, [i], :], ys[:, [i], :]
            # fit a model for each time series
            learned_params, _, D = self._fit_one(rng_key, x, y, niter)
            # save the learned parameters and the original data
            self._models[i] = learned_params, D

    def _fit_one(self, rng_key, x, y, niter):
        # here we use GPs to model the time series
        D = gpx.Dataset(X=x.reshape(-1, 1), y=y.reshape(-1, 1))
        sgpr, q, likelihood = self._model(rng_key, D.n)

        parameter_state = gpx.initialise(sgpr, rng_key)
        negative_elbo = jax.jit(sgpr.elbo(D, negative=True))
        optimiser = optax.adam(learning_rate=5e-3)
        inference_state = gpx.fit(
            objective=negative_elbo,
            parameter_state=parameter_state,
            optax_optim=optimiser,
            num_iters=niter,
        )
        learned_params, training_history = inference_state.unpack()
        return learned_params, training_history, D

    @staticmethod
    def _model(rng_key, n):
        z = random.uniform(rng_key, (20, 1))
        prior = gpx.Prior(mean_function=gpx.Constant(), kernel=gpx.RBF())
        likelihood = gpx.Gaussian(num_datapoints=n)
        posterior = prior * likelihood
        q = gpx.CollapsedVariationalGaussian(
            prior=prior,
            likelihood=likelihood,
            inducing_inputs=z,
        )
        sgpr = gpx.CollapsedVI(posterior=posterior, variational_family=q)
        return sgpr, q, likelihood

    def posterior_predictive(self, rng_key, xs_test: Array):
        """Compute the joint
        posterior predictive distribution of all timeseries at xs_test"""
        chex.assert_rank(xs_test, 3)

        q = xs_test.shape[1]
        means = [None] * q
        covs = [None] * q
        for i in np.arange(q):
            x_test = xs_test[:, [i], :].reshape(-1, 1)
            learned_params, D = self._models[i]
            _, q, likelihood = self._model(rng_key, D.n)
            latent_dist = q(learned_params, D)(x_test)
            predictive_dist = likelihood(learned_params, latent_dist)
            means[i] = predictive_dist.mean()
            cov = jnp.linalg.cholesky(predictive_dist.covariance_matrix)
            covs[i] = cov.reshape((1, *cov.shape))

        # here we stack the means and covariance functions of all
        # GP models we used
        means = jnp.vstack(means)
        covs = jnp.vstack(covs)

        # here we use a single distrax distribution to model the predictive
        # posterior of _all_ models
        posterior_predictive = distrax.MultivariateNormalTri(means, covs)
        return posterior_predictive

    def predictive_posterior_probability(
        self, rng_key: PRNGKey, ys_test: Array, xs_test: Array
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

    def _group_names():
        hierarchy = ["A:10", "A:20", "B:10", "B:20", "B:30"]

        return pd.DataFrame.from_dict({"h1": hierarchy})

    return _sample_timeseries(100, 5), _group_names()


def run():
    (b, x), groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    all_timeseries = grouping.all_timeseries(b)
    all_features = jnp.tile(x, [1, all_timeseries.shape[1], 1])

    forecaster = GPForecaster()
    forecaster.fit(
        random.PRNGKey(1),
        all_timeseries[:, :, :90],
        all_features[:, :, :90],
    )

    recon = ProbabilisticReconciliation(grouping, forecaster)
    # do reconciliation via sampling
    _ = recon.sample_reconciled_posterior_predictive(
        random.PRNGKey(1), all_features, n_iter=100, n_warmup=50
    )
    # do reconciliation via optimization of the energy score
    _ = recon.fit_reconciled_posterior_predictive(
        random.PRNGKey(1), all_features, n_samples=100
    )


if __name__ == "__main__":
    run()
