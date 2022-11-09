from typing import List

import chex
import distrax
import gpjax as gpx
import jax
import numpy as np
import optax
from chex import Array, PRNGKey
from data import sample_hierarchical_timeseries
from jax import numpy as jnp
from jax import random

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping
from reconcile.probabilistic_reconciliation import ProbabilisticReconciliation


class GPForecaster(Forecaster):
    def __init__(self):
        self._models: List = []
        self._xs: Array = None
        self._ys: Array = None

    @property
    def data(self):
        return self._ys, self._xs

    def fit(self, rng_key: PRNGKey, ys: Array, xs: Array) -> None:
        self._xs = xs
        self._ys = ys
        chex.assert_rank([ys, xs], [3, 3])
        chex.assert_equal_shape([ys, xs])

        p = xs.shape[1]
        self._models = [None] * p
        for i in np.arange(p):
            x, y = xs[:, [i], :], ys[:, [i], :]
            learned_params, _, D = self._fit_one(rng_key, x, y)
            self._models[i] = learned_params, D

    def _fit_one(self, rng_key, x, y):
        D = gpx.Dataset(X=x.reshape(-1, 1), y=y.reshape(-1, 1))
        sgpr, q, likelihood = self._model(rng_key, D.n)

        parameter_state = gpx.initialise(sgpr, rng_key)
        negative_elbo = jax.jit(sgpr.elbo(D, negative=True))
        optimiser = optax.adam(learning_rate=5e-3)
        inference_state = gpx.fit(
            objective=negative_elbo,
            parameter_state=parameter_state,
            optax_optim=optimiser,
            n_iters=20,
        )
        learned_params, training_history = inference_state.unpack()
        return learned_params, training_history, D

    @staticmethod
    def _model(rng_key, n):
        z = random.uniform(rng_key, (20, 1))
        prior = gpx.Prior(kernel=gpx.RBF())
        likelihood = gpx.Gaussian(num_datapoints=n)
        posterior = prior * likelihood
        q = gpx.CollapsedVariationalGaussian(
            prior=prior,
            likelihood=likelihood,
            inducing_inputs=z,
        )
        sgpr = gpx.CollapsedVI(posterior=posterior, variational_family=q)
        return sgpr, q, likelihood

    def posterior_predictive(self, rng_key, xs_test: Array) -> Array:
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
        means = jnp.vstack(means)
        covs = jnp.vstack(covs)
        posterior_predictive = distrax.MultivariateNormalTri(means, covs)
        return posterior_predictive

    def predictive_posterior_probability(
        self, rng_key: PRNGKey, ys_test: Array, xs_test: Array
    ) -> Array:
        chex.assert_rank([ys_test, xs_test], [3, 3])
        chex.assert_equal_shape([ys_test, xs_test])

        preds = self.posterior_predictive(rng_key, xs_test)
        y_test_pred = jnp.zeros(ys_test.shape[1])
        for i, pred in enumerate(preds):
            lp = preds[i].log_prob(jnp.squeeze(ys_test[:, i, :]))
            y_test_pred.at[i].set(lp)
        return jnp.asarray(y_test_pred)


def run():
    (b, x), groups = sample_hierarchical_timeseries()
    grouping = Grouping(groups)
    all_timeseries = grouping.all_timeseries(b)
    all_features = jnp.tile(x, [1, all_timeseries.shape[1], 1])

    forecaster = GPForecaster()
    forecaster.fit(random.PRNGKey(1), all_timeseries, all_features)
    forecaster.posterior_predictive(random.PRNGKey(1), all_features)
    forecaster.predictive_posterior_probability(
        random.PRNGKey(1), all_timeseries, all_features
    )

    recon = ProbabilisticReconciliation(grouping, forecaster)
    recon.sample_reconciled_posterior_predictive(
        random.PRNGKey(1), all_features
    )
    recon.fit_reconciled_posterior_predictive(random.PRNGKey(1), all_features)


if __name__ == "__main__":
    run()
