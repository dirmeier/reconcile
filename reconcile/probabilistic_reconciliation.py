"""Probabilistic reconciliation module."""

import logging
from collections.abc import Callable

import blackjax
import jax
import optax
from einops import rearrange
from flax import linen as nn
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random as jr

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping

logger = logging.getLogger(__name__)


# pylint: disable=too-many-arguments,too-many-locals,arguments-differ
class ProbabilisticReconciliation:
    """Probabilistic reconcilation of hierarchical time series class."""

    def __init__(self, grouping: Grouping, forecaster: Forecaster):
        """Construct a ProbabilisticReconciliation object."""
        self._forecaster = forecaster
        self._grouping = grouping

    def sample_reconciled_posterior_predictive(
        self,
        rng_key: jr.PRNGKey,
        xs_test: jax.Array,
        n_chains=4,
        n_iter=2000,
        n_warmup=1000,
    ):
        """Probabilistic reconciliation using Markov Chain Monte Carlo.

        Compute the reconciled bottom time series forecast by sampling from
        the joint density of bottom and upper predictive
        densities. The implementation and method loosely follow [1]_ but
        is not the same method (!).

        Args:
            rng_key: a key for random number generation
            xs_test: a (1 x P x N)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (N) are the time points for which predictions
                are made. The second axis, P, needs to have as many elements as
                the original training data
            n_chains: number of chains to sample from
            n_iter: number of samples to take per chain
            n_warmup: number of samples to discard as burn-in from the chain

        Returns:
            returns a posterior sample of shape (n_iter x n_chains x P x N)
            representing the reconciled bottom time series forecast

        References:
            .. [1] Zambon, Lorenzo, et al. "Probabilistic reconciliation of
                forecasts via importance sampling." arXiv:2210.02286 (2022).
        """

        def _logprob_fn(b):
            if b.ndim == 2:
                b = b.reshape((1, *b.shape))
            u = self._grouping.upper_time_series(b)
            y = jnp.concatenate([u, b], axis=1)
            x = jnp.tile(xs_test, [y.shape[0], 1, 1])
            lp = self._forecaster.predictive_posterior_probability(
                rng_key, y, x
            )
            return jnp.sum(lp)

        def lp(x):
            return _logprob_fn(**x)

        curr_key, rng_key = jr.split(rng_key, 2)
        initial_positions = self._forecaster.posterior_predictive(
            curr_key,
            xs_test,
        ).sample(seed=rng_key, sample_shape=(n_chains,))
        initial_positions = {
            "b": self._grouping.extract_bottom_timeseries(initial_positions)
        }

        init_keys = jr.split(rng_key, n_chains)
        warmup = blackjax.window_adaptation(blackjax.nuts, lp)
        initial_states, kernel_params = jax.vmap(
            lambda seed, param: warmup.run(seed, param)[0]
        )(init_keys, initial_positions)

        kernel_params = {k: v[0] for k, v in kernel_params.items()}
        _, kernel = blackjax.nuts(lp, **kernel_params)

        def _inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def _step(states, rng_key):
                keys = jr.split(rng_key, n_chains)
                states, infos = jax.vmap(kernel)(keys, states)
                return states, (states, infos)

            curr_keys = jr.split(rng_key, num_samples)
            _, (states, _) = jax.lax.scan(_step, initial_state, curr_keys)
            return states

        states = _inference_loop(rng_key, kernel, initial_states, n_iter)
        b_samples = states.position["b"].block_until_ready()
        b_samples = b_samples[n_warmup:, ...]
        b_samples = rearrange(b_samples, "... f b t-> ... (f b) t")
        return b_samples

    def fit_reconciled_posterior_predictive(
        self,
        rng_key: jr.PRNGKey,
        xs_test: jax.Array,
        n_samples=2000,
        net: Callable = None,
        n_iter: int = None,
    ):
        """Probabilistic reconciliation using energy score optimization.

        Compute the reconciled bottom time series forecast by optimization of
        an energy score. The implementation and method loosely follow [1]_ but
        is not the exactly same method.

        Args:
            rng_key: a key for random number generation
            xs_test: a (1 x P x N)-dimensional array of time points where
                the second axis (P) corresponds to the different time series
                and the last axis (N) are the time points for which predictions
                are made. The second axis, P, needs to have as many elements as
                the original training data
            n_samples: number of samples to return
            net: a flax neural network that is used for the projection or None
                to use the linear projection from [1]
            n_iter: number of iterations to train the network or None for
            early stopping

        Returns:
            returns a posterior sample of shape (n_samples x P x N)
            representing the reconciled bottom time series forecast

        References:
            .. [1] Panagiotelis, Anastasios, et al. "Probabilistic forecast
                reconciliation: Properties, evaluation and score
                optimisation." European Journal of Operational Research (2022).
        """

        def _projection(output_dim):
            class _network(nn.Module):
                @nn.compact
                def __call__(self, x: jax.Array):
                    x = x.swapaxes(-2, -1)
                    x = nn.Sequential(
                        [
                            nn.Dense(output_dim * 2),
                            nn.gelu,
                            nn.Dense(output_dim * 2),
                            nn.gelu,
                            nn.Dense(output_dim),
                        ]
                    )(x)
                    x = x.swapaxes(-2, -1)
                    return x

            return _network() if net is None else net()

        def _loss(
            y: jax.Array, y_reconciled_0: jax.Array, y_reconciled_1: jax.Array
        ):
            y = y.reshape((1, *y.shape))
            y = jnp.tile(y, [y_reconciled_0.shape[0], 1, 1, 1])
            lhs = jnp.linalg.norm(y_reconciled_0 - y, axis=2, keepdims=True)
            rhs = 0.5 * jnp.linalg.norm(
                y_reconciled_0 - y_reconciled_1, axis=2, keepdims=True
            )
            loss = jnp.mean(lhs - rhs, axis=0, keepdims=True)
            loss = jnp.sum(loss, axis=-1)
            return loss

        def _step(state, y_batched, y_predictive_batched):
            def loss_fn(params):
                b_reconciled_pred = state.apply_fn(params, y_predictive_batched)
                y_reconciled_pred = self._grouping.all_timeseries(
                    b_reconciled_pred
                )
                loss = _loss(
                    y_batched,
                    y_reconciled_pred[:, [0], :, :],
                    y_reconciled_pred[:, [1], :, :],
                )
                return jnp.sum(loss)

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)

            return new_state, loss

        ys, xs = self._forecaster.data
        predictive = self._forecaster.posterior_predictive(rng_key, xs)

        model = _projection(self._grouping.n_bottom_timeseries)
        params = model.init(rng_key, ys)
        tx = optax.adam(0.001)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        batch_size = 64
        early_stop = EarlyStopping(min_delta=0.1, patience=10)
        itr = 0
        while True:
            sample_key, rng_key = jr.split(rng_key)
            y_predictive_batch = predictive.sample(
                seed=sample_key,
                sample_shape=(batch_size, 2),
            )
            state, loss = _step(state, ys, y_predictive_batch)
            logger.info("Loss after batch update %d", loss)
            early_stop = early_stop.update(loss)
            if early_stop.should_stop and n_iter is None:
                logger.info("Met early stopping criteria, breaking...")
                break
            if n_iter is not None and itr == n_iter:
                break
            itr += 1

        predictive = self._forecaster.posterior_predictive(rng_key, xs_test)
        y_predictive = predictive.sample(
            seed=rng_key, sample_shape=(n_samples,)
        )
        b_reconciled = state.apply_fn(state.params, y_predictive)
        b_reconciled = rearrange(b_reconciled, "... f b t-> ... (f b) t")
        return b_reconciled
