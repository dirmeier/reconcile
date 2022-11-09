import logging

import blackjax
import jax
import optax
from chex import Array, PRNGKey
from flax import linen as nn
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping

logger = logging.getLogger(__name__)


class ProbabilisticReconciliation:
    def __init__(self, grouping: Grouping, forecaster: Forecaster):
        self._forecaster = forecaster
        self._grouping = grouping

    def sample_reconciled_posterior_predictive(
        self,
        rng_key: PRNGKey,
        xs_test: Array,
        n_chains=4,
        n_iter=2000,
        n_warmup=1000,
    ):
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

        curr_key, rng_key = random.split(rng_key, 2)
        initial_positions = self._forecaster.posterior_predictive(
            curr_key,
            xs_test,
        ).sample(seed=rng_key, sample_shape=(n_chains,))
        initial_positions = {
            "b": self._grouping.extract_bottom_timeseries(initial_positions)
        }

        (curr_keys), rng_key = random.split(rng_key, n_chains + 1)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            lambda x: _logprob_fn(**x),
            n_warmup,
        )
        initial_states = jax.vmap(
            lambda seed, param: warmup.run(seed, param)[0]
        )(curr_keys, initial_positions)
        warmup_init = {"b": initial_positions["b"][0]}
        _, kernel, _ = warmup.run(rng_key, warmup_init)

        def _inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def _step(states, rng_key):
                keys = jax.random.split(rng_key, n_chains)
                states, infos = jax.vmap(kernel)(keys, states)
                return states, (states, infos)

            curr_keys = jax.random.split(rng_key, num_samples)
            _, (states, infos) = jax.lax.scan(_step, initial_state, curr_keys)
            return states

        states = _inference_loop(rng_key, kernel, initial_states, n_iter)
        b_samples = states.position["b"].block_until_ready()
        b_samples = b_samples[n_warmup:, ...]
        return b_samples

    def fit_reconciled_posterior_predictive(
        self,
        rng_key: PRNGKey,
        xs_test: Array,
        n_iter=2000,
    ):
        def _projection(output_dim):
            class _network(nn.Module):
                @nn.compact
                def __call__(self, x: Array):
                    x = x.swapaxes(-2, -1)
                    x = nn.Dense(features=output_dim)(x)
                    x = x.swapaxes(-2, -1)
                    return x

            return _network()

        def _loss(y: Array, y_reconciled_0: Array, y_reconciled_1: Array):
            y = y.reshape((1, *y.shape))
            y = jnp.tile(y, [y_reconciled_0.shape[0], 1, 1, 1])
            lhs = jnp.linalg.norm(y_reconciled_0 - y, axis=2, keepdims=True)
            rhs = 0.5 * jnp.linalg.norm(
                y_reconciled_0 - y_reconciled_1, axis=2, keepdims=True
            )
            loss = jnp.mean(lhs - rhs, axis=0, keepdims=True)
            loss = jnp.sum(loss, axis=-1)
            return loss

        def _step(state, epoch_key, y_batched, y_predictive_batched):
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
        early_stop = EarlyStopping(min_delta=0.1, patience=5)
        while True:
            sample_key, epoch_key, rng_key = random.split(rng_key, 3)
            y_predictive_batch = predictive.sample(
                seed=sample_key,
                sample_shape=(batch_size, 2),
            )
            state, loss = _step(state, epoch_key, ys, y_predictive_batch)
            logger.info("Loss after batch update %d", loss)
            _, early_stop = early_stop.update(loss)
            if early_stop.should_stop:
                logger.info("Met early stopping criteria, breaking...")
                break

        predictive = self._forecaster.posterior_predictive(rng_key, xs_test)
        y_predictive = predictive.sample(seed=rng_key, sample_shape=(n_iter,))
        b_reconciled = state.apply_fn(state.params, y_predictive)

        return b_reconciled
