import blackjax
import jax
from chex import PRNGKey, Array
from jax import numpy as jnp

from reconcile.forecast import Forecaster
from reconcile.grouping import Grouping


class ProbabilisticReconciliation:
    def __init__(self, grouping: Grouping, forecaster: Forecaster):
        self._forecaster = forecaster
        self._grouping = grouping

    def reconciled_posterior_predictive(
        self,
        rng_key: PRNGKey,
        xs_test: Array,
        n_chains=4,
        n_iter=2000,
        n_warmup=1000,
    ):
        def _fn(b):
            if b.ndim == 2:
                b = b.reshape((1, *b.shape))
            u = self._grouping.upper_time_series(b)
            y = jnp.concatenate([u, b], axis=1)
            x = jnp.tile(xs_test, [y.shape[0], 1, 1])
            lp = self._forecaster.predictive_posterior_probability(
                rng_key, y, x
            )
            return jnp.sum(lp)

        initial_positions = self._forecaster.posterior_predictive(
            rng_key,
            xs_test,
        ).sample(seed=rng_key, sample_shape=(4,))
        initial_positions = self._grouping.extract_bottom_timeseries(
            initial_positions
        )

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            _fn,
            10,
        )

        keys = jax.random.split(rng_key, n_chains)
        initial_states = jax.vmap(
            lambda seed, param: warmup.run(seed, param)[0]
        )(keys, initial_positions)
        _, kernel, _ = warmup.run(rng_key, initial_positions[0])

        def _inference_loop(rng_key, kernel, initial_state, num_samples):
            @jax.jit
            def _step(states, rng_key):
                keys = jax.random.split(rng_key, n_chains)
                states, infos = jax.vmap(kernel)(keys, states)
                return states, (states, infos)

            keys = jax.random.split(rng_key, num_samples)
            _, (states, infos) = jax.lax.scan(_step, initial_states, keys)
            return states

        states = _inference_loop(rng_key, kernel, initial_states, 10)
        loc_samples = states.position.block_until_ready()
        print(loc_samples)
