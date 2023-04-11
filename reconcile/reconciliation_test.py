# pylint: skip-file

import chex
from jax import random


def test_fit_reconciliation(reconciliator):
    (_, all_features), recon = reconciliator
    fit_recon = recon.fit_reconciled_posterior_predictive(
        random.PRNGKey(1), all_features, n_samples=100
    )
    chex.assert_shape(fit_recon, (100, 5, 100))


def test_sample_reconciliation(reconciliator):
    (_, all_features), recon = reconciliator
    fit_recon = recon.sample_reconciled_posterior_predictive(
        random.PRNGKey(1), all_features, n_warmup=50, n_iter=100
    )
    chex.assert_shape(fit_recon, (50, 4, 5, 100))
