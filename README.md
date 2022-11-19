# reconcile

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/reconcile/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/reconcile/actions/workflows/ci.yaml)
[![codacy badge](https://app.codacy.com/project/badge/Grade/f0a254348e894c7c85b4e979bc81f1d9)](https://www.codacy.com/gh/dirmeier/reconcile/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dirmeier/reconcile&amp;utm_campaign=Badge_Grade)
[![codacy badge](https://app.codacy.com/project/badge/Coverage/f0a254348e894c7c85b4e979bc81f1d9)](https://www.codacy.com/gh/dirmeier/reconcile/dashboard?utm_source=github.com&utm_medium=referral&utm_content=dirmeier/reconcile&utm_campaign=Badge_Coverage)
[![version](https://img.shields.io/pypi/v/probabilistic-reconciliation.svg?colorB=black&style=flat)](https://pypi.org/project/probabilistic-reconciliation/)

> Probabilistic reconciliation of time series forecasts

## About

Reconcile implements probabilistic time series forecast reconciliation methods introduced in

1) Zambon, Lorenzo, Dario Azzimonti, and Giorgio Corani. ["Probabilistic reconciliation of forecasts via importance sampling."](https://doi.org/10.48550/arXiv.2210.02286) arXiv preprint arXiv:2210.02286 (2022).
2) Panagiotelis, Anastasios, et al. ["Probabilistic forecast reconciliation: Properties, evaluation and score optimisation."](https://doi.org/10.1016/j.ejor.2022.07.040) European Journal of Operational Research (2022).

The package implements

- methods to compute summing/aggregation matrices for grouped and hierarchical time series,
- an abstract base forecasting class,
- reconciliation methods for forecasts based on sampling and optimization

An example application can be found in `examples/reconciliation.py`

## Installation


To install from PyPI, call:

```bash
pip install probabilistic-reconciliation
```

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/dirmeier/reconcile@<RELEASE>
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
