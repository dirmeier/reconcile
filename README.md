# reconcile

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/reconcile/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/reconcile/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/probabilistic-reconciliation.svg?colorB=black&style=flat)](https://pypi.org/project/probabilistic-reconciliation/)

> Probabilistic reconciliation of time series forecasts

## About

Reconcile implements probabilistic time series forecast reconciliation methods introduced in

1) Zambon, Lorenzo, Dario Azzimonti, and Giorgio Corani. ["Probabilistic reconciliation of forecasts via importance sampling."](https://doi.org/10.48550/arXiv.2210.02286) arXiv preprint arXiv:2210.02286 (2022).
2) Panagiotelis, Anastasios, et al. ["Probabilistic forecast reconciliation: Properties, evaluation and score optimisation."](https://doi.org/10.1016/j.ejor.2022.07.040) European Journal of Operational Research (2022).

The package implements methods to compute summing/aggregation matrices for grouped and hierarchical time series and reconciliation methods for probabilistic forecasts based on sampling and optimization,
and in the near future also some recent forecasting methods, such as proposed in [Benavoli, *et al.* (2021)](https://doi.org/10.1007/978-3-030-91445-5_2) or [Corani *et al.*, (2020)](https://arxiv.org/abs/2009.08102) via [GPJax](https://github.com/JaxGaussianProcesses/GPJax).

## Examples

An example timeseries forecast application using GPs can be found in `examples/reconciliation.py` and a **case study on probabilistic forecast reconciliation of stock index data** can be found [here](https://dirmeier.github.io/etudes/probabilistic_reconciliation.html).

## Installation

Make sure to have a working `JAX` installation. Depending whether you want to use CPU/GPU/TPU,
please follow [these instructions](https://github.com/google/jax#installation).

To install the package from PyPI, call:

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
