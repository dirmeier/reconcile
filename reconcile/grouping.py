"""Grouping module."""

import warnings
from itertools import chain

import numpy as np
import pandas as pd
from jax import numpy as jnp
from scipy import sparse


# pylint: disable=missing-function-docstring
def _as_list(maybe_list):
    return maybe_list if isinstance(maybe_list, list) else [maybe_list]


# pylint: disable=missing-function-docstring,too-many-locals,unnecessary-comprehension  # noqa: E501
class Grouping:
    """Class that represents a grouping/hierarchy of a time series."""

    def __init__(self, groups: pd.DataFrame):
        """Initialize a grouping.

        Args:
            groups: pd.DataFrame
        """
        self._p = groups.shape[0]
        self._groups = groups
        self._group_names = list(groups.columns)

        if len(self._group_names) > 1:
            warnings.warn("Grouped timeseries is poorly tested. Use with care!")
            gmat = self._gts_create_g_mat()
            gmat = self._gts_gmat_as_integer(gmat)
            self._labels = None
        else:
            out_edges_per_level, labels, _ = self._hts_create_nodes()
            gmat = self._hts_create_g_mat(out_edges_per_level)
            labels = [_as_list(labels[key]) for key in sorted(labels.keys())]
            self._labels = list(chain(*labels))
        self._s_matrix = self._smatrix(gmat)
        self._n_all_timeseries = self._s_matrix.shape[0]

    def all_timeseries_column_names(self):
        """Getter for column names of all time series."""
        return self._labels

    def bottom_timeseries_column_names(self):
        """Getter for column names of bottom time series."""
        return self._labels[self.n_upper_timeseries :]

    @property
    def n_groups(self):
        """Getter for number of groups."""
        return self._groups.shape[1]

    @property
    def n_all_timeseries(self):
        """Getter for number of all time series."""
        return self._n_all_timeseries

    @property
    def n_bottom_timeseries(self):
        """Getter for number of bottom time series."""
        return self._p

    @property
    def n_upper_timeseries(self):
        """Getter for number of upper time series."""
        return self.n_all_timeseries - self.n_bottom_timeseries

    def all_timeseries(self, b: jnp.ndarray):
        """Getter for all time series."""
        return jnp.einsum("...ijk,jl->...ilk", b, self._s_matrix.T.toarray())

    def summing_matrix(self):
        """Getter for the summing matrix."""
        return self._s_matrix

    def extract_bottom_timeseries(self, y):
        """Getter for the bottom time series."""
        return y[..., self.n_upper_timeseries :, :]

    def upper_time_series(self, b):
        """Getter for upper time series."""
        y = self.all_timeseries(b)
        return y[..., : self.n_upper_timeseries, :]

    @staticmethod
    def _paste0(a, b):
        return np.array([":".join([e, k]) for e, k in zip(a, b)])

    def _gts_create_g_mat(self):
        """Compute the G Matrix.

        This is a direct transpilation of the method
        'CreateGmat' of the R package 'hts' (version 6.0.2)
        """
        total_len = len(self._group_names)
        sub_len = [0]
        for group_name in self._group_names:
            group = self._groups[group_name].values
            sub_len.append(len(group[0].split(":")))
        cs = np.cumsum(sub_len)

        temp_tokens = [None] * self._p
        for i, r in self._groups.iterrows():
            k = ":".join([g for g in r]).split(":")
            temp_tokens[i] = np.array(k)
        temp_tokens = np.vstack(temp_tokens).T

        token = [None] * total_len
        for i in range(total_len):
            token[i] = []

        for i in range(total_len):
            token[i].append(temp_tokens[cs[i],])
            if sub_len[i + 1] >= 2:
                for j in range(1, sub_len[i + 1]):
                    col = self._paste0(
                        token[i][j - 1],
                        temp_tokens[cs[i] + j,],
                    )
                    token[i].append(col)
            token[i] = np.vstack(token[i])

        cn = np.vstack(np.triu_indices(2, 1))
        groups = []
        for i in range(cn.shape[1]):
            bigroups = [token[cn[:, i][0]], token[cn[:, i][1]]]
            nr1, nr2 = bigroups[0].shape[0], bigroups[1].shape[0]
            tmp_groups = [None] * nr1
            for j in range(nr1):
                tmp_groups[j] = self._paste0(
                    bigroups[0][j, :], bigroups[1][0, :]
                )
                if nr2 >= 2:
                    for k in range(1, nr2):
                        tmp_groups[j] = np.vstack(
                            [
                                tmp_groups[j],
                                self._paste0(
                                    bigroups[0][j, :], bigroups[1][k, :]
                                ),
                            ]
                        )
            groups.append(tmp_groups[0])
            if nr1 >= 2:
                for h in range(1, nr1):
                    groups[i] = np.vstack([groups[i], tmp_groups[h]])

        g_matrix = np.vstack(token + groups)
        indexes = np.unique(g_matrix, axis=0, return_index=True)[1]
        g_matrix = g_matrix[sorted(indexes), :]
        g_matrix = np.vstack([np.repeat("Root", g_matrix.shape[1]), g_matrix])

        return g_matrix

    @staticmethod
    def _gts_gmat_as_integer(gmat):
        gmat = np.vstack(
            [
                pd.factorize(
                    gmat[i, :],
                )[0]
                for i in range(gmat.shape[0])
            ]
        )
        gmat = gmat.astype(np.int32)
        return gmat

    def _hts_create_nodes(self):
        tokens_per_level = {}
        for i, r in self._groups.iterrows():
            els = ":".join([g for g in r]).split(":")
            for j, _ in enumerate(els):
                if j not in tokens_per_level:
                    tokens_per_level[j] = []
                tokens_per_level[j].append(":".join(els[: (j + 1)]))

        unique_tokens_per_level = {
            i: np.unique(tokens) for i, tokens in tokens_per_level.items()
        }

        out_edges_per_level = {1: len(unique_tokens_per_level[0])}
        for i in range(1, len(unique_tokens_per_level.keys())):
            out_edges_per_level[i + 1] = {}
            els = [
                ":".join(el.split(":")[:i]) for el in unique_tokens_per_level[i]
            ]
            values, counts = np.unique(els, return_counts=True)
            for v, c in zip(values, counts):
                out_edges_per_level[i + 1][v] = c

        labels = {0: "Total"}
        for k, v in unique_tokens_per_level.items():
            labels[k + 1] = list(v)

        a = labels[len(labels) - 1]
        b = tokens_per_level[len(tokens_per_level) - 1]
        idxs = [b.index(x) if x in b else None for x in a]

        return out_edges_per_level, labels, idxs

    @staticmethod
    def _hts_create_g_mat(out_edges_per_level):
        n_bottom_levels = sum(
            out_edges_per_level[len(out_edges_per_level)].values()
        )
        n_elements_per_level = {
            k: len(v) if isinstance(v, dict) else 1
            for k, v in out_edges_per_level.items()
        }

        els = np.zeros(((len(out_edges_per_level) + 1), n_bottom_levels))
        els[len(out_edges_per_level)] = np.arange(n_bottom_levels)
        if len(out_edges_per_level) > 1:
            level = out_edges_per_level[len(out_edges_per_level)]
            repcount = (
                list(level.values()) if isinstance(level, dict) else [level]
            )
            for i in np.arange(len(out_edges_per_level) - 1, 0, -1):
                v = np.arange(n_elements_per_level[i + 1])
                els[i] = np.repeat(v, repcount)
                times = out_edges_per_level[i]
                times = (
                    list(times.values()) if isinstance(times, dict) else [times]
                )
                x = pd.DataFrame(
                    {
                        "r": repcount,
                        "g": np.repeat(
                            np.arange(n_elements_per_level[i]), times
                        ),
                    }
                )
                x = x.groupby("g").sum()
                repcount = list(x["r"].values)
        return els.astype(np.int32)

    @staticmethod
    def _smatrix(gmat):
        mats = [None] * gmat.shape[0]
        for i in range(gmat.shape[0]):
            ia = gmat[i, :]
            ra = np.ones(gmat.shape[1])
            ja = np.arange(gmat.shape[1])
            m = sparse.csr_matrix((ra, (ia, ja)))
            mats[i] = m
        mat = sparse.vstack(mats)
        return mat
