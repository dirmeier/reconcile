import numpy as np
from scipy import sparse
import pandas as pd
from jax import numpy as jnp


class GroupedTimeseries:
    def __init__(self, y: jnp.array, groups: pd.DataFrame):
        assert y.shape[1] == groups.shape[0]
        self._y = y
        self._p = y.shape[1]
        self._groups = groups
        self._group_names = list(groups.columns)

        gmat = self._create_g_mat()
        gmat = self._gmat_as_integer(gmat)
        self._s_matrix = self._smatrix(gmat)

    def bottom_timeseries(self):
        return self._y

    def all_timeseries(self):
        return self._y @ self._s_matrix.T

    def summing_matrix(self):
        return self._s_matrix

    def reconcile(self, predictions):
        pass

    def _create_g_mat(self):
        """
        Compute the 'G Matrix'. This is a direct transpilation of the method
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
            token[i].append(
                temp_tokens[
                    cs[i],
                ]
            )
            if sub_len[i + 1] >= 2:
                for j in range(1, sub_len[i + 1]):
                    col = np.array(
                        [
                            ":".join([e, k])
                            for e, k in zip(
                                token[i][j - 1],
                                temp_tokens[
                                    cs[i] + j,
                                ],
                            )
                        ]
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
    def _paste0(a, b):
        return np.array([":".join([e, k]) for e, k in zip(a, b)])

    @staticmethod
    def _gmat_as_integer(gmat):
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

    @staticmethod
    def _smatrix(gmat):
        mats = [None] * gmat.shape[0]
        for i in range(gmat.shape[0]):
            ia = gmat[
                i,
            ]
            ra = np.ones(gmat.shape[1])
            ja = np.arange(gmat.shape[1])
            m = sparse.csr_matrix((ra, (ia, ja)))
            mats[i] = m

        from scipy.sparse import vstack

        mat = vstack(mats)

        return mat
