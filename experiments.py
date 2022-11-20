from abc import abstractmethod
from typing import List, Union, Optional

import numpy as np
import scipy.sparse

from mpyc.runtime import mpc

SparseMatrixListType = List[List[int]]
ScipySparseMatType = scipy.sparse._coo.coo_matrix


class SecureTuple:
    def __init__(self, tup):
        self._tup = tuple(tup)

    def __lt__(self, other):
        if len(self._tup) != len(other._tup):
            raise ValueError("Tuples must be of same size")
        return SecureTuple._lt_tuples(self._tup, other._tup)

    def _lt_tuples(tup1, tup2):
        first_comp = tup1[0] < tup2[0]
        if len(tup1) > 1:
            equal_comp = tup1[0] = tup2[0]
            recursive_comp = SecureTuple._lt_tuples(tup1[1:], tup2[1:])
            return first_comp or (recursive_comp and equal_comp)
        else:
            return first_comp


class SecureMatrix:
    _mat: Optional[SparseMatrixListType] = None

    def __init__(self, sectype=None):
        if sectype is None:
            self.sectype = mpc.SecInt(64)
        else:
            self.sectype = sectype

    @abstractmethod
    def dot(self, other):
        raise NotImplementedError


class SparseMatrixCOO(SecureMatrix):
    def __init__(
        self,
        sparse_mat: Union[ScipySparseMatType, SparseMatrixListType],
        sectype=None,
    ):
        super().__init__(sectype)

        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        if isinstance(sparse_mat, ScipySparseMatType):
            to_sec_int = lambda x: self.sectype(int(x))
            self._mat = []
            for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
                self._mat.append([to_sec_int(i), to_sec_int(j), to_sec_int(v)])
        elif isinstance(sparse_mat, list):
            for tup in sparse_mat:
                assert len(tup) == 3
        else:
            raise ValueError("Invalid input")

    def dot(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Can only multiply SparseMatrixCOO with SparseMatrixCOO")

        # TODO

    async def print(self):
        for i in range(len(self._mat)):
            print(
                await mpc.output(self._mat[i][0]),
                await mpc.output(self._mat[i][1]),
                await mpc.output(self._mat[i][2]),
            )

    async def to_numpy_dense(self):
        ...  # TODO


class SparseMatrixColumn(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            print(i, j, v)
            self._mat[j].append((to_sec_int(i), to_sec_int(v)))


class SparseMatrixRow(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i].append((to_sec_int(j), to_sec_int(v)))


async def main():
    n_dim = 400
    secint = mpc.SecInt(64)
    x_sparse = scipy.sparse.random(n_dim, n_dim, density=0.1, dtype=np.int16).astype(
        int
    )
    x_dense = x_sparse.todense()
    l = np.vectorize(lambda x: secint(int(x)))(x_dense)
    l = l.tolist()
    print("here")
    z = mpc.matrix_prod(l, l)
    print(await mpc.output(z[0][0]))
    print(await mpc.output(l[0][0]))
    print(x_sparse.dot(x_sparse).todense()[0, 0])


if __name__ == "__main__":
    mpc.run(main())
