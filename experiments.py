from abc import abstractmethod
from typing import List, Union, Optional, Tuple

import numpy as np
import scipy.sparse

from mpyc.runtime import mpc

SparseMatrixListType = List[List[int]]
ScipySparseMatType = scipy.sparse._coo.coo_matrix


class SortableTuple:
    def __init__(self, tup):
        self._tup = tuple(tup)

    def __lt__(self, other):
        if len(self._tup) != len(other._tup):
            raise ValueError("Tuples must be of same size")
        return SortableTuple._lt_tuples(self._tup, other._tup)

    def __ge__(self, other):
        return ~self.__lt__(other)

    def _lt_tuples(tup1, tup2):
        first_comp = tup1[0] < tup2[0]
        if len(tup1) > 1:
            equal_comp = tup1[0] == tup2[0]
            recursive_comp = SortableTuple._lt_tuples(tup1[1:], tup2[1:])
            # first_comp or (recursive_comp and equal_comp)
            return first_comp | (recursive_comp & equal_comp)
            # NB: "+" represent the OR without a negative term (i.e., A or B = A + B - A*B)
            # because the two terms cannot be true at the same time
        else:
            return first_comp


class SecureMatrix:
    _mat: Optional[SparseMatrixListType] = None
    shape: Tuple[int, int]

    def __init__(self, sectype=None):
        if sectype is None:
            self.sectype = mpc.SecInt(64)
        else:
            self.sectype = sectype

    @abstractmethod
    def dot(self, other):
        raise NotImplementedError

    @abstractmethod
    async def print(self):
        raise NotImplementedError


class SparseMatrixCOO(SecureMatrix):
    def __init__(
        self,
        sparse_mat: Union[ScipySparseMatType, SparseMatrixListType],
        sectype=None,
        shape=None,
    ):
        super().__init__(sectype)

        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        if isinstance(sparse_mat, ScipySparseMatType):
            assert shape is None
            self.shape = sparse_mat.shape
            to_sec_int = lambda x: self.sectype(int(x))
            self._mat = []
            for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
                self._mat.append([to_sec_int(i), to_sec_int(j), to_sec_int(v)])
        elif isinstance(sparse_mat, list):
            assert isinstance(shape, tuple) and len(shape) == 2

            for tup in sparse_mat:
                assert len(tup) == 3
            self._mat = sparse_mat
            self.shape = shape
        else:
            raise ValueError("Invalid input")

    def dot(self, other) -> "SparseMatrixCOO":
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Can only multiply SparseMatrixCOO with SparseMatrixCOO")

        # TODO: execute either multiply-then-sort or sort-then-multiply depending on the sparsity rate

    def __eq__(self, other):
        if not isinstance(other, SparseMatrixCOO):
            raise ValueError("Can only compare SparseMatrixCOO with SparseMatrixCOO")
        if len(self._mat) != len(other._mat):
            return self.sectype(0)
        mat1 = mpc.sorted(self._mat, key=SortableTuple)
        mat2 = mpc.sorted(other._mat, key=SortableTuple)
        res = self.sectype(1)
        for i in range(len(self._mat)):
            res = (
                res
                & (mat1[i][0] == mat2[i][0])
                & (mat1[i][1] == mat2[i][1])
                & (mat1[i][2] == mat2[i][2])
            )
        return res

    def __ne__(self, other):
        return ~(self.__eq__(other))

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
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[j].append((to_sec_int(i), to_sec_int(v)))

    async def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseMatrixRow):
            res = []
            for k in range(self.shape[1]):
                for left_i, left_value in self._mat[k]:
                    for right_j, right_value in other._mat[k]:

                        res.append([left_i, right_j, left_value * right_value])

            res = mpc.sorted(res, SortableTuple)

            for i in range(1, len(res)):
                sec_comp_res = (res[i - 1][0] == res[i][0]) & (
                    res[i - 1][1] == res[i][1]
                )
                res[i][2] = mpc.if_else(
                    sec_comp_res, res[i - 1][2] + res[i][2], res[i][2]
                )

                res[i - 1][0] = mpc.if_else(sec_comp_res, -1, res[i - 1][0])
                # Only need one placeholder per tuple to make it invalid

            mpc.random.shuffle(self.sectype, res)

            final_res = []
            for i in range(len(res)):
                if await mpc.output(res[i][0] != -1):
                    final_res.append(res[i])
                # Here, we leak the number of non-zero elements in the output matrix

            return SparseMatrixCOO(
                final_res, sectype=self.sectype, shape=(self.shape[0], other.shape[1])
            )

        raise ValueError("Can only multiply SparseMatrixColumn with this object")

    async def print(self):
        for j, col in enumerate(self._mat):
            print(f"Column {j}: [", end="")
            for i, val in col:
                print(
                    "(", await mpc.output(i), ", ", await mpc.output(val), ")", end=","
                )
            print("]")


class SparseMatrixRow(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i].append((to_sec_int(j), to_sec_int(v)))

    async def print(self):
        for i, col in enumerate(self._mat):
            print(f"Row {i}: [", end="")
            for j, val in col:
                print(
                    "(", await mpc.output(j), ", ", await mpc.output(val), ")", end=","
                )
                print("]")


async def main():
    n_dim = 10
    secint = mpc.SecInt(64)

    x_sparse = scipy.sparse.random(n_dim, n_dim, density=0.3, dtype=np.int16).astype(
        int
    )
    sec_x = SparseMatrixColumn(x_sparse, secint)
    sec_y = SparseMatrixRow(x_sparse, secint)
    sec_z = await sec_x.dot(sec_y)

    z = x_sparse.dot(x_sparse).tocoo()
    print("===")
    await sec_z.print()
    print("===")
    print(z)

    sec_z_real = SparseMatrixCOO(z, secint)
    assert await mpc.output(sec_z == sec_z_real)


if __name__ == "__main__":
    mpc.run(main())
