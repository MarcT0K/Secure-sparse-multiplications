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
        return 1 - 1 * self.__lt__(other)

    def _lt_tuples(tup1, tup2):
        first_comp = tup1[0] < tup2[0]
        if len(tup1) > 1:
            equal_comp = tup1[0] == tup2[0]
            recursive_comp = SortableTuple._lt_tuples(tup1[1:], tup2[1:])
            # first_comp or (recursive_comp and equal_comp)
            return first_comp + recursive_comp * equal_comp
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
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            print(i, j, v)
            self._mat[j].append((to_sec_int(i), to_sec_int(v)))

    def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")

        if isinstance(other, SparseMatrixRow):
            res = []
            for _k in range(self.shape[1]):
                for left_j in range(self.shape[0]):
                    left_i, left_value = self._mat[left_j]
                    for right_i in range(other.shape[1]):
                        right_j, right_value = other._mat[right_i]
                        res.append([left_i, right_j, left_value * right_value])
            res = mpc.sorted(res, key=SortableTuple)

            for i in range(1, len(res)):
                sec_comp_res = res[i - 1][0] == res[i][0] and res[i - 1][1] == res[i][1]
                res[i][2] = sec_comp_res * res[i - 1][2] + res[i][2]

                res[i - 1][0] = mpc.if_else(sec_comp_res, -1, res[i - 1][0])
                # Only need one placeholder per tuple to make it invalid

            mpc.random.shuffle(res)

            final_res = []
            for i in range(len(res)):
                if res[i][0] != -1:
                    final_res.append(res[i])
                # Here, we leak the number of non-zero elements in the output matrix

            return SparseMatrixCOO(
                final_res, sectype=self.sectype, shape=(self.shape[0], other.shape[1])
            )

        else:
            raise ValueError("Can only multiply SparseMatrixCOO with SparseMatrixCOO")


class SparseMatrixRow(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i].append((to_sec_int(j), to_sec_int(v)))


# async def main():
#     n_dim = 400
#     secint = mpc.SecInt(64)
#     x_sparse = scipy.sparse.random(n_dim, n_dim, density=0.1, dtype=np.int16).astype(
#         int
#     )
#     x_dense = x_sparse.todense()
#     l = np.vectorize(lambda x: secint(int(x)))(x_dense)
#     l = l.tolist()
#     print("here")
#     z = mpc.matrix_prod(l, l)
#     print(await mpc.output(z[0][0]))
#     print(await mpc.output(l[0][0]))
#     print(x_sparse.dot(x_sparse).todense()[0, 0])


async def main():
    secint = mpc.SecInt(64)
    l = [
        [secint(2), secint(0), secint(0)],
        [secint(1), secint(0), secint(0)],
        [secint(1), secint(2), secint(0)],
        [secint(1), secint(0), secint(3)],
    ]

    async def print_mat(mat):
        for i in range(len(mat)):
            print(
                await mpc.output(mat[i][0]),
                await mpc.output(mat[i][1]),
                await mpc.output(mat[i][2]),
            )

    await print_mat(l)

    print("---")

    l_sorted = mpc.sorted(l, SortableTuple)

    await print_mat(l_sorted)
    print("---")

    mpc.random.shuffle(secint, l_sorted)
    await print_mat(l_sorted)
    print("---")

    mpc.random.shuffle(secint, l_sorted)
    await print_mat(l_sorted)
    print("---")

    mpc.random.shuffle(secint, l_sorted)
    await print_mat(l_sorted)
    print("---")


if __name__ == "__main__":
    mpc.run(main())
