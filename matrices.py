from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from sortable_tuple import SortableTuple
from shuffle import np_shuffle

SparseMatrixListType = List[List[int]]
ScipySparseMatType = scipy.sparse._coo.coo_matrix


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
    def __init__(
        self, sparse_mat: ScipySparseMatType, sectype=None, sort_coroutine=None
    ):
        super().__init__(sectype)

        self.sort_coroutine = (
            SparseMatrixColumn.default_sort
            if sort_coroutine is None
            else sort_coroutine
        )
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[j].append((to_sec_int(i), to_sec_int(v)))

    @staticmethod
    async def default_sort(unsorted, _sectype, key=None):
        return mpc.sorted(unsorted, key)

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

            res = await self.sort_coroutine(res, self.sectype, key=SortableTuple)

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


class SparseMatrixColumnNumpy(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)

        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[j] += [to_sec_int(i), to_sec_int(v)]

        self._mat = [
            mpc.np_reshape(mpc.np_fromlist(self._mat[k]), (len(self._mat[k]) // 2, 2))
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]

    async def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseMatrixRowNumpy):
            res = None

            for k in range(self.shape[1]):
                if self._mat[k] is None or other._mat[k] is None:
                    continue

                i_vect = []
                j_vect = []
                for i in range(self._mat[k].shape[0]):
                    curr_i = self._mat[k][i, 0]
                    for j in range(other._mat[k].shape[0]):
                        i_vect.append(curr_i)
                        j_vect.append(other._mat[k][j, 0])

                i_vect = mpc.np_fromlist(i_vect)
                j_vect = mpc.np_fromlist(j_vect)
                mult_res_k = mpc.np_flatten(
                    mpc.np_outer(self._mat[k][:, 1], other._mat[k][:, 1])
                )
                res_k = mpc.np_transpose(mpc.np_vstack((i_vect, j_vect, mult_res_k)))
                if res is None:
                    res = res_k
                else:
                    res = mpc.np_vstack((res, res_k))

            sorting_keys = res[:, 0] * (other.shape[1] + 1) + res[:, 1]
            res = mpc.np_column_stack((mpc.np_transpose(sorting_keys), res))

            res = mpc.np_sort(res, axis=0, key=lambda tup: tup[0])

            comp = res[0 : res.shape[0] - 1, 0] == res[1 : res.shape[0], 0]
            col_val = [res[0, 3]]
            col_i = []
            for i in range(res.shape[0] - 1):
                col_val.append(
                    mpc.if_else(comp[i], res[i, 3] + res[i + 1, 3], res[i + 1, 3])
                )
                col_i.append(mpc.if_else(comp[i], -1, res[i, 1]))
                # Only need one placeholder per tuple to make it invalid
            col_i.append(res[-1, 1])

            # I do a unique update because I had issue with iterative updates.
            mpc.np_update(res, (range(len(col_val)), 3), mpc.np_fromlist(col_val))
            mpc.np_update(res, (range(len(col_i)), 1), mpc.np_fromlist(col_i))

            res = res[:, 1:]  # We remove the sorting key
            res = await np_shuffle(self.sectype, res)

            final_res = []
            zero_test = await mpc.np_is_zero_public(
                res[:, 0] + 1
            )  # Here, we leak the number of non-zero elements in the output matrix
            mask = [i for i, test in enumerate(zero_test) if not test]
            final_res = mpc.np_tolist(res[mask, :])

            # TODO: create a numpy equivalent of this class
            return SparseMatrixCOO(
                final_res, sectype=self.sectype, shape=(self.shape[0], other.shape[1])
            )

        raise ValueError("Can only multiply SparseMatrixColumn with this object")


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


class SparseMatrixRowNumpy(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i] += [to_sec_int(j), to_sec_int(v)]
        self._mat = [
            mpc.np_reshape(mpc.np_fromlist(self._mat[k]), (len(self._mat[k]) // 2, 2))
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]


class DenseMatrix(SecureMatrix):
    def __init__(self, mat, sectype=None):
        super().__init__(sectype)
        if isinstance(mat, list):
            self._mat = mat
            self.shape = (len(mat), len(mat[0]))
        else:
            self.shape = mat.shape
            temp_mat = np.vectorize(lambda x: self.sectype(int(x)))(
                mat
            )  # TODO: find a way to avoid the hardcoded type
            self._mat = temp_mat.tolist()

    def dot(self, other):
        if not isinstance(other, DenseMatrix):
            raise ValueError("Can only multiply dense with dense")

        return DenseMatrix(mpc.matrix_prod(self._mat, other._mat), sectype=self.sectype)

    async def print(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print(await mpc.output(self._mat[i][j]), end=" ")
            print("")

    def get(self, i, j):
        return self._mat[i][j]


class DenseMatrixNaive(DenseMatrix):
    def __init__(self, mat, sectype=None):
        super().__init__(mat, sectype)

    def dot(self, other):
        if not isinstance(other, DenseMatrixNaive):
            raise ValueError("Can only multiply dense with dense")

        res = [[0] * other.shape[1]] * self.shape[0]
        for k in range(self.shape[1]):
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    res[i][j] += self._mat[i][k] * other._mat[k][j]

        return DenseMatrix(res)
