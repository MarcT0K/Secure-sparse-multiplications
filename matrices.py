import math
from abc import abstractmethod
from random import randint
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from quicksort import parallel_quicksort
from resharing import np_shuffle_3PC
from radix_sort import radix_sort
from sortable_tuple import SortableTuple

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
        to_sec_int = lambda x: mpc.input(self.sectype(int(x)), senders=0)
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

            if len(mpc.parties) == 3:
                res = await np_shuffle_3PC(res)
            else:
                mpc.random.shuffle(self.sectype, res)

            final_res = [
                tup for tup in range(len(res)) if await mpc.output(tup[0] != -1)
            ]
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
        to_sec_int = lambda x: mpc.input(self.sectype(int(x)), senders=0)
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

            if res.shape[0] == 1:
                return SparseMatrixCOO(
                    mpc.np_tolist(res),
                    sectype=self.sectype,
                    shape=(self.shape[0], other.shape[1]),
                )

            sorting_keys = res[:, 0] * (other.shape[1] + 1) + res[:, 1]

            res = mpc.np_column_stack((mpc.np_transpose(sorting_keys), res))

            if len(mpc.parties) == 3:
                assert self.shape[0] * other.shape[1] * 10**3  # Prevent overflow
                if mpc.pid == 0:
                    rand_vect = [
                        randint(0, 10**3) for _i in range(sorting_keys.shape[0])
                    ]
                else:
                    rand_vect = None

                rand_vect = mpc.np_fromlist(
                    [self.sectype(i) for i in await mpc.transfer(rand_vect, senders=0)]
                )
                rand_sorting_keys = sorting_keys * 10**3 + rand_vect
                res = mpc.np_column_stack((mpc.np_transpose(rand_sorting_keys), res))
                res = await parallel_quicksort(res, key=lambda tup: tup[0])
                res = res[:, 1:]
            else:
                res = mpc.np_sort(res, axis=0, key=lambda tup: tup[0])

            comp = res[0 : res.shape[0] - 1, 0] == res[1 : res.shape[0], 0]
            res = res[:, 1:]  # We remove the sorting key

            col_val = mpc.np_tolist(res[:, -1])
            col_i = res[:-1, 0] * (1 - comp) - comp

            for i in range(res.shape[0] - 1):
                col_val[i + 1] = col_val[i + 1] + comp[i] * col_val[i]
            col_i = mpc.np_hstack((col_i, res[-1, 1:2]))
            col_val = mpc.np_transpose(mpc.np_fromlist(col_val))
            col_i = mpc.np_transpose(col_i)

            res = mpc.np_transpose(mpc.np_vstack((col_i, res[:, 1], col_val)))

            await mpc.barrier()

            if len(mpc.parties) == 3:
                res = await np_shuffle_3PC(res)
            else:
                mpc.random.shuffle(self.sectype, res)

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
        to_sec_int = lambda x: mpc.input(self.sectype(int(x)), senders=0)
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
        to_sec_int = lambda x: mpc.input(self.sectype(int(x)), senders=0)
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
            self._mat = [[sectype(i) for i in row] for row in mat.tolist()]

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


class DenseMatrixNumpy(SecureMatrix):
    def __init__(self, mat, sectype=None):
        super().__init__(sectype)
        if isinstance(mat, mpc.SecureArray):
            self._mat = mat
            self.shape = (len(mat), len(mat[0]))
        else:
            self.shape = mat.shape
            temp_mat = [sectype(i) for i in mat.flatten().tolist()[0]]
            self._mat = mpc.input(
                mpc.np_reshape(mpc.np_fromlist(temp_mat), self.shape), senders=0
            )

    def dot(self, other):
        if not isinstance(other, DenseMatrixNumpy):
            raise ValueError("Can only multiply dense with dense")

        return DenseMatrixNumpy(
            mpc.np_matmul(self._mat, other._mat), sectype=self.sectype
        )

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


class OptimizedSparseMatrixColumn(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)

        self.shape = sparse_mat.shape
        self.key_bit_length = int(math.log(self.shape[0], 2)) + 1
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[j] += self.int_to_secure_bits(i) + [
                to_sec_int(i),
                to_sec_int(v),
            ]

        self._mat = [
            mpc.input(
                mpc.np_reshape(
                    mpc.np_fromlist(self._mat[k]),
                    (
                        len(self._mat[k]) // (self.key_bit_length + 2),
                        self.key_bit_length + 2,
                    ),
                ),
                senders=0,
            )
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]

    def int_to_secure_bits(self, number):
        bitstring = format(number, f"0{self.key_bit_length}b")
        return [self.sectype(int(c)) for c in bitstring][::-1]

    async def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, OptimizedSparseMatrixRow):
            res = None

            sorting_key_length = self.key_bit_length + other.key_bit_length

            for k in range(self.shape[1]):
                if self._mat[k] is None or other._mat[k] is None:
                    continue

                coord_mat = []  # TODO: replace with numpy-like array
                for i in range(self._mat[k].shape[0]):
                    curr_decomp_i = self._mat[k][i, :-2]
                    curr_i = mpc.np_fromlist([self._mat[k][i, -2]])
                    for j in range(other._mat[k].shape[0]):
                        curr_decomp_j = other._mat[k][j, :-2]
                        curr_j = mpc.np_fromlist([other._mat[k][j, -2]])
                        curr_coord = mpc.np_hstack(
                            (curr_decomp_i, curr_decomp_j, curr_i, curr_j)
                        )
                        coord_mat.extend(mpc.np_tolist(curr_coord))

                coord_mat = mpc.np_reshape(
                    mpc.np_fromlist(coord_mat),
                    (
                        len(coord_mat) // (sorting_key_length + 2),
                        (sorting_key_length + 2),
                    ),
                )

                mult_res_k = mpc.np_flatten(
                    mpc.np_outer(self._mat[k][:, -1], other._mat[k][:, -1])
                )
                res_k = mpc.np_transpose(
                    mpc.np_vstack((mpc.np_transpose(coord_mat), mult_res_k))
                )

                if res is None:
                    res = res_k
                else:
                    res = mpc.np_vstack((res, res_k))

            if res.shape[0] == 1:
                return SparseMatrixCOO(
                    mpc.np_tolist(res),
                    sectype=self.sectype,
                    shape=(self.shape[0], other.shape[1]),
                )

            if len(mpc.parties) == 3:
                res = await radix_sort(res, sorting_key_length, already_decomposed=True)
                res = res[:, -3:]
            else:
                raise NotImplementedError

            comp = mpc.np_multiply(
                res[0 : res.shape[0] - 1, 0] == res[1 : res.shape[0], 0],
                res[0 : res.shape[0] - 1, 1] == res[1 : res.shape[0], 1],
            )

            col_val = mpc.np_tolist(res[:, -1])
            col_i = res[:-1, 0] * (1 - comp) - comp

            for i in range(res.shape[0] - 1):
                col_val[i + 1] = col_val[i + 1] + comp[i] * col_val[i]
            col_i = mpc.np_hstack((col_i, res[-1, 1:2]))
            col_val = mpc.np_transpose(mpc.np_fromlist(col_val))
            col_i = mpc.np_transpose(col_i)

            res = mpc.np_transpose(mpc.np_vstack((col_i, res[:, 1], col_val)))

            if len(mpc.parties) == 3:
                res = await np_shuffle_3PC(res)
            else:
                mpc.random.shuffle(self.sectype, res)

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


class OptimizedSparseMatrixRow(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype)
        self.shape = sparse_mat.shape
        self.key_bit_length = int(math.log(self.shape[1], 2)) + 1
        to_sec_int = lambda x: self.sectype(int(x))
        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i] += self.int_to_secure_bits(j) + [
                to_sec_int(j),
                to_sec_int(v),
            ]

        self._mat = [
            mpc.input(
                mpc.np_reshape(
                    mpc.np_fromlist(self._mat[k]),
                    (
                        len(self._mat[k]) // (self.key_bit_length + 2),
                        self.key_bit_length + 2,
                    ),
                ),
                senders=0,
            )
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]

    def int_to_secure_bits(self, number):
        bitstring = format(number, f"0{self.key_bit_length}b")
        return [self.sectype(int(c)) for c in bitstring][::-1]
