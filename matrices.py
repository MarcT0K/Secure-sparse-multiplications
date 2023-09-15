import math
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc
from mpyc.seclists import seclist

from radix_sort import radix_sort
from resharing import np_shuffle_3PC
from sortable_tuple import SortableTuple

SparseMatrixListType = List[List[int]]
ScipySparseMatType = scipy.sparse._coo.coo_matrix


class SecureMatrix:
    _mat: Optional[SparseMatrixListType] = None
    shape: Tuple[int, int]

    def __init__(self, sectype=None, shape=None):
        if sectype is None:
            self.sectype = mpc.SecFxp(64)
        else:
            self.sectype = sectype

        if shape is None or not isinstance(shape, tuple) or len(shape) < 1:
            raise ValueError("Invalid shape")
        self.shape = shape
        self.row_bit_length = int(math.log(self.shape[0], 2)) + 1
        self.col_bit_length = int(math.log(self.shape[1], 2)) + 1

    @staticmethod
    def int_to_secure_bits(number, sectype, nb_bits):
        bitstring = format(number, f"0{nb_bits}b")
        return [sectype(int(c)) for c in bitstring][::-1]

    @staticmethod
    def to_secint(sectype, x):  # TODO: replace this?
        return sectype(int(x))

    @abstractmethod
    def dot(self, other):
        raise NotImplementedError

    @abstractmethod
    async def print(self):
        raise NotImplementedError


## DENSE CLASSES


class DenseMatrix(SecureMatrix):
    def __init__(self, mat, sectype=None):
        shape = (
            (len(mat), len(mat[0])) if isinstance(mat, mpc.SecureArray) else mat.shape
        )
        super().__init__(sectype, shape)
        if isinstance(mat, mpc.SecureArray):
            self._mat = mat
        else:
            temp_mat = [sectype(i) for i in mat.flatten().tolist()[0]]
            self._mat = mpc.input(
                mpc.np_reshape(mpc.np_fromlist(temp_mat), self.shape), senders=0
            )

    def dot(self, other):
        if isinstance(other, DenseMatrix) or isinstance(other, DenseVector):
            return DenseMatrix(
                mpc.np_matmul(self._mat, other._mat), sectype=self.sectype
            )
        raise ValueError("Can only multiply dense with dense")

    async def print(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print(await mpc.output(self._mat[i][j]), end=" ")
            print("")

    def get(self, i, j):
        return self._mat[i][j]


class DenseVector(DenseMatrix):
    def __init__(self, mat, sectype=None):
        if mat.shape[1] != 1 and mat.shape[0] != 1:
            raise ValueError("Input must be a vector")
        super().__init__(mat, sectype)

    def dot(self, other):
        if isinstance(other, DenseVector):
            res = super().dot(other)
            assert res.shape == (1, 1)
            return res.get(0, 0)
        else:
            raise NotImplementedError


## SPARSE CLASSES


class SparseVector(SecureMatrix):
    def __init__(self, sparse_mat, sectype=None, shape=None):
        assert (
            (isinstance(sparse_mat, mpc.SecureArray) and shape is not None)
            or (sparse_mat == [] and shape is not None)
            or shape is None
        )

        if isinstance(sparse_mat, mpc.SecureArray) or sparse_mat == []:
            super().__init__(sectype, shape)
            self._mat = sparse_mat
        else:
            super().__init__(sectype, sparse_mat.shape)

            self._mat = []
            for i, _j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
                self._mat.extend(
                    SecureMatrix.int_to_secure_bits(
                        i, self.sectype, self.row_bit_length
                    )
                )
                print(float(i), type(i))
                self._mat.append(self.sectype(int(i)))
                self._mat.append(self.sectype(float(v)))

            if self._mat:
                np_mat = mpc.np_reshape(
                    mpc.np_fromlist(self._mat),
                    (
                        len(self._mat) // (self.row_bit_length + 2),
                        self.row_bit_length + 2,
                    ),
                )
                self._mat = mpc.input(np_mat, senders=0)

        if self.shape[1] != 1:
            raise ValueError("Input must be a vector")

    async def dot(self, other):
        if isinstance(other, SparseVector):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")

            if not self._mat or not other._mat:
                return self.sectype(0)

            print("_mat", self._mat.integral, other._mat.integral)
            unsorted2 = mpc.np_vstack((self._mat, other._mat))
            print(unsorted2.integral, unsorted2.shape)
            unsorted = mpc.np_column_stack((self._mat, other._mat))
            print(unsorted.integral, unsorted.shape)
            sorted_array = await radix_sort(
                unsorted, self.row_bit_length, already_decomposed=True
            )

            n = sorted_array.shape[0]
            mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
            comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
            return mpc.np_sum(mult_vect * comp_vect)
        elif isinstance(other, DenseVector):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")

            s = self.sectype(0)
            if not self._mat or not other._mat:
                return s

            dense_vect_list = seclist(
                mpc.np_tolist(mpc.np_transpose(other._mat))[0], self.sectype
            )

            nnz = self._mat.shape[0]
            unit_matrix = None
            for i in range(nnz):
                sparse_coord = self._mat[i, -2]
                unit_vector = mpc.np_unit_vector(sparse_coord, self.shape[0])
                if unit_matrix is None:
                    unit_matrix = unit_vector
                else:
                    unit_matrix = mpc.np_vstack((unit_matrix, unit_vector))
            temp = mpc.np_matmul(unit_matrix, other._mat)
            s = mpc.np_matmul(mpc.np_transpose(temp), self._mat[:, -1:])
            return s
        else:
            raise NotImplementedError

    async def print(self):
        print(await mpc.output(self._mat[:, self.row_bit_length :]))


class SparseMatrixCOO(SecureMatrix):
    def __init__(
        self,
        sparse_mat: Union[ScipySparseMatType, SparseMatrixListType],
        sectype=None,
        shape=None,
    ):
        # https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
        if isinstance(sparse_mat, ScipySparseMatType):
            super().__init__(sectype, sparse_mat.shape)
            self._mat = []
            for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
                self._mat.append(
                    [
                        SecureMatrix.to_secint(self.sectype, i),
                        SecureMatrix.to_secint(self.sectype, j),
                        SecureMatrix.to_secint(self.sectype, v),
                    ]
                )
        elif isinstance(sparse_mat, list):
            super().__init__(sectype, shape)

            for tup in sparse_mat:
                assert len(tup) == 3
            self._mat = sparse_mat
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
        super().__init__(sectype, sparse_mat.shape)

        self._mat = [[] for i in range(sparse_mat.shape[1])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[j] += SecureMatrix.int_to_secure_bits(
                i, self.sectype, self.row_bit_length
            ) + [
                SecureMatrix.to_secint(self.sectype, i),
                SecureMatrix.to_secint(self.sectype, v),
            ]

        self._mat = [
            mpc.input(
                mpc.np_reshape(
                    mpc.np_fromlist(self._mat[k]),
                    (
                        len(self._mat[k]) // (self.row_bit_length + 2),
                        self.row_bit_length + 2,
                    ),
                ),
                senders=0,
            )
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]

    async def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseMatrixRow):
            res = None

            sorting_key_length = self.row_bit_length + other.col_bit_length

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
                    mpc.np_tolist(res[:, -3:]),
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
            # If the test is false, the value of this column is -1 (i.e., a placeholder)

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

            return SparseMatrixCOO(
                final_res, sectype=self.sectype, shape=(self.shape[0], other.shape[1])
            )
        raise NotImplementedError


class SparseMatrixRow(SecureMatrix):
    def __init__(self, sparse_mat: ScipySparseMatType, sectype=None):
        super().__init__(sectype, sparse_mat.shape)

        self._mat = [[] for i in range(sparse_mat.shape[0])]
        for i, j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[i] += SecureMatrix.int_to_secure_bits(
                j, self.sectype, self.col_bit_length
            ) + [
                SecureMatrix.to_secint(self.sectype, j),
                SecureMatrix.to_secint(self.sectype, v),
            ]

        self._mat = [
            mpc.input(
                mpc.np_reshape(
                    mpc.np_fromlist(self._mat[k]),
                    (
                        len(self._mat[k]) // (self.col_bit_length + 2),
                        self.col_bit_length + 2,
                    ),
                ),
                senders=0,
            )  # Each row is represented as a sparse vector
            if len(self._mat[k])
            else None
            for k in range(len(self._mat))
        ]

    async def _matrix_vector_prod(self, other) -> SparseVector:
        if not other._mat:
            return SparseVector([], self.sectype, shape=(self.shape[0], 1))

        ### NUMPY-LIKE MATRIX PREPARATION (i.e., for parallelized operations)
        padded_matrix = []
        for i in range(self.shape[0]):
            if not self._mat[i]:
                continue

            bin_row_ind = (
                SecureMatrix.int_to_secure_bits(i, self.sectype, self.row_bit_length)
                * self._mat[i].shape[0]
            )
            bin_row_ind = mpc.np_reshape(
                mpc.np_fromlist(bin_row_ind),
                (len(bin_row_ind) // self.row_bit_length, self.row_bit_length),
            )
            int_row_ind = self.sectype.array(
                i * np.ones((self._mat[i].shape[0], 1), dtype=int)
            )
            ones = self.sectype.array(np.ones((self._mat[i].shape[0], 1), dtype=int))
            curr_row_mat = mpc.np_hstack(
                (  # We place the column binary representation first to sort based on columns
                    ones,
                    self._mat[i][:, : self.col_bit_length],
                    self._mat[i][:, -2:-1],  # Integer column indices
                    bin_row_ind,
                    int_row_ind,
                    self._mat[i][:, -1:],
                )
            )

            padded_matrix.append(curr_row_mat)

        if not padded_matrix:
            return SparseVector([], self.sectype, shape=(self.shape[0], 1))

        padded_matrix = mpc.np_vstack(
            padded_matrix
        )  # We concatenate (vertically) all row matrices

        zeros_for_vect = self.sectype.array(
            np.zeros((other._mat.shape[0], 1), dtype=int)
        )
        bin_placeholders_for_vect = self.sectype.array(
            np.zeros((other._mat.shape[0], self.row_bit_length), dtype=int)
        )
        placeholder_for_vect = self.sectype.array(
            -np.ones((other._mat.shape[0], 1), dtype=int)
        )  # Replace the integer row indices by placeholders so they can be removed at the end of the function

        padded_vect = mpc.np_hstack(
            (
                zeros_for_vect,
                other._mat[:, : self.col_bit_length],
                other._mat[:, -2:-1],
                bin_placeholders_for_vect,
                placeholder_for_vect,
                other._mat[:, -1:],
            )
        )
        # We add an extra zero bit to differentiate the matrix values from the vector values

        res = mpc.np_vstack((padded_vect, padded_matrix))

        ### MULTIPLICATION STEP
        if len(mpc.parties) == 3:
            res = await radix_sort(
                res,
                self.col_bit_length + 1,
                already_decomposed=True,
                keep_bin_keys=True,
            )
        else:
            raise NotImplementedError

        last_vect_val = self.sectype(0)
        last_vect_col = self.sectype(-1)
        val_col = []
        for i in range(res.shape[0]):
            is_vect_elem = 1 - res[i, 0]
            same_col = res[i, self.col_bit_length + 1] == last_vect_col
            mult_cond = same_col * res[i, 0]

            last_vect_val = (1 - is_vect_elem) * last_vect_val + is_vect_elem * res[
                i, -1
            ]
            last_vect_col = (1 - is_vect_elem) * last_vect_col + is_vect_elem * res[
                i, self.col_bit_length + 1
            ]
            val_col.append(res[i, -1] * last_vect_val * mult_cond)

        val_col = mpc.np_reshape(mpc.np_fromlist(val_col), (res.shape[0], 1))

        res = mpc.np_hstack(
            (res[:, 0:1], res[:, self.col_bit_length + 2 : -1], val_col)
        )
        # We remove the column indices but keep the differentiating bit

        # AGGREGATION STEP
        if len(mpc.parties) == 3:
            res = await radix_sort(
                res,
                self.row_bit_length + 1,
                already_decomposed=True,
                keep_bin_keys=True,
            )
        else:
            raise NotImplementedError

        # We compare the integer representation of the column index for all pairs of consecutive elements
        comp = res[0 : res.shape[0] - 1, -2] == res[1 : res.shape[0], -2]

        col_val = mpc.np_tolist(res[:, -1])
        col_i = res[:-1, -2] * (1 - comp) + (-1) * comp
        # If the test is false, the value of this column is -1 (i.e., a placeholder)

        for i in range(res.shape[0] - 1):
            col_val[i + 1] = col_val[i + 1] + comp[i] * col_val[i]
        col_i = mpc.np_hstack((col_i, res[-1, 1:2]))
        col_val = mpc.np_reshape(mpc.np_fromlist(col_val), (res.shape[0], 1))
        col_i = mpc.np_reshape(col_i, (res.shape[0], 1))

        res = mpc.np_hstack((res[:, :-2], col_i, col_val))

        if len(mpc.parties) == 3:
            res = await np_shuffle_3PC(res)
        else:
            mpc.random.shuffle(self.sectype, res)

        final_res = []
        zero_test = await mpc.np_is_zero_public(
            res[:, -2] + 1
        )  # Here, we leak the number of non-zero elements in the output matrix
        zero_val_test = await mpc.np_is_zero_public(
            res[:, -1]
        )  # TODO: refactor this test?

        mask = [
            i for i, test in enumerate(zero_test) if not test and not zero_val_test[i]
        ]
        final_res = res[mask, 1:]  # We remove the differentiating bit

        return SparseVector(final_res, sectype=self.sectype, shape=(self.shape[0], 1))

    async def dot(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseVector):
            return await self._matrix_vector_prod(other)
        raise NotImplementedError
