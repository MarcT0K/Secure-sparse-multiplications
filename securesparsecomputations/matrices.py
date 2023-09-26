import math
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from .radix_sort import radix_sort
from .resharing import np_shuffle_3PC

ScipySparseMatType = scipy.sparse._coo.coo_matrix


class SecureMatrix:
    _mat = None
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

    @abstractmethod
    def dot(self, other):
        raise NotImplementedError

    async def print(self):
        raise NotImplementedError


## DENSE CLASSES


class DenseMatrix(SecureMatrix):
    def __init__(self, mat, sectype=None):
        shape = (
            (len(mat), len(mat[0])) if isinstance(mat, mpc.SecureArray) else mat.shape
        )
        super().__init__(sectype, shape)
        self._mat = mat

    def dot(self, other):
        if isinstance(other, DenseMatrix) or isinstance(other, DenseVector):
            return type(other)(
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

    def transpose(self):
        return DenseMatrix(mpc.np_transpose(self._mat), self.sectype)


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
        elif isinstance(other, SparseVector):
            s = self.sectype(0)
            if other.nnz == 0:
                return s

            unit_matrix = None
            for i in range(other.nnz):
                sparse_coord = other._mat[i, -2]
                unit_vector = mpc.np_unit_vector(sparse_coord, other.shape[0])
                if unit_matrix is None:
                    unit_matrix = unit_vector
                else:
                    unit_matrix = mpc.np_vstack((unit_matrix, unit_vector))
            temp = mpc.np_matmul(unit_matrix, self._mat)
            s = mpc.np_matmul(mpc.np_transpose(temp), other._mat[:, -1:])
            return s
        else:
            raise NotImplementedError


def from_numpy_dense_matrix(dense_matrix, sectype) -> Union[DenseMatrix, DenseVector]:
    shape = dense_matrix.shape

    flattened = dense_matrix.flatten()
    flattened = (
        flattened.tolist() if len(flattened.shape) == 1 else flattened.tolist()[0]
    )
    temp_mat = [sectype(i) for i in flattened]
    secure_mat = mpc.input(
        mpc.np_reshape(mpc.np_fromlist(temp_mat), dense_matrix.shape), senders=0
    )

    retcls = (
        DenseMatrix
        if len(shape) > 1 and shape[1] != 1 and shape[0] != 1
        else DenseVector
    )
    return retcls(secure_mat, sectype)


## SPARSE CLASSES


class SparseVector(SecureMatrix):
    def __init__(self, sparse_mat: Optional[mpc.SecureArray], shape, sectype=None):
        assert isinstance(sparse_mat, mpc.SecureArray) or sparse_mat is None

        super().__init__(sectype, shape)
        self._mat = sparse_mat
        self.nnz = self._mat.shape[0] if self._mat is not None else 0

        if self.shape[1] != 1:
            raise ValueError("Input must be a vector")
        # NB: the implementation can be generalized to handle vectors completely
        # even if they are permuted. Our work focused on the most complex operation:
        # the inner product. The outer product is trivial.

    async def dot(self, other):
        if self.shape != other.shape:
            raise ValueError("Incompatible vector size")

        if self.nnz == 0:
            return self.sectype(0)

        if isinstance(other, SparseVector):
            if other.nnz == 0:
                return self.sectype(0)

            unsorted = mpc.np_vstack((self._mat, other._mat))
            sorted_array = await radix_sort(
                unsorted, self.row_bit_length, already_decomposed=True
            )

            n = sorted_array.shape[0]
            mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
            comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
            return mpc.np_sum(mult_vect * comp_vect)
        elif isinstance(other, DenseVector):
            return other.dot(self)
        else:
            raise NotImplementedError

    async def print(self):
        print(await mpc.output(self._mat[:, self.row_bit_length :]))


def from_scipy_sparse_vect(sparse_vect, sectype):
    if sparse_vect.format != "coo":
        sparse_vect = sparse_vect.tocoo()

    if sparse_vect.shape[1] != 1:
        sparse_vect = sparse_vect.T

    row_bit_length = int(math.log(sparse_vect.shape[0], 2)) + 1
    secure_mat = []

    for i, _j, v in zip(sparse_vect.row, sparse_vect.col, sparse_vect.data):
        secure_mat.extend(SecureMatrix.int_to_secure_bits(i, sectype, row_bit_length))
        secure_mat.append(sectype(int(i)))
        secure_mat.append(sectype(float(v)))

    if secure_mat:
        np_mat = mpc.np_reshape(
            mpc.np_fromlist(secure_mat),
            (
                len(secure_mat) // (row_bit_length + 2),
                row_bit_length + 2,
            ),
        )
        secure_mat = mpc.input(np_mat, senders=0)
    else:
        secure_mat = None

    return SparseVector(secure_mat, sparse_vect.shape, sectype)


class SparseMatrixCOO(SecureMatrix):
    def __init__(
        self,
        sparse_mat: List[
            Tuple[mpc.SecureFixedPoint, mpc.SecureFixedPoint, mpc.SecureFixedPoint]
        ],
        sectype=None,
        shape=None,
    ):
        super().__init__(sectype, shape)

        for tup in sparse_mat:
            assert len(tup) == 3
        self._mat = sparse_mat
        self.nnz = len(self._mat)

    def dot(self, other) -> "SparseMatrixCOO":
        raise NotImplementedError

    async def print(self):
        for i in range(len(self._mat)):
            print(
                await mpc.output(self._mat[i][0]),
                await mpc.output(self._mat[i][1]),
                await mpc.output(self._mat[i][2]),
            )


class SparseMatrixColumn(SecureMatrix):
    def __init__(self, list_of_secure_arrays, shape, sectype=None):
        super().__init__(sectype, shape)
        self._mat = list_of_secure_arrays
        # ADVICE: generate list_of_secure_arrays using the function from_scipy_sparse_mat

    def get_column(self, k):
        return self._mat[k]

    async def dot(self, other) -> SparseMatrixCOO:
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseMatrixRow):
            res = None

            sorting_key_length = self.row_bit_length + other.col_bit_length

            for k in range(self.shape[1]):
                curr_row = self.get_column(k)
                curr_col = other.get_row(k)

                if curr_row.nnz == 0 or curr_col.nnz == 0:
                    continue

                coord_mat = []
                for i in range(curr_row.nnz):
                    curr_decomp_i = curr_row._mat[i, :-2]
                    curr_i = mpc.np_fromlist([curr_row._mat[i, -2]])
                    for j in range(curr_col.nnz):
                        curr_decomp_j = curr_col._mat[j, :-2]
                        curr_j = mpc.np_fromlist([curr_col._mat[j, -2]])
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
                    mpc.np_outer(curr_row._mat[:, -1], curr_col._mat[:, -1])
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
        elif isinstance(SparseVector, other):
            return await _matrix_vector_prod(self, other)
        raise NotImplementedError

    def transpose(self) -> "SparseMatrixRow":
        return SparseMatrixRow(self._mat, (self.shape[1], self.shape[0]), self.sectype)


class SparseMatrixRow(SecureMatrix):
    def __init__(self, list_of_secure_arrays, shape, sectype=None):
        super().__init__(sectype, shape)
        self._mat = list_of_secure_arrays
        # ADVICE: generate list_of_secure_arrays using the static method from_scipy_sparse_mat

    def get_row(self, k):
        return self._mat[k]

    async def dot(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("Invalid dimensions")
        if self.sectype != other.sectype:
            raise ValueError("Incompatible secure types")

        if isinstance(other, SparseVector):
            return await _matrix_vector_prod(self, other)
        raise NotImplementedError

    def transpose(self) -> SparseMatrixColumn:
        return SparseMatrixColumn(
            self._mat, (self.shape[1], self.shape[0]), self.sectype
        )


async def _matrix_vector_prod(mat, vect) -> SparseVector:
    assert isinstance(SparseVector, vect)

    if vect.nnz == 0:
        return SparseVector(None, shape=(mat.shape[0], 1), sectype=mat.sectype)

    ### NUMPY-LIKE MATRIX PREPARATION (i.e., for parallelized operations)
    padded_matrix = []

    if isinstance(SparseMatrixRow, mat):
        for i in range(mat.shape[0]):
            curr_row = mat.get_row(i)

            if curr_row.nnz == 0:
                continue

            bin_row_ind = (
                SecureMatrix.int_to_secure_bits(i, mat.sectype, mat.row_bit_length)
                * curr_row.nnz
            )
            bin_row_ind = mpc.np_reshape(
                mpc.np_fromlist(bin_row_ind),
                (len(bin_row_ind) // mat.row_bit_length, mat.row_bit_length),
            )
            int_row_ind = mat.sectype.array(i * np.ones((curr_row.nnz, 1), dtype=int))
            ones = mat.sectype.array(np.ones((curr_row.nnz, 1), dtype=int))
            curr_row_mat = mpc.np_hstack(
                (  # We place the column binary representation first to sort based on columns
                    ones,
                    curr_row._mat[:, : mat.col_bit_length],
                    curr_row._mat[:, -2:-1],  # Integer column indices
                    bin_row_ind,
                    int_row_ind,
                    curr_row._mat[:, -1:],
                )
            )

            padded_matrix.append(curr_row_mat)
    elif isinstance(SparseMatrixColumn, mat):
        for j in range(mat.shape[0]):
            curr_col = mat.get_column(j)

            if curr_col.nnz == 0:
                continue

            bin_col_ind = (
                SecureMatrix.int_to_secure_bits(j, mat.sectype, mat.col_bit_length)
                * curr_col.nnz
            )
            bin_col_ind = mpc.np_reshape(
                mpc.np_fromlist(bin_col_ind),
                (len(bin_col_ind) // mat.col_bit_length, mat.col_bit_length),
            )
            int_col_ind = mat.sectype.array(j * np.ones((curr_col.nnz, 1), dtype=int))
            ones = mat.sectype.array(np.ones((curr_col.nnz, 1), dtype=int))
            curr_col_mat = mpc.np_hstack(
                (  # We place the column binary representation first to sort based on columns
                    ones,
                    bin_col_ind,
                    int_col_ind,
                    curr_col._mat[:, : mat.row_bit_length],
                    curr_col._mat[:, -2:-1],  # Integer row indices
                    curr_col._mat[:, -1:],
                )
            )

            padded_matrix.append(curr_col_mat)

    if not padded_matrix:
        return SparseVector(None, shape=(mat.shape[0], 1), sectype=mat.sectype)

    padded_matrix = mpc.np_vstack(
        padded_matrix
    )  # We concatenate (vertically) all row/column matrices

    zeros_for_vect = mat.sectype.array(np.zeros((vect.nnz, 1), dtype=int))
    bin_placeholders_for_vect = mat.sectype.array(
        np.zeros((vect.nnz, mat.row_bit_length), dtype=int)
    )
    placeholder_for_vect = mat.sectype.array(
        -np.ones((vect.nnz, 1), dtype=int)
    )  # Replace the integer row indices by placeholders so they can be removed at the end of the function

    padded_vect = mpc.np_hstack(
        (
            zeros_for_vect,
            vect._mat[:, : mat.col_bit_length],
            vect._mat[:, -2:-1],
            bin_placeholders_for_vect,
            placeholder_for_vect,
            vect._mat[:, -1:],
        )
    )
    # We add an extra zero bit to differentiate the matrix values from the vector values

    res = mpc.np_vstack((padded_vect, padded_matrix))

    ### MULTIPLICATION STEP
    if len(mpc.parties) == 3:
        res = await radix_sort(
            res,
            mat.col_bit_length + 1,
            already_decomposed=True,
            keep_bin_keys=True,
        )
    else:
        raise NotImplementedError

    last_vect_val = mat.sectype(0)
    last_vect_col = mat.sectype(-1)
    val_col = []
    for i in range(res.shape[0]):
        is_vect_elem = 1 - res[i, 0]
        same_col = res[i, mat.col_bit_length + 1] == last_vect_col
        mult_cond = same_col * res[i, 0]

        last_vect_val = (1 - is_vect_elem) * last_vect_val + is_vect_elem * res[i, -1]
        last_vect_col = (1 - is_vect_elem) * last_vect_col + is_vect_elem * res[
            i, mat.col_bit_length + 1
        ]
        val_col.append(res[i, -1] * last_vect_val * mult_cond)

    val_col = mpc.np_reshape(mpc.np_fromlist(val_col), (res.shape[0], 1))

    res = mpc.np_hstack((res[:, 0:1], res[:, mat.col_bit_length + 2 : -1], val_col))
    # We remove the column indices but keep the differentiating bit

    # AGGREGATION STEP
    if len(mpc.parties) == 3:
        res = await radix_sort(
            res,
            mat.row_bit_length + 1,
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
        raise NotImplementedError

    final_res = []
    zero_test = await mpc.np_is_zero_public(
        res[:, -2] + 1
    )  # Here, we leak the number of non-zero elements in the output matrix
    zero_val_test = await mpc.np_is_zero_public(res[:, -1])  # TODO: refactor this test?

    mask = [i for i, test in enumerate(zero_test) if not test and not zero_val_test[i]]
    final_res = res[mask, 1:]  # We remove the differentiating bit

    return SparseVector(final_res, sectype=mat.sectype, shape=(mat.shape[0], 1))


def from_scipy_sparse_mat(sparse_mat: ScipySparseMatType, sectype, leakage_axis=0):
    assert leakage_axis in [0, 1]

    if leakage_axis == 0:
        bit_length = int(math.log(sparse_mat.shape[1], 2)) + 1
    else:
        bit_length = int(math.log(sparse_mat.shape[0], 2)) + 1

    secure_mat = []
    sparse_mat = sparse_mat.tocsr()
    for i in range(sparse_mat.shape[leakage_axis]):
        curr_vect = sparse_mat[i, :].T if leakage_axis == 0 else sparse_mat[:, i]
        secure_mat.append(from_scipy_sparse_vect(curr_vect, sectype))

    retcls = SparseMatrixRow if leakage_axis == 0 else SparseMatrixColumn
    return retcls(secure_mat, sparse_mat.shape, sectype)
