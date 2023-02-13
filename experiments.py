from abc import abstractmethod
from typing import List, Union, Optional, Tuple

import numpy as np
import scipy.sparse

from mpyc.runtime import mpc

from datetime import datetime

from sparse_dot_vector import *

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
        if len(tup1) > 2:  # The last element is not subject of the sort in our case
            equal_comp = tup1[0] == tup2[0]
            recursive_comp = SortableTuple._lt_tuples(tup1[1:], tup2[1:])
            # first_comp or (recursive_comp and equal_comp)
            return first_comp | (recursive_comp & equal_comp)
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

            # Unoptimized algorithm
            # res = self._mat[0][0] * other._mat[0][0]
            # print("LEN:", self.shape[1])
            # for i in range(self.shape[1]):
            #     res += self._mat[0][i] * other._mat[i][0]
            # return res
        else:
            raise NotImplementedError


class SparseVector(SecureMatrix):
    def __init__(self, sparse_mat, sectype=None):
        if sparse_mat.shape[1] != 1:
            raise ValueError("Input must be a vector")

        super().__init__(sectype)
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))

        self.shape = sparse_mat.shape
        self._mat = []
        for i, _j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat.append([to_sec_int(i), to_sec_int(v)])

    def dot(self, other):
        if isinstance(other, SparseVector):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return sparse_vector_dot(self._mat, other._mat)
        else:
            raise NotImplementedError

    async def print(self):
        for i, v in self._mat:
            print(f"({await mpc.output(i)}, {await mpc.output(v)})")


class SparseVectorNaive(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)

    def dot(self, other):
        if isinstance(other, SparseVectorNaive):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return sparse_vector_dot_naive(self._mat, other._mat)
        else:
            raise NotImplementedError


class SparseVectorNaivePSI(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        if sparse_mat.shape[1] != 1:
            raise ValueError("Input must be a vector")

        super().__init__(sparse_mat, sectype)

    async def dot(self, other):
        if isinstance(other, SparseVectorNaivePSI):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return await sparse_vector_dot_psi(self._mat, other._mat, self.sectype)
        else:
            raise NotImplementedError


class SparseVectorNaivePSIOpti(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)

    async def dot(self, other):
        if isinstance(other, SparseVectorNaivePSIOpti):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return await sparse_vector_dot_psi_opti(self._mat, other._mat, self.sectype)
        else:
            raise NotImplementedError


class SparseVectorQuicksort(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)

    async def dot(self, other):
        if isinstance(other, SparseVectorQuicksort):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return await sparse_vector_dot_quicksort(
                self._mat, other._mat, self.sectype, key=SortableTuple
            )
        else:
            raise NotImplementedError


class SparseVectorORAM(SecureMatrix):
    def __init__(self, sparse_mat, sectype=None):
        if sparse_mat.shape[1] != 1:
            raise ValueError("Input must be a vector")

        super().__init__(sectype)
        self.shape = sparse_mat.shape
        to_sec_int = lambda x: self.sectype(int(x))

        self.shape = sparse_mat.shape
        self._mat = [mpc.seclist([], self.sectype), mpc.seclist([], self.sectype)]
        for i, _j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat[0].append(to_sec_int(i))
            self._mat[1].append(to_sec_int(v))

    async def dot(self, other):
        # TODO: fix => invalid output
        if isinstance(other, SparseVectorORAM):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return sparse_vector_dot_merge(
                self._mat, other._mat, self.sectype, key=SortableTuple
            )
        else:
            raise NotImplementedError

    async def print(self):
        for i, v in self._mat:
            print(f"({await mpc.output(i)}, {await mpc.output(v)})")


async def main():
    n_dim = 1000
    density = 0.1
    secint = mpc.SecInt(64)

    x_sparse = scipy.sparse.random(n_dim, 1, density=density, dtype=np.int16).astype(
        int
    )
    y_sparse = scipy.sparse.random(n_dim, 1, density=density, dtype=np.int16).astype(
        int
    )

    dense_x = x_sparse.astype(int).todense()
    dense_y = y_sparse.astype(int).todense()
    print("Real result:", dense_x.transpose().dot(dense_y)[0, 0])
    sec_dense_x = DenseVector(dense_x.transpose(), sectype=secint)
    sec_dense_y = DenseVector(dense_y, sectype=secint)

    print("===")
    start = datetime.now()
    z = sec_dense_x.dot(sec_dense_y)
    print(await mpc.output(z))
    end = datetime.now()
    delta_dense = end - start
    print("Time for dense:", delta_dense.total_seconds())

    sec_x = SparseVector(x_sparse, secint)
    sec_y = SparseVector(y_sparse, secint)
    start = datetime.now()
    z = sec_x.dot(sec_y)
    print(await mpc.output(z))
    end = datetime.now()
    delta_sparse = end - start
    print("===")
    print("Time for sparse:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaive(x_sparse, secint)
    # sec_y = SparseVectorNaive(y_sparse, secint)
    # start = datetime.now()
    # z = sec_x.dot(sec_y)
    # print(await mpc.output(z))
    # end = datetime.now()
    # delta_sparse = end - start
    # print("===")
    # print("Time for sparse naive:", delta_sparse.total_seconds())

    sec_x = SparseVectorNaivePSI(x_sparse, secint)
    sec_y = SparseVectorNaivePSI(y_sparse, secint)
    print("===")
    start = datetime.now()
    z = await sec_x.dot(sec_y)
    print(await mpc.output(z))
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse psi:", delta_sparse.total_seconds())

    sec_x = SparseVectorNaivePSIOpti(x_sparse, secint)
    sec_y = SparseVectorNaivePSIOpti(y_sparse, secint)
    print("===")
    start = datetime.now()
    z = await sec_x.dot(sec_y)
    print(await mpc.output(z))
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse psi optimized:", delta_sparse.total_seconds())

    # sec_x = SparseVectorORAM(x_sparse, secint)
    # sec_y = SparseVectorORAM(y_sparse, secint)
    # print("===")
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # print(await mpc.output(z))
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse ORAM:", delta_sparse.total_seconds())

    sec_x = SparseVectorQuicksort(x_sparse, secint)
    sec_y = SparseVectorQuicksort(y_sparse, secint)
    print("===")
    start = datetime.now()
    z = await sec_x.dot(sec_y)
    print(await mpc.output(z))
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse quicksort:", delta_sparse.total_seconds())


async def benchmark_sparse_sparse_mat_mult(n_dim, m_dim=100, sparsity=0.001):
    secint = mpc.SecInt(64)
    print("Started experiment with n =", n_dim)
    x_sparse = scipy.sparse.random(
        n_dim, m_dim, density=sparsity, dtype=np.int16
    ).astype(int)

    dense_mat = x_sparse.astype(int).todense()
    sec_dense_t = DenseMatrix(dense_mat.transpose(), sectype=secint)
    sec_dense = DenseMatrix(dense_mat, sectype=secint)

    start = datetime.now()
    z = sec_dense_t.dot(sec_dense)
    end = datetime.now()
    delta_dense = end - start
    print("Time for dense:", delta_dense.total_seconds())

    sec_x = SparseMatrixColumn(x_sparse.transpose(), secint)
    sec_y = SparseMatrixRow(x_sparse, secint)

    start = datetime.now()
    z = await sec_x.dot(sec_y)
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse with batcher sort:", delta_sparse.total_seconds())

    sec_x = SparseMatrixColumn(x_sparse.transpose(), secint, quicksort)
    sec_y = SparseMatrixRow(x_sparse, secint)

    start = datetime.now()
    z = await sec_x.dot(sec_y)
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse with quick sort:", delta_sparse.total_seconds())
    print("=== END")


if __name__ == "__main__":
    mpc.run(main())
    # mpc.run(benchmark_sparse_sparse_mat_mult(1000))
    # mpc.run(benchmark_sparse_sparse_mat_mult(10000))
    # mpc.run(benchmark_sparse_sparse_mat_mult(1000000))

# Results
# Started experiment with n = 1000
# ===
# Time for dense: 4.506629
# ===
# Time for sparse: 3.391966
# Started experiment with n = 10000
# ===
# Time for dense: 75.327001
# ===
# Time for sparse: 179.795847
# Started experiment with n = 100000
# ===
# Time for dense: 725.376901
# ===
# Time for sparse: 20029.160868

# Problems:
# - If we have a comparison x100 more expensive than a multiplication, the dot product will never be profitable for sparsity above 1%
# - Public inequality contrary to public equality cannot bring improvement

# Current directions:
# - public comparison via square root (Secure sqrt: https://eprint.iacr.org/2012/405)
# - sparse-dense multiplication using DORAM
# - merging network to improve multiplications
# - DORAM use for sparse matrix covariance
