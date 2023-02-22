from abc import abstractmethod
from typing import List, Union, Optional, Tuple

import numpy as np
import scipy.sparse

from mpyc.runtime import mpc

from datetime import datetime

from sparse_dot_vector import *
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


class DenseVectorNaive(DenseVector):
    def __init__(self, mat, sectype=None):
        super().__init__(mat, sectype)

    def dot(self, other):
        if isinstance(other, DenseVector):
            res = self._mat[0][0] * other._mat[0][0]
            for i in range(1, self.shape[1]):
                res += self._mat[0][i] * other._mat[i][0]
            return res
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


class SparseVectorNumpy(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)
        np_mat = []
        for i in range(len(self._mat)):
            np_mat += self._mat[i]
        np_mat = mpc.np_reshape(mpc.np_fromlist(np_mat), (len(self._mat), 2))
        self._mat = np_mat

    def dot(self, other):
        if isinstance(other, SparseVectorNumpy):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return sparse_vector_dot_np(self._mat, other._mat)
        else:
            raise NotImplementedError


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


class SparseVectorNaiveOpti(SparseVector):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)

    def dot(self, other):
        if isinstance(other, SparseVectorNaiveOpti):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return sparse_vector_dot_naive_opti(self._mat, other._mat)
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


class SparseVectorParallelQuicksort(SparseVectorNumpy):
    def __init__(self, sparse_mat, sectype=None):
        super().__init__(sparse_mat, sectype)

    async def dot(self, other):
        if isinstance(other, SparseVectorParallelQuicksort):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")
            return await sparse_vector_dot_parallel_quicksort(
                self._mat, other._mat, self.sectype, key=lambda tup: tup[0]
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


async def benchmark_dot_product(n_dim=10**5, density=0.001):
    print("Sparse dot benchmark: n=", n_dim, " density=", density)
    secint = mpc.SecInt(64)

    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, 1, density=density, dtype=np.int16
        ).astype(int)
        y_sparse = scipy.sparse.random(
            n_dim, 1, density=density, dtype=np.int16
        ).astype(int)
    else:
        x_sparse = None
        y_sparse = None

    x_sparse = await mpc.transfer(x_sparse, senders=0)
    y_sparse = await mpc.transfer(y_sparse, senders=0)

    dense_x = x_sparse.astype(int).todense()
    dense_y = y_sparse.astype(int).todense()
    real_res = dense_x.transpose().dot(dense_y)[0, 0]
    print("Real result:", real_res)

    sec_dense_x = DenseVector(dense_x.transpose(), sectype=secint)
    sec_dense_y = DenseVector(dense_y, sectype=secint)
    start = datetime.now()
    z = sec_dense_x.dot(sec_dense_y)
    assert await mpc.output(z) == real_res
    end = datetime.now()
    delta_dense = end - start
    print("Time for dense:", delta_dense.total_seconds())

    # sec_dense_x = DenseVectorNaive(dense_x.transpose(), sectype=secint)
    # sec_dense_y = DenseVectorNaive(dense_y, sectype=secint)
    # start = datetime.now()
    # z = sec_dense_x.dot(sec_dense_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_dense = end - start
    # print("Time for dense unoptimized:", delta_dense.total_seconds())

    # sec_x = SparseVector(x_sparse, secint)
    # sec_y = SparseVector(y_sparse, secint)
    # start = datetime.now()
    # z = sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse:", delta_sparse.total_seconds())

    sec_x = SparseVectorNumpy(x_sparse, secint)
    sec_y = SparseVectorNumpy(y_sparse, secint)
    start = datetime.now()
    z = sec_x.dot(sec_y)
    assert await mpc.output(z) == real_res
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse with np optim.:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaive(x_sparse, secint)
    # sec_y = SparseVectorNaive(y_sparse, secint)
    # start = datetime.now()
    # z = sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse naive:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaiveOpti(x_sparse, secint)
    # sec_y = SparseVectorNaiveOpti(y_sparse, secint)
    # start = datetime.now()
    # z = sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse naive opti:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaivePSI(x_sparse, secint)
    # sec_y = SparseVectorNaivePSI(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse psi:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaivePSIOpti(x_sparse, secint)
    # sec_y = SparseVectorNaivePSIOpti(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse psi optimized:", delta_sparse.total_seconds())

    # sec_x = SparseVectorORAM(x_sparse, secint)
    # sec_y = SparseVectorORAM(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert(await mpc.output(z) == real_res)
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse ORAM:", delta_sparse.total_seconds())

    # sec_x = SparseVectorQuicksort(x_sparse, secint)
    # sec_y = SparseVectorQuicksort(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse quicksort:", delta_sparse.total_seconds())

    sec_x = SparseVectorParallelQuicksort(x_sparse, secint)
    sec_y = SparseVectorParallelQuicksort(y_sparse, secint)
    start = datetime.now()
    z = await sec_x.dot(sec_y)
    assert await mpc.output(z) == real_res
    end = datetime.now()
    delta_sparse = end - start
    print("Time for sparse parallel quicksort:", delta_sparse.total_seconds())
    print("===END")


async def benchmark_sparse_sparse_mat_mult(n_dim=1000, m_dim=10**5, sparsity=0.001):
    secint = mpc.SecInt(64)
    print(
        f"Started experiment with sparse matrix multiplication ({n_dim}x{m_dim}), sparsity={sparsity}"
    )
    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, m_dim, density=sparsity, dtype=np.int16
        ).astype(int)
    else:
        x_sparse = None

    x_sparse = await mpc.transfer(x_sparse, senders=0)

    dense_mat = x_sparse.astype(int).todense()
    sec_dense_t = DenseMatrix(dense_mat.transpose(), sectype=secint)
    sec_dense = DenseMatrix(dense_mat, sectype=secint)

    start = datetime.now()
    z = sec_dense_t.dot(sec_dense)
    await mpc.output(z.get(0, 0))
    end = datetime.now()
    delta_dense = end - start
    print("Time for dense:", delta_dense.total_seconds())

    # sec_dense_t = DenseMatrixNaive(dense_mat.transpose(), sectype=secint)
    # sec_dense = DenseMatrixNaive(dense_mat, sectype=secint)

    # start = datetime.now()
    # z = sec_dense_t.dot(sec_dense)
    # end = datetime.now()
    # delta_dense = end - start
    # print("Time for dense naive:", delta_dense.total_seconds())

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


async def main():
    await mpc.start()
    await benchmark_dot_product()
    # await benchmark_sparse_sparse_mat_mult(m_dim=1000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=10000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=100000)
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main())


# Started experiment with sparse matrix multiplication (1000x1000), sparsity=0.001
# Time for dense: 139.481098
# Time for sparse with batcher sort: 170.852584
