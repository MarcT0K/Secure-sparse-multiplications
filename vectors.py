from matrices import DenseMatrix, DenseMatrixNumpy, SecureMatrix
from sparse_dot_vector import *


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


class DenseVectorNumpy(DenseMatrixNumpy):
    def __init__(self, mat, sectype=None):
        if mat.shape[1] != 1 and mat.shape[0] != 1:
            raise ValueError("Input must be a vector")
        super().__init__(mat, sectype)

    def dot(self, other):
        if isinstance(other, DenseVectorNumpy):
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
            self._mat.append(to_sec_int(i))
            self._mat.append(to_sec_int(v))
        self._mat = mpc.input(self._mat, senders=0)
        self._mat = [
            [self._mat[i], self._mat[i + 1]] for i in range(0, len(self._mat) // 2, 2)
        ]

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
        to_sec_int = lambda x: mpc.input(self.sectype(int(x)), senders=0)

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
