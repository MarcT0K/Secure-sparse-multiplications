import math

from mpyc.runtime import mpc

from matrices import DenseMatrix, SecureMatrix
from radix_sort import radix_sort


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


class SparseVector(SecureMatrix):
    def __init__(self, sparse_mat, sectype=None):
        if sparse_mat.shape[1] != 1:
            raise ValueError("Input must be a vector")

        super().__init__(sectype)
        self.shape = sparse_mat.shape

        self.key_bit_length = int(math.log(self.shape[0], 2)) + 1

        self._mat = []
        for i, _j, v in zip(sparse_mat.row, sparse_mat.col, sparse_mat.data):
            self._mat.extend(
                SparseVector.int_to_secure_bits(i, self.sectype, self.key_bit_length)
            )
            self._mat.append(SparseVector.to_secint(self.sectype, i))
            self._mat.append(SparseVector.to_secint(self.sectype, v))

        if self._mat:
            np_mat = mpc.np_reshape(
                mpc.np_fromlist(self._mat),
                (len(self._mat) // (self.key_bit_length + 2), self.key_bit_length + 2),
            )
            self._mat = mpc.input(np_mat, senders=0)

    @staticmethod
    def to_secint(sectype, x):
        return sectype(int(x))

    @staticmethod
    def int_to_secure_bits(number, sectype, nb_bits):
        bitstring = format(number, f"0{nb_bits}b")
        return [sectype(int(c)) for c in bitstring][::-1]

    async def dot(self, other):
        if isinstance(other, SparseVector):
            if self.shape != other.shape:
                raise ValueError("Incompatible vector size")

            if not self._mat or not other._mat:
                return self.sectype(0)

            unsorted = mpc.np_vstack((self._mat, other._mat))
            sorted_array = await radix_sort(
                unsorted, self.key_bit_length, already_decomposed=True
            )

            n = sorted_array.shape[0]
            mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
            comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
            return mpc.np_sum(mult_vect * comp_vect)
        else:
            raise NotImplementedError

    async def print(self):
        print(await mpc.output(self._mat[:, self.key_bit_length :]))
