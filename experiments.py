from datetime import datetime

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from matrices import (
    DenseMatrix,
    DenseMatrixNaive,
    DenseMatrixNumpy,
    SparseMatrixColumn,
    SparseMatrixRow,
    SparseMatrixRowNumpy,
    SparseMatrixColumnNumpy,
)

from vectors import (
    DenseVector,
    DenseVectorNumpy,
    SparseVector,
    SparseVectorNaive,
    SparseVectorNaiveOpti,
    SparseVectorNaivePSI,
    SparseVectorNaivePSIOpti,
    SparseVectorNumpy,
    SparseVectorORAM,
    SparseVectorParallelQuicksort,
    SparseVectorQuicksort,
)


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

    # sec_dense_x = DenseVector(dense_x.transpose(), sectype=secint)
    # sec_dense_y = DenseVector(dense_y, sectype=secint)
    # start = datetime.now()
    # z = sec_dense_x.dot(sec_dense_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_dense = end - start
    # print("Time for dense:", delta_dense.total_seconds())

    sec_dense_x = DenseVectorNumpy(dense_x.transpose(), sectype=secint)
    sec_dense_y = DenseVectorNumpy(dense_y, sectype=secint)
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

    # sec_x = SparseVectorParallelQuicksort(x_sparse, secint)
    # sec_y = SparseVectorParallelQuicksort(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse parallel quicksort:", delta_sparse.total_seconds())
    # print("===END")


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
    dense_mat = x_sparse.todense().astype(int)

    # sec_dense_t = DenseMatrix(dense_mat.transpose(), sectype=secint)
    # sec_dense = DenseMatrix(dense_mat, sectype=secint)

    # start = datetime.now()
    # z = sec_dense_t.dot(sec_dense)
    # await mpc.output(z.get(0, 0))
    # await mpc.barrier()
    # end = datetime.now()
    # delta_dense = end - start
    # print("Time for dense:", delta_dense.total_seconds())

    # sec_dense_t = DenseMatrixNaive(dense_mat.transpose(), sectype=secint)
    # sec_dense = DenseMatrixNaive(dense_mat, sectype=secint)

    # start = datetime.now()
    # z = sec_dense_t.dot(sec_dense)
    # end = datetime.now()
    # delta_dense = end - start
    # print("Time for dense naive:", delta_dense.total_seconds())

    sec_dense_t = DenseMatrixNumpy(dense_mat.transpose(), sectype=secint)
    sec_dense = DenseMatrixNumpy(dense_mat, sectype=secint)

    start = datetime.now()
    z = sec_dense_t.dot(sec_dense)
    await mpc.output(z.get(0, 0))
    await mpc.barrier()
    end = datetime.now()
    delta_dense = end - start
    print("Time for dense with numpy optimization:", delta_dense.total_seconds())

    sec_x = SparseMatrixColumnNumpy(x_sparse.transpose(), secint)
    sec_y = SparseMatrixRowNumpy(x_sparse, secint)

    start = datetime.now()
    z = await sec_x.dot(sec_y)
    await mpc.barrier()
    end = datetime.now()
    delta_sparse = end - start
    print(
        "Time for sparse with numpy-optimized batcher sort:",
        delta_sparse.total_seconds(),
    )

    # sec_x = SparseMatrixColumn(x_sparse.transpose(), secint)
    # sec_y = SparseMatrixRow(x_sparse, secint)

    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # await mpc.barrier()
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse with batcher sort:", delta_sparse.total_seconds())

    print("=== END")


async def main():
    await mpc.start()
    await benchmark_dot_product(n_dim=10**3)
    await benchmark_dot_product(n_dim=10**4)
    await benchmark_dot_product(n_dim=10**5)
    await benchmark_dot_product(n_dim=10**6)
    # await benchmark_sparse_sparse_mat_mult(m_dim=100)
    # await benchmark_sparse_sparse_mat_mult(m_dim=500)
    # await benchmark_sparse_sparse_mat_mult(m_dim=1000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=5000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=10000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=100000)
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main())


# Started experiment with sparse matrix multiplication (1000x1000), sparsity=0.001
# Time for dense: 139.481098
# Time for sparse with batcher sort: 170.852584
