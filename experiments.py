from contextlib import asynccontextmanager
from csv import DictWriter
from datetime import datetime
from typing import Optional

import resource


import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from matrices import (
    DenseMatrix,
    DenseMatrixNaive,
    DenseMatrixNumpy,
    SparseMatrixColumn,
    SparseMatrixColumnNumpy,
    SparseMatrixRow,
    SparseMatrixRowNumpy,
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

CSV_FIELDS = [
    "Algorithm",
    "Nb. rows",
    "Nb. columns",
    "Density",
    "Memory overflow",
    "Runtime",
    "Communication cost",
]

active_exp_env: Optional["ExperimentEnvironment"] = None


class ExperimentEnvironment:
    def __init__(self, filename, csv_fields):
        if mpc.pid == 0:
            self._file = open(filename, "w", encoding="utf-8")
            self._csv_writer = DictWriter(self._file, csv_fields)
            self._csv_writer.writeheader()

        self._experiment_count = 0
        self.prev_soft, self.prev_hard = resource.getrlimit(resource.RLIMIT_AS)

    @staticmethod
    def current_sent_bytes():
        return sum(
            [
                peer.protocol.nbytes_sent if peer.pid != mpc.pid else 0
                for peer in mpc.parties
            ]
        )

    async def start(self):
        global active_exp_env
        if active_exp_env is not None:
            raise ValueError("There is already an ongoing experiment")
        active_exp_env = self

        with open("/proc/meminfo", "r") as mem:  # We estimate the free memory
            free_memory = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                    free_memory += int(sline[1])

        resource.setrlimit(  # We limite the memory usage to prevent memory overflow from crashing the PC
            resource.RLIMIT_AS,
            (int(free_memory * 1024 * 0.9 / len(mpc.parties)), self.prev_hard),
        )

        await mpc.start()

    @asynccontextmanager
    async def benchmark(self, parameters):
        self._experiment_count += 1
        start_ts = datetime.now()
        start_bytes = ExperimentEnvironment.current_sent_bytes()
        try:
            yield self
            # await mpc.barrier(f"Experiment {self._experiment_count}")
        except MemoryError:
            parameters["Memory overflow"] = True
        else:
            end_ts = datetime.now()
            end_bytes = ExperimentEnvironment.current_sent_bytes()
            parameters["Memory overflow"] = False
            parameters["Runtime"] = (end_ts - start_ts).total_seconds()
            parameters["Communication cost"] = end_bytes - start_bytes

        if mpc.pid == 0:
            self._csv_writer.writerow(parameters)
            self._file.flush()
        print(f"Algorithm {parameters['Algorithm']} DONE")

    @asynccontextmanager
    async def control_memory_usage(self):
        try:
            yield self
        except MemoryError:
            raise ValueError

    async def shutdown(self):
        await mpc.shutdown()
        self._file.close()
        resource.setrlimit(resource.RLIMIT_AS, (self.prev_soft, self.prev_hard))

        global active_exp_env
        active_exp_env = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_args):
        await self.shutdown()


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

    async with active_exp_env.control_memory_usage():
        sec_dense_x = DenseVectorNumpy(dense_x.transpose(), sectype=secint)
        sec_dense_y = DenseVectorNumpy(dense_y, sectype=secint)

    params = {
        "Algorithm": "Dense",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    async with active_exp_env.benchmark(params):
        z = sec_dense_x.dot(sec_dense_y)
        assert await mpc.output(z) == real_res

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
    params = {
        "Algorithm": "Sparse w/ Batcher",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    async with active_exp_env.benchmark(params):
        z = sec_x.dot(sec_y)
        assert await mpc.output(z) == real_res

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
    params = {
        "Algorithm": "Sparse w/ Quicksort",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    async with active_exp_env.benchmark(params):
        z = await sec_x.dot(sec_y)
        assert await mpc.output(z) == real_res

    print("===END")


async def benchmark_sparse_sparse_mat_mult(n_dim=1000, m_dim=10**5, density=0.001):
    secint = mpc.SecInt(64)
    print(
        f"Started experiment with sparse matrix multiplication ({n_dim}x{m_dim}), density={density}"
    )
    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, m_dim, density=density, dtype=np.int16
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

    params = {
        "Algorithm": "Dense",
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    async with active_exp_env.benchmark(params):
        z = sec_dense_t.dot(sec_dense)
    print("Dense with numpy optimization: DONE")

    sec_x = SparseMatrixColumnNumpy(x_sparse.transpose(), secint)
    sec_y = SparseMatrixRowNumpy(x_sparse, secint)

    start = datetime.now()
    z = await sec_x.dot(sec_y)
    # await z.print()
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
    async with ExperimentEnvironment("dot_product.csv", CSV_FIELDS):
        for i in range(3, 6):
            for j in range(1, 10):
                await benchmark_dot_product(n_dim=j * 10**i, density=0.001)

    # await benchmark_sparse_sparse_mat_mult(m_dim=100)
    # await benchmark_sparse_sparse_mat_mult(m_dim=500)
    # await benchmark_sparse_sparse_mat_mult(m_dim=1000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=5000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=10000)
    # await benchmark_sparse_sparse_mat_mult(m_dim=100000)


if __name__ == "__main__":
    mpc.run(main())


# Started experiment with sparse matrix multiplication (1000x1000), density=0.001
# Time for dense: 139.481098
# Time for sparse with batcher sort: 170.852584
