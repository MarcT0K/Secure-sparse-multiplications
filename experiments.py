from contextlib import asynccontextmanager
from csv import DictWriter
from datetime import datetime
from typing import Optional
from itertools import product

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


class ExperimentalEnvironment:
    def __init__(self, filename, csv_fields):
        if mpc.pid == 0:
            self._file = open(filename, "w", encoding="utf-8")
            self._csv_writer = DictWriter(self._file, csv_fields)
            self._csv_writer.writeheader()

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
        await mpc.start()

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

    @asynccontextmanager
    async def benchmark(self, parameters):
        await mpc.barrier(f"Before Experiment {parameters['Algorithm']}")

        start_ts = datetime.now()
        start_bytes = ExperimentalEnvironment.current_sent_bytes()
        try:
            yield self
            await mpc.barrier(f"Experiment {parameters['Algorithm']}")
        except MemoryError:
            parameters["Memory overflow"] = True
        else:
            end_ts = datetime.now()
            end_bytes = ExperimentalEnvironment.current_sent_bytes()
            parameters["Memory overflow"] = False
            parameters["Runtime"] = (end_ts - start_ts).total_seconds()
            parameters["Communication cost"] = end_bytes - start_bytes

        if mpc.pid == 0:
            self._csv_writer.writerow(parameters)
            self._file.flush()
        print(f"Algorithm {parameters['Algorithm']} DONE")

    async def end(self):
        await mpc.shutdown()

        if mpc.pid == 0:
            self._file.close()
        resource.setrlimit(resource.RLIMIT_AS, (self.prev_soft, self.prev_hard))

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_args):
        await self.end()


async def benchmark_dot_product(exp_env, n_dim=10**5, density=0.001):
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

    params = {
        "Algorithm": "Dense sharing",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        sec_dense_x = DenseVectorNumpy(dense_x.transpose(), sectype=secint)
        sec_dense_y = DenseVectorNumpy(dense_y, sectype=secint)

    for i in range(1):
        params = {
            "Algorithm": "Dense",
            "Nb. rows": n_dim,
            "Nb. columns": 1,
            "Density": density,
        }
        async with exp_env.benchmark(params):
            z = sec_dense_x.dot(sec_dense_y)
            assert await mpc.output(z) == real_res

    del dense_x, dense_y

    # params = {
    #     "Algorithm": "Sparse sharing",
    #     "Nb. rows": n_dim,
    #     "Nb. columns": 1,
    #     "Density": density,
    # }
    # async with exp_env.benchmark(params):
    #     sec_x = SparseVectorNumpy(x_sparse, secint)
    #     sec_y = SparseVectorNumpy(y_sparse, secint)

    # params = {
    #     "Algorithm": "Sparse w/ Batcher",
    #     "Nb. rows": n_dim,
    #     "Nb. columns": 1,
    #     "Density": density,
    # }
    # for i in range(1):
    #     async with exp_env.benchmark(params):
    #         z = sec_x.dot(sec_y)
    #         assert await mpc.output(z) == real_res

    # sec_x = SparseVectorNaiveOpti(x_sparse, secint)
    # sec_y = SparseVectorNaiveOpti(y_sparse, secint)
    # start = datetime.now()
    # z = sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse naive opti:", delta_sparse.total_seconds())

    # sec_x = SparseVectorNaivePSIOpti(x_sparse, secint)
    # sec_y = SparseVectorNaivePSIOpti(y_sparse, secint)
    # start = datetime.now()
    # z = await sec_x.dot(sec_y)
    # assert await mpc.output(z) == real_res
    # end = datetime.now()
    # delta_sparse = end - start
    # print("Time for sparse psi optimized:", delta_sparse.total_seconds())

    params = {
        "Algorithm": "Sparse sharing",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        sec_x = SparseVectorParallelQuicksort(x_sparse, secint)
        sec_y = SparseVectorParallelQuicksort(y_sparse, secint)

    params = {
        "Algorithm": "Sparse w/ Quicksort",
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }
    for i in range(1):
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)
            assert await mpc.output(z) == real_res

    print("=== END")


async def benchmark_sparse_sparse_mat_mult(
    exp_env, n_dim=1000, m_dim=10**5, density=0.001
):
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

    params = {
        "Algorithm": "Dense sharing",
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        sec_dense_t = DenseMatrixNumpy(dense_mat.transpose(), sectype=secint)
        sec_dense = DenseMatrixNumpy(dense_mat, sectype=secint)
    del dense_mat

    params = {
        "Algorithm": "Dense",
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        z = sec_dense_t.dot(sec_dense)
        assert z._mat.shape == (m_dim, m_dim)
    z_clear = await mpc.output(z._mat)
    nb_non_zeros = (z_clear != 0).sum()

    del sec_dense_t, sec_dense, z, z_clear

    params = {
        "Algorithm": "Sparse sharing",
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        sec_x = SparseMatrixColumnNumpy(x_sparse.transpose(), secint)
        sec_y = SparseMatrixRowNumpy(x_sparse, secint)

    params = {
        "Algorithm": "Sparse w/ Quicksort",
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    async with exp_env.benchmark(params):
        z = await sec_x.dot(sec_y)

    nb_non_zeros2 = len(z._mat)
    assert nb_non_zeros == nb_non_zeros2  #
    print("=== END")


async def main():
    async with ExperimentalEnvironment("dot_product.csv", CSV_FIELDS) as exp_env:
        for i, j, density in product(range(3, 6), range(1, 10), [0.001, 0.005, 0.01]):
            await benchmark_dot_product(exp_env, n_dim=j * 10**i, density=density)

    async with ExperimentalEnvironment("mat_mult.csv", CSV_FIELDS) as exp_env:
        for i, j, density in product(range(2, 4), range(1, 10, 2), [0.001]):
            await benchmark_sparse_sparse_mat_mult(
                exp_env, n_dim=10**2, m_dim=j * 10**i, density=density
            )


if __name__ == "__main__":
    mpc.run(main())
