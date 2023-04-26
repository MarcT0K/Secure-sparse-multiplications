import argparse
import random

from contextlib import asynccontextmanager
from csv import DictWriter
from datetime import datetime

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from matrices import (
    DenseMatrixNumpy,
    SparseMatrixColumnNumpy,
    SparseMatrixRowNumpy,
)
from vectors import (
    DenseVectorNumpy,
    SparseVectorParallelQuicksort,
)

from resharing import np_shuffle_3PC
from shuffle import np_shuffle

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Nb. rows",
    "Nb. columns",
    "Density",
    "Memory overflow",
    "Runtime",
    "Communication cost",
]


class ExperimentalEnvironment:
    def __init__(self, filename, csv_fields, seed=None):
        if mpc.pid == 0:
            self._file = open(filename, "a", encoding="utf-8")
            self._csv_writer = DictWriter(self._file, csv_fields)
            if self._file.tell() == 0:
                self._csv_writer.writeheader()

        self.seed = seed
        if seed is None:
            self.seed = int.from_bytes(random.randbytes(4), "big")
        random.seed(self.seed)
        np.random.seed(self.seed)

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

    @asynccontextmanager
    async def benchmark(self, parameters):
        await mpc.barrier(f"Before Experiment {parameters['Algorithm']}")

        parameters["Initial seed"] = await mpc.transfer(self.seed)

        start_ts = datetime.now()
        start_bytes = ExperimentalEnvironment.current_sent_bytes()
        parameters["Timestamp"] = start_ts
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

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_args):
        await self.end()


async def benchmark_dot_product(exp_env, n_dim, density, alg_choice=None):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "dense", "sparse"]

    print(
        f"Dot product benchmark: dimension={n_dim}, density={density} algorithm={alg_choice}"
    )
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
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }

    if alg_choice in ["*", "dense"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_dense_x = DenseVectorNumpy(dense_x.transpose(), sectype=secint)
            sec_dense_y = DenseVectorNumpy(dense_y, sectype=secint)

        params["Algorithm"] = "Dense"
        async with exp_env.benchmark(params):
            z = sec_dense_x.dot(sec_dense_y)
            assert await mpc.output(z) == real_res

        del dense_x, dense_y

    if alg_choice in ["*", "sparse"]:
        params["Algorithm"] = "Sparse sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseVectorParallelQuicksort(x_sparse, secint)
            sec_y = SparseVectorParallelQuicksort(y_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)
            assert await mpc.output(z) == real_res


async def benchmark_sparse_sparse_mat_mult(
    exp_env, n_dim, m_dim, density, alg_choice=None
):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "dense", "sparse"]

    secint = mpc.SecInt(64)
    print(
        f"Matrix multiplication benchmark: dimensions={n_dim}x{m_dim}, density={density}, algorithm={alg_choice}"
    )
    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, m_dim, density=density, dtype=np.int16
        ).astype(int)
    else:
        x_sparse = None

    x_sparse = await mpc.transfer(x_sparse, senders=0)
    nb_non_zeros = len((x_sparse.T @ x_sparse).data)
    dense_mat = x_sparse.todense().astype(int)

    params = {
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    if alg_choice in ["dense", "*"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_dense_t = DenseMatrixNumpy(dense_mat.transpose(), sectype=secint)
            sec_dense = DenseMatrixNumpy(dense_mat, sectype=secint)
        del dense_mat

        params["Algorithm"] = "Dense"
        async with exp_env.benchmark(params):
            z = sec_dense_t.dot(sec_dense)
            assert z._mat.shape == (m_dim, m_dim)
        z_clear = await mpc.output(z._mat)
        dense_nb_non_zeros = (z_clear != 0).sum()
        assert nb_non_zeros == dense_nb_non_zeros

        del sec_dense_t, sec_dense, z, z_clear

    if alg_choice in ["sparse", "*"]:
        params["Algorithm"] = "Sparse sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseMatrixColumnNumpy(x_sparse.transpose(), secint)
            sec_y = SparseMatrixRowNumpy(x_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)

        sparse_nb_non_zeros = len(z._mat)
        assert nb_non_zeros == sparse_nb_non_zeros


async def benchmark_oblivious_shuffle(exp_env, n_dim, alg_choice=None):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "MPyC", "3PC"]

    print(f"Oblivious shuffle benchmark: dimensions={n_dim}, algorithm={alg_choice}")

    secint = mpc.SecInt(64)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 2**62) for _ in range(n_dim)]
    else:
        rand_list = [0] * n_dim

    rand_list = mpc.np_fromlist([mpc.input(secint(i), senders=0) for i in rand_list])
    init_list = await mpc.output(rand_list)

    if alg_choice in ["*", "MPyC"]:
        params = {
            "Algorithm": "MPyC shuffle",
            "Nb. rows": n_dim,
        }
        async with exp_env.benchmark(params):
            await np_shuffle(rand_list)
            new_list = await mpc.output(rand_list)
            assert (new_list != init_list).any()
            assert set(new_list) == set(init_list)

    if alg_choice in ["*", "3PC"]:
        params = {
            "Algorithm": "3PC shuffle",
            "Nb. rows": n_dim,
        }
        async with exp_env.benchmark(params):
            z = await np_shuffle_3PC(rand_list)
            new_list = await mpc.output(z)
            assert (new_list != init_list).any()
            assert set(new_list) == set(init_list)


def check_args(args, fields):
    for field in fields:
        if args[field] is None:
            raise ValueError("Missing value for " + field)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--nb-rows", type=int)
    parser.add_argument("--nb-cols", type=int)
    parser.add_argument("--density", type=float)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--algo")
    args, _rest = parser.parse_known_args()
    args = vars(args)

    if args["benchmark"] == "dot_product":
        check_args(args, ["nb_rows", "density"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_dot_product(
                exp_env,
                n_dim=args["nb_rows"],
                density=args["density"],
                alg_choice=args["algo"],
            )
    elif args["benchmark"] == "mat_mult":
        check_args(args, ["nb_rows", "nb_cols", "density", "algo"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_sparse_sparse_mat_mult(
                exp_env,
                n_dim=args["nb_rows"],
                m_dim=args["nb_cols"],
                density=args["density"],
                alg_choice=args["algo"],
            )
    elif args["benchmark"] == "shuffle":
        check_args(args, ["nb_rows"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_oblivious_shuffle(
                exp_env, n_dim=args["nb_rows"], alg_choice=args["algo"]
            )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    mpc.run(main())
