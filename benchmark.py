import argparse
import math
import random
from contextlib import asynccontextmanager
from csv import DictWriter
from datetime import datetime

import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from matrices import (
    DenseMatrix,
    DenseVector,
    SparseMatrixColumn,
    SparseMatrixRow,
    SparseVector,
)


CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Nb. rows",
    "Nb. columns",
    "Density",
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

        yield self
        await mpc.barrier(f"Experiment {parameters['Algorithm']}")

        end_ts = datetime.now()
        end_bytes = ExperimentalEnvironment.current_sent_bytes()
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


async def benchmark_vect_mult(exp_env, n_dim, density, alg_choice=None):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "dense", "sparse"]

    print(
        f"Vector multiplication benchmark: dimension={n_dim}, density={density} algorithm={alg_choice}"
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

    x_dense = x_sparse.astype(int).todense()
    y_dense = y_sparse.astype(int).todense()
    real_res = x_dense.transpose().dot(y_dense)[0, 0]
    print("Real result:", real_res)

    params = {
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }

    if alg_choice in ["*", "dense"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_x_dense = DenseVector(x_dense.transpose(), sectype=secint)
            sec_y_dense = DenseVector(y_dense, sectype=secint)

        params["Algorithm"] = "Dense"
        async with exp_env.benchmark(params):
            z = sec_x_dense.dot(sec_y_dense)
            dense_res = await mpc.output(z)
            assert dense_res == real_res

        del x_dense, y_dense

    if alg_choice in ["*", "sparse"]:
        params["Algorithm"] = "Sparse sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseVector(x_sparse, secint)
            sec_y = SparseVector(y_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)
            sparse_res = await mpc.output(z)
            assert sparse_res == real_res


async def benchmark_sparse_dense_vect_mult(exp_env, n_dim, density, alg_choice=None):
    if alg_choice is None:
        alg_choice = "*"

    assert alg_choice in ["*", "dense", "sparse", "sparse-dense"]

    print(
        f"Vector multiplication benchmark: dimension={n_dim}, density={density} algorithm={alg_choice}"
    )
    secint = mpc.SecInt(64)

    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, 1, density=density, dtype=np.int16
        ).astype(int)
        y_sparse = scipy.sparse.random(n_dim, 1, density=1, dtype=np.int16).astype(int)
    else:
        x_sparse = None
        y_sparse = None

    x_sparse = await mpc.transfer(x_sparse, senders=0)
    y_sparse = await mpc.transfer(y_sparse, senders=0)

    x_dense = x_sparse.astype(int).todense()
    y_dense = y_sparse.astype(int).todense()
    real_res = x_dense.transpose().dot(y_dense)[0, 0]
    print("Real result:", real_res)

    params = {
        "Nb. rows": n_dim,
        "Nb. columns": 1,
        "Density": density,
    }

    if alg_choice in ["*", "dense"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_x_dense = DenseVector(x_dense.transpose(), sectype=secint)

        sec_y_dense = DenseVector(y_dense, sectype=secint)

        params["Algorithm"] = "Dense"
        async with exp_env.benchmark(params):
            z = sec_x_dense.dot(sec_y_dense)
            dense_res = await mpc.output(z)
            assert dense_res == real_res

        del x_dense, y_dense

    if alg_choice in ["*", "sparse"]:
        params["Algorithm"] = "Sparse sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseVector(x_sparse, secint)

        sec_y = SparseVector(y_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)
            sparse_res = await mpc.output(z)
            assert sparse_res == real_res

    if alg_choice in ["*", "sparse-dense"]:
        params["Algorithm"] = "Sparse-dense sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseVector(x_sparse, secint)

        sec_y = DenseVector(y_dense, secint)
        # We assume the dense vector is reused for multiple operations
        # So we only measure the sharing cost of the first vector

        params["Algorithm"] = "Sparse-dense"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)
            sparse_res = await mpc.output(z)
            assert sparse_res == real_res


async def benchmark_mat_vector_mult(exp_env, n_dim, density, alg_choice=None):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "dense", "sparse"]

    print(
        f"Matrix-vector multiplication benchmark: dimension={n_dim}, density={density} algorithm={alg_choice}"
    )
    secint = mpc.SecInt(64)

    if mpc.pid == 0:
        X_sparse = scipy.sparse.random(
            n_dim, n_dim, density=density, dtype=np.int16
        ).astype(int)
        y_sparse = scipy.sparse.random(
            n_dim, 1, density=density, dtype=np.int16
        ).astype(int)
    else:
        X_sparse = None
        y_sparse = None

    X_sparse = await mpc.transfer(X_sparse, senders=0)
    y_sparse = await mpc.transfer(y_sparse, senders=0)
    nb_non_zeros = len((X_sparse @ y_sparse).data)

    x_dense = X_sparse.astype(int).todense()
    y_dense = y_sparse.astype(int).todense()

    params = {
        "Nb. rows": n_dim,
        "Nb. columns": n_dim,
        "Density": density,
    }

    if alg_choice in ["*", "dense"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_x_dense = DenseMatrix(x_dense, sectype=secint)
            sec_y_dense = DenseVector(y_dense, sectype=secint)

        params["Algorithm"] = "Dense"
        async with exp_env.benchmark(params):
            z = sec_x_dense.dot(sec_y_dense)
            z_clear = await mpc.output(z._mat)
            dense_nb_non_zeros = (z_clear != 0).sum()
            assert nb_non_zeros == dense_nb_non_zeros

        del x_dense, y_dense

    if alg_choice in ["*", "sparse"]:
        params["Algorithm"] = "Sparse sharing"
        async with exp_env.benchmark(params):
            sec_x = SparseMatrixRow(X_sparse, secint)
            sec_y = SparseVector(y_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)

            sparse_nb_non_zeros = len(z._mat)
            assert nb_non_zeros == sparse_nb_non_zeros


async def benchmark_sparse_sparse_mat_mult(
    exp_env, n_dim, m_dim, density, alg_choice=None
):
    if alg_choice is None:
        alg_choice = "*"
    assert alg_choice in ["*", "dense", "sparse"]

    secint = mpc.SecInt(64)
    print(
        f"Matrix-matrix multiplication benchmark: dimensions={n_dim}x{m_dim}, density={density}, algorithm={alg_choice}"
    )
    if mpc.pid == 0:
        X_sparse = scipy.sparse.random(
            n_dim, m_dim, density=density, dtype=np.int16
        ).astype(int)
    else:
        X_sparse = None

    X_sparse = await mpc.transfer(X_sparse, senders=0)
    nb_non_zeros = len((X_sparse.T @ X_sparse).data)
    dense_mat = X_sparse.todense().astype(int)

    params = {
        "Nb. rows": n_dim,
        "Nb. columns": m_dim,
        "Density": density,
    }
    if alg_choice in ["dense", "*"]:
        params["Algorithm"] = "Dense sharing"
        async with exp_env.benchmark(params):
            sec_dense_t = DenseMatrix(dense_mat.transpose(), sectype=secint)
            sec_dense = DenseMatrix(dense_mat, sectype=secint)
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
            sec_x = SparseMatrixColumn(X_sparse.transpose(), secint)
            sec_y = SparseMatrixRow(X_sparse, secint)

        params["Algorithm"] = "Sparse"
        async with exp_env.benchmark(params):
            z = await sec_x.dot(sec_y)

        sparse_nb_non_zeros = len(z._mat)
        assert nb_non_zeros == sparse_nb_non_zeros


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

    if args["benchmark"] == "vect_mult":
        check_args(args, ["nb_rows", "density"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_vect_mult(
                exp_env,
                n_dim=args["nb_rows"],
                density=args["density"],
                alg_choice=args["algo"],
            )
    elif args["benchmark"] == "sparse_dense_vect_mult":
        check_args(args, ["nb_rows", "density"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_sparse_dense_vect_mult(
                exp_env,
                n_dim=args["nb_rows"],
                density=args["density"],
                alg_choice=args["algo"],
            )

    elif args["benchmark"] == "mat_vect_mult":
        check_args(args, ["nb_rows", "density"])
        async with ExperimentalEnvironment(
            args["benchmark"] + ".csv", CSV_FIELDS, seed=args.get("seed")
        ) as exp_env:
            await benchmark_mat_vector_mult(
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
    else:
        raise NotImplementedError


if __name__ == "__main__":
    mpc.run(main())
