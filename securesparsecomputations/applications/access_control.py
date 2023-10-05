# https://www.kaggle.com/code/shagkala/analysis-part-1-comparing-different-encodings

from abc import abstractmethod

import numpy as np
import pandas as pd

from mpyc.runtime import mpc
from sklearn.preprocessing import OneHotEncoder

from ..matrices import (
    from_scipy_sparse_mat,
    from_numpy_dense_matrix,
    DenseMatrix,
    DenseVector,
    SparseVector,
)
from ..benchmark import ExperimentalEnvironment
from ..resharing import np_shuffle_3PC
from ..radix_sort import radix_sort

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Runtime",
    "Communication cost",
]

NB_TRAINING_SAMPLES = 1000


def extract_dataset():
    access_log = pd.read_csv("../datasets/amazon.csv")
    encoder = OneHotEncoder(sparse_output=True)

    label = access_log["ACTION"].to_numpy().reshape(-1, 1)
    features = access_log.drop("ACTION", axis=1).drop("ROLE_CODE", axis=1)
    sparse_mat = encoder.fit_transform(features)
    return sparse_mat, label, encoder


class LDA:
    _proportion = None
    _mean_vectors = None
    _corr_matrix = None

    def _compute_proportion(self, y):
        self._proportion = mpc.np_sum(y._mat) / y.shape[0]

    @abstractmethod
    async def _compute_means(self, X, y):
        raise NotImplementedError

    async def _compute_covariance(self, X):
        if isinstance(X, DenseMatrix):
            self._corr_matrix = X.transpose().dot(X)
        else:
            self._corr_matrix = await X.transpose().dot(X)

    async def fit(self, X, y):
        self._compute_proportion(y)
        await self._compute_means(X, y)
        await self._compute_covariance(X)

    def predict(self, x):
        ...


class SparseLDA(LDA):
    async def _compute_means(self, X, y):
        merged_matrix = []
        for i in range(X.shape[0]):
            curr_row = X._mat[i]
            if curr_row.nnz != 0:
                label_vect = X.sectype.array(np.ones((curr_row.nnz, 1))) * y._mat[i]
                merged_matrix.append(mpc.np_hstack((label_vect, curr_row._mat)))
        merged_matrix = mpc.np_vstack(merged_matrix)

        res = await radix_sort(
            merged_matrix,
            X.col_bit_length + 1,
            already_decomposed=True,
            keep_bin_keys=True,
        )

        comp = mpc.np_multiply(
            res[0 : res.shape[0] - 1, -2]
            == res[1 : res.shape[0], -2],  # Equal coordinate
            res[0 : res.shape[0] - 1, 0] == res[1 : res.shape[0], 0],  # Equal label
        )

        col_val = mpc.np_tolist(res[:, -1])
        col_i = res[:-1, -2] * (1 - comp) + (-1) * comp
        # If the test is false, the value of this column is -1 (i.e., a placeholder)

        for i in range(res.shape[0] - 1):
            col_val[i + 1] = col_val[i + 1] + comp[i] * col_val[i]
        col_i = mpc.np_hstack((col_i, res[-1, -2:-1]))
        col_val = mpc.np_reshape(mpc.np_fromlist(col_val), (res.shape[0], 1))
        col_i = mpc.np_reshape(col_i, (res.shape[0], 1))

        res = mpc.np_hstack((res[:, :-2], col_i, col_val))

        res = await np_shuffle_3PC(res)

        zero_test = await mpc.np_is_zero_public(
            res[:, -2] + 1
        )  # Here, we leak the number of non-zero elements in the output matrix
        zero_val_test = await mpc.np_is_zero_public(res[:, -1])

        mask = [
            i for i, test in enumerate(zero_test) if not test and not zero_val_test[i]
        ]
        final_res = res[mask, :]

        label = await mpc.output(final_res[:, 0])
        mean_vector_1 = SparseVector(
            final_res[label == 0, 1:], (X.shape[1], 1), X.sectype
        )
        mean_vector_2 = SparseVector(
            final_res[label == 1, 1:], (X.shape[1], 1), X.sectype
        )

        self._mean_vectors = (mean_vector_1, mean_vector_2)


class DenseLDA(LDA):
    async def _compute_means(self, X, y):
        merged_matrix = mpc.np_hstack((X._mat, y._mat))
        merged_matrix = await np_shuffle_3PC(merged_matrix)
        cleartext_label = await mpc.output(merged_matrix[:, -1])

        class_1_matrix = merged_matrix[cleartext_label == 0, :-1]
        class_2_matrix = merged_matrix[cleartext_label == 1, :-1]

        self._mean_vectors = (
            DenseVector(mpc.np_sum(class_1_matrix, axis=0).reshape(-1, 1), X.sectype),
            DenseVector(mpc.np_sum(class_2_matrix, axis=0).reshape(-1, 1), X.sectype),
        )


async def experiment():
    async with ExperimentalEnvironment(
        "access_control.csv", CSV_FIELDS, seed=452179
    ) as exp_env:
        sec_fxp = mpc.SecFxp(64)
        X, y, _encoder = extract_dataset()
        print("Extraction done.")
        X = X[:NB_TRAINING_SAMPLES, :]
        y = y[:NB_TRAINING_SAMPLES]

        sparse_model = SparseLDA()
        print("Sparse sharing starts...")
        sec_sparse_X = from_scipy_sparse_mat(X, sec_fxp)
        print("Sparse sharing done.")
        sec_y = from_numpy_dense_matrix(y, sec_fxp)
        async with exp_env.benchmark({"Algorithm": "Sparse"}):
            await sparse_model.fit(sec_sparse_X, sec_y)

        dense_model = DenseLDA()
        print("Dense sharing starts...")
        sec_dense_X = from_numpy_dense_matrix(X.todense(), sec_fxp)
        print("Dense sharing done.")
        async with exp_env.benchmark({"Algorithm": "Dense"}):
            await dense_model.fit(sec_dense_X, sec_y)


def run():
    mpc.run(experiment())
