# https://www.kaggle.com/code/shagkala/analysis-part-1-comparing-different-encodings

from abc import abstractmethod

import pandas as pd

from mpyc.runtime import mpc
from sklearn.preprocessing import OneHotEncoder

from ..matrices import from_scipy_sparse_mat, from_numpy_dense_matrix
from ..benchmark import ExperimentalEnvironment

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Runtime",
    "Communication cost",
]


def extract_dataset():
    access_log = pd.read_csv("datasets/amazon.csv")
    encoder = OneHotEncoder(sparse=True)

    label = access_log["ACTION"]
    features = access_log.drop("ACTION", axis=1).drop("ROLE_CODE", axis=1)
    sparse_mat = encoder.fit_transform(features)
    return sparse_mat, label, encoder


class LDA:
    _proportion = None
    _mean_vector = None
    _corr_matrix = None

    @abstractmethod
    def _compute_proportion(self, y):
        ...

    @abstractmethod
    def _compute_means(self, X, y):
        ...

    @abstractmethod
    def _compute_covariance(self, X):
        ...

    def fit(self, X, y):
        self._compute_proportion(X)
        self._compute_means(X, y)
        self._compute_covariance(X)

    def predict(self, x):
        raise NotImplementedError


class SparseLDA(LDA):
    def _compute_proportion(self, y):
        ...

    def _compute_means(self, X, y):
        ...

    def _compute_covariance(self, X):
        ...


class DenseLDA(LDA):
    def _compute_proportion(self, y):
        ...

    def _compute_means(self, X, y):
        ...

    def _compute_covariance(self, X):
        ...


def experiment():
    async with ExperimentalEnvironment(
        "access_control.csv", CSV_FIELDS, seed=452179
    ) as exp_env:
        sec_fxp = mpc.SecFxp(64)
        X, y, _encoder = extract_dataset()

        sparse_model = SparseLDA()
        sec_sparse_X = from_scipy_sparse_mat(X)
        sec_y = from_numpy_dense_matrix(y)
        async with exp_env.benchmark({"Algorithm": "Sparse"}):
            sparse_model.fit(sec_sparse_X, sec_y)

        dense_model = DenseLDA()
        sec_dense_X = from_numpy_dense_matrix(X.todense())
        async with exp_env.benchmark({"Algorithm": "Dense"}):
            dense_model.fit(sec_dense_X, sec_y)


def run():
    mpc.run(experiment())
