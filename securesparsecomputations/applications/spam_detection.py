import random

import pandas as pd
import scipy.sparse

from mpyc.runtime import mpc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from ..matrices import (
    from_numpy_dense_matrix,
    from_scipy_sparse_vect,
    SparseVector,
)
from ..benchmark import ExperimentalEnvironment

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Density",
    "Runtime",
    "Communication cost",
]


def extract_dataset():
    df = pd.read_csv("datasets/spam.csv", sep="\t", names=["Label", "Message"])
    vect = CountVectorizer(stop_words="english")
    vect.fit(df["Message"])
    X_sparse = vect.fit_transform(df["Message"])
    y = df.Label.map({"ham": 0, "spam": 1})
    return X_sparse, y


def train_logreg(X, y):
    logreg = LogisticRegression(solver="liblinear")
    logreg.fit(X, y)
    return logreg


class SecureLogisticRegression:
    def __init__(self, logreg, sectype):
        self.coef_ = from_numpy_dense_matrix(logreg.coef_.T, sectype)
        self.intercept_ = sectype(logreg.intercept_[0])

    async def predict(self, secure_input):
        if isinstance(secure_input, SparseVector):
            return (await secure_input.dot(self.coef_)) + self.intercept_
        else:
            return secure_input.dot(self.coef_) + self.intercept_


class SecureSparseLogisticRegression:
    def __init__(self, logreg, sectype):
        sparse_coef = scipy.sparse.coo_matrix(logreg.coef_.T)
        self.coef_ = from_scipy_sparse_vect(sparse_coef, sectype)
        self.intercept_ = sectype(logreg.intercept_[0])

    async def predict(self, secure_input):
        return (await secure_input.dot(self.coef_)) + self.intercept_


async def experiment():
    async with ExperimentalEnvironment(
        "spam_detection.csv", CSV_FIELDS, seed=548941
    ) as exp_env:
        sec_fxp = mpc.SecFxp(64)

        X_sparse, y = extract_dataset()
        model = train_logreg(X_sparse, y)

        sec_model = SecureLogisticRegression(model, sec_fxp)
        sec_sparse_model = SecureSparseLogisticRegression(model, sec_fxp)

        if mpc.pid == 0:
            samples = random.sample(range(X_sparse.shape[0]), k=10)
        else:
            samples = None
        samples = await mpc.transfer(samples, senders=0)

        for user_id in samples:
            user_input = X_sparse[user_id, :]
            density = 1 - user_input.nnz / user_input.shape[1]
            async with exp_env.benchmark(
                {"Algorithm": "Dense sharing", "Density": density}
            ):
                sec_input = from_numpy_dense_matrix(
                    user_input.todense(), sectype=sec_fxp
                )
            async with exp_env.benchmark({"Algorithm": "Dense", "Density": density}):
                sec_res = await sec_model.predict(sec_input)
                res_dense = await mpc.output(sec_res)

            async with exp_env.benchmark(
                {"Algorithm": "Sparse-dense sharing", "Density": density}
            ):
                sec_input = from_scipy_sparse_vect(user_input, sectype=sec_fxp)
            async with exp_env.benchmark(
                {"Algorithm": "Sparse-dense", "Density": density}
            ):
                sec_res = await sec_model.predict(sec_input)
                res_sparse_dense = await mpc.output(sec_res)

            async with exp_env.benchmark(
                {"Algorithm": "Sparse sharing", "Density": density}
            ):
                sec_input = from_scipy_sparse_vect(user_input, sectype=sec_fxp)
            async with exp_env.benchmark({"Algorithm": "Sparse", "Density": density}):
                sec_res = await sec_sparse_model.predict(sec_input)
                res_sparse = await mpc.output(sec_res)

            assert res_dense == res_sparse_dense and res_sparse == res_dense


def run():
    mpc.run(experiment())
