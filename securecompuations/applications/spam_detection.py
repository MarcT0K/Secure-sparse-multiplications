import random

import pandas as pd

from mpyc.runtime import mpc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from ..matrices import (
    from_numpy_dense_matrix,
    from_scipy_sparse_vect,
    DenseVector,
    SparseVector,
)


def extract_dataset():
    df = pd.read_csv("spam.csv", sep="\t", names=["Label", "Message"])
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
        self.sectype = sectype
        self.coef_ = from_numpy_dense_matrix(logreg.coef_[0], sectype)
        self.intercept_ = sectype(logreg.intercept_[0])

    async def predict(self, secure_input):
        if isinstance(secure_input, SparseVector):
            return (await secure_input.dot(self.coef_)) + self.intercept_
        else:
            return secure_input.dot(self.coef_) + self.intercept_


async def experiment(sparse=True):
    random.setstate(476528)
    sec_fxp = mpc.SecFxp(64)

    X_sparse, y = extract_dataset()
    model = train_logreg(X_sparse, y)

    sec_model = SecureLogisticRegression(model, sec_fxp)

    samples = random.sample(range(X_sparse.shape[0]), k=100)
    for user_id in samples:
        user_input = X_sparse[user_id, :]
        if sparse:
            sec_input = from_scipy_sparse_vect(user_input)
        else:
            sec_input = from_numpy_dense_matrix(user_input.todense())

        sec_res = sec_model.predict(sec_input)
        _res = await mpc.output(sec_res)


if __name__ == "__main__":
    mpc.run(experiment())
