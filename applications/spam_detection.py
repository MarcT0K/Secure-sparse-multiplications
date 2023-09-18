import pandas as pd

from mpyc.runtime import mpc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


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


def experiment():
    sec_fxp = mpc.SecFxp(64)

    X_sparse, y = extract_dataset()
    model = train_logreg(X_sparse, y)

    sec_coefs = sec_fxp.array(model.coef_[0])
    sec_intercept = sec_fxp(model.intercept_[0])

    # TODO: implement experiment


if __name__ == "__main__":
    experiment()
