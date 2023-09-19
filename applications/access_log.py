# https://www.kaggle.com/code/shagkala/analysis-part-1-comparing-different-encodings

import pandas as pd

from mpyc.runtime import mpc
from sklearn.preprocessing import OneHotEncoder


def extract_dataset():
    access_log = pd.read_csv("amazon.csv")
    encoder = OneHotEncoder(sparse=True)

    label = access_log["ACTION"]
    features = access_log.drop("ACTION", axis=1).drop("ROLE_CODE", axis=1)
    sparse_mat = encoder.fit_transform(features)
    return sparse_mat, label, encoder


def experiment():
    sec_fxp = mpc.SecFxp(64)


if __name__ == "__main__":
    experiment()
