# https://www.kaggle.com/code/shagkala/analysis-part-1-comparing-different-encodings

import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

CSV_FIELDS = [
    "Timestamp",
    "Mitigation technique",
    "Storage cost",
    "Ratio non-repeating",
    "Ratio unique",
]


def threshold_padding(sp_matrix, threshold): ...


def max_padding(sp_matrix): ...


def sparse_storage_cost(sp_matrix): ...


def privacy_metrics(sp_matrix): ...


def extract_access_dataset():
    access_log = pd.read_csv("../datasets/amazon.csv")
    encoder = OneHotEncoder(sparse=True)

    label = access_log["ACTION"].to_numpy().reshape(-1, 1)
    features = access_log.drop("ACTION", axis=1).drop("ROLE_CODE", axis=1)
    sparse_mat = encoder.fit_transform(features)
    return sparse_mat


def extract_recommendation_dataset():
    ratings = pd.read_csv(
        "../datasets/BX-Book-Ratings.csv",
        sep=";",
        encoding="latin-1",
    )
    ratings.columns = ["user", "isbn", "rating"]

    user_set = ratings.user.unique()
    user_to_index = {o: i for i, o in enumerate(user_set)}
    book_set = ratings.isbn.unique()
    isbn_to_index = {o: i for i, o in enumerate(book_set)}

    ratings["user_ind"] = ratings["user"].map(user_to_index)
    ratings["book_ind"] = ratings["isbn"].map(isbn_to_index)

    n_users = len(user_set)
    n_books = len(book_set)

    sparse_mat = scipy.sparse.dok_matrix((n_users, n_books), dtype=int)
    for entry in ratings.itertuples():
        sparse_mat[entry[4], entry[5]] = entry[3] + 1

    return sparse_mat


def extract_spam_dataset():
    df = pd.read_csv("../datasets/spam.csv", sep="\t", names=["Label", "Message"])
    vect = CountVectorizer(stop_words="english")
    vect.fit(df["Message"])
    sparse_mat = vect.fit_transform(df["Message"])
    y = df.Label.map({"ham": 0, "spam": 1})
    return sparse_mat
