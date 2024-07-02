import math
import random

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


class TupleFormat:
    def __init__(self, list_of_lists, shape):
        self.shape = shape
        self._mat = list_of_lists

    @staticmethod
    def from_scipy_sparse(sp_matrix):
        list_matrix = []
        for row_id in range(sp_matrix.shape[0]):
            row = sp_matrix[row_id, :].tocoo()
            list_row = [(row.col[i], row.data[i]) for i in range(row.getnnz())]
            list_matrix.append(list_row)
        return TupleFormat(list_matrix, sp_matrix.shape)

    @staticmethod
    def pad_row(row, threshold, row_size):
        assert threshold >= len(row)  # We cannot remove element
        padded_row = row.copy()
        nb_dummies = threshold - len(row)

        nnz_ind = [ind for ind, _val in row]
        zero_ind = set(range(row_size)).difference(nnz_ind)

        dummies = random.sample(list(zero_ind), nb_dummies)
        for ind_dummy in dummies:
            padded_row.append((ind_dummy, 0))
        return padded_row

    def threshold_padding(self, threshold) -> "TupleFormat":
        assert threshold <= self.shape[1]
        treshold_multiple = lambda x: math.ceil(x / threshold) * threshold
        list_matrix = [
            TupleFormat.pad_row(row, treshold_multiple(len(row)), self.shape[1])
            for row in self._mat
        ]
        return TupleFormat(list_matrix, self.shape)

    def max_padding(self) -> "TupleFormat":
        nnz_per_row = [len(row) for row in self._mat]
        max_nnz = max(nnz_per_row)
        list_matrix = [
            TupleFormat.pad_row(row, max_nnz, self.shape[1]) for row in self._mat
        ]
        return TupleFormat(list_matrix, self.shape)

    def storage_cost(self):
        unit_bit_size = 64
        total_bit_size = 0
        for row in self._mat:
            total_bit_size += 2 * unit_bit_size * len(row)
        return total_bit_size

    def non_repeating_ratio(self):
        nnz_per_row = [len(row) for row in self._mat]
        nnz_dict = {}
        for nnz_count in nnz_per_row:
            nnz_dict[nnz_count] = nnz_dict.get(nnz_count, 0) + 1

        non_repeating = [
            nnz_count for nnz_count, repetition in nnz_dict.items() if repetition == 1
        ]
        return len(non_repeating) / len(self._mat)

    def uniqueness_ratio(self):
        set_nnz_per_row = set([len(row) for row in self._mat])
        return len(set_nnz_per_row) / len(self._mat)


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


def benchmark(): ...
