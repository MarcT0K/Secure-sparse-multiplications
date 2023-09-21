import pandas as pd
import scipy.sparse

from mpyc.runtime import mpc

from ..matrices import SparseMatrixRow, SparseVector


def extract_dataset():
    ratings = pd.read_csv(
        "BX-Book-Ratings.csv",
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

    return sparse_mat, user_to_index, isbn_to_index


def inference(model: SparseMatrixRow, input: SparseVector):
    ...


def experiment():
    sec_fxp = mpc.SecFxp(64)


def run():
    experiment()
