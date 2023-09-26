import math
import random

import pandas as pd
import scipy.sparse

from mpyc.runtime import mpc

from ..matrices import (
    from_scipy_sparse_mat,
    from_numpy_dense_matrix,
    from_scipy_sparse_vect,
    SecureMatrix,
    SparseVector,
    DenseVector,
)
from ..benchmark import ExperimentalEnvironment
from ..quicksort import np_quicksort

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Runtime",
    "Communication cost",
]


def extract_dataset():
    ratings = pd.read_csv(
        "datasets/BX-Book-Ratings.csv",
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


def custom_np_reciprocal(vector):
    # NB: MPyC np_reciprocal is not yet working for fixed-point arrays
    res = [mpc._rec(el) for el in mpc.np_tolist(vector)]
    res = mpc.np_reshape(mpc.np_fromlist(res), (vector.shape[0], 1))
    return res


class KNNRecommenderSystem:
    def __init__(self, training_dataset, sectype, k=5):
        self._dataset = from_scipy_sparse_mat(training_dataset, sectype=sectype)
        self._sectype = sectype
        self._k = k

    @property
    def nb_books(self):
        return self._dataset.shape[1]

    async def predict(self, secure_book_id):
        ...


class DenseKNNRecommenderSystem:
    def __init__(self, training_dataset, sectype, k=5):
        self._dataset = from_numpy_dense_matrix(training_dataset, sectype=sectype)
        self._sectype = sectype
        self._k = k

    @property
    def nb_books(self):
        return self._dataset.shape[1]

    async def predict(self, secure_book_id):
        selection_vector = DenseVector(
            mpc.np_reshape(
                mpc.np_unit_vector(secure_book_id, self.nb_books),
                (self.nb_books, 1),
            ),
            self._sectype,
        )
        selected_movie = self._dataset.dot(selection_vector)

        selected_movie_norm = mpc.np_sum(mpc.np_pow(selected_movie._mat, 2))
        movie_norms = mpc.np_sum(mpc.np_pow(self._dataset._mat, 2), axis=0)
        movie_norms *= selected_movie_norm
        # We only compute the square of the cosine similarity to avoid computing a square-root.
        # This trick does no change the KNN output.

        movie_norms += self._sectype(2 ** (-self._sectype.frac_length))
        # We add a negligible value for robustness (i.e., to avoid divisions by 0)
        inv_norms = custom_np_reciprocal(movie_norms)

        movie_inner_products = self._dataset.transpose().dot(selected_movie)
        similarities = mpc.np_multiply(movie_inner_products._mat, inv_norms)

        ind_vector = mpc.np_reshape(
            mpc.np_fromlist([self._sectype(i) for i in range(self.nb_books)]),
            (self.nb_books, 1),
        )
        temp = mpc.np_hstack((similarities, ind_vector))

        sorted_results = await np_quicksort(temp, key=lambda tup: tup[0])
        # We use quicksort as it does not require bit decomposition.
        # This implementation could be slightly sped up thanks to radix sort.
        return sorted_results[-self._k :, 1]


async def experiment():
    async with ExperimentalEnvironment(
        "recommender_system.csv", CSV_FIELDS, seed=734868
    ) as exp_env:
        sec_fxp = mpc.SecFxp(64)

        X_sparse, _user_to_index, _isbn_to_index = extract_dataset()
        X_sparse = X_sparse[:101, :100]

        if mpc.pid == 0:
            samples = random.sample(range(X_sparse.shape[1]), k=10)
        else:
            samples = None
        samples = await mpc.transfer(samples, senders=0)

        # sparse_model = KNNRecommenderSystem(X_sparse, sectype=sec_fxp)
        # for book_id in samples:
        #     async with exp_env.benchmark({"Algorithm": "Sparse sharing"}):
        #         bit_length = int(math.log(X_sparse.shape[1], 2)) + 1
        #         sec_input = [
        #             SecureMatrix.int_to_secure_bits(book_id, sec_fxp, bit_length)
        #         ] + [sec_fxp(book_id), sec_fxp(1)]
        #         sec_input = mpc.np_reshape(sec_input, (1, X_sparse.shape[1]))
        #         sec_input = mpc.input(mpc.np_fromlist(sec_input), senders=0)
        #         sec_input = SparseVector(
        #             sec_input, shape=(1, X_sparse.shape[1]), sectype=sec_fxp
        #         )
        #     async with exp_env.benchmark({"Algorithm": "Sparse"}):
        #         sec_res = await sparse_model.predict(sec_input)
        #         _res_sparse = await mpc.output(sec_res)

        dense_model = DenseKNNRecommenderSystem(X_sparse.todense(), sectype=sec_fxp)
        for book_id in samples:
            async with exp_env.benchmark({"Algorithm": "Dense sharing"}):
                sec_input = mpc.input(sec_fxp(book_id), senders=0)
            async with exp_env.benchmark({"Algorithm": "Dense"}):
                sec_res = await dense_model.predict(sec_input)
                _res_dense = await mpc.output(sec_res)
                print(_res_dense)


def run():
    mpc.run(experiment())
