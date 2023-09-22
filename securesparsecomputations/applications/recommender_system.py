import random

import pandas as pd
import scipy.sparse

from mpyc.runtime import mpc

from ..matrices import (
    from_scipy_sparse_mat,
    from_numpy_dense_matrix,
    from_scipy_sparse_vect,
)
from ..benchmark import ExperimentalEnvironment

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


class KNNRecommenderSystem:
    def __init__(self, training_dataset, sectype, k=5):
        self._dataset = from_scipy_sparse_mat(training_dataset, sectype=sectype)
        self._sectype = sectype
        self._k = k

        # NB: We only compute the square of the cosine similarity to avoid computing a square-root.
        # This trick does no change the KNN output.

    async def predict(self, secure_book_id):
        ...


class DenseKNNRecommenderSystem:
    def __init__(self, training_dataset, sectype, k=5):
        self._dataset = from_numpy_dense_matrix(training_dataset, sectype=sectype)
        self._sectype = sectype
        self._k = k

    async def predict(self, secure_book_id):
        print("Norms", await mpc.output(self._dataset._mat[:20].sum(axis=1)))
        selection_vector = mpc.np_unit_vector(secure_book_id, self._dataset.shape[1])
        selected_movie = mpc.np_matmul(self._dataset._mat, selection_vector)
        selected_movie_norm = mpc.np_sum(mpc.np_pow(selected_movie, 2))
        movie_norms = mpc.np_sum(mpc.np_pow(self._dataset._mat, 2), axis=0)
        movie_norms *= selected_movie_norm
        # We add a negligible value for robustness (i.e., to avoid divisions by 0)
        movie_norms += self._sectype(2 ** (-self._sectype.frac_length))
        print("Norms", await mpc.output(movie_norms[:10]))
        inv_norms = mpc.np_reciprocal(movie_norms)
        print("Inv", await mpc.output(inv_norms[:10]))

        movie_inner_products = mpc.np_matmul(
            mpc.np_transpose(self._dataset._mat), selected_movie
        )
        print(
            "Inner",
            movie_inner_products.shape,
            await mpc.output(movie_inner_products[:5]),
        )
        similarities = movie_inner_products  # * inv_norms
        print("Sim", similarities.shape, await mpc.output(similarities))

        ind_vector = mpc.np_fromlist(
            [self._sectype(i) for i in range(self._dataset.shape[1])]
        )
        print("Ind", ind_vector.shape)
        print(await mpc.output(ind_vector[:5]))

        temp = mpc.np_transpose(
            mpc.np_vstack((similarities, mpc.np_transpose(ind_vector)))
        )
        print("Temp", temp.shape, await mpc.output(temp[:10]))
        sorted_results = mpc.np_sort(temp, axis=0, key=lambda tup: tup[0])
        print("Sorted", sorted_results.shape)
        print(await mpc.output(sorted_results[-5:, :]))
        input()
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
        #         sec_input = mpc.input(sec_fxp(book_id), senders=0)
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
