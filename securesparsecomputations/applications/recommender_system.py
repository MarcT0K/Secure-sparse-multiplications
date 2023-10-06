import math
import random

import numpy as np
import pandas as pd
import scipy.sparse

from mpyc.runtime import mpc

from ..matrices import (
    from_scipy_sparse_mat,
    from_numpy_dense_matrix,
    SecureMatrix,
    SparseVector,
    DenseVector,
    SparseMatrixRow,
)
from ..radix_sort import radix_sort
from ..resharing import np_shuffle_3PC
from ..benchmark import ExperimentalEnvironment
from ..quicksort import np_quicksort

CSV_FIELDS = [
    "Timestamp",
    "Initial seed",
    "Algorithm",
    "Runtime",
    "Communication cost",
]

NB_TRAINING_SAMPLES = 1000


def extract_dataset():
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

    return sparse_mat, user_to_index, isbn_to_index


def custom_np_reciprocal(vector):
    # NB: MPyC np_reciprocal is not yet working for fixed-point arrays
    res = [mpc._rec(el) for el in mpc.np_tolist(vector)]
    res = mpc.np_reshape(mpc.np_fromlist(res), (vector.shape[0], 1))
    return res


async def norm_sparse_matrix(matrix: SparseMatrixRow):  # column-wise
    assert isinstance(matrix, SparseMatrixRow)

    merged_matrix = []
    for i in range(matrix.shape[0]):
        curr_row = matrix._mat[i]
        if curr_row.nnz != 0:
            merged_matrix.append(curr_row._mat)
    merged_matrix = mpc.np_vstack(merged_matrix)

    square_val = merged_matrix[:, -1] ** 2
    merged_matrix = mpc.np_hstack((merged_matrix[:, :-1], square_val.reshape(-1, 1)))

    padding_coord = []
    for i in range(matrix.shape[1]):
        padding_coord.extend(
            SecureMatrix.int_to_secure_bits(i, matrix.sectype, matrix.col_bit_length)
        )
        padding_coord.append(matrix.sectype(int(i)))
    padding_coord = mpc.np_reshape(
        mpc.np_fromlist(padding_coord),
        (matrix.shape[1], matrix.col_bit_length + 1),
    )
    padding_val = matrix.sectype.array(np.zeros((matrix.shape[1], 1)))
    padding = mpc.np_hstack((padding_coord, padding_val))
    padded = mpc.np_vstack((merged_matrix, padding))

    res = await radix_sort(
        padded,
        matrix.col_bit_length,
        already_decomposed=True,
        keep_bin_keys=True,
    )
    res = padded[:, -2:]

    # We compare the integer representation of the column index for all pairs of consecutive elements
    comp = res[0 : res.shape[0] - 1, 0] == res[1 : res.shape[0], 0]

    col_val = mpc.np_tolist(res[:, -1])
    col_i = res[:-1, 0] * (1 - comp) + (-1) * comp
    # If the test is false, the value of this column is -1 (i.e., a placeholder)

    for i in range(res.shape[0] - 1):
        col_val[i + 1] = col_val[i + 1] + comp[i] * col_val[i]
    col_i = mpc.np_hstack((col_i, res[-1, 0:1]))
    col_val = mpc.np_reshape(mpc.np_fromlist(col_val), (res.shape[0], 1))
    col_i = mpc.np_reshape(col_i, (res.shape[0], 1))

    res = mpc.np_hstack((col_i, col_val))

    res = await np_shuffle_3PC(res)

    zero_test = await mpc.np_is_zero_public(
        res[:, 0] + 1
    )  # Here, we leak the number of non-zero elements in the output matrix
    zero_val_test = await mpc.np_is_zero_public(res[:, -1])

    mask = [i for i, test in enumerate(zero_test) if not test and not zero_val_test[i]]
    final_res = res[mask, :]

    public_coord = await mpc.output(final_res[:, 0])
    permutation = np.empty_like(public_coord, dtype=int)
    permutation[public_coord.astype(int)] = np.arange(len(permutation), dtype=int)

    return final_res[permutation, 1]


def norm_sparse_vector(vect):
    assert isinstance(vect, SparseVector)
    s = 0
    for i in range(vect.nnz):
        s += vect._mat[i, -1] ** 2
    return s


def compute_sparse_similarities(inner_products, inv_norms):
    if inner_products.nnz == 0:
        return inner_products.sectype.array([[0, -1]])  # returns a placeholder

    unit_matrix = []
    sparse_coord = inner_products._mat[:, -2]
    for i in range(inner_products.nnz):
        unit_vector = mpc.np_unit_vector(sparse_coord[i], inv_norms.shape[0])
        unit_matrix.append(unit_vector)

    unit_matrix = mpc.np_vstack(unit_matrix)
    val_vect = mpc.np_matmul(unit_matrix, inv_norms)
    similarities = mpc.np_multiply(val_vect, inner_products._mat[i, -1] ** 2)

    return mpc.np_hstack((similarities, sparse_coord.reshape(-1, 1)))


class KNNRecommenderSystem:
    def __init__(self, training_dataset, sectype, k=5):
        self._dataset = from_scipy_sparse_mat(training_dataset, sectype=sectype)
        self._sectype = sectype
        self._k = k

    @property
    def nb_books(self):
        return self._dataset.shape[1]

    async def predict(self, secure_book_id):
        assert isinstance(secure_book_id, SparseVector)
        selected_movie = await self._dataset.dot(secure_book_id)
        print("Number of nnz in the selected vector:", selected_movie.nnz)
        # Remark: We could improve the privacy of this protocol and (obliviously)
        # randomly pad the movie vector during the multiplication to avoid identifying
        # the movie based on the number of non-zero elements => Possible but out of scope

        selected_movie_norm = norm_sparse_vector(selected_movie)
        movie_norms = await norm_sparse_matrix(self._dataset)
        movie_norms *= selected_movie_norm
        # We only compute the square of the cosine similarity to avoid computing a square-root.
        # This trick does no change the KNN output.

        movie_norms += self._sectype(2 ** (-self._sectype.frac_length))
        # We add a negligible value for robustness (i.e., to avoid divisions by 0)
        inv_norms = custom_np_reciprocal(movie_norms)

        movie_inner_products = await self._dataset.transpose().dot(selected_movie)
        similarities = compute_sparse_similarities(movie_inner_products, inv_norms)

        sorted_results = await np_quicksort(similarities, key=lambda tup: tup[0])
        # We use quicksort as it does not require bit decomposition.
        # This implementation could be slightly sped up thanks to radix sort.
        return sorted_results[-self._k :, 1]


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
        similarities = mpc.np_multiply(movie_inner_products._mat**2, inv_norms)

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
        X_sparse = X_sparse[:NB_TRAINING_SAMPLES, :]

        if mpc.pid == 0:
            samples = random.sample(range(X_sparse.shape[1]), k=10)
        else:
            samples = None
        samples = await mpc.transfer(samples, senders=0)

        sparse_model = KNNRecommenderSystem(X_sparse, sectype=sec_fxp)
        for book_id in samples:
            async with exp_env.benchmark({"Algorithm": "Sparse sharing"}):
                bit_length = int(math.log(X_sparse.shape[1], 2)) + 1
                sec_input = SecureMatrix.int_to_secure_bits(
                    book_id, sec_fxp, bit_length
                )
                sec_input += [sec_fxp(book_id), sec_fxp(1)]
                sec_input = mpc.np_fromlist(sec_input)
                sec_input = mpc.np_reshape(sec_input, (1, bit_length + 2))
                assert sec_input.shape[1] == sparse_model._dataset._mat[0]._mat.shape[1]
                sec_input = mpc.input(sec_input, senders=0)
                sec_input = SparseVector(
                    sec_input, shape=(X_sparse.shape[1], 1), sectype=sec_fxp
                )
            async with exp_env.benchmark({"Algorithm": "Sparse"}):
                sec_res = await sparse_model.predict(sec_input)
                _res_sparse = await mpc.output(sec_res)
                print(_res_sparse)

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
