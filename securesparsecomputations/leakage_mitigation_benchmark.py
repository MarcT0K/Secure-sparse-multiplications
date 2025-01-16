import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse

from cycler import cycler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

params = {
    "text.usetex": True,
    "font.size": 15,
    "axes.labelsize": 25,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.7,
    "scatter.marker": "x",
    "text.latex.preamble": r"\usepackage{amsmath}",
}
plt.style.use("seaborn-v0_8-colorblind")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "-"])
    ),
)

texture_1 = {"hatch": "/"}
texture_2 = {"hatch": "\\"}
texture_3 = {"hatch": "."}
texture_4 = {"hatch": "x"}
texture_5 = {"hatch": "o"}
plt.rcParams.update(params)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


class PerRowKnowledge:
    def __init__(self, nb_nnz_array, shape):
        assert nb_nnz_array.shape == (shape[0],)
        self.matrix_shape = shape
        self._nb_nnz_per_row = nb_nnz_array

    @staticmethod
    def from_scipy_sparse(sp_matrix):
        public_knowledge_list = (sp_matrix != 0).sum(axis=1).ravel()
        public_knowledge_list = np.array(public_knowledge_list)[0]
        assert public_knowledge_list.shape[0] == sp_matrix.shape[0]
        return PerRowKnowledge(public_knowledge_list, sp_matrix.shape)

    def threshold_padding(self, threshold) -> "PerRowKnowledge":
        nb_nnz_padded = np.ceil(self._nb_nnz_per_row / threshold) * threshold
        nb_nnz_padded[nb_nnz_padded > self.matrix_shape[1]] = self.matrix_shape[1]
        return PerRowKnowledge(nb_nnz_padded, self.matrix_shape)

    def max_padding(self) -> "PerRowKnowledge":
        max_nnz = self._nb_nnz_per_row.max()
        return PerRowKnowledge(
            np.array([max_nnz] * len(self._nb_nnz_per_row)), self.matrix_shape
        )

    def matrix_templating(self, nb_data_owners) -> "PerRowKnowledge":
        raise NotImplementedError

    def storage_cost(self):
        unit_byte_size = 8
        bytes_count = 2 * self._nb_nnz_per_row * unit_byte_size
        return bytes_count.sum()

    def inverse_uniqueness(self):
        set_nnz_per_row = set(self._nb_nnz_per_row.tolist())
        return 1 / len(set_nnz_per_row)


def extract_access_dataset():
    access_log = pd.read_csv("../datasets/amazon.csv")
    encoder = OneHotEncoder(sparse_output=True)

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


DATASET_DICT = {
    "Spam": extract_spam_dataset,
    "Bookcrossing": extract_recommendation_dataset,
    "Access control": extract_access_dataset,
}


def benchmark():
    # Storage cost
    no_mitigation_cost = {}
    threshold_10_cost = {}
    threshold_100_cost = {}
    threshold_max_cost = {}
    dense_cost = {}

    # Uniqueness ratio
    no_mitigation_unique = {}
    threshold_10_unique = {}
    threshold_100_unique = {}
    threshold_max_unique = {}
    dense_unique = {}

    print("Extracting results")
    for name, extraction in DATASET_DICT.items():
        print(f"Benchmarking {name} dataset...")
        dataset = extraction()

        matrix_no_mitigation = PerRowKnowledge.from_scipy_sparse(dataset)
        no_mitigation_cost[name] = matrix_no_mitigation.storage_cost()
        no_mitigation_unique[name] = matrix_no_mitigation.inverse_uniqueness()

        padded_matrix = matrix_no_mitigation.threshold_padding(10)
        threshold_10_cost[name] = padded_matrix.storage_cost()
        threshold_10_unique[name] = padded_matrix.inverse_uniqueness()

        padded_matrix = matrix_no_mitigation.threshold_padding(100)
        threshold_100_cost[name] = padded_matrix.storage_cost()
        threshold_100_unique[name] = padded_matrix.inverse_uniqueness()

        padded_matrix = matrix_no_mitigation.max_padding()
        threshold_max_cost[name] = padded_matrix.storage_cost()
        threshold_max_unique[name] = padded_matrix.inverse_uniqueness()

        dense_cost[name] = (
            matrix_no_mitigation.matrix_shape[0]
            * matrix_no_mitigation.matrix_shape[1]
            * 8
        )
        dense_unique[name] = 1

    print("\nPreparing the plots...")

    # Plot 1: storage cost
    labels = list(no_mitigation_cost.keys())

    no_mitigation = list(no_mitigation_cost.values())
    threshold_10 = list(threshold_10_cost.values())
    threshold_100 = list(threshold_100_cost.values())
    threshold_max = list(threshold_max_cost.values())
    dense = list(dense_cost.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figwidth(7)
    rects1 = ax.bar(
        x - 2 * width,
        no_mitigation,
        width,
        capsize=4,
        label="No mitigation",
        **texture_1,
    )
    rects2 = ax.bar(
        x - width,
        threshold_10,
        width,
        capsize=4,
        label="Padding $p=10$",
        **texture_2,
    )
    rects3 = ax.bar(
        x, threshold_100, width, capsize=4, label="Padding $p=100$", **texture_3
    )
    rects4 = ax.bar(
        x + width,
        threshold_max,
        width,
        capsize=4,
        label="Padding $p=\\text{max}$",
        **texture_4,
    )
    rects5 = ax.bar(x + 2 * width, dense, width, capsize=4, label="Dense", **texture_5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set(xlabel="Dataset", ylabel="Memory footprint (B)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", prop={"size": 12}, framealpha=0.98)
    fig.tight_layout()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set_yscale("log")
    fig.savefig("public_knowledge_minimization_cost.png", dpi=400)

    # Plot 2: uniqueness
    labels = list(no_mitigation_unique.keys())

    no_mitigation = list(no_mitigation_unique.values())
    threshold_10 = list(threshold_10_unique.values())
    threshold_100 = list(threshold_100_unique.values())
    threshold_max = list(threshold_max_unique.values())
    dense = list(dense_unique.values())

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figwidth(7)
    rects1 = ax.bar(
        x - 2 * width,
        no_mitigation,
        width,
        capsize=4,
        label="No mitigation",
        **texture_1,
    )
    rects2 = ax.bar(
        x - width,
        threshold_10,
        width,
        capsize=4,
        label="Padding $p=10$",
        **texture_2,
    )
    rects3 = ax.bar(
        x, threshold_100, width, capsize=4, label="Padding $p=100$", **texture_3
    )
    rects4 = ax.bar(
        x + width,
        threshold_max,
        width,
        capsize=4,
        label="Padding $p=\\text{max}$",
        **texture_4,
    )
    rects5 = ax.bar(x + 2 * width, dense, width, capsize=4, label="Dense", **texture_5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set(xlabel="Dataset", ylabel="Inverse uniqueness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right", prop={"size": 12}, framealpha=0.98)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig("public_knowledge_minimization_uniqueness.png", dpi=400)


if __name__ == "__main__":
    benchmark()
