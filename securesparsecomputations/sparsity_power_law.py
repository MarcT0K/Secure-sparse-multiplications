import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer


def extract_bookcrossing_dataset():
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
        sparse_mat[entry.user_ind, entry.book_ind] = entry.rating

    return sparse_mat


def extract_movielens_dataset():
    ratings = pd.read_csv(
        "../datasets/movielens.csv",
    )
    ratings = ratings[: 10**7]  # TODO: remove after debugging
    ratings.columns = ["userId", "movieId", "rating", "timestamp"]

    user_set = ratings.userId.unique()
    user_to_index = {o: i for i, o in enumerate(user_set)}
    movie_set = ratings.movieId.unique()
    movie_to_index = {o: i for i, o in enumerate(movie_set)}

    ratings["user_ind"] = ratings["userId"].map(user_to_index)
    ratings["movie_ind"] = ratings["movieId"].map(movie_to_index)

    n_users = len(user_set)
    n_movies = len(movie_set)

    sparse_mat = scipy.sparse.dok_matrix((n_users, n_movies), dtype=float)
    for entry in ratings.itertuples():
        sparse_mat[entry.user_ind, entry.movie_ind] = entry.rating

    return sparse_mat


def extract_spam_dataset():
    df = pd.read_csv("../datasets/spam.csv", sep="\t", names=["Label", "Message"])
    vect = CountVectorizer(stop_words="english")
    vect.fit(df["Message"])
    sparse_mat = vect.fit_transform(df["Message"])
    y = df.Label.map({"ham": 0, "spam": 1})
    return sparse_mat


def extract_dorothea_dataset():
    sparse_mat = scipy.sparse.dok_matrix((1950, 10**5), dtype=int)
    with open("../datasets/dorothea.data", "r") as dorothea_file:
        for i, line in enumerate(dorothea_file):
            feat_list = line.split()
            for feat in feat_list:
                feat_int = int(feat) - 1
                sparse_mat[i, feat_int] = 1

    return sparse_mat


def plot_sparsity_distribution(sparse_dataset, dataset_name):
    sparsity_distrib = (sparse_dataset != 0).sum(axis=1)

    curr_density = (sparsity_distrib > 0).mean()
    density = [curr_density]
    nnz = [0]
    curr_nnz = 0

    max_nnz = sparse_dataset.shape[1]
    max_nnz_density = (sparsity_distrib >= max_nnz).mean()
    while curr_nnz != max_nnz:
        next_nnz = max_nnz
        min_next_nnz = curr_nnz
        density_min_nnz = (sparsity_distrib > min_next_nnz).mean()
        density_next_nnz = max_nnz_density
        while min_next_nnz + 1 != next_nnz:
            mid_nnz = (next_nnz + min_next_nnz + 1) // 2
            density_mid_nnz = (sparsity_distrib > mid_nnz).mean()
            if density_mid_nnz == density_min_nnz:
                min_next_nnz = mid_nnz
                density_min_nnz = density_mid_nnz
            else:
                next_nnz = mid_nnz
                density_next_nnz = density_mid_nnz

        for i in range(curr_nnz + 1, next_nnz):
            nnz.append(i)
            density.append(curr_density)

        nnz.append(next_nnz)
        density.append(density_next_nnz)

        curr_density = density_next_nnz
        curr_nnz = next_nnz

    plt.plot(nnz, density)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1, max_nnz)
    plt.xlabel("Number of non-zeros per row")
    plt.ylabel("Proportion")
    plt.title("Pre-row sparsity distribution in " + dataset_name)
    plt.savefig("../figures/sparsity_distrib_" + dataset_name.lower() + ".png")
    plt.close()


def main():
    print("Start Dorothea dataset analysis...")
    dataset = extract_dorothea_dataset()
    print("Dorothea extracted")
    plot_sparsity_distribution(dataset, "Dorothea")
    print("End Dorothea dataset.")

    print("Start Spam dataset analysis...")
    dataset = extract_spam_dataset()
    print("Dataset extracted")
    plot_sparsity_distribution(dataset, "Spam")
    print("End Spam dataset.")

    print("Start Movielens analysis...")
    dataset = extract_movielens_dataset()
    print("Dataset extracted")
    plot_sparsity_distribution(dataset, "MovieLens")
    print("End Movielens.")

    print("Start Bookcrossing analysis...")
    dataset = extract_bookcrossing_dataset()
    print("Dataset extracted")
    plot_sparsity_distribution(dataset, "Bookcrossing")
    print("End Bookcrossing.")


if __name__ == "__main__":
    main()
