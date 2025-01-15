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

    curr_density = (sparsity_distrib >= 0).mean()
    density = []
    nnz = []
    curr_nnz = 0

    max_nnz = sparse_dataset.shape[1]
    max_nnz_density = (sparsity_distrib >= max_nnz).mean()
    while max_nnz > curr_nnz:
        nnz.append(curr_nnz)
        density.append(curr_density)

        lower_bound_next_nnz = curr_nnz
        upper_bound_next_nnz = max_nnz
        density_lower_bound = curr_density
        density_upper_bound = max_nnz_density

        # Find the highest nnz with the same density as the current nnz
        while lower_bound_next_nnz != upper_bound_next_nnz:  # Binary search
            if lower_bound_next_nnz == upper_bound_next_nnz - 1:
                if density_upper_bound == curr_density:
                    lower_bound_next_nnz = upper_bound_next_nnz
                else:
                    upper_bound_next_nnz = lower_bound_next_nnz
                continue

            mid_nnz = (lower_bound_next_nnz + upper_bound_next_nnz) // 2
            density_mid_nnz = (sparsity_distrib > mid_nnz).mean()
            if density_mid_nnz == density_lower_bound:
                lower_bound_next_nnz = mid_nnz
                density_lower_bound = density_mid_nnz
            else:
                upper_bound_next_nnz = mid_nnz
                density_upper_bound = density_mid_nnz

        next_nnz = lower_bound_next_nnz + 1
        next_nnz_density = (sparsity_distrib > next_nnz).mean()

        for i in range(curr_nnz + 1, next_nnz):
            nnz.append(i)
            density.append(curr_density)

        curr_density = next_nnz_density
        curr_nnz = next_nnz

    assert nnz == list(range(max_nnz + 1))  # Consistency check
    assert len(density) == len(nnz)

    # Bounds the x-axis to focus on the
    min_nnz_plot = None
    max_nnz_plot = None
    for i in range(len(nnz)):
        if density[i] == 1.0:
            min_nnz_plot = nnz[i]

        if density[i] == 0.0 and max_nnz_plot is None:
            # Works because the density is by definition sorted
            max_nnz_plot = nnz[i]

    assert min_nnz_plot is not None and max_nnz_plot is not None
    assert max_nnz_plot > min_nnz_plot

    plt.plot(nnz, density)
    plt.yscale("log")
    plt.xscale("log")
    print(min_nnz_plot, max_nnz_plot)
    plt.xlim(min_nnz_plot, max_nnz_plot)
    plt.xlabel(f"Number of non-zeros per row (Row size= {sparse_dataset.shape[1]})")
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
