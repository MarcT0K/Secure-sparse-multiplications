from cycler import cycler
import matplotlib.pyplot as plt
import pandas as pd

params = {
    "text.usetex": True,
    "text.latex.preamble": [r"\usepackage{amssymb}", r"\usepackage{amsmath}"],
    "font.size": 15,
    "axes.labelsize": 25,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.7,
    "scatter.marker": "x",
}
plt.style.use("seaborn-colorblind")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "-"])
    ),
)
plt.rcParams.update(params)


def plot_sharing_experiment(csv_name, rows_or_col, xlabel):
    df = pd.read_csv(csv_name + ".csv")
    dense_sharing = df[(df["Algorithm"] == "Dense sharing") & (df["Density"] == 0.001)]
    sparse_sharing_01 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.001)
    ]
    sparse_sharing_05 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.005)
    ]
    sparse_sharing_1 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.01)
    ]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(dense_sharing[f"Nb. {rows_or_col}"], dense_sharing[col], label="Dense")
        ax.plot(
            sparse_sharing_01[f"Nb. {rows_or_col}"],
            sparse_sharing_01[col],
            label=r"Sparse (99.9\%)",
        )
        ax.plot(
            sparse_sharing_05[f"Nb. {rows_or_col}"],
            sparse_sharing_05[col],
            label=r"Sparse (99.5\%)",
        )
        ax.plot(
            sparse_sharing_1[f"Nb. {rows_or_col}"],
            sparse_sharing_1[col],
            label=r"Sparse (99\%)",
        )
        ax.set(xlabel=xlabel, ylabel=f"{col} ({unit})")
        ax.legend()

        ax.set_yscale("log")
        ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_sharing_{col.lower().replace(' ', '_')}.png",
            dpi=400,
        )

        plt.close(fig)

    figure_per_col("Runtime", "s")
    figure_per_col("Communication cost", "bytes")


def plot_mult_experiment(csv_name, rows_or_col, xlabel):
    df = pd.read_csv(csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.001)]
    sparse_mult_01 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.001)
    ]
    sparse_mult_05 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.005)
    ]
    sparse_mult_1 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.01)
    ]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(dense_mult[f"Nb. {rows_or_col}"], dense_mult[col], label="Dense")
        ax.plot(
            sparse_mult_01[f"Nb. {rows_or_col}"],
            sparse_mult_01[col],
            label=r"Sparse (99.9\%)",
        )
        ax.plot(
            sparse_mult_05[f"Nb. {rows_or_col}"],
            sparse_mult_05[col],
            label=r"Sparse (99.5\%)",
        )
        ax.plot(
            sparse_mult_1[f"Nb. {rows_or_col}"],
            sparse_mult_1[col],
            label=r"Sparse (99\%)",
        )
        ax.set(xlabel=xlabel, ylabel=f"{col} ({unit})")
        ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_mult_{col.lower().replace(' ', '_')}.png",
            dpi=400,
        )
        plt.close(fig)

    figure_per_col("Runtime", "s")
    figure_per_col("Communication cost", "bytes")


def plot_mult_and_sharing_experiment(csv_name, rows_or_col, xlabel):
    df = pd.read_csv(csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.001)]
    sparse_mult_01 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.001)
    ]
    sparse_mult_05 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.005)
    ]
    sparse_mult_1 = df[
        (df["Algorithm"] == "Sparse w/ Quicksort") & (df["Density"] == 0.01)
    ]

    dense_sharing = df[(df["Algorithm"] == "Dense sharing") & (df["Density"] == 0.001)]
    sparse_sharing_01 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.001)
    ]
    sparse_sharing_05 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.005)
    ]
    sparse_sharing_1 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.01)
    ]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(
            dense_mult[f"Nb. {rows_or_col}"],
            dense_mult[col].to_numpy() + dense_sharing[col].to_numpy(),
            label="Dense",
        )
        ax.plot(
            sparse_mult_01[f"Nb. {rows_or_col}"],
            sparse_mult_01[col].to_numpy() + sparse_sharing_01[col].to_numpy(),
            label=r"Sparse (99.9\%)",
        )
        ax.plot(
            sparse_mult_05[f"Nb. {rows_or_col}"],
            sparse_mult_05[col].to_numpy() + sparse_sharing_05[col].to_numpy(),
            label=r"Sparse (99.5\%)",
        )
        ax.plot(
            sparse_mult_1[f"Nb. {rows_or_col}"],
            sparse_mult_1[col].to_numpy() + sparse_sharing_1[col].to_numpy(),
            label=r"Sparse (99\%)",
        )
        ax.set(xlabel=xlabel, ylabel=f"{col} ({unit})")
        ax.legend()

        ax.set_yscale("log")
        ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_mult_and_sharing_{col.lower().replace(' ', '_')}.png",
            dpi=400,
        )
        plt.close(fig)

    figure_per_col("Runtime", "s")
    figure_per_col("Communication cost", "bytes")


def gen_all_figures(csv_name, rows_or_col, xlabel):
    plot_sharing_experiment(csv_name, rows_or_col, xlabel)
    plot_mult_experiment(csv_name, rows_or_col, xlabel)
    plot_mult_and_sharing_experiment(csv_name, rows_or_col, xlabel)


def generate_shuffle_experiment():
    df = pd.read_csv("shuffle.csv")
    mpyc_shuffle = df[(df["Algorithm"] == "MPyC shuffle")]
    laur_shuffle = df[(df["Algorithm"] == "3PC shuffle")]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(mpyc_shuffle["Nb. rows"], mpyc_shuffle[col], label="MPyC shuffle")
        ax.plot(
            laur_shuffle["Nb. rows"],
            laur_shuffle[col],
            label="Laur et al. shuffle",
        )
        ax.set(xlabel="List length", ylabel=f"{col} ({unit})")
        ax.legend()

        ax.set_yscale("log")
        ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(
            f"shuffle_{col.lower().replace(' ', '_')}.png",
            dpi=400,
        )
        plt.close(fig)

    figure_per_col("Runtime", "s")
    figure_per_col("Communication cost", "bytes")


def generate_sorting_experiment():
    df = pd.read_csv("sort.csv")
    quicksort = df[(df["Algorithm"] == "Quicksort")]
    radix_8bits = df[(df["Algorithm"] == "Radix sort") & (df["Key bit length"] == 8)]
    radix_16bits = df[(df["Algorithm"] == "Radix sort") & (df["Key bit length"] == 16)]
    radix_32bits = df[(df["Algorithm"] == "Radix sort") & (df["Key bit length"] == 32)]
    radix_48bits = df[(df["Algorithm"] == "Radix sort") & (df["Key bit length"] == 48)]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(quicksort["Nb. rows"], quicksort[col], label="Quick sort")
        ax.plot(
            radix_8bits["Nb. rows"],
            radix_8bits[col],
            label="Radix sort (8-bit sorting keys)",
        )
        ax.plot(
            radix_16bits["Nb. rows"],
            radix_16bits[col],
            label="Radix sort (16-bit sorting keys)",
        )
        ax.plot(
            radix_32bits["Nb. rows"],
            radix_32bits[col],
            label="Radix sort (32-bits sorting keys)",
        )
        ax.plot(
            radix_48bits["Nb. rows"],
            radix_48bits[col],
            label="Radix sort (48-bits sorting keys)",
        )

        ax.set(xlabel="List length", ylabel=f"{col} ({unit})")
        ax.legend()

        ax.set_yscale("log")
        ax.set_xscale("log")

        fig.tight_layout()
        fig.savefig(
            f"sort_{col.lower().replace(' ', '_')}.png",
            dpi=400,
        )
        plt.close(fig)

    figure_per_col("Runtime", "s")
    figure_per_col("Communication cost", "bytes")


if __name__ == "__main__":
    generate_sorting_experiment()
    # generate_shuffle_experiment()
    # gen_all_figures("dot_product", "rows", "Vector length")
    # gen_all_figures("mat_mult", "columns", "Number of columns")
    plt.close("all")