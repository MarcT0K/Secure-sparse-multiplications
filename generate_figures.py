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

    def figure_per_col(col, unit, log_scale_x, log_scale_y):
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
        if log_scale_y:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        if log_scale_x:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_sharing_{col.lower().replace(' ', '_')}_{'log' if log_scale_x else 'lin'}_{'log' if log_scale_y else 'lin'}.png",
            dpi=400,
        )

        plt.close(fig)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Runtime", "s", b1, b2)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Communication cost", "bytes", b1, b2)


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

    def figure_per_col(col, unit, log_scale_x, log_scale_y):
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
        if log_scale_y:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        if log_scale_x:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_mult_{col.lower().replace(' ', '_')}_{'log' if log_scale_x else 'lin'}_{'log' if log_scale_y else 'lin'}.png",
            dpi=400,
        )
        plt.close(fig)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Runtime", "s", b1, b2)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Communication cost", "bytes", b1, b2)


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

    def figure_per_col(col, unit, log_scale_x, log_scale_y):
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
        if log_scale_y:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        if log_scale_x:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        fig.tight_layout()
        fig.savefig(
            f"{csv_name}_mult_and_sharing_{col.lower().replace(' ', '_')}_{'log' if log_scale_x else 'lin'}_{'log' if log_scale_y else 'lin'}.png",
            dpi=400,
        )
        plt.close(fig)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Runtime", "s", b1, b2)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Communication cost", "bytes", b1, b2)


def gen_all_figures(csv_name, rows_or_col, xlabel):
    plot_sharing_experiment(csv_name, rows_or_col, xlabel)
    plot_mult_experiment(csv_name, rows_or_col, xlabel)
    plot_mult_and_sharing_experiment(csv_name, rows_or_col, xlabel)


def generate_shuffle_experiment():
    df = pd.read_csv("shuffle.csv")
    mpyc_shuffle = df[(df["Algorithm"] == "MPyC shuffle")]
    laur_shuffle = df[(df["Algorithm"] == "3PC shuffle")]

    def figure_per_col(col, unit, log_scale_x, log_scale_y):
        fig, ax = plt.subplots()

        ax.plot(mpyc_shuffle["Nb. rows"], mpyc_shuffle[col], label="MPyC shuffle")
        ax.plot(
            laur_shuffle["Nb. rows"],
            laur_shuffle[col],
            label="Laur et al. shuffle",
        )
        ax.set(xlabel="List length", ylabel=f"{col} ({unit})")
        ax.legend()
        if log_scale_y:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")

        if log_scale_x:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")

        fig.tight_layout()
        fig.savefig(
            f"shuffle_{col.lower().replace(' ', '_')}_{'log' if log_scale_x else 'lin'}_{'log' if log_scale_y else 'lin'}.png",
            dpi=400,
        )
        plt.close(fig)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Runtime", "s", b1, b2)

    for b1, b2 in [(i, j) for i in [True, False] for j in [True, False]]:
        figure_per_col("Communication cost", "bytes", b1, b2)


if __name__ == "__main__":
    generate_shuffle_experiment()
    gen_all_figures("dot_product", "rows", "Vector length")
    gen_all_figures("mat_mult", "columns", "Number of columns")
    plt.close("all")
