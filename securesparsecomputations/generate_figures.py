import os
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

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
    sparse_sharing_001 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.0001)
    ]
    sparse_sharing_01 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.001)
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
            sparse_sharing_001[f"Nb. {rows_or_col}"],
            sparse_sharing_001[col],
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


def plot_mult_experiment(csv_name, rows_or_col, xlabel, until_overflow=False):
    df = pd.read_csv(csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.001)]
    sparse_mult_001 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.0001)]
    sparse_mult_01 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.001)]
    sparse_mult_1 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.01)]

    def figure_per_col(col, unit, until_overflow=False):
        fig, ax = plt.subplots()

        if until_overflow:
            ax.scatter(
                dense_mult[f"Nb. {rows_or_col}"].to_numpy()[-1],
                dense_mult[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
                label="Memory overflow",
            )
            ax.scatter(
                sparse_mult_001[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_001[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )
            ax.scatter(
                sparse_mult_01[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_01[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )
            ax.scatter(
                sparse_mult_1[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_1[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )

        ax.plot(dense_mult[f"Nb. {rows_or_col}"], dense_mult[col], label="Dense")
        ax.plot(
            sparse_mult_001[f"Nb. {rows_or_col}"],
            sparse_mult_001[col],
            label=r"Sparse (99.99\%)",
        )
        ax.plot(
            sparse_mult_01[f"Nb. {rows_or_col}"],
            sparse_mult_01[col],
            label=r"Sparse (99.9\%)",
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

    figure_per_col("Runtime", "s", until_overflow)
    figure_per_col("Communication cost", "bytes", until_overflow)


def plot_mult_and_sharing_experiment(csv_name, rows_or_col, xlabel):
    df = pd.read_csv(csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.001)]
    sparse_mult_001 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.0001)]
    sparse_mult_01 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.001)]
    sparse_mult_1 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.01)]

    dense_sharing = df[(df["Algorithm"] == "Dense sharing") & (df["Density"] == 0.001)]
    sparse_sharing_001 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.0001)
    ]
    sparse_sharing_01 = df[
        (df["Algorithm"] == "Sparse sharing") & (df["Density"] == 0.001)
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
            sparse_mult_001[f"Nb. {rows_or_col}"],
            sparse_mult_001[col].to_numpy() + sparse_sharing_001[col].to_numpy(),
            label=r"Sparse (99.99\%)",
        )
        ax.plot(
            sparse_mult_01[f"Nb. {rows_or_col}"],
            sparse_mult_01[col].to_numpy() + sparse_sharing_01[col].to_numpy(),
            label=r"Sparse (99.9\%)",
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


def gen_all_figures(csv_name, rows_or_col, xlabel, until_overflow=False):
    plot_sharing_experiment(csv_name, rows_or_col, xlabel)
    plot_mult_experiment(csv_name, rows_or_col, xlabel, until_overflow=until_overflow)
    plot_mult_and_sharing_experiment(csv_name, rows_or_col, xlabel)


# Sparse-dense plot
def plot_sparse_dense_experiment(csv_name="sparse_dense"):
    df = pd.read_csv(csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.0001)]
    sparse_mult = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.0001)]
    sparse_dense_mult = df[
        (df["Algorithm"] == "Sparse-dense") & (df["Density"] == 0.0001)
    ]

    def figure_per_col(col, unit):
        fig, ax = plt.subplots()

        ax.plot(dense_mult["Nb. rows"], dense_mult[col], label="Dense")
        ax.plot(
            sparse_mult["Nb. rows"],
            sparse_mult[col],
            label="Sparse",
        )
        ax.plot(
            sparse_dense_mult["Nb. rows."],
            sparse_dense_mult[col],
            label="Sparse-dense",
        )

        ax.set(xlabel="Vector length", ylabel=f"{col} ({unit})")
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


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")
    os.chdir("figures")

    gen_all_figures("vect_mult", "rows", "Vector length")
    gen_all_figures("mat_mult", "columns", "Number of columns", until_overflow=True)
    gen_all_figures(
        "mat_vect_mult", "columns", "Number of columns", until_overflow=True
    )
    plot_sparse_dense_experiment()
    plt.close("all")
