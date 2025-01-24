import os
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from typing import List, Optional

from .knowledge_minization import knowledge_minimization_benchmark
from .sparsity_distrib import sparsity_distribution_plots

params = {
    "text.usetex": True,
    "font.size": 15,
    "axes.labelsize": 25,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.7,
    "scatter.marker": "x",
}
plt.style.use("seaborn-v0_8-colorblind")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "-"])
    ),
)
plt.rcParams.update(params)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


def plot_mult_experiment(csv_name, rows_or_col, xlabel, until_overflow=None):
    df = pd.read_csv("../data/" + csv_name + ".csv")
    dense_mult = df[(df["Algorithm"] == "Dense") & (df["Density"] == 0.001)]
    sparse_mult_001 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.0001)]
    sparse_mult_01 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.001)]
    sparse_mult_1 = df[(df["Algorithm"] == "Sparse") & (df["Density"] == 0.01)]

    col = "Communication cost"
    unit = "bytes"

    fig, ax = plt.subplots()

    if until_overflow is not None:
        if until_overflow[0]:
            ax.scatter(
                dense_mult[f"Nb. {rows_or_col}"].to_numpy()[-1],
                dense_mult[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
                label="Memory overflow",
            )

        if until_overflow[1]:
            ax.scatter(
                sparse_mult_001[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_001[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )

        if until_overflow[2]:
            ax.scatter(
                sparse_mult_01[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_01[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )

        if until_overflow[3]:
            ax.scatter(
                sparse_mult_1[f"Nb. {rows_or_col}"].to_numpy()[-1],
                sparse_mult_1[col].to_numpy()[-1],
                marker="X",
                color="red",
                s=100,
                zorder=1000,
            )

    ax.plot(
        dense_mult[f"Nb. {rows_or_col}"],
        dense_mult[col],
        label="[Baseline] Dense",
        marker="x",
    )
    ax.plot(
        sparse_mult_001[f"Nb. {rows_or_col}"],
        sparse_mult_001[col],
        label=r"Sparse (99.99\%)",
        marker="x",
    )
    ax.plot(
        sparse_mult_01[f"Nb. {rows_or_col}"],
        sparse_mult_01[col],
        label=r"Sparse (99.9\%)",
        marker="x",
    )
    ax.plot(
        sparse_mult_1[f"Nb. {rows_or_col}"],
        sparse_mult_1[col],
        label=r"Sparse (99\%)",
        marker="x",
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


def plot_plaintext_experiment():
    df = pd.read_csv("../data/plaintext_comparison.csv")
    dense_mult = df[(df["Algorithm"] == "Dense")]
    sparse_mult = df[(df["Algorithm"] == "Sparse")]

    dense_overflow = dense_mult["Nb. columns"].iloc[-1] != 9 * 10**7
    sparse_overflow = sparse_mult["Nb. columns"].iloc[-1] != 9 * 10**7

    fig, ax = plt.subplots()

    overflow_line = None
    if dense_overflow:
        overflow_line = ax.scatter(
            dense_mult["Nb. columns"].iloc[-1],
            dense_mult["Runtime"].iloc[-1],
            marker="X",
            color="red",
            s=100,
            zorder=1000,
            label="Memory overflow",
        )

    if sparse_overflow:
        ax.scatter(
            sparse_mult["Nb. columns"].iloc[-1],
            sparse_mult["Runtime"].iloc[-1],
            marker="X",
            color="red",
            s=100,
            zorder=1000,
        )

    (plain_dense,) = ax.plot(
        dense_mult["Nb. columns"],
        dense_mult["Runtime"],
        label="NumPy plaintext dense",
        marker="x",
    )
    (plain_sparse,) = ax.plot(
        sparse_mult["Nb. columns"],
        sparse_mult["Runtime"],
        label="SciPy plaintext sparse",
        marker="+",
    )

    secure_sparse = ax.axvline(
        x=10**6,
        label="Memory overflow with\nour MPC sparse alg.",
        linestyle="--",
        color=colors[2],
    )  # HARDCODED BASED ON THE OTHER EXPERIMENTS
    secure_dense = ax.axvline(
        x=10**3,
        label="Memory overflow with\nclassic MPC dense alg.",
        linestyle="-",
        color=colors[5],
    )  # HARDCODED BASED ON THE OTHER EXPERIMENTS

    ax.set(xlabel="Number of columns", ylabel="Runtime (s)")
    plain_lines = (
        [plain_sparse, plain_dense]
        if overflow_line is None
        else [plain_sparse, plain_dense, overflow_line]
    )
    secure_lines = [secure_sparse, secure_dense]
    legend1 = ax.legend(handles=plain_lines, loc=2, prop={"size": 13})
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=secure_lines, loc=4, prop={"size": 13})
    ax.add_artist(legend2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(top=10**3)
    ax.set_xlim(right=10**8)

    fig.tight_layout()
    fig.savefig("plaintext_mult_runtime.png", dpi=400)
    plt.close(fig)


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")
    os.chdir("figures")

    plot_plaintext_experiment()
    plot_mult_experiment(
        "mat_vect_mult",
        "columns",
        "Number of columns and rows",
        until_overflow=[True, True, True, True],  # HARCODED
    )
    plot_mult_experiment(
        "mat_mult",
        "columns",
        "Number of columns",
        until_overflow=[True, False, False, False],  # HARDCODED
    )
    plt.close("all")

    knowledge_minimization_benchmark()
    sparsity_distribution_plots()


if __name__ == "__main__":
    main()
