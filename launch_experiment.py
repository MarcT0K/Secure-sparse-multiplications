import os
import threading

from subprocess import Popen, PIPE, STDOUT
from itertools import product

import psutil


def generate_seed():
    return int.from_bytes(os.urandom(4), "big")


def log_stdout(process):
    with process.stdout:
        for line in iter(process.stdout.readline, b""):
            print(line[:-1].decode())


def track_memory(process: Popen, memory_usage_threshold=0.9) -> bool:
    log_thread = threading.Thread(target=log_stdout, args=(process,))
    log_thread.start()

    memory_overflow = False

    while process.poll() is None:
        memory_usage = (
            1 - psutil.virtual_memory().available / psutil.virtual_memory().total
        )

        if memory_usage > memory_usage_threshold and not memory_overflow:
            memory_overflow = True
            print("Killed the process")
            psutil_proc = psutil.Process(process.pid)
            for proc in psutil_proc.children(recursive=True):
                proc.kill()
            psutil_proc.kill()
    log_thread.join()
    return memory_overflow


def shuffle_experiments():
    mpyc_failed = False
    threepc_failed = False
    for (
        i,
        j,
    ) in product(range(1, 4), range(1, 10)):
        if mpyc_failed and threepc_failed:
            print("Both algorithms failed")
            break

        seed = generate_seed()
        list_length = j * 10**i
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "shuffle",
            "--seed",
            str(seed),
            "--nb-rows",
            str(list_length),
        ]
        if mpyc_failed == 0:
            subp = Popen(
                base_args + ["--algo", "MPyC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            mpyc_failed = track_memory(subp)
        else:
            print("Skipped MPyC")

        if mpyc_failed == 0:
            subp = Popen(
                base_args + ["--algo", "3PC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            threepc_failed = track_memory(subp)
        else:
            print("Skipped 3PC")
    print("FINISHED ALL SHUFFLE EXPERIMENTS")


def dot_product_experiments():
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(range(3, 6), range(1, 10), [0.001, 0.005, 0.01]):
        if dense_failed and sparse_failed:
            print("Both algorithms failed")
            break

        seed = generate_seed()
        nb_cols = j * 10**i
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "dot_product",
            "--seed",
            str(seed),
            "--nb-rows",
            str(10**2),
            "--nb-cols",
            str(nb_cols),
            "--density",
            str(density),
        ]

        if dense_failed == 0:
            subp = Popen(
                base_args + ["--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            print("Skipped dense experiments")

        if sparse_failed == 0:
            subp = Popen(
                base_args + ["--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse_failed = track_memory(subp)
        else:
            print("Skipped sparse experiments")


def matmult_experiments():
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(range(2, 6), range(1, 10, 2), [0.001, 0.005, 0.01]):
        if dense_failed and sparse_failed:
            print("Both algorithms failed")
            break

        seed = generate_seed()
        nb_cols = j * 10**i
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "mat_mult",
            "--seed",
            str(seed),
            "--nb-rows",
            str(10**2),
            "--nb-cols",
            str(nb_cols),
            "--density",
            str(density),
        ]

        if not dense_failed:
            subp = Popen(
                base_args + ["--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            print("Skipped dense experiments")

        input()

        if sparse_failed == 0:
            subp = Popen(
                base_args + ["--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse_failed = track_memory(subp)
        else:
            print("Skipped sparse experiments")


def main():
    shuffle_experiments()
    dot_product_experiments()
    matmult_experiments()


if __name__ == "__main__":
    main()
