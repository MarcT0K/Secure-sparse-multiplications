import logging
import os
import threading

from subprocess import Popen, PIPE, STDOUT
from itertools import product

import colorlog
import psutil


logger = colorlog.getLogger()


def setup_logger():
    logger.handlers = []  # Reset handlers
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s %(levelname)s]%(reset)s %(white)s%(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    logger.addHandler(handler)
    fhandler = logging.FileHandler("experiments.log")
    fhandler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)


def generate_seed():
    return int.from_bytes(os.urandom(4), "big")


def log_stdout(process):
    with process.stdout:
        for line in iter(process.stdout.readline, b""):
            logger.debug(line[:-1].decode())


def track_memory(process: Popen, memory_usage_threshold=0.95) -> bool:
    log_thread = threading.Thread(target=log_stdout, args=(process,))
    log_thread.start()

    memory_overflow = False

    while process.poll() is None:
        memory_usage = (
            1 - psutil.virtual_memory().available / psutil.virtual_memory().total
        )

        if memory_usage > memory_usage_threshold and not memory_overflow:
            memory_overflow = True
            logger.error("Killed the process")
            psutil_proc = psutil.Process(process.pid)
            for proc in psutil_proc.children(recursive=True):
                proc.kill()
            psutil_proc.kill()
    log_thread.join()
    return memory_overflow


def shuffle_experiments():
    logger.info("START SHUFFLE EXPERIMENTS")
    mpyc_failed = False
    threepc_failed = False
    for (
        i,
        j,
    ) in product(range(1, 4), range(1, 10)):
        if mpyc_failed and threepc_failed:
            logger.warning("Both algorithms failed")
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
        logger.info("Shuffle experiment: seed=%d, length=%d", seed, list_length)

        if mpyc_failed == 0:
            subp = Popen(
                base_args + ["--algo", "MPyC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            mpyc_failed = track_memory(subp)
        else:
            logger.warning("Skipped MPyC")

        if mpyc_failed == 0:
            subp = Popen(
                base_args + ["--algo", "3PC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            threepc_failed = track_memory(subp)
        else:
            logger.warning("Skipped 3PC")
    logger.info("FINISHED ALL SHUFFLE EXPERIMENTS")


def dot_product_experiments():
    logger.info("START DOT PRODUCT EXPERIMENTS")
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(range(2, 6), range(1, 10), [0.001, 0.005, 0.01]):
        if dense_failed and sparse_failed:
            logger.warning("Both algorithms failed")
            break

        seed = generate_seed()
        nb_rows = j * 10**i
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "dot_product",
            "--seed",
            str(seed),
            "--nb-rows",
            str(nb_rows),
            "--density",
            str(density),
        ]
        logger.info(
            "Dot product experiment: seed=%d, dimensions=%d, density=%.3f",
            seed,
            nb_rows,
            density,
        )

        if dense_failed == 0:
            subp = Popen(
                base_args + ["--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            logger.warning("Skipped dense experiments")

        if sparse_failed == 0:
            subp = Popen(
                base_args + ["--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse experiments")

    logger.info("FINISHED ALL DOT PRODUCT EXPERIMENTS")


def matmult_experiments():
    logger.info("START MATRIX MULTIPLICATION EXPERIMENTS")
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(range(2, 6), range(1, 10, 2), [0.001, 0.005, 0.01]):
        if dense_failed and sparse_failed:
            logger.warning("Both algorithms failed")
            break

        seed = generate_seed()
        nb_cols = j * 10**i
        nb_rows = 10**2
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "mat_mult",
            "--seed",
            str(seed),
            "--nb-rows",
            str(),
            "--nb-cols",
            str(nb_cols),
            "--density",
            str(density),
        ]
        logger.info(
            "Dot product experiment: seed=%d, dimensions=%dx%d, density=%.3f",
            seed,
            nb_rows,
            nb_cols,
            density,
        )

        if not dense_failed:
            subp = Popen(
                base_args + ["--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            logger.warning("Skipped dense experiments")

        if sparse_failed == 0:
            subp = Popen(
                base_args + ["--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse experiments")

    logger.info("FINISHED ALL MATRIX MULTIPLICATION EXPERIMENTS")


def main():
    setup_logger()

    shuffle_experiments()
    dot_product_experiments()
    matmult_experiments()


if __name__ == "__main__":
    main()
