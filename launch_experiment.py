import glob
import logging
import os
import random
import shutil
import threading
import time
from itertools import product
from subprocess import PIPE, STDOUT, Popen

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
    return int.from_bytes(random.randbytes(4), "big")


def log_stdout(process):
    with process.stdout:
        for line in iter(process.stdout.readline, b""):
            logger.debug(line[:-1].decode())


def archive_experiment_results():
    existing_results = glob.glob("*.csv") + glob.glob("*.log")

    if not existing_results:
        return

    if not os.path.exists("archive"):
        os.makedirs("archive")
    archive_dir = "archive/" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    os.makedirs(archive_dir)

    for file in existing_results:
        shutil.move(file, archive_dir)


def track_memory(
    process: Popen, memory_usage_threshold=0.95, time_threshold=36000
) -> bool:
    log_thread = threading.Thread(target=log_stdout, args=(process,))
    log_thread.start()
    start_time = time.time()

    killed = False

    while process.poll() is None:
        memory_usage = (
            1 - psutil.virtual_memory().available / psutil.virtual_memory().total
        )
        elapsed_time = time.time() - start_time

        memory_overflow = memory_usage > memory_usage_threshold
        timeout = elapsed_time > time_threshold

        if not killed and (memory_overflow or timeout):
            killed = True
            if memory_overflow:
                logger.error("Memory overflow")
            if timeout:
                logger.error("Timeout")
            logger.error("Killed the process")
            psutil_proc = psutil.Process(process.pid)
            for proc in psutil_proc.children(recursive=True):
                proc.kill()
            psutil_proc.kill()

    log_thread.join()
    return killed


def shuffle_experiments():
    logger.info("START SHUFFLE EXPERIMENTS")
    mpyc_failed = False
    threepc_failed = False
    for (
        i,
        j,
    ) in product(range(1, 5), range(1, 10, 2)):
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

        if not mpyc_failed:
            subp = Popen(
                base_args + ["--algo", "MPyC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            mpyc_failed = track_memory(subp)
        else:
            logger.warning("Skipped MPyC")

        if not threepc_failed:
            subp = Popen(
                base_args + ["--algo", "3PC"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            threepc_failed = track_memory(subp)
        else:
            logger.warning("Skipped 3PC")
    logger.info("FINISHED ALL SHUFFLE EXPERIMENTS")


def sorting_experiments():
    logger.info("START SORTING EXPERIMENTS")

    batcher_failed = False
    quicksort_failed = False
    radixsort_failed = False

    for (
        i,
        j,
    ) in product(range(1, 5), range(1, 10, 2)):
        if batcher_failed and quicksort_failed and radixsort_failed:
            logger.warning("All algorithms failed")
            break

        seed = generate_seed()
        list_length = j * 10**i
        default_key_bit_length = 16
        base_args = [
            "python3",
            "benchmark.py",
            "-M3",
            "--benchmark",
            "sort",
            "--seed",
            str(seed),
            "--nb-rows",
            str(list_length),
            "--sorting-bit-length",
            str(default_key_bit_length),
        ]
        logger.info("Sorting experiment: seed=%d, length=%d", seed, list_length)

        if not quicksort_failed:
            subp = Popen(
                base_args + ["--algo", "quick"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            quicksort_failed = track_memory(subp)
        else:
            logger.warning("Skipped Quicksort")

        if not batcher_failed:
            subp = Popen(
                base_args + ["--algo", "batcher"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            batcher_failed = track_memory(subp)
        else:
            logger.warning("Skipped Batcher sort")

        for bit_length in [8, 16, 32, 48]:
            base_args = base_args[:-1] + [str(bit_length)]
            if not radixsort_failed:
                subp = Popen(
                    base_args + ["--algo", "radix"],
                    stdout=PIPE,
                    stderr=STDOUT,
                )
                radixsort_failed = track_memory(subp)
            else:
                logger.warning("Skipped radix sort")

    logger.info("FINISHED ALL SORTING EXPERIMENTS")


def dot_product_experiments():
    logger.info("START DOT PRODUCT EXPERIMENTS")
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(
        range(1, 9), range(1, 10, 2), [0.0001, 0.001, 0.005, 0.01]
    ):
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
            "vect_mult",
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

        if not dense_failed:
            subp = Popen(
                base_args + ["--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            logger.warning("Skipped dense experiments")

        if not sparse_failed:
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
    sparse001_failed = False
    sparse01_failed = False
    sparse05_failed = False
    sparse1_failed = False
    for i, j in product(range(2, 7), range(1, 10, 2)):
        if (
            dense_failed
            and sparse001_failed
            and sparse01_failed
            and sparse05_failed
            and sparse1_failed
        ):
            logger.warning("All algorithms failed")
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
            str(nb_rows),
            "--nb-cols",
            str(nb_cols),
            "--density",
        ]
        logger.info(
            "Matrix multiplication experiment: seed=%d, dimensions=%dx%d",
            seed,
            nb_rows,
            nb_cols,
        )

        if not dense_failed:
            subp = Popen(
                base_args + [str(0.001), "--algo", "dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            dense_failed = track_memory(subp)
        else:
            logger.warning("Skipped dense experiments")

        if not sparse001_failed:
            subp = Popen(
                base_args + [str(0.0001), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse001_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 0.01 percent experiments")

        if not sparse01_failed:
            subp = Popen(
                base_args + [str(0.001), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse01_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 0.1 percent experiments")

        if not sparse05_failed:
            subp = Popen(
                base_args + [str(0.005), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse05_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 0.5 percent experiments")

        if not sparse1_failed:
            subp = Popen(
                base_args + [str(0.01), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse1_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 1 percent experiments")

    logger.info("FINISHED ALL MATRIX MULTIPLICATION EXPERIMENTS")


def main():
    random.seed(74589312)
    archive_experiment_results()
    setup_logger()

    try:
        # sorting_experiments()
        # shuffle_experiments()
        # dot_product_experiments()
        matmult_experiments()
    except KeyboardInterrupt:  # To avoid memory leakage
        psutil_proc = psutil.Process(os.getpid())
        for proc in psutil_proc.children(recursive=True):
            proc.kill()
        raise


if __name__ == "__main__":
    main()
