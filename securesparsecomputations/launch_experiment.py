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


def vect_mult_experiments():
    logger.info("START VECTOR MULTIPLICATION EXPERIMENTS")
    dense_failed = False
    sparse_failed = False
    for i, j, density in product(range(4, 7), range(1, 10, 2), [0.0001, 0.001, 0.01]):
        if dense_failed and sparse_failed:
            logger.warning("Both algorithms failed")
            break

        seed = generate_seed()
        nb_rows = j * 10**i
        base_args = [
            "benchmark",
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
            "Vector multiplication experiment: seed=%d, dimensions=%d, density=%.4f",
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

    logger.info("FINISHED ALL VECTOR MULTIPLICATION EXPERIMENTS")


def sparse_dense_vect_mult_experiments():
    logger.info("START SPARSE-DENSE VECTOR MULTIPLICATION EXPERIMENTS")
    dense_failed = False
    sparse_failed = False
    sparse_dense_failed = False
    for i, j, density in product(range(4, 6), range(1, 10, 2), [0.0001]):
        if dense_failed and sparse_failed and sparse_dense_failed:
            logger.warning("All algorithms failed")
            break

        seed = generate_seed()
        nb_rows = j * 10**i
        base_args = [
            "benchmark",
            "-M3",
            "--benchmark",
            "sparse_dense_vect_mult",
            "--seed",
            str(seed),
            "--nb-rows",
            str(nb_rows),
            "--density",
            str(density),
        ]
        logger.info(
            "Vector multiplication experiment: seed=%d, dimensions=%d, density=%.4f",
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

        if not sparse_dense_failed:
            subp = Popen(
                base_args + ["--algo", "sparse-dense"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse_dense_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse-dense experiments")

    logger.info("FINISHED ALL SPARSE-DENSE VECTOR MULTIPLICATION EXPERIMENTS")


def mat_vect_mult_experiments():
    logger.info("START MATRIX-VECTOR MULTIPLICATION EXPERIMENTS")
    dense_failed = False
    sparse001_failed = False
    sparse01_failed = False
    sparse1_failed = False

    for i, j in product(range(2, 6), range(1, 10, 2)):
        if dense_failed and sparse001_failed and sparse01_failed and sparse1_failed:
            logger.warning("All algorithms failed")
            break

        seed = generate_seed()
        nb_rows = j * 10**i
        base_args = [
            "benchmark",
            "-M3",
            "--benchmark",
            "mat_vect_mult",
            "--seed",
            str(seed),
            "--nb-rows",
            str(nb_rows),
            "--density",
        ]
        logger.info(
            "Matrix-vector multiplication experiment: seed=%d, dimensions=(%d, %d)",
            seed,
            nb_rows,
            nb_rows,
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

        if not sparse1_failed:
            subp = Popen(
                base_args + [str(0.01), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse1_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 1 percent experiments")

    logger.info("FINISHED ALL MATRIX-VECTOR MULTIPLICATION EXPERIMENTS")


def matmult_experiments():
    logger.info("START MATRIX-MATRIX MULTIPLICATION EXPERIMENTS")
    dense_failed = False
    sparse001_failed = False
    sparse01_failed = False
    sparse1_failed = False
    for i, j in product(range(2, 6), range(1, 10, 2)):
        if dense_failed and sparse001_failed and sparse01_failed and sparse1_failed:
            logger.warning("All algorithms failed")
            break

        seed = generate_seed()
        nb_cols = j * 10**i
        nb_rows = 10**2
        base_args = [
            "benchmark",
            "-M3",
            "--no-prss",  # Reason: https://github.com/lschoe/mpyc/issues/78
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
            "Matrix multiplication experiment: seed=%d, dimensions=(%d, %d)",
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

        if not sparse1_failed:
            subp = Popen(
                base_args + [str(0.01), "--algo", "sparse"],
                stdout=PIPE,
                stderr=STDOUT,
            )
            sparse1_failed = track_memory(subp)
        else:
            logger.warning("Skipped sparse 1 percent experiments")

    logger.info("FINISHED ALL MATRIX-MATRIX MULTIPLICATION EXPERIMENTS")


def spam_detection_application():
    logger.info("START SPAM DETECTION EXPERIMENT")
    subp = Popen(
        ["benchmark_spam_detection", "-M3"],
        stdout=PIPE,
        stderr=STDOUT,
    )
    track_memory(subp)
    logger.info("FINISHED SPAM DETECTION EXPERIMENT")


def recommender_system_application():
    logger.info("START RECOMMENDER SYSTEM EXPERIMENT")
    subp = Popen(
        ["benchmark_recommender", "-M3"],
        stdout=PIPE,
        stderr=STDOUT,
    )
    track_memory(subp)
    logger.info("FINISHED RECOMMENDER SYSTEM EXPERIMENT")


def access_control_application():
    logger.info("START ACCESS CONTROL EXPERIMENT")
    subp = Popen(
        ["benchmark_access_control", "-M3", "--no-prss"],
        stdout=PIPE,
        stderr=STDOUT,
    )
    track_memory(subp)
    logger.info("FINISHED ACCESS CONTROL EXPERIMENT")


def clean_csv():
    """Removes from the CSV files all lines for experiments that crashed during computation times.

    Hence, we remove the isolated secret-sharing results."""
    for fname in glob.glob("*.csv"):
        with open(fname, "r", encoding="utf-8") as csv_file:
            lines = csv_file.readlines()

        with open(fname + ".old", "w", encoding="utf-8") as archive_file:
            archive_file.writelines(fname)  # Keep the unfiltered version

        filtered_lines = [
            lines[i]
            for i in range(len(lines) - 1)
            if not ("sharing" in lines[i] and "sharing" in lines[i + 1])
        ]

        if "sharing" not in lines[-1]:
            filtered_lines.append(lines[-1])

        with open(fname, "w", encoding="utf-8") as csv_file:
            csv_file.writelines(filtered_lines)


def main():
    random.seed(74589312)
    if not os.path.exists("data"):
        os.makedirs("data")
    os.chdir("data")
    archive_experiment_results()
    setup_logger()

    try:
        vect_mult_experiments()
        sparse_dense_vect_mult_experiments()
        mat_vect_mult_experiments()
        matmult_experiments()
        spam_detection_application()
        # recommender_system_application()
        # access_control_application()
    except KeyboardInterrupt:  # To avoid memory leakage
        psutil_proc = psutil.Process(os.getpid())
        for proc in psutil_proc.children(recursive=True):
            proc.kill()
        raise

    clean_csv()
