from mpyc.runtime import mpc
import scipy.sparse
import numpy as np
from datetime import datetime
import random


class SortableTuple:
    def __init__(self, tup):
        self._tup = tuple(tup)

    def __lt__(self, other):
        if len(self._tup) != len(other._tup):
            raise ValueError("Tuples must be of same size")
        return SortableTuple._lt_tuples(self._tup, other._tup)

    def __ge__(self, other):
        return ~self.__lt__(other)

    def _lt_tuples(tup1, tup2):
        first_comp = tup1[0] < tup2[0]
        if len(tup1) > 2:  # The last element is not subject of the sort in our case
            equal_comp = tup1[0] == tup2[0]
            recursive_comp = SortableTuple._lt_tuples(tup1[1:], tup2[1:])
            # first_comp or (recursive_comp and equal_comp)
            return first_comp | (recursive_comp & equal_comp)
        else:
            return first_comp


def sparse_vector_dot_naive(vect1, vect2):
    # Sparse vector: [(ind, val),...]
    res = 0
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            sec_comp = vect1[i][0] == vect2[j][0]
            curr_mult_res = vect1[i][1] * vect2[j][1]
            curr_mult_res = mpc.if_else(sec_comp, curr_mult_res, 0)
            res += curr_mult_res
    return res


def sparse_vector_dot_naive_no_comp(vect1, vect2):
    # Sparse vector: [(ind, val),...]
    res = 0
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            sec_comp = mpc.SecInt(64)(1)
            curr_mult_res = vect1[i][1] * vect2[j][1]
            curr_mult_res = mpc.if_else(sec_comp, curr_mult_res, 0)
            res += curr_mult_res
    return res


def sparse_vector_dot(vect1, vect2):
    unsorted = vect1 + vect2
    sorted_list = mpc.sorted(unsorted, key=SortableTuple)
    res = 0
    for i in range(len(sorted_list) - 1):
        temp = sorted_list[i][1] * sorted_list[i + 1][1]
        sec_comp = sorted_list[i][0] == sorted_list[i + 1][0]
        temp = mpc.if_else(sec_comp, temp, 0)
        res += temp
    return res


def sparse_vector_dot_no_sort(vect1, vect2):
    unsorted = vect1 + vect2
    sorted_list = unsorted  # !!NO SORT!!
    res = 0
    for i in range(len(sorted_list) - 1):
        temp = sorted_list[i][1] * sorted_list[i + 1][1]
        sec_comp = sorted_list[i][0] == sorted_list[i + 1][0]
        temp = mpc.if_else(sec_comp, temp, 0)
        res += temp
    return res


async def bottleneck_comparison(n_dim):
    density = 0.01
    print(
        f"Bottlneck comparisons with parameters dimension={n_dim} and density={density}"
    )
    sectype = mpc.SecInt(64)
    to_sec_int = lambda x: sectype(int(x))

    x_sparse = scipy.sparse.random(n_dim, 1, density=density, dtype=np.int16).astype(
        int
    )
    y_sparse = scipy.sparse.random(n_dim, 1, density=density, dtype=np.int16).astype(
        int
    )

    x_sec = []
    for i, _j, v in zip(x_sparse.row, x_sparse.col, x_sparse.data):
        x_sec.append([to_sec_int(i), to_sec_int(v)])
    y_sec = []
    for i, _j, v in zip(y_sparse.row, y_sparse.col, y_sparse.data):
        y_sec.append([to_sec_int(i), to_sec_int(v)])

    start = datetime.now()
    z = sparse_vector_dot_no_sort(x_sec, y_sec)
    end = datetime.now()
    delta_sparse = end - start
    print("Time without sort:", delta_sparse.total_seconds())

    start = datetime.now()
    z = sparse_vector_dot(x_sec, y_sec)
    end = datetime.now()
    delta_sparse = end - start
    print("Time with sort:", delta_sparse.total_seconds())

    start = datetime.now()
    z = sparse_vector_dot_naive(x_sec, y_sec)
    end = datetime.now()
    delta_sparse = end - start
    print("Time naive dot with comparsion:", delta_sparse.total_seconds())

    start = datetime.now()
    z = sparse_vector_dot_naive_no_comp(x_sec, y_sec)
    end = datetime.now()
    delta_sparse = end - start
    print("Time naive dot without comparison:", delta_sparse.total_seconds())
    print("===")


async def other_measurements():
    print("Other measurements")
    sectype = mpc.SecInt(64)
    rand_list = [sectype(random.randint(0, 1024)) for _ in range(100)]

    start = datetime.now()
    mpc.random.shuffle(sectype, rand_list)
    end = datetime.now()
    delta_sparse = end - start
    print(f"Shuffle runtime ({len(rand_list)} elements):", delta_sparse.total_seconds())

    start = datetime.now()
    mpc.sorted(rand_list)
    end = datetime.now()
    delta_sparse = end - start
    print(f"Sort runtime ({len(rand_list)} elements):", delta_sparse.total_seconds())

    rand_list_tuple = [
        [sectype(random.randint(0, 1024)), sectype(0)] for _ in range(100)
    ]
    start = datetime.now()
    mpc.sorted(rand_list_tuple, key=SortableTuple)
    end = datetime.now()
    delta_sparse = end - start
    print(f"Sort runtime ({len(rand_list)} tuples):", delta_sparse.total_seconds())

    NB_REP = 1000
    start = datetime.now()
    for _i in range(NB_REP):
        rand_list[1] == rand_list[2]
    end = datetime.now()
    delta_sparse = end - start
    print("Average equality runtime:", delta_sparse.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in range(NB_REP):
        rand_list[1] < rand_list[2]
    end = datetime.now()
    delta_sparse = end - start
    print("Average inequality runtime:", delta_sparse.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in range(NB_REP):
        mpc.eq_public(rand_list[1], rand_list[2])
    end = datetime.now()
    delta_sparse = end - start
    print("Average public equality runtime:", delta_sparse.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in range(NB_REP):
        _ = rand_list[1] * rand_list[2]
    end = datetime.now()
    delta_sparse = end - start
    print("Average multiplication runtime:", delta_sparse.total_seconds() / NB_REP)


if __name__ == "__main__":
    mpc.run(bottleneck_comparison(1000))
    mpc.run(bottleneck_comparison(10000))
    mpc.run(bottleneck_comparison(100000))
    mpc.run(other_measurements())
