from mpyc.runtime import mpc
import mpyc.numpy as sec_np
import scipy.sparse
import numpy as np
from datetime import datetime
import random
import gmpy2
from quicksort import quicksort
import tqdm


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


async def other_measurements_32bits():
    print("Other measurements")
    sectype = mpc.SecInt(32)
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


async def investigation():
    print("Investigation")
    sectype = mpc.SecInt(32)
    x = sectype(-2)
    start = datetime.now()
    for _i in range(100):
        y = x >> 64
    end = datetime.now()
    delta = end - start
    print("Average shift runtime:", delta.total_seconds() / 100)

    start = datetime.now()
    for _i in range(100):
        y = x % 2**64
    end = datetime.now()
    delta = end - start
    print("Average modulo runtime:", delta.total_seconds() / 100)


async def overflow():
    sectype = mpc.SecInt(64)

    i = 2**62
    x = sectype(i)
    y = x >> 64
    res1 = await mpc.output(y)
    print(res1)
    x = sectype(-i)
    y = x >> 64
    res2 = await mpc.output(y)
    cond = res1 == 0 and res2 != 0
    print(res2)


@mpc.coroutine
async def bsgn_0(a):
    """Compute binary sign of a securely.
    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing (2a+1 | p).
    Legendre symbols (a | p) for secret a are computed securely by evaluating
    (a s r^2 | p) in the clear for secret random sign s and secret random r modulo p,
    and outputting secret s * (a s r^2 | p).
    """
    stype = type(a)
    await mpc.returnType(stype)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 1, signed=True)  # random sign
    r = mpc._random(Zp)
    r = mpc.prod([r, r])  # random square modulo p
    a, s, r = await mpc.gather(a, s, r)
    b = await mpc.prod([2 * a + 1, s[0], r])
    b = await mpc.output(b)
    return s[0] * legendre_p(b)


@mpc.coroutine
async def bsgn_2(a):
    """Compute binary sign of a securely.
    Binary sign of a (1 if a>=0 else -1) is obtained by securely computing
    (t | p), with t = sum((2a+1+2i | p) for i=-2,-1,0,1,2).
    """
    stype = type(a)
    await mpc.returnType(stype)
    Zp = stype.field
    p = Zp.modulus
    legendre_p = lambda a: gmpy2.legendre(a.value, p)

    s = mpc.random_bits(Zp, 6, signed=True)  # 6 random signs
    r = mpc._randoms(Zp, 6)
    r = mpc.schur_prod(r, r)  # 6 random squares modulo p
    a, s, r = await mpc.gather(a, s, r)
    y = [b + 2 * i for b in (2 * a + 1,) for i in (-2, -1, 0, 1, 2)]
    y = await mpc.schur_prod(y, s[:-1])
    y.append(s[-1])
    y = await mpc.schur_prod(y, r)
    y = await mpc.output(y)
    t = sum(s[i] * legendre_p(y[i]) for i in range(5))
    t = await mpc.output(t * y[-1])
    return s[-1] * legendre_p(t)


async def benchmark_bsgn():
    secint = mpc.SecInt(14, p=15569949805843283171)
    x = secint(-3)
    NB_REP = 1000
    start = datetime.now()
    for _i in range(NB_REP):
        s = bsgn_2(x)
    end = datetime.now()
    delta = end - start
    print("Average bsgn2 runtime:\t", delta.total_seconds() / NB_REP)
    start = datetime.now()
    for _i in range(NB_REP):
        s = x < 0
    end = datetime.now()
    delta = end - start
    print("Average sgn runtime:\t", delta.total_seconds() / NB_REP)

    secint = mpc.SecInt(14, p=3546374752298322551)
    x = secint(-3)
    NB_REP = 1000
    start = datetime.now()
    for _i in range(NB_REP):
        s = bsgn_0(x)
    end = datetime.now()
    delta = end - start
    print("Average bsgn0 runtime:\t", delta.total_seconds() / NB_REP)
    start = datetime.now()
    for _i in range(NB_REP):
        s = x < 0
    end = datetime.now()
    delta = end - start
    print("Average sgn runtime:\t", delta.total_seconds() / NB_REP)


async def benchmark_sort(n_dim):
    print("Benchmark for oblivious sort coroutines")
    sectype = mpc.SecInt(64)
    rand_list = [sectype(random.randint(0, 1024)) for _ in range(n_dim)]
    NB_REP = 10
    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Batcher sort"):
        mpc.sorted(rand_list)
    end = datetime.now()
    delta = end - start
    print("Average batcher sort runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Quick sort"):
        await quicksort(rand_list, sectype)
    end = datetime.now()
    delta = end - start
    print("Average batcher sort runtime:\t", delta.total_seconds() / NB_REP)
    print("===END")


async def benchmark_vectorized_comp(n_dim):
    print("Benchmark for vectorized comparison")
    rand_list = [random.randint(-1024, 1024) for _ in range(n_dim)]

    sectype = mpc.SecInt(64)
    sec_list = sectype.array(sec_np.np.array(rand_list))

    NB_REP = 10
    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Naive comparison"):
        for j in range(len(sec_list)):
            x = mpc.sgn(sec_list[j], LT=True)
    end = datetime.now()
    delta = end - start
    print("Average naive inequality runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Vectorized comparison"):
        res = mpc.np_sgn(sec_list, LT=True)

    end = datetime.now()
    delta = end - start
    print("Average vectorized inequality runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Naive comparison"):
        for j in range(len(sec_list)):
            x = mpc.sgn(sec_list[j], EQ=True)
    end = datetime.now()
    delta = end - start
    print("Average naive equality runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Vectorized comparison"):
        res = mpc.np_sgn(sec_list, EQ=True)

    end = datetime.now()
    delta = end - start
    print("Average vectorized equality runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Naive comparison"):
        for j in range(len(sec_list)):
            x = await mpc.is_zero_public(sec_list[j])
    end = datetime.now()
    delta = end - start
    print("Average naive public equality runtime:\t", delta.total_seconds() / NB_REP)

    start = datetime.now()
    for _i in tqdm.tqdm(iterable=range(NB_REP), desc="Vectorized comparison"):
        res = await mpc.np_is_zero_public(sec_list)

    end = datetime.now()
    delta = end - start
    print(
        "Average vectorized public equality runtime:\t", delta.total_seconds() / NB_REP
    )
    print("===END")


if __name__ == "__main__":
    # mpc.run(bottleneck_comparison(1000))
    # mpc.run(bottleneck_comparison(10000))
    # mpc.run(bottleneck_comparison(100000))
    # mpc.run(other_measurements())
    # mpc.run(other_measurements_32bits())
    # mpc.run(investigation())
    # mpc.run(overflow())
    # mpc.run(benchmark_bsgn())
    # mpc.run(benchmark_sort(100))
    mpc.run(benchmark_vectorized_comp(1000))
