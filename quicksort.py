"""Implementation of the oblivious quicksort by Hamada et al. (2013).

DOI: 10.1007/978-3-642-37682-5_15
"""

from mpyc.runtime import mpc
import datetime
import random

from shuffle import np_shuffle


async def quicksort(sec_list, sectype, rec_call=False, key=None):
    if not rec_call:
        mpc.random.shuffle(sectype, sec_list)

    if 1 < len(sec_list):
        p, curr_list = await partition(sec_list, key=key)
        left_part = await quicksort(curr_list[:p], sectype, True, key=key)
        right_part = await quicksort(curr_list[p + 1 :], sectype, True, key=key)
        return left_part + [curr_list[p]] + right_part
    else:
        return sec_list


async def partition(sec_list, key=None):
    res = sec_list.copy()
    i = -1
    for j in range(len(res) - 1):
        if key is None:
            sec_comp = res[j] <= res[len(res) - 1]
        else:
            sec_comp = key(res[j]) <= key(res[len(res) - 1])

        comp = await mpc.output(sec_comp)
        if comp:
            i += 1
            res[i], res[j] = res[j], res[i]
    p = i + 1
    res[p], res[len(res) - 1] = res[len(res) - 1], res[p]
    return p, res


async def parallel_quicksort(sec_arr, sectype, key=None):
    """Parallel implementation of the oblivious quicksort."""
    res = mpc.np_copy(sec_arr)
    n, m = res.shape
    if n == 1:
        return res

    res = mpc.np_tolist(await np_shuffle(sectype, res))

    key_func = (lambda x: x) if key is None else key
    pivots = [-1, len(res)]

    while len(pivots) - 2 != len(res):
        end_of_partitions = [
            pivots[ind + 1] - 1
            for ind in range(len(pivots) - 1)
            for _i in range(pivots[ind] + 1, pivots[ind + 1] - 1)
        ]
        end_of_partitions_vect = mpc.np_fromlist(
            [key_func(res[p]) for p in end_of_partitions]
        )
        val_vect = mpc.np_fromlist(
            [
                key_func(res[i])
                for i in range(len(res))
                if i not in pivots and i not in end_of_partitions
            ]
        )
        sec_comp = end_of_partitions_vect >= val_vect
        plaintext_comp = await mpc.output(sec_comp)

        i = -1
        pivot_count = 0
        for j in range(len(res)):
            if j in pivots:
                pivot_count += 1
                i = j
            elif j in end_of_partitions:
                p = i + 1
                pivot_count += 1
                res[p], res[j] = res[j], res[p]
                pivots.append(p)
            else:
                if plaintext_comp[j - pivot_count]:
                    i += 1
                    res[i], res[j] = res[j], res[i]

        pivots.sort()
        i = 0
        while i < len(pivots) - 1:  # To remove paritions of size 1
            if pivots[i + 1] - pivots[i] == 2:
                pivots = pivots[: i + 1] + [pivots[i] + 1] + pivots[i + 1 :]
            i += 1
    temp = []
    for tup in res:
        temp += tup
    return mpc.np_reshape(mpc.np_fromlist(temp), (n, m))


async def test():
    await mpc.start()
    sectype = mpc.SecInt(32)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 1024) for _ in range(10)]
    else:
        rand_list = []

    rand_list = await mpc.transfer(rand_list, senders=0)

    l = mpc.np_reshape(
        mpc.np_fromlist([sectype(i) for i in rand_list]), (len(rand_list) // 2, 2)
    )
    print("Initial:", await mpc.output(l))
    l_p = await parallel_quicksort(l, sectype, key=lambda tup: tup[0])
    print("Resultat", await mpc.output(l_p))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
