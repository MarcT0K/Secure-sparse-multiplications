"""Implementation of the oblivious quicksort by Hamada et al. (2013).

DOI: 10.1007/978-3-642-37682-5_15
"""

from mpyc.runtime import mpc
import datetime
import random


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


async def parallel_quicksort(sec_list, sectype, key=None):
    """Parallel implementation of the oblivious quicksort."""

    start = datetime.datetime.now()
    res = sec_list.copy()
    if len(res) == 1:
        return res

    start_shuffle = datetime.datetime.now()
    mpc.random.shuffle(sectype, res)
    total_shuffle = datetime.datetime.now() - start_shuffle

    key_func = lambda x: x if key is None else key
    pivots = [-1, len(res)]

    total_comp = start - start
    count = 0
    while len(pivots) - 2 != len(res):
        count += 1
        print(len(pivots))
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
        start_comp = datetime.datetime.now()
        sec_comp = end_of_partitions_vect >= val_vect
        await mpc.barrier()
        end_comp = datetime.datetime.now()
        total_comp += end_comp - start_comp
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
    end = datetime.datetime.now()
    print(f"Comparison time: {total_comp.total_seconds()}s")
    print(f"Shuffle time: {total_shuffle.total_seconds()}s")
    print(f"Total: {(end - start).total_seconds()}s")
    print("Number of rounds: ", count)
    return res


async def test():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 1024) for _ in range(1000)]
    else:
        rand_list = []

    rand_list = await mpc.transfer(rand_list, senders=0)

    l = [sectype(i) for i in rand_list]
    print("Initial:", await mpc.output(l))
    l_p = await parallel_quicksort(l, sectype)
    print("Resultat", await mpc.output(l_p))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
