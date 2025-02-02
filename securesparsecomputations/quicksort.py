"""Implementation of the oblivious quicksort by Hamada et al. (2013).

DOI: 10.1007/978-3-642-37682-5_15
"""
from mpyc.runtime import mpc

from .resharing import np_shuffle_3PC


async def np_quicksort(sec_arr, key=None):
    """Parallel implementation of the oblivious quicksort."""
    res = mpc.np_copy(sec_arr)
    init_shape = res.shape

    if init_shape[0] == 1:
        return res

    if len(mpc.parties) == 3:
        res = mpc.np_tolist(await np_shuffle_3PC(res))
    else:
        raise NotImplementedError

    key_func = (lambda x: x) if key is None else key
    pivots = [-1, len(res)]

    key_vect = [key_func(res[i]) * 1000 for i in range(len(res))]

    while len(pivots) - 2 != len(res):
        end_of_partitions = [
            pivots[ind + 1] - 1
            for ind in range(len(pivots) - 1)
            for _i in range(pivots[ind] + 1, pivots[ind + 1] - 1)
        ]
        end_of_partitions_vect = mpc.np_fromlist(
            [key_vect[p] for p in end_of_partitions]
        )
        val_vect = mpc.np_fromlist(
            [
                key_vect[i]
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
                key_vect[p], key_vect[j] = key_vect[j], key_vect[p]
                pivots.append(p)
            else:
                if plaintext_comp[j - pivot_count]:
                    i += 1
                    res[i], res[j] = res[j], res[i]
                    key_vect[i], key_vect[j] = key_vect[j], key_vect[i]

        pivots.sort()
        i = 0
        while i < len(pivots) - 1:  # To remove paritions of size 1
            if pivots[i + 1] - pivots[i] == 2:
                pivots = pivots[: i + 1] + [pivots[i] + 1] + pivots[i + 1 :]
            i += 1
    temp = []
    for el in res:
        if len(init_shape) == 1:
            temp.append(el)
        else:
            temp += el

    return mpc.np_reshape(mpc.np_fromlist(temp), init_shape)
