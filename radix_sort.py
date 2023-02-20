"""Implementation of the oblivious sort by Hamada et al. (2014)

URL: https://eprint.iacr.org/2014/121
"""

from mpyc.runtime import mpc
from mpyc.random import shuffle


def bin_vec_to_B(bin_vect):
    return [[bit, 1 - bit] for bit in bin_vect]


def dest_comp(B):
    n = len(B)
    m = len(B[0])
    S = [[0] * m] * n

    temp = 0
    for j in range(m):
        for i in range(n):
            temp += B[i, j]
            S[i, j] = temp

    T = mpc.np_multiply(S, B)
    return mpc.np_sum(T, axis=1)


async def reveal_sort(keys, data, sectype):
    assert len(keys) == len(data)
    merged = [[keys[i]] + data[i] for i in range(len(data))]
    shuffle(sectype, merged)
    shuffled_keys = [merged[i][0] for i in range(len(data))]
    plaintext_keys = await mpc.output(shuffled_keys)
    merged = [[plaintext_keys[i]] + data[i] for i in range(len(data))]
    sorted_data = sorted(merged, key=lambda tup: tup[0])
    return [tup[1:] for tup in sorted_data]  # remove the plaintext key
