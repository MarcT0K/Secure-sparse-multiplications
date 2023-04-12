"""Implementation of the oblivious sort by Hamada et al. (2014)

URL: https://eprint.iacr.org/2014/121
"""

import math
import random

from mpyc.random import shuffle
from mpyc.runtime import mpc

from shuffle import np_shuffle


async def bin_vec_to_B(bin_vect):
    neg_vect = 1 - bin_vect
    B_t = mpc.np_vstack((bin_vect, neg_vect))
    return mpc.np_transpose(B_t)


def dest_comp(B, sectype):
    n, m = B.shape
    S = mpc.np_fromlist([sectype(0)] * n * m)
    S = mpc.np_reshape(S, (n, m))

    temp = 0
    for j in range(m):
        for i in range(n):
            temp += mpc.np_getitem(B, (i, j))
            mpc.np_update(S, (i, j), temp)

    T = mpc.np_multiply(S, B)
    return mpc.np_transpose(mpc.np_sum(T, axis=1))


async def reveal_sort(keys, data, sectype):
    assert keys.shape[0] == data.shape[0]

    if len(data.shape) == 1:
        merged = mpc.np_transpose(mpc.np_vstack((keys, data)))
    else:
        merged = mpc.np_transpose(
            mpc.np_vstack((mpc.np_transpose(keys), mpc.np_transpose(data)))
        )

    merged = await np_shuffle(sectype, merged)
    shuffled_keys = merged[:, 0]

    plaintext_keys = await mpc.output(shuffled_keys)

    sorted_indices = list(
        ind for (ind, _key) in sorted(enumerate(plaintext_keys), key=lambda tup: tup[1])
    )

    sorted_data = mpc.np_copy(merged)
    mpc.np_update(sorted_data, sorted_indices, merged)
    return sorted_data[:, 1:]  # remove the plaintext key


def int_to_secure_bits(number, sectype, nb_bits):
    bitstring = format(number, f"0{nb_bits}b")
    return [sectype(int(c)) for c in bitstring]


async def radix_sort(data, sectype, nb_bits=None):
    n, l = data.shape
    if nb_bits is None:
        l -= 1  # the last dimension is the complete key
    else:
        l = nb_bits

    assert n > 0 and l > 0

    h = mpc.np_fromlist([sectype(i) for i in range(n)])
    h_j = mpc.np_copy(h)
    bp_j = data[:, 0]

    for i in range(l + 1):
        B_j = await bin_vec_to_B(bp_j)
        c_j = dest_comp(B_j, sectype)
        cp_j = await reveal_sort(h_j, c_j, sectype)

        if i < l:
            b_jpp = data[:, i + 1]
            concat_res = mpc.np_transpose(mpc.np_vstack((b_jpp, h_j)))
            sort_res = await reveal_sort(cp_j, concat_res, sectype)

            h_j = sort_res[:, -1]
            bp_j = mpc.np_transpose(sort_res[:, :-1])
        else:
            res = await reveal_sort(cp_j, data, sectype)
    return res


async def main():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 1) for _ in range(1000)]
    else:
        rand_list = []

    rand_list = await mpc.transfer(rand_list, senders=0)

    nb_bits = int(math.log(max(rand_list), 2)) + 1
    print(nb_bits)

    l = mpc.np_vstack(
        [
            mpc.np_fromlist(int_to_secure_bits(r, sectype, nb_bits) + [sectype(r)])
            for r in rand_list
        ]
    )
    print(l)
    print(l.shape)
    print(await mpc.output(l))

    s = await radix_sort(l, sectype=sectype)
    print(await mpc.output(s))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main())
