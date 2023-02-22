"""Implementation of the oblivious sort by Hamada et al. (2014)

URL: https://eprint.iacr.org/2014/121
"""

from mpyc.runtime import mpc
from mpyc.random import shuffle

import random
import math


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
    return mpc.np_sum(T, axis=1)


async def reveal_sort(keys, data, sectype):
    # TODO: make a difference when data is a vector and when it is a matrix
    assert len(keys) == len(data)
    merged = [[keys[i]] + [data[i]] for i in range(len(data))]
    shuffle(sectype, merged)
    shuffled_keys = [merged[i][0] for i in range(len(data))]
    plaintext_keys = await mpc.output(shuffled_keys)
    merged = [[plaintext_keys[i]] + [data[i]] for i in range(len(data))]
    sorted_data = sorted(merged, key=lambda tup: tup[0])
    return [tup[1:] for tup in sorted_data]  # remove the plaintext key


def int_to_secure_bits(number, sectype, nb_bits):
    bitstring = format(number, f"0{nb_bits}b")
    return [sectype(int(c)) for c in bitstring]


async def radix_sort(data, sectype):
    assert len(data[0]) > 0  # list of tuples [key, values...]
    assert len(data[0][0]) > 0  # Key is a tuple of secure bits

    l = len(data[0][0])
    n = len(data)
    h = mpc.np_fromlist([sectype(i) for i in range(n)])
    bit_keys = [mpc.np_fromlist([data[i][0][j] for i in range(n)]) for j in range(l)]
    h_j = h.copy()
    bp_j = bit_keys[0]
    for i in range(l + 1):
        print("HERE")
        B_j = await bin_vec_to_B(bp_j)
        c_j = dest_comp(B_j, sectype)
        print("HEREA")
        cp_j = await reveal_sort(h_j, c_j, sectype)
        print("HEREB")

        if i < l:
            b_jpp = bit_keys[i + 1]
            concat_res = [b_jpp[k] + [h_j[k]] for k in range(n)]
            sort_res = await reveal_sort(cp_j, concat_res, sectype)
            h_j = []
            bp_j = []
            for k in range(n):  # We extract h_{j+1} and b_{j+1}
                h_j.append(sort_res[k][-1])
                bp_j.append(sort_res[k][:-1])
        else:
            res = await reveal_sort(cp_j, data, sectype)
    return res


async def main():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 1024) for _ in range(1000)]
    else:
        rand_list = []

    rand_list = await mpc.transfer(rand_list, senders=0)

    nb_bits = int(math.log(max(rand_list), 2)) + 1
    print(nb_bits)

    l = [
        [
            int_to_secure_bits(r, sectype, nb_bits),
            sectype(r),
        ]
        for r in rand_list
    ]
    sorted = await radix_sort(l, sectype=sectype)
    print(await mpc.output([c for _, c in sorted]))
    await mpc.shutdown()


# TODO: We should now improve the shuffle because it is the main bottleneck for the radix sort
# TODO: implement the radix sort of Bogdanov et al.

if __name__ == "__main__":
    mpc.run(main())
