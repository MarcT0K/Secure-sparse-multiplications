"""Implementation of the oblivious sort by Hamada et al. (2014)

URL: https://eprint.iacr.org/2014/121
"""

import math
import random

from mpyc.runtime import mpc

from shuffle import np_shuffle
from resharing import np_shuffle_3PC


async def bin_vec_to_B(bin_vect):
    neg_vect = 1 - bin_vect
    B_t = mpc.np_vstack((bin_vect, neg_vect))
    return mpc.np_transpose(B_t)


async def dest_comp(B):
    sectype = type(B).sectype
    n, m = B.shape
    S = []

    temp = sectype(0)
    B_arr = await mpc.gather(B)
    for j in range(m):
        for i in range(n):
            temp += B[i, j]
            S.append(temp)

    S = mpc.np_transpose(mpc.np_reshape(mpc.np_fromlist(S), (m, n)))

    T = mpc.np_multiply(S, B)
    return mpc.np_sum(T, axis=1) - 1


async def reveal_sort(keys, data, sectype):
    assert keys.shape[0] == data.shape[0]

    if len(data.shape) == 1:
        merged = mpc.np_transpose(mpc.np_vstack((keys, data)))
    else:
        merged = mpc.np_transpose(
            mpc.np_vstack((mpc.np_transpose(keys), mpc.np_transpose(data)))
        )

    if len(mpc.parties) != 3:
        await np_shuffle(merged)
    else:
        merged = await np_shuffle_3PC(merged)

    plaintext_keys = await mpc.output(merged[:, 0])

    # NB: In the radix sort, the plaintext keys are already the indices
    sorted_indices = [i for i in plaintext_keys]

    sorted_data = mpc.np_copy(merged)
    sorted_data = mpc.np_update(sorted_data, sorted_indices, merged)
    return sorted_data[:, 1:]  # remove the plaintext key


def int_to_secure_bits(number, sectype, nb_bits):
    bitstring = format(number, f"0{nb_bits}b")
    return [sectype(int(c)) for c in bitstring][::-1]


async def radix_sort(
    data, key_bitlength, sectype, desc=False, already_decomposed=False
):  # TODO: handle the descending order
    n, l = data.shape

    if already_decomposed:
        assert l > key_bitlength
    else:
        raise NotImplementedError

    h = mpc.np_fromlist([sectype(i) for i in range(n)])
    h_j = mpc.np_copy(h)
    bp_j = data[:, 0]

    res = None

    for i in range(key_bitlength):
        B_j = await bin_vec_to_B(bp_j)
        c_j = await dest_comp(B_j)
        cp_j = await reveal_sort(h_j, c_j, sectype)

        if i < key_bitlength - 1:
            b_jpp = data[:, i + 1]
            concat_res = mpc.np_transpose(mpc.np_vstack((b_jpp, h)))
            sort_res = await reveal_sort(cp_j, concat_res, sectype)

            h_j = sort_res[:, -1]
            bp_j = mpc.np_transpose(sort_res[:, :-1])
        else:
            res = await reveal_sort(cp_j, data, sectype)

    if already_decomposed:
        res = res[:, key_bitlength:]

    if desc:  # Descending order
        return res

    return res[::-1, :]


async def main():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        rand_list = [random.randint(0, 10**5) for _ in range(1000)]
    else:
        rand_list = []

    rand_list = await mpc.transfer(rand_list, senders=0)

    nb_bits = int(math.log(max(rand_list), 2)) + 1

    l = mpc.np_vstack(
        [
            mpc.np_fromlist(int_to_secure_bits(r, sectype, nb_bits) + [sectype(r)])
            for r in rand_list
        ]
    )

    print("INIT:", await mpc.output(l))

    s = await radix_sort(l, nb_bits, sectype=sectype, already_decomposed=True)
    print(await mpc.output(s))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main())
