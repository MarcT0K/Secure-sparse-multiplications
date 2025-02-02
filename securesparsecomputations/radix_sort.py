"""Implementation of the oblivious radix sort by Hamada et al. (2014)

URL: https://eprint.iacr.org/2014/121
"""

from mpyc.runtime import mpc

from .resharing import np_shuffle_3PC


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


async def reveal_sort(keys, data):
    assert keys.shape[0] == data.shape[0]

    if len(data.shape) == 1:
        merged = mpc.np_transpose(mpc.np_vstack((keys, data)))
    else:
        merged = mpc.np_transpose(
            mpc.np_vstack((mpc.np_transpose(keys), mpc.np_transpose(data)))
        )

    if len(mpc.parties) != 3:
        raise NotImplementedError
    else:
        merged = await np_shuffle_3PC(merged)

    plaintext_keys = await mpc.output(merged[:, 0])

    # NB: In the radix sort, the plaintext keys are already the indices
    sorted_indices = [int(i) for i in plaintext_keys]

    sorted_data = mpc.np_copy(merged)
    sorted_data = mpc.np_update(sorted_data, sorted_indices, merged)
    return sorted_data[:, 1:]  # remove the plaintext key


def int_to_secure_bits(number, sectype, nb_bits):
    bitstring = format(number, f"0{nb_bits}b")
    return [sectype(int(c)) for c in bitstring][::-1]


async def radix_sort(
    data, key_bitlength, desc=False, already_decomposed=False, keep_bin_keys=False
):
    assert not (not already_decomposed and keep_bin_keys)

    n, l = data.shape

    if already_decomposed:
        assert l > key_bitlength
    else:
        raise NotImplementedError

    sectype = type(data).sectype
    h = mpc.np_fromlist([sectype(i) for i in range(n)])
    h_j = mpc.np_copy(h)
    bp_j = data[:, 0]

    res = None

    for i in range(key_bitlength):
        B_j = await bin_vec_to_B(bp_j)
        c_j = await dest_comp(B_j)
        cp_j = await reveal_sort(h_j, c_j)

        if i < key_bitlength - 1:
            b_jpp = data[:, i + 1]
            concat_res = mpc.np_transpose(mpc.np_vstack((b_jpp, h)))
            sort_res = await reveal_sort(cp_j, concat_res)

            h_j = sort_res[:, -1]
            bp_j = mpc.np_transpose(sort_res[:, :-1])
        else:
            res = await reveal_sort(cp_j, data)

    if already_decomposed:
        if not keep_bin_keys:
            res = res[:, key_bitlength:]

    if desc:  # Descending order
        return res

    return res[::-1, :]
