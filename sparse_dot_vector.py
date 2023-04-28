from mpyc.runtime import mpc

from quicksort import parallel_quicksort, quicksort
from sortable_tuple import SortableTuple
from radix_sort import radix_sort


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


def sparse_vector_dot_np(vect1, vect2):
    unsorted = mpc.np_vstack((vect1, vect2))
    sorted_array = mpc.np_sort(unsorted, axis=0, key=lambda tup: tup[0])

    n = sorted_array.shape[0]
    mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
    comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
    return mpc.np_sum(mult_vect * comp_vect)


async def sparse_vector_dot_radix(vect1, vect2, key_bit_length):
    unsorted = mpc.np_vstack((vect1, vect2))
    sorted_array = await radix_sort(unsorted, key_bit_length, already_decomposed=True)

    n = sorted_array.shape[0]
    mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
    comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
    return mpc.np_sum(mult_vect * comp_vect)


async def sparse_vector_dot_quicksort(vect1, vect2, sectype, key=None):
    unsorted = vect1 + vect2
    sorted_list = await quicksort(unsorted, sectype, False, key=lambda tup: tup[0])

    res = 0
    for i in range(len(sorted_list) - 1):
        temp = sorted_list[i][1] * sorted_list[i + 1][1]
        sec_comp = sorted_list[i][0] == sorted_list[i + 1][0]
        temp = mpc.if_else(sec_comp, temp, 0)
        res += temp
    return res


async def sparse_vector_dot_parallel_quicksort(vect1, vect2, sectype, key=None):
    unsorted = mpc.np_vstack((vect1, vect2))
    n = unsorted.shape[0]

    sorted_array = await parallel_quicksort(unsorted, key)

    mult_vect = sorted_array[0 : n - 1, 1] * sorted_array[1:n, 1]
    comp_vect = sorted_array[0 : n - 1, 0] == sorted_array[1:n, 0]
    return mpc.np_sum(mult_vect * comp_vect)


def sparse_vector_dot_naive(vect1, vect2):
    res = 0
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            temp = vect1[i][1] * vect2[j][1]
            sec_comp = vect1[i][0] == vect2[j][0]
            temp = mpc.if_else(sec_comp, temp, 0)
            res += temp
    return res


def sparse_vector_dot_naive_opti(vect1, vect2):
    val1 = [vect1[i][1] for i in range(len(vect1))]
    ind1 = [vect1[i][0] for i in range(len(vect1))]
    val1_ext = mpc.np_fromlist(val1 * len(vect2))
    ind1_ext = mpc.np_fromlist(ind1 * len(vect2))

    val2 = [vect2[i][1] for i in range(len(vect1))]
    ind2 = [vect2[i][0] for i in range(len(vect1))]

    val2_ext = []
    ind2_ext = []
    for j in range(len(vect2)):
        val2_ext += [val2[j]] * len(vect1)
        ind2_ext += [ind2[j]] * len(vect1)
    ind2_ext = mpc.np_fromlist(ind2_ext)
    val2_ext = mpc.np_fromlist(val2_ext)

    eq_res = mpc.np_equal(ind1_ext, ind2_ext)

    mult_res = mpc.np_multiply(val2_ext, val1_ext)
    mult_res = mpc.np_multiply(mult_res, eq_res)
    res = mpc.np_sum(mult_res)
    return res


async def sparse_vector_dot_psi(vect1, vect2, sectype):
    res = sectype(0)
    mpc.random.shuffle(sectype, vect1)
    mpc.random.shuffle(sectype, vect2)
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            temp = vect1[i][1] * vect2[j][1]
            comp = await mpc.is_zero_public(vect1[i][0] - vect2[j][0])
            if comp:
                res += temp
    return res


async def sparse_vector_dot_psi_opti(vect1, vect2, sectype):
    res = sectype(0)

    mpc.random.shuffle(sectype, vect1)
    mpc.random.shuffle(sectype, vect2)

    val1 = [vect1[i][1] for i in range(len(vect1))]
    ind1 = [vect1[i][0] for i in range(len(vect1))]
    val1_ext = mpc.np_fromlist(val1 * len(vect2))
    ind1_ext = mpc.np_fromlist(ind1 * len(vect2))

    val2 = [vect2[i][1] for i in range(len(vect1))]
    ind2 = [vect2[i][0] for i in range(len(vect1))]

    val2_ext = []
    ind2_ext = []
    for j in range(len(vect2)):
        val2_ext += [val2[j]] * len(vect1)
        ind2_ext += [ind2[j]] * len(vect1)
    ind2_ext = mpc.np_fromlist(ind2_ext)
    val2_ext = mpc.np_fromlist(val2_ext)

    eq_ind = await mpc.np_is_zero_public(mpc.np_subtract(ind1_ext, ind2_ext))

    if eq_ind.any():
        mult_res = mpc.np_multiply(val2_ext[eq_ind], val1_ext[eq_ind])
        res = mpc.np_sum(mult_res)
    else:
        res = sectype(0)
    return res


def merge_oram(mat1, mat2, sectype):
    n_tot = len(mat1[0]) + len(mat2[0])
    ind1 = sectype(0)
    ind2 = sectype(0)
    res = []
    for _ in range(n_tot):
        v1 = mat1[1][ind1]
        i1 = mat1[0][ind1]
        v2 = mat2[1][ind2]
        i2 = mat2[0][ind2]
        comp = i1 < i2  # works for tuples of length 2 ONLY
        res.append([0, 0])
        res[-1][0] = mpc.if_else(comp, ind1, ind2)
        res[-1][1] = mpc.if_else(comp, v1, v2)
        ind1 = ind1 + comp
        ind2 = ind2 + 1 - comp
    return res


def sparse_vector_dot_merge(vect1, vect2, sectype, key=None):  # TODO: test
    sorted_list = merge_oram(vect1, vect2, sectype)
    res = 0
    for i in range(len(sorted_list) - 1):
        temp = sorted_list[i][1] * sorted_list[i + 1][1]
        sec_comp = sorted_list[i][0] == sorted_list[i + 1][0]
        temp = mpc.if_else(sec_comp, temp, 0)
        res += temp
    return res
