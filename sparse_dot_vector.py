from mpyc.runtime import mpc
from experiments import SortableTuple
from quicksort import quicksort


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


async def sparse_vector_dot_quicksort(vect1, vect2, sectype, key=None):
    unsorted = vect1 + vect2
    sorted_list = await quicksort(unsorted, sectype, False, key)
    # sorted_list = unsorted
    res = 0
    for i in range(len(sorted_list) - 1):
        temp = sorted_list[i][1] * sorted_list[i + 1][1]
        sec_comp = sorted_list[i][0] == sorted_list[i + 1][0]
        temp = mpc.if_else(sec_comp, temp, 0)
        res += temp
    return res


def sparse_vector_dot_naive(vect1, vect2):
    res = 0
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            temp = vect1[i][1] * vect2[j][1]
            sec_comp = vect1[i][0] == vect2[j][0]
            temp = mpc.if_else(sec_comp, temp, 0)
            res += temp
    return res


async def sparse_vector_dot_psi(vect1, vect2, sectype):
    res = sectype(0)
    mpc.random.shuffle(sectype, vect1)
    mpc.random.shuffle(sectype, vect2)
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            temp = vect1[i][1] * vect2[j][1]
            # rand = mpc.from_bits(
            #     mpc.random_bits(sectype, sectype.bit_length, signed=True)
            # )  # Bottleneck is here

            # sec_comp = (vect1[i][0] - vect2[j][0]) * rand
            # if await mpc.output(sec_comp) == 0:

            comp = await mpc.is_zero_public(vect1[i][0] - vect2[j][0])
            # print(comp)
            if comp:
                res += temp
    return res


def merge_oram(vect1, vect2, sectype):
    n_tot = len(vect1) + len(vect2)
    ind1 = sectype(0)
    ind2 = sectype(0)
    res = []
    for _ in range(n_tot):
        v1 = vect1[ind1]
        v2 = vect2[ind2]
        comp = v1[0] < v2[0]  # works for tuples of length 2 ONLY
        res.append(mpc.if_else(comp, v1, v2))
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


def sparse_vector_dot_naive_bis(vect1, vect2):
    res = 0
    for i in range(len(vect1)):
        for j in range(len(vect2)):
            temp = vect1[i][1] * vect2[j][1]
            sec_comp = mpc.SecInt(64)(1) // ((vect1[i][0] - vect2[j][0]) ** 2 + 1)
            # Idea: use a fixed-point number to compute a floor_div
            res += temp * sec_comp
    return res


# IDEE pour sparse-dense mult: en plus du vecteur sparse, l'utilisateur fourni une permutation aléatoire. Le vecteur dense est permuté et les indices sont révélés en clair.
# => Problème: la permutation doit être plus chere qu'une produit vectoriel
