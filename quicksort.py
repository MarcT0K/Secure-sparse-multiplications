from mpyc.runtime import mpc


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


# IDEA: we can replace the comparison a < b by revealing the result of (a-b)*rand_positive_integer
# PROBLEM: to avoid an overflow, we would need to reveal some info about the data
# => Partial solution: the overflow is problematic for the comparison case (overflow may affect the sign) but not for the equality case
# IDEA: we only need to reveal the sign bit of (a-b) => it is what is done by the comparison function I believe
