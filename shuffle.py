from mpyc.runtime import mpc
from mpyc.random import random_unit_vector, shuffle

import random
import datetime


async def random_unit_vector_custom(sectype, n):
    """Uniformly random secret rotation of [1] + [0]*(n-1).

    Expected number of secret random bits needed is ceil(log_2 n) + c,
    with c a small constant, c < 3.
    """

    if n == 1:
        return [sectype(1)]

    b = n - 1
    k = b.bit_length()
    x = mpc.random_bits(sectype, k)
    i = k - 1
    u = [x[i], 1 - x[i]]
    while i:
        i -= 1
        if (b >> i) & 1:
            v = mpc.scalar_mul(x[i], u)
            v.extend(mpc.vector_sub(u, v))
            u = v
        elif await mpc.output(u[0] * x[i]):  # TODO: mul_public
            # restart, keeping unused secret random bits x[:i]
            x[i:] = mpc.random_bits(sectype, k - i)
            i = k - 1
            u = [x[i], 1 - x[i]]
        else:
            v = mpc.scalar_mul(x[i], u[1:])
            v.extend(mpc.vector_sub(u[1:], v))
            u[1:] = v
    return u


async def np_random_unit_vector(sectype, n):
    """Uniformly random secret rotation of [1] + [0]*(n-1).

    Expected number of secret random bits needed is ceil(log_2 n) + c,
    with c a small constant, c < 3.
    """
    if n == 1:
        return mpc.np_fromlist([sectype(1)])

    b = n - 1
    k = b.bit_length()
    x = mpc.np_random_bits(sectype, k)
    i = k - 1
    u = mpc.np_fromlist([x[i], 1 - x[i]])
    while i:
        i -= 1
        if (b >> i) & 1:
            v = x[i] * u
            v = mpc.np_hstack((v, u - v))
            u = v
        elif await mpc.output(u[0] * x[i]):  # TODO: mul_public
            # restart, keeping unused secret random bits x[:i]
            x = mpc.np_hstack((x[:i], await np_random_unit_vector(sectype, k - i)))
            i = k - 1
            u = mpc.np_fromlist([x[i], 1 - x[i]])
        else:
            v = x[i] * u[1:]
            v = mpc.np_hstack((v, u[1:] - v))
            u = mpc.np_hstack((u[:1], v))
    return u


async def np_shuffle(sectype, x):
    """Shuffle list x secretly in-place, and return None.

    Given list x may contain public or secret elements.
    Elements of x are all numbers or all lists (of the same length) of numbers.
    """
    n = len(x)
    # assume same type for all elts of x
    x_i_is_list = isinstance(x[0], list)
    if not x_i_is_list:
        # elements of x are numbers
        if not isinstance(x[0], sectype):
            for i in range(n):
                x[i] = sectype(x[i])
        x = mpc.np_fromlist(x)

        for i in range(n - 1):
            # print("x", await mpc.output(x))
            u = await np_random_unit_vector(sectype, n - i)
            # print("u", await mpc.output(u))
            x_u = mpc.np_matmul(x[i:], u)
            # print("x_u", await mpc.output(x_u))
            d = (x[i] - x_u) * u
            # print("d", await mpc.output(d))
            mpc.np_update(x, i, x_u)
            # print("x", await mpc.output(x))
            x = mpc.np_hstack((x[:i], mpc.np_add(x[i:], d)))
            # print("x ", await mpc.output(x))
            # print("===")
            # input()
        return mpc.np_tolist(x)

    # elements of x are lists of numbers
    m = len(x[0])
    temp = []
    for i in range(n):
        for j in range(m):
            if not isinstance(x[i][j], sectype):
                temp.append(sectype(x[i][j]))
            else:
                temp.append(x[i][j])
    x = mpc.np_reshape(mpc.np_fromlist(temp), (n, m))

    for i in range(n - 1):
        u = mpc.np_transpose(await np_random_unit_vector(sectype, n - i))
        x_u = u @ x[i:]
        d = mpc.np_outer(u, (x[i] - x_u))
        mpc.np_update(x, i, x_u)
        x = mpc.np_vstack((x[:i, ...], mpc.np_add(x[i:, ...], d)))
    return mpc.np_tolist(x)


async def test():
    await mpc.start()
    sectype = mpc.SecInt(64)

    if mpc.pid == 0:
        l = [[sectype(i), sectype(random.randint(0, 1024))] for i in range(1000)]
    else:
        l = None
    l = await mpc.transfer(l, senders=0)

    start = datetime.datetime.now()
    x = await np_shuffle(sectype, l)
    print(await mpc.output(x))
    delta_np = datetime.datetime.now() - start
    print("np shuffle: ", delta_np.total_seconds())

    start = datetime.datetime.now()
    shuffle(sectype, l)
    print("[")
    for i in range(len(l)):
        print(await mpc.output(l[i]), end=", ")
    print("]")
    delta = datetime.datetime.now() - start
    print("np shuffle: ", delta_np.total_seconds())
    print("shuffle: ", delta.total_seconds())
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
