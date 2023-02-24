import datetime
import random

from mpyc.random import random_unit_vector, shuffle
from mpyc.runtime import mpc


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
    n = x.shape[0]

    for i in range(n - 1):
        u = mpc.np_transpose(await np_random_unit_vector(sectype, n - i))
        x_u = mpc.np_matmul(u, x[i:])
        if len(x.shape) > 1:
            d = mpc.np_outer(u, (x[i] - x_u))
            x = mpc.np_vstack((x[:i, ...], mpc.np_add(x[i:, ...], d)))
        else:
            d = u * (x[i] - x_u)
            x = mpc.np_hstack((x[:i, ...], mpc.np_add(x[i:, ...], d)))
        mpc.np_update(x, i, x_u)

    return x


async def test():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        l = [[sectype(i), sectype(random.randint(0, 1024))] for i in range(10000)]
    else:
        l = None
    l = await mpc.transfer(l, senders=0)
    l_arr = []
    for tup in l:
        l_arr += tup
    l_arr = mpc.np_reshape(mpc.np_fromlist(l_arr), (len(l), len(l[0])))

    print(await mpc.output(l_arr))
    start = datetime.datetime.now()
    x = await np_shuffle(sectype, l_arr)
    print(await mpc.output(x))
    delta_np = datetime.datetime.now() - start

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
