import datetime
import random

from mpyc.random import shuffle
from mpyc.runtime import Future, mpc, mpc_coro


@mpc_coro
async def np_random_unit_vector(sectype, n):
    """Uniformly random secret rotation of [1] + [0]*(n-1).
    Expected number of secret random bits needed is ceil(log_2 n) + c,
    with c a small constant, c < 3.
    """

    await mpc.returnType((sectype.array, True, (n,)))

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
            x = mpc.np_hstack((x[:i], mpc.np_random_bits(sectype, k - i)))
            i = k - 1
            u = mpc.np_fromlist([x[i], 1 - x[i]])
        else:
            v = x[i] * u[1:]
            v = mpc.np_hstack((v, u[1:] - v))
            u = mpc.np_hstack((u[:1], v))
    return u


async def np_shuffle(a, axis=None):
    """Shuffle numpy-like array x secretly in-place, and return None.
    Given array x may contain public or secret elements.
    """
    sectype = type(a).sectype

    if len(a.shape) > 2:
        raise ValueError("Can only shuffle 1D and 2D arrays")

    if axis is None:
        axis = 0

    if axis not in (0, 1, -1):
        raise ValueError("Invalid axis")

    x = mpc.np_copy(a)

    if axis != 0:
        x = mpc.np_transpose(x)

    n = x.shape[0]

    for i in range(n - 1):
        u = mpc.np_transpose(np_random_unit_vector(sectype, n - i))
        x_u = mpc.np_matmul(u, x[i:])
        if len(x.shape) > 1:
            d = mpc.np_outer(u, (x[i] - x_u))
            x = mpc.np_vstack((x[:i, ...], mpc.np_add(x[i:, ...], d)))
        else:
            d = u * (x[i] - x_u)
            x = mpc.np_hstack((x[:i, ...], mpc.np_add(x[i:, ...], d)))
        x = mpc.np_update(x, i, x_u)

    if axis != 0:
        x = mpc.np_transpose(x)

    x = await mpc.gather(x)
    mpc.np_update(a, range(a.shape[0]), x)


async def test():
    await mpc.start()
    sectype = mpc.SecInt(64)
    if mpc.pid == 0:
        # l = [[sectype(i), sectype(random.randint(0, 1024))] for i in range(1000)]
        l = [sectype(i) for i in range(1000)]
    else:
        l = None
    l = await mpc.transfer(l, senders=0)

    # l_arr = []
    # for tup in l:
    #     l_arr += tup
    # l_arr = mpc.np_reshape(mpc.np_fromlist(l_arr), (len(l), len(l[0])))
    l_arr = mpc.np_fromlist(l)

    print(await mpc.output(l_arr))
    start = datetime.datetime.now()
    await np_shuffle(l_arr, axis=0)
    print("l_arr:", await mpc.output(l_arr))
    delta_np = datetime.datetime.now() - start

    # start = datetime.datetime.now()
    # shuffle(sectype, l)
    # print("[")
    # for i in range(len(l)):
    #     print(await mpc.output(l[i]), end=", ")
    # print("]")
    # delta = datetime.datetime.now() - start
    print("np shuffle: ", delta_np.total_seconds())
    # print("shuffle: ", delta.total_seconds())
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
