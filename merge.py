import datetime
import random

from mpyc.numpy import np
from mpyc.runtime import mpc


async def np_merge(l1, l2, axis=-1, key=None):
    assert l1.shape == l2.shape  # TODO 1: extend to asymetric merge
    # TODO 2: extend to multidimensional arrays

    n1 = l1.shape[0]
    a = mpc.np_concatenate((l1[0:1], l2[0:1]))
    for i in range(1, n1):
        a = mpc.np_append(a, l1[i : i + 1])
        a = mpc.np_append(a, l2[i : i + 1])

    if axis is None:
        a = mpc.np_flatten(a)
        axis = 0
    else:
        a = mpc.np_copy(a)
    if key is None:
        key = lambda a: a
    n = a.shape[axis]
    if a.size == 0 or n <= 1:
        return a

    # n >= 2
    a = mpc.np_swapaxes(a, axis, -1)  # switch to last axis

    t = (n - 1).bit_length()
    p = 1
    d, q, r = p, 1 << t - 1, 0
    while d:
        I = np.fromiter((i for i in range(n - d) if i & p == r), dtype=int)
        b0 = a[..., I]
        b1 = a[..., I + d]
        h = (key(b1) < key(b0)) * (b1 - b0)
        b0, b1 = b0 + h, b1 - h
        a = mpc.np_update(a, (..., I), b0)
        a = mpc.np_update(a, (..., I + d), b1)
        d, q, r = q - p, q >> 1, p

    a = mpc.np_swapaxes(a, axis, -1)  # restore original axis
    return a


async def test(size=1000):
    await mpc.start()
    sectype = mpc.SecInt(64)

    if mpc.pid == 0:
        l1 = [random.randint(0, 1024) for i in range(size)]
        l2 = [random.randint(0, 1024) for i in range(size)]
        l1.sort()
        l2.sort()
        m = l1 + l2
        m.sort()
        l1 = [sectype(i) for i in l1]
        l2 = [sectype(i) for i in l2]
    else:
        l1 = None
        l2 = None
        m = None

    l1 = await mpc.transfer(l1, senders=0)
    l2 = await mpc.transfer(l2, senders=0)
    m = await mpc.transfer(m, senders=0)

    l1 = mpc.np_fromlist(l1)
    l2 = mpc.np_fromlist(l2)

    start = datetime.datetime.now()
    z = await np_merge(l1, l2)
    res = await mpc.output(z)
    assert res.tolist() == m
    end = datetime.datetime.now()
    print("Merge time:", (end - start).total_seconds())

    start = datetime.datetime.now()
    a = mpc.np_concatenate((l1, l2))
    z = mpc.np_sort(a)
    res = await mpc.output(z)
    assert res.tolist() == m
    end = datetime.datetime.now()
    print("Sort time:", (end - start).total_seconds())

    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
