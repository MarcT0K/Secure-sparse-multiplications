import asyncio
import pickle
import random

import numpy as np

from mpyc import thresha
from mpyc.runtime import mpc
from mpyc import sectypes
from mpyc import finfields
from mpyc import asyncoro


@asyncoro.mpc_coro
async def partial_reshare(x, ignore=None):
    if ignore is None:
        ignore = []
    assert isinstance(ignore, list)

    x_is_list = isinstance(x, list)
    if not x_is_list:
        x = [x]
    if x == []:
        return []

    sftype = type(x[0])  # all elts assumed of same type
    if issubclass(sftype, mpc.SecureObject):
        if not sftype.frac_length:
            if issubclass(sftype, sectypes.SecureArray):
                rettype = (sftype, x[0].shape)
            else:
                rettype = sftype
        else:
            if issubclass(sftype, sectypes.SecureArray):
                rettype = (sftype, x[0].integral, x[0].shape)
            else:
                rettype = (sftype, x[0].integral)
        if x_is_list:
            await mpc.returnType(rettype, len(x))
        else:
            await mpc.returnType(rettype)
        x = await mpc.gather(x)
    else:
        await mpc.returnType(asyncio.Future)

    t = mpc.threshold
    if t == 0:
        if not x_is_list:
            x = x[0]
        return x

    if isinstance(x[0], finfields.FiniteFieldElement):
        field = type(x[0])
        x = [a.value for a in x]
        shape = None
    else:
        field = x[0].field
        x = x[0].value
        shape = x.shape
        x = x.reshape(-1)  # in-place flatten

    m = len(mpc.parties)
    if shape is None or mpc.options.mix32_64bit:
        shares = thresha.random_split(field, x, t, m)
        shares = [field.to_bytes(elts) for elts in shares]
    else:
        shares = thresha.np_random_split(field, x, t, m)
        shares = [pickle.dumps(elts) for elts in shares]
    # Recombine the first 2t+1 output_shares.
    shares = mpc._exchange_shares(shares)
    shares = await mpc.gather(shares[: 2 * t + 1 + len(ignore)])
    if shape is None or mpc.options.mix32_64bit:
        points = [
            (j + 1, field.from_bytes(s))
            for j, s in enumerate(shares)
            if j not in ignore
        ]
        y = thresha.recombine(field, points)
    else:
        points = [
            (j + 1, pickle.loads(s)) for j, s in enumerate(shares) if j not in ignore
        ]
        y = thresha.np_recombine(field, points)
    if shape is None:
        y = [field(a) for a in y]
    elif mpc.options.mix32_64bit:
        y = [field.array(y).reshape(shape)]
    else:
        y = [y.reshape(shape)]
    if not x_is_list:
        y = y[0]
    return y


def permute(input_list, seed, inv=False):
    random.seed(seed)
    res = input_list.copy()
    permutation = list(range(len(input_list)))
    random.shuffle(permutation)

    if inv:
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
        permutation = inv

    res = [res[i] for i in permutation]
    return res


async def test():
    await mpc.start()
    secint = mpc.SecInt(64)

    x = [secint(8), secint(42), secint(1337)]
    x = mpc._reshare(x)

    if mpc.pid != 2:
        x = [x[1], x[2], x[0]]

    x = partial_reshare(x, ignore=[2])
    print(await mpc.output(x))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(test())
