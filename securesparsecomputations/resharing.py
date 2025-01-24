import asyncio
import os
import pickle
import random

import numpy as np
from mpyc import asyncoro, finfields, sectypes, thresha
from mpyc.runtime import mpc


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


def np_permute(input_list, seed, axis, inv=False):
    random.seed(seed)
    permutation = list(range(len(input_list)))
    random.shuffle(permutation)

    if inv:
        inv = np.empty_like(permutation)
        inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
        permutation = inv

    res = input_list[permutation]
    return res


async def np_shuffle_3PC(input_list, axis=-1):
    assert len(mpc.parties) == 3
    output_list = mpc.np_copy(input_list)

    # Sharing random seeds between pairs of parties
    seeds = [None] * 3
    self_seed = int.from_bytes(os.urandom(16), "big")
    other_seed = await mpc.transfer(
        self_seed, sender_receivers=[(0, 1), (1, 2), (2, 0)]
    )
    seeds[(mpc.pid - 1) % 3] = self_seed
    seeds[(mpc.pid + 1) % 3] = other_seed[0]

    # Permuting the lists
    for i in range(3):
        if mpc.pid != i:
            output_list = np_permute(output_list, seeds[i], axis)

        output_list = partial_reshare(output_list, ignore=[i])
    return output_list
