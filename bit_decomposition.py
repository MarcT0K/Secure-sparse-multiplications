"""Extends MPyC codebase to parallelize bit decomposition
"""

import numpy as np

from mpyc import asyncoro
from mpyc.runtime import mpc


def np_add_bits(x_in, y_in):
    """Secure binary addition of bit vectors x and y."""
    x = np.transpose(x_in)
    y = np.transpose(y_in)

    def f(i, j, high=False):
        n = j - i
        if n == 1:
            c[i] = x[i] * y[i]
            if high:
                d[i] = x[i] + y[i] - c[i] * 2
        else:
            h = i + n // 2
            f(i, h, high=high)
            f(h, j, high=True)
            c[h:j] = c[h:j] + c[h - 1] * d[h:j, :]
            if high:
                d[h:j] = d[h:j] * d[h - 1, :]

    n, m = x.shape
    c = np.array([[None] * m] * n)
    if n >= 1:
        d = np.array([[None] * m] * n)
        f(0, n)
    # c = prefix carries for addition of x and y
    for i in range(n - 1, -1, -1):
        c[i] = x[i] + y[i] - c[i] * 2 + (c[i - 1] if i > 0 else 0)

    c = np.transpose(c)
    return c


@asyncoro.mpc_coro
async def np_to_bits(a, l=None):
    """Secure extraction of l (or all) least significant bits of a."""  # a la [ST06].
    # TODO: other cases than characteristic=2 case, see mpc.to_bits()
    stype = type(a).sectype
    if l is None:
        l = stype.bit_length
    assert l <= stype.bit_length + stype.frac_length
    shape = a.shape + (l,)
    n = a.size
    await mpc.returnType((type(a), True, shape))
    field = stype.field
    f = stype.frac_length
    rshift_f = f and a.integral  # optimization for integral fixed-point numbers
    if rshift_f:
        # f least significant bits of a are all 0
        if f >= l:
            return [field(0) for _ in range(l)]

        l -= f

    r_bits = await mpc.np_random_bits(field, n * l)
    r_bits = r_bits.reshape(shape)
    shifts = np.arange(l)
    r_modl = np.sum(r_bits.value << shifts, axis=a.ndim)

    if issubclass(stype, mpc.SecureFiniteField):
        if field.characteristic == 2:
            a = await mpc.gather(a)
            c = await mpc.output(a + r_modl)
            c = np.vectorize(int, otypes="O")(c.value)
            c_bits = np.right_shift.outer(c, shifts) & 1
            return c_bits + r_bits

        if field.ext_deg > 1:
            raise TypeError("Binary field or prime field required.")

        raise NotImplementedError

    #     a = mpc.convert(a, mpc.SecInt(l=1 + stype.field.order.bit_length()))
    #     a_bits = mpc.to_bits(a)
    #     return mpc.convert(a_bits, stype)

    k = mpc.options.sec_param
    r_divl = mpc._np_randoms(field, n, 1 << (stype.bit_length + k - l)).value
    a = await mpc.gather(a)
    if rshift_f:
        a = a >> f
    c = await mpc.output(a + ((1 << stype.bit_length) + (r_divl << l) - r_modl))
    c = np.vectorize(int, otypes="O")(c.value % (1 << l))
    c_bits = np.right_shift.outer(c, shifts) & 1
    c_bits = c_bits.reshape(shape)
    r_bits = r_bits.value  # TODO: drop .value, fix secfxp(r) if r field elt
    a_bits = np_add_bits(r_bits, c_bits)
    if rshift_f:
        a_bits = [field(0) for _ in range(f)] + a_bits
    return stype.array(a_bits)


async def main():
    await mpc.start()

    secint = mpc.SecInt(64)
    c = secint.array(np.array([i for i in range(10)]))
    print(await mpc.output(c))
    y = np_to_bits(c, 5)
    print(type(y), y.shape)
    print(await mpc.output(y))
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(main())
