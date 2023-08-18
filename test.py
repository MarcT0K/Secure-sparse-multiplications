import numpy as np
import scipy.sparse
from mpyc.runtime import mpc

from matrices import (
    DenseMatrixNumpy,
)


async def benchmark_large_matrix():
    await mpc.start()
    # n_dim = 32 * 10**3
    # m_dim = 16 * 10**5

    n_dim = 3 * 10**3
    m_dim = 300 * 10**3
    density = 0.00001
    secint = mpc.SecInt(64)

    if mpc.pid == 0:
        x_sparse = scipy.sparse.random(
            n_dim, m_dim, density=density, dtype=np.int16
        ).astype(int)
    else:
        x_sparse = None

    x_sparse = await mpc.transfer(x_sparse, senders=0)

    dense_mat = x_sparse.astype(int).todense()
    sec_dense = DenseMatrixNumpy(dense_mat, sectype=secint)
    z = sec_dense + sec_dense
    await mpc.barrier()
    await mpc.shutdown()


if __name__ == "__main__":
    mpc.run(benchmark_large_matrix())
