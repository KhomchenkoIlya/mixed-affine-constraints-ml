import numpy as np


def admm_shared_quadratic(
    b: np.ndarray,
    rho: float = 1.0,
    steps: int = 200,
    x0: np.ndarray | None = None,
    z0: np.ndarray | None = None,
    u0: np.ndarray | None = None,
):
    """
    ADMM for shared-variable consensus:

        minimize   sum_i  1/2 ||x_i - b_i||^2
        subject to x_i = z   for all i

    Here:
      - x_i \in R^d  : local variable at agent i
      - z   \in R^d  : shared (global) variable
      - u_i \in R^d  : scaled dual variables

    Closed-form updates for quadratic objective:
      x_i^{k+1} = (b_i + rho*(z^k - u_i^k)) / (1 + rho)
      z^{k+1}   = average_i (x_i^{k+1} + u_i^k)
      u_i^{k+1} = u_i^k + x_i^{k+1} - z^{k+1}

    Returns:
      opt_err  : ||z - mean(b)|| / ||mean(b)||
      cons_err : avg_i ||x_i - z|| / ||z||
      x, z, u
    """
    n, d = b.shape
    b_mean = b.mean(axis=0)

    x = np.zeros((n, d)) if x0 is None else np.array(x0, dtype=float, copy=True)
    z = np.zeros(d) if z0 is None else np.array(z0, dtype=float, copy=True)
    u = np.zeros((n, d)) if u0 is None else np.array(u0, dtype=float, copy=True)

    opt_err = []
    cons_err = []

    for _ in range(steps):
        # x-update (separable across agents)
        x = (b + rho * (z - u)) / (1.0 + rho)

        # z-update
        z = (x + u).mean(axis=0)

        # u-update
        u = u + x - z

        # metrics
        oe = np.linalg.norm(z - b_mean) / (np.linalg.norm(b_mean) + 1e-12)
        ce = np.mean(np.linalg.norm(x - z, axis=1)) / (np.linalg.norm(z) + 1e-12)

        opt_err.append(oe)
        cons_err.append(ce)

        if oe < 1e-12 and ce < 1e-12:
            break

    return np.array(opt_err), np.array(cons_err), x, z, u
