import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def oriented_incidence_matrix(G: nx.Graph) -> sp.csr_matrix:
    """
    Oriented incidence matrix B for an undirected graph.
    For each edge e=(i,j): row has +1 at i, -1 at j (orientation fixed by node order).
    Shape: (m, n)
    """
    n = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)

    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(edges):
        if i > j:
            i, j = j, i
        rows += [k, k]
        cols += [i, j]
        data += [1.0, -1.0]

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()


def admm_edge_quadratic(
    G: nx.Graph,
    b: np.ndarray,
    rho: float = 1.0,
    steps: int = 200,
):
    """
    ADMM (scaled) for edge/coupled constraints:

        minimize   sum_i  1/2 ||x_i - b_i||^2
        subject to x_i - x_j = 0  for all edges (i,j) in E

    Compact: B x = 0, where B is oriented incidence matrix.
    Solve per dimension: (I + rho * B^T B) x = b - rho * B^T u.

    Returns:
      feas_err: ||B x|| / ||x||
      cons_err: avg_i ||x_i - mean(x)|| / ||mean(x)||
      x, u
    """
    n, d = b.shape
    B = oriented_incidence_matrix(G)  # (m, n)
    m = B.shape[0]

    x = np.zeros((n, d))
    u = np.zeros((m, d))  # scaled dual

    L = (B.T @ B).tocsr()  # Laplacian
    A = (sp.eye(n, format="csr") + rho * L).tocsc()  # сразу CSC (убираем warning)

    solve_A = spla.factorized(A)

    feas_err = []
    cons_err = []

    for _ in range(steps):
        rhs = b - rho * (B.T @ u)  # (n, d)

        for j in range(d):
            x[:, j] = solve_A(rhs[:, j])

        r = B @ x
        u = u + r

        x_mean = x.mean(axis=0)

        fe = np.linalg.norm(r) / (np.linalg.norm(x) + 1e-12)
        ce = np.mean(np.linalg.norm(x - x_mean, axis=1)) / (np.linalg.norm(x_mean) + 1e-12)

        feas_err.append(fe)
        cons_err.append(ce)

        if fe < 1e-10 and ce < 1e-10:
            break

    return np.array(feas_err), np.array(cons_err), x, u
