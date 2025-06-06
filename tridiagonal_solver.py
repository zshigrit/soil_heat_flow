import numpy as np

def tridiagonal_solver(a, b, c, d):
    """Solve a tridiagonal system for u given a, b, c, d."""
    n = len(d)
    e = np.zeros(n)
    f = np.zeros(n)

    e[0] = c[0] / b[0]
    for i in range(1, n-1):
        e[i] = c[i] / (b[i] - a[i] * e[i-1])

    f[0] = d[0] / b[0]
    for i in range(1, n):
        f[i] = (d[i] - a[i] * f[i-1]) / (b[i] - a[i] * e[i-1])

    u = np.zeros(n)
    u[-1] = f[-1]
    for i in range(n-2, -1, -1):
        u[i] = f[i] - e[i] * u[i+1]

    return u
