import numpy as np
import pandas as pd
import time

# -------------------------------------------------
# ТЕСТОВІ ФУНКЦІЇ
# -------------------------------------------------

def f(x, t, **kwargs):
    return 0.0

def mu(t, **kwargs):
    return 0.0

def nu(t, **kwargs):
    return 0.0

def phi(x, **kwargs):
    return np.sin(np.pi * x / kwargs['l'])


# -------------------------------------------------
# ІТЕРАТИВНА ВЕРСІЯ
# -------------------------------------------------

def FDA_P_solver_iterative(f, mu, nu, phi,
                           nxst, ntst, mx, mt, kiter,
                           etalon=None, **kwargs):

    Nx = nxst
    Nt = ntst

    for _ in range(kiter):
        h = kwargs['l'] / Nx
        tau = kwargs['T'] / Nt
        alpha = kwargs['a'] ** 2 * tau / (h * h)

        x = np.linspace(0, kwargs['l'], Nx + 1)
        t = np.linspace(0, kwargs['T'], Nt + 1)

        taufij = np.zeros((Nt, Nx + 1))
        for j in range(Nt):
            for i in range(1, Nx):
                taufij[j, i] = tau * f(x[i], t[j], **kwargs)

        u = np.zeros((Nt + 1, Nx + 1))
        u[0, :] = phi(x, **kwargs)

        for j in range(1, Nt + 1):
            u[j, 0] = mu(t[j], **kwargs)
            u[j, Nx] = nu(t[j], **kwargs)
            for i in range(1, Nx):
                u[j, i] = (
                    alpha * (u[j - 1, i - 1] + u[j - 1, i + 1])
                    + (1 - 2 * alpha) * u[j - 1, i]
                    + taufij[j - 1, i]
                )

        Nx *= mx
        Nt *= mt

    return u


# -------------------------------------------------
# РЕКУРСИВНА ВЕРСІЯ
# -------------------------------------------------

def compute_time_layer(j, u, t, x, Nx, alpha,
                       mu, nu, taufij, **kwargs):

    if j == 0:
        return

    compute_time_layer(j - 1, u, t, x, Nx, alpha,
                       mu, nu, taufij, **kwargs)

    u[j, 0] = mu(t[j], **kwargs)
    u[j, Nx] = nu(t[j], **kwargs)

    for i in range(1, Nx):
        u[j, i] = (
            alpha * (u[j - 1, i - 1] + u[j - 1, i + 1])
            + (1 - 2 * alpha) * u[j - 1, i]
            + taufij[j - 1, i]
        )


def FDA_P_solver_recursive(f, mu, nu, phi,
                           nxst, ntst, mx, mt, kiter,
                           etalon=None, **kwargs):

    Nx = nxst
    Nt = ntst

    for _ in range(kiter):
        h = kwargs['l'] / Nx
        tau = kwargs['T'] / Nt
        alpha = kwargs['a'] ** 2 * tau / (h * h)

        x = np.linspace(0, kwargs['l'], Nx + 1)
        t = np.linspace(0, kwargs['T'], Nt + 1)

        taufij = np.zeros((Nt, Nx + 1))
        for j in range(Nt):
            for i in range(1, Nx):
                taufij[j, i] = tau * f(x[i], t[j], **kwargs)

        u = np.zeros((Nt + 1, Nx + 1))
        u[0, :] = phi(x, **kwargs)

        compute_time_layer(
            Nt, u, t, x, Nx, alpha,
            mu, nu, taufij, **kwargs
        )

        Nx *= mx
        Nt *= mt

    return u


# -------------------------------------------------
# ПОРІВНЯННЯ ШВИДКОДІЇ
# -------------------------------------------------

if __name__ == "__main__":

    params = {
        'l': 1.0,
        'T': 0.5,
        'a': 1.0
    }

    nx = 40
    nt = 200
    mx = 1
    mt = 1
    kiter = 1

    print("Запуск ітеративної версії...")
    start = time.perf_counter()
    u_iter = FDA_P_solver_iterative(
        f, mu, nu, phi,
        nx, nt, mx, mt, kiter,
        **params
    )
    t_iter = time.perf_counter() - start

    print("Запуск рекурсивної версії...")
    start = time.perf_counter()
    u_rec = FDA_P_solver_recursive(
        f, mu, nu, phi,
        nx, nt, mx, mt, kiter,
        **params
    )
    t_rec = time.perf_counter() - start

    print("\n--- РЕЗУЛЬТАТ ---")
    print(f"Ітеративна версія : {t_iter:.6f} c")
    print(f"Рекурсивна версія : {t_rec:.6f} c")
    print(f"Уповільнення     : {t_rec / t_iter:.2f} раз(и)")
