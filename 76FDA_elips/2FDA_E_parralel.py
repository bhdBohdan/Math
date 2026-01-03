import concurrent.futures
import numpy as np
import pandas as pd
from scipy import linalg

from FDA_Esolver import set_matrix, set_vector

def f_func(x, y, **kwargs):
    return 2 * (x**2 + y**2)

def phi1(y, **kwargs):
    return y**2

def phi2(y, **kwargs):
    return 1 + y**2

def psi1(x, **kwargs):
    return x**2

def psi2(x, **kwargs):
    return 1 + x**2

def exact_solution(x, y, **kwargs):
    return x**2 + y**2

# Функція-воркер також має бути тут
def compute_2d_grid_task(Nx, Ny, f, p1, p2, s1, s2, params):
    # Тут виконуються обчислення...
    # (код set_matrix, set_vector тощо має бути доступним)
    hx = params['a'] / Nx
    hy = params['b'] / Ny
    alpha, beta = 1/(hx*hx), 1/(hy*hy)
    
    c = set_matrix(alpha, beta, Nx, Ny)
    x = np.linspace(0, params['a'], Nx+1)
    y = np.linspace(0, params['b'], Ny+1)
    
    ps1, ps2 = s1(x, **params), s2(x, **params)
    ph1, ph2 = p1(y, **params), p2(y, **params)
    
    g = set_vector(f, ph1, ph2, ps1, ps2, x, y, alpha, beta, Nx, Ny, **params)
    u_flat = linalg.solve(c, g)
    return u_flat.reshape(((Nx-1), (Ny-1)))

# --- 2. Оновлений паралельний розв'язувач з кращою обробкою помилок ---

def FDA_E_parallel_solver(f, p1, p2, s1, s2, nxst, nyst, mx, my, kiter, etalon_func=None, **params):
    grid_params = [(nxst * (mx**k), nyst * (my**k)) for k in range(kiter)]
    results = [None] * kiter

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(compute_2d_grid_task, n[0], n[1], f, p1, p2, s1, s2, params): i 
            for i, n in enumerate(grid_params)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result() # Тут вилетить помилка, якщо воркер впав
            except Exception as exc:
                print(f"Помилка в процесі на сітці {idx}: {exc}")

    # Перевірка: чи всі результати отримані
    if any(res is None for res in results):
        raise RuntimeError("Деякі обчислення не завершилися успішно. Перевірте консоль на помилки воркерів.")

    # Логіка формування solutions (залишається вашою)
    solutions = pd.DataFrame()
    stx, sty = 1, 1
    for k in range(kiter):
        uij = results[k]
        Nx_k, Ny_k = grid_params[k]
        solutions[f'u^{k}'] = uij[stx-1:Nx_k:stx, sty-1:Ny_k:sty].reshape((nxst-1)*(nyst-1))
        stx *= mx
        sty *= my
        
    return results[-1], solutions

# --- 3. Блок запуску ---
if __name__ == "__main__":
    params = {'a': 1, 'b': 1, 'alpha0': 0, 'alpha1': 0, 'beta0': 1, 'beta1': 1, 'gamma0': 0, 'gamma1': 0}
    
    # Виклик тепер спрацює, бо функції доступні для pickle
    u, solutions = FDA_E_parallel_solver(
        f_func, phi1, phi2, psi1, psi2,
        5, 5, 2, 2, 4,
        etalon_func=exact_solution,
        **params
    )
    print(solutions.head())