import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FDA_solver import tridiagonal_matrix_algorithm, f_func, p_func, q_func, u_star

def set_matrix_diagonals_and_vector(f, p, q, x, n, **kwargs):
    h = np.abs(x[1] - x[0])
    h2 = h**2
    
    c = np.empty(n + 1, dtype='float64')
    a = np.empty(n + 1, dtype='float64')
    b = np.empty(n + 1, dtype='float64')
    d = np.empty(n + 1, dtype='float64')

    for i in range(1, n):
        p_val = p(x[i])
        # Коефіцієнти для схеми u'' + p(x)u' + q(x)u = f(x)
        a[i] = 1 - (h / 2) * p_val
        c[i] = h2 * q(x[i]) - 2
        b[i] = 1 + (h / 2) * p_val
        d[i] = h2 * f(x[i])

    # Крайові умови (alpha*u + beta*u' = gamma)
    c[0] = kwargs['alpha0'] * h - kwargs['beta0']
    b[0] = kwargs['beta0']
    a[0] = 0
    d[0] = h * kwargs['gamma0']

    a[n] = -kwargs['beta1']
    c[n] = kwargs['alpha1'] * h + kwargs['beta1']
    b[n] = 0
    d[n] = h * kwargs['gamma1']

    return c, a, b, d

def compute_on_grid(n, f, p, q, params):
    """Виконує повний цикл обчислення для заданого n"""
    a, b = params['a'], params['b']
    xk = np.linspace(a, b, n + 1)
    
    # Формування матриці та розв'язання (використовуємо ваші функції)
    c, low, up, d = set_matrix_diagonals_and_vector(f, p, q, xk, n, **params)
    uk = tridiagonal_matrix_algorithm(low, up, c, d)
    return xk, uk

def FDA_parallel_solver(f, p, q, nstrt, kiter, etalon_func=None, **params):
    """Паралельна версія FDA_solver з використанням concurrent.futures"""
    
    # Створюємо список значень n для кожної ітерації
    n_values = [nstrt * (2**k) for k in range(kiter)]
    
    x_results = [None] * kiter
    u_results = [None] * kiter

    # Використання ProcessPoolExecutor для паралельних обчислень
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Запускаємо задачі паралельно
        future_to_n = {executor.submit(compute_on_grid, n, f, p, q, params): i 
                       for i, n in enumerate(n_values)}
        
        for future in concurrent.futures.as_completed(future_to_n):
            idx = future_to_n[future]
            try:
                xk, uk = future.result()
                x_results[idx] = xk
                u_results[idx] = uk
            except Exception as exc:
                print(f'Обчислення для сітки {idx} викликало помилку: {exc}')

    # Формування результатів у DataFrame
    solutions = pd.DataFrame(index=x_results[0])
    ist = 1
    for k in range(kiter):
        solutions[f'u^{k}'] = u_results[k][::ist]
        ist *= 2
    
    if etalon_func:
        solutions['ux'] = etalon_func(x_results[0])
        
    return u_results, x_results, solutions

# --- Приклад запуску (використовуємо ваші функції p, q, f з попереднього кроку) ---
if __name__ == "__main__":
    # Параметри задачі 5
    params = {
        'a': 0, 'b': 1,
        'alpha0': 1, 'beta0': 1, 'gamma0': 1,
        'alpha1': 1, 'beta1': 0, 'gamma1': np.cos(1)
    }
    
    u, x, sol = FDA_parallel_solver(f_func, p_func, q_func, 20, 5, etalon_func=u_star, **params)
    print("Паралельні обчислення завершено.")
    print(sol.head())