import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def SequencealApproach(f, K, x_grid, kiter, etalon_func=None, **kwargs):
    """
    Розв'язування інтегрального рівняння методом послідовних наближень
    u_{n+1}(x) = f(x) + integral_{0}^{x} K(x, t) u_n(t) dt
    """
    n_points = len(x_grid)
    solutions = pd.DataFrame(index=x_grid)
    
    # Початкове наближення u_0(x) = 1 (згідно з умовою)
    u_current = np.ones(n_points)
    solutions['u^0'] = u_current
    
    for k in range(1, kiter + 1):
        u_next = np.zeros(n_points)
        for i in range(n_points):
            xi = x_grid[i]
            if i == 0:
                u_next[i] = f(xi, **kwargs)
            else:
                # Сітка для інтегрування від 0 до xi
                t_subgrid = x_grid[:i+1]
                u_sub = u_current[:i+1]
                
                # Підінтегральна функція: K(xi, t) * u_n(t)
                integrand = K(xi, t_subgrid, **kwargs) * u_sub
                
                # Чисельне інтегрування методом трапецій на підвідрізку [0, xi]
                integral = np.trapz(integrand, t_subgrid)
                u_next[i] = f(xi, **kwargs) + integral
        
        u_current = u_next.copy()
        solutions[f'u^{k}'] = u_current

    if etalon_func:
        solutions['u_exact'] = etalon_func(x_grid)
        errors = pd.DataFrame(index=x_grid)
        for k in range(kiter + 1):
            errors[f'e^{k}'] = np.abs(solutions['u_exact'] - solutions[f'u^{k}'])
        return solutions, errors
    
    return solutions

# --- Функції задачі згідно з зображенням ---

def f(x, **kwargs):
    return 1.0

def K(x, t, **kwargs):
    # Ядро рівняння K(x, t) = t
    return t

def exact_solution(x):
    return np.exp(x**2 / 2)

if __name__ == "__main__":
    # Параметри: розв'язуємо на проміжку [0, 2]
    params = {'a': 0, 'b': 2, 'n_points': 100}
    x_grid = np.linspace(params['a'], params['b'], params['n_points'])
    kiter = 6  # Кількість ітерацій

    solutions, errors = SequencealApproach(f, K, x_grid, kiter, etalon_func=exact_solution, **params)

    print("Таблиця наближень (перші 5 вузлів):")
    print(solutions.iloc[::20, :].head()) # Вивід кожного 20-го вузла для компактності

    # Візуалізація наближень
    plt.figure(figsize=(10, 6))
    for k in range(kiter + 1):
        alpha = 0.3 + 0.7 * (k / kiter)
        plt.plot(x_grid, solutions[f'u^{k}'], label=f'Наближення u^{k}', alpha=alpha)
    
    plt.plot(x_grid, solutions['u_exact'], 'k--', label='Точний розв\'язок (exp(x²/2))', linewidth=2)
    plt.title("Метод послідовних наближень для рівняння Вольтерри")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Графік похибки
    errors.plot(title="Зменшення похибки з кожною ітерацією (log scale)", 
                xlabel='x', ylabel='|u_exact - u_k|', logy=True)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()