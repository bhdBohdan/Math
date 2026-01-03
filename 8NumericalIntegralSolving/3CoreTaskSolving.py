import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Degenerate_IE_solver(f_func, a_funcs, b_funcs, lmbd, a, b, n_points=100):
    """
    Розв'язування інтегрального рівняння Фредгольма II роду
    методом заміни ядра на вироджене: K(x, t) ≈ Σ a_j(x) * b_j(t)
    """
    n_terms = len(a_funcs)
    # Побудова матриці системи (I - λM)c = F
    M = np.zeros((n_terms, n_terms))
    F = np.zeros(n_terms)
    
    t_grid = np.linspace(a, b, n_points)
    
    for i in range(n_terms):
        # Обчислення вектора вільних членів F_i = ∫ b_i(t) * f(t) dt
        f_vals = f_func(t_grid)
        integrand_f = b_funcs[i](t_grid) * f_vals
        F[i] = np.trapezoid(integrand_f, t_grid)
        
        for k in range(n_terms):
            # Обчислення елементів матриці M_ik = ∫ b_i(t) * a_k(t) dt
            integrand_m = b_funcs[i](t_grid) * a_funcs[k](t_grid)
            M[i, k] = np.trapezoid(integrand_m, t_grid)
            
    # Розв'язання СЛАР для коефіцієнтів c
    A_matrix = np.eye(n_terms) - lmbd * M
    c = np.linalg.solve(A_matrix, F)
    
    # Побудова розв'язку u(x) = f(x) + λ * Σ c_j * a_j(x)
    def u_solution(x):
        res = f_func(x)
        sum_terms = 0
        for j in range(n_terms):
            sum_terms += c[j] * a_funcs[j](x)
        return res + lmbd * sum_terms
    
    return u_solution

# --- Наближення ядра K(x, t) = tan(x / (C + t)) ---
# Використовуємо розклад tan(z) ≈ z + z^3/3
def get_tan_expansion(C):
    # a_j(x) - функції від x, b_j(t) - функції від t
    a1 = lambda x: x
    b1 = lambda t: 1 / (C + t)
    
    a2 = lambda x: x**3
    b2 = lambda t: 1 / (3 * (C + t)**3)
    
    return [a1, a2], [b1, b2]

if __name__ == "__main__":
    # Параметри задачі
    a, b = 0, 1
    lmbd = 1.0
    f_func = lambda x: 1 + x
    x_plot = np.linspace(a, b, 20)
    
    # Створюємо DataFrame для збереження результатів
    all_results = pd.DataFrame({'x': x_plot})
    
    plt.figure(figsize=(12, 8))
    
    # Цикл i від 1 до 10
    for i_val in range(1, 11):
        a_f, b_f = get_tan_expansion(i_val)
        u_sol = Degenerate_IE_solver(f_func, a_f, b_f, lmbd, a, b)
        
        y_vals = u_sol(x_plot)
        col_name = f'i={i_val}'
        all_results[col_name] = y_vals
        
        # Градієнт кольору від синього до червоного
        color_val = (i_val - 1) / 9
        plt.plot(x_plot, y_vals, label=col_name, color=(color_val, 0.2, 1 - color_val))

    plt.title(r"Розв'язки $u(x) - \int_0^1 \tan\left(\frac{x}{i+t}\right)u(t)dt = 1+x$ для $i \in [1, 10]$")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Вивід підсумкової таблиці
    print("Результати розрахунків для кожного i:")
    pd.set_option('display.precision', 6)
    print(all_results.iloc[::2, :].to_string(index=False))