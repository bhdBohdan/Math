import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Оновлена функція формування матриці для рівняння u'' + p(x)u' + q(x)u = f(x)
def set_matrix_diagonals_and_vector(f, p, q, x, n, **kwargs):
  
    """ функцiя задає 3 дiагоналi матрицi i вектор вiльних членiв СЛАР """
    h = np.abs(x[1] - x[0])
    h2 = h**2
    
    c = np.empty(n + 1, dtype='float64') # головна діагональ
    a = np.empty(n + 1, dtype='float64') # нижня діагональ
    b = np.empty(n + 1, dtype='float64') # верхня діагональ
    d = np.empty(n + 1, dtype='float64') # вектор правої частини

    # Внутрішні вузли i = 1 ... n-1
    for i in range(1, n):
        p_val = p(x[i])
        a[i] = 1 - (h / 2) * p_val
        c[i] = h2 * q(x[i]) - 2
        b[i] = 1 + (h / 2) * p_val
        d[i] = h2 * f(x[i])

    # Ліва грань (x=0): alpha0*u + beta0*u' = gamma0
    # u0(alpha0*h - beta0) + u1(beta0) = h*gamma0
    c[0] = kwargs['alpha0'] * h - kwargs['beta0']
    b[0] = kwargs['beta0']
    a[0] = 0
    d[0] = h * kwargs['gamma0']

    # Права грань (x=1): alpha1*u + beta1*u' = gamma1
    # u_{n-1}(-beta1) + u_n(alpha1*h + beta1) = h*gamma1
    a[n] = -kwargs['beta1']
    c[n] = kwargs['alpha1'] * h + kwargs['beta1']
    b[n] = 0
    d[n] = h * kwargs['gamma1']

    return c, a, b, d

def tridiagonal_matrix_algorithm(a, b, c, g):
    """ метод прогонки для розв'язування СЛАР
    з 3-дiагональною матрицею
    вектор с-головна дiагональ
    вектори a i b - нижня i верхня дiагоналi, паралельнi головнiй
    вектор g - вiльнi члени
    """
    
    n1 = c.size
    alpha = np.empty(n1, dtype=float)
    beta = np.empty(n1, dtype=float)
    
    alpha[0] = -b[0] / c[0]
    beta[0] = g[0] / c[0]
    
    for i in range(1, n1):
        w = a[i] * alpha[i-1] + c[i]
        alpha[i] = -b[i] / w
        beta[i] = (g[i] - a[i] * beta[i-1]) / w
        
    x = np.empty(n1, dtype=float)
    n = n1 - 1
    x[n] = beta[n]
    for i in range(n-1, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]
    return x

def FDA_solver(f, p, q, nstrt, kiter, etalon_func, **kwargs):
    """ Розв'язування крайових задач для ЗДР
    методом скiнчених рiзниць
    """  
    n = nstrt
    x_list, u_list = [], []
    
    for k in range(kiter):
        xk = np.linspace(kwargs['a'], kwargs['b'], n + 1)
        c, a, b, d = set_matrix_diagonals_and_vector(f, p, q, xk, n, **kwargs)
        uk = tridiagonal_matrix_algorithm(a, b, c, d)
        x_list.append(xk)
        u_list.append(uk)
        n *= 2
        
    # Таблиця розв'язків у вузлах початкової сітки
    solutions = pd.DataFrame(index=x_list[0])
    ist = 1
    for k in range(kiter):
        solutions[f'u^{k}'] = u_list[k][::ist]
        ist *= 2
    
    solutions['ux'] = etalon_func(x_list[0])
    errors = pd.DataFrame(index=x_list[0])
    for k in range(kiter):
        errors[f'e^{k}'] = np.abs(solutions['ux'] - solutions[f'u^{k}'])
        
    return u_list, x_list, solutions, errors

# --- Параметри задачі 5 ---
def f_func(x): return -np.sin(2 + x**2) - x * np.cos(x)
def p_func(x): return x
def q_func(x): return -1
def u_star(x): return x * np.cos(x)

params = {
    'a': 0, 'b': 1,
    'alpha0': 1, 'beta0': 1, 'gamma0': 1,
    'alpha1': 1, 'beta1': 0, 'gamma1': np.cos(1)
}

nstrt, kiter = 20, 5
u, x, solutions, errors = FDA_solver(f_func, p_func, q_func, nstrt, kiter, u_star, **params)

# Перевірка точності
max_err = errors[f'e^{kiter-1}'].max()
print(f"Максимальна похибка на останній ітерації: {max_err:.2e}")
print(f"Чи досягнута точність 10^-4: {'Так' if max_err < 1e-4 else 'Ні'}")

print("Розв'язки на сітках з подвоєнням кількості вузлів:")
print(solutions.head(5)
)

solutions.plot(xlabel='x', title='Розв\'язки задачi Кошi у вузлах початкової сiтки')
plt.show()

print("Абсолютні похибки розв'язків:")
print(errors.head(5))

# Візуалізація
plt.figure(figsize=(10, 5))
plt.plot(x[-1], u[-1], label='Numerical (final grid)')
plt.plot(x[-1], u_star(x[-1]), '--', label='Exact solution')
plt.title('Розв\'язок крайової задачі')
plt.legend()
plt.grid(True)
plt.show()

# Графік похибки
errors.plot(title='Похибки на різних сітках (log scale)', logy=True)
plt.show()