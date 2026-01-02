import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def norm_3(a):
    return np.sqrt(np.sum(a**2))

def Newton_iteration(f, invJ, x0, eps, kmax=1000):
    x_prev = x0.copy()
    k = 0
    while k < kmax:
        k += 1
        # Формула Ньютона: x_new = x_prev - J^-1 * f(x_prev)
        delta = invJ(x_prev).dot(f(x_prev))
        x_new = x_prev - delta
        
        if norm_3(x_new - x_prev) < eps:
            return k, x_new
        x_prev = x_new
        
    raise Exception(f'Методом Ньютона точнiсть {eps} не досягнута за {kmax} iтерацiй')

# Вектор-функція системи
def f(x_vec):
    x, y = x_vec[0], x_vec[1]
    f0 = x**2 * y**2 - 3*x**3 - 6*y**3 + 8
    f1 = x**4 - 9*y + 2
    return np.array([f0, f1])

# Обернена матриця Якобі
def inverse_Jacobian_matrix(x_vec):
    x, y = x_vec[0], x_vec[1]
    df00 = 2*x * y**2 - 9*x**2
    df01 = 2*x**2 * y - 18*y**2
    df10 = 4*x**3
    df11 = -9
    
    J = np.array([[df00, df01], [df10, df11]])
    return linalg.inv(J)

# Початкове наближення (визначене графічно раніше)
x0 = np.array([1.3, 0.6], dtype='float64')

print(f"{'Точність':>10} | {'Ітерацій':>8} | {'Розв’язок x':>18} | {'Розв’язок y':>18}")
print("-" * 75)

try:
    for n in range(3, 12, 2):
        eps = 10**(-n)
        k, xk = Newton_iteration(f, inverse_Jacobian_matrix, x0, eps)
        print(f"10^{(-n):<3} | {k:>8} | {xk[0]:>18.15f} | {xk[1]:>18.15f}")
except Exception as e:
    print(e)