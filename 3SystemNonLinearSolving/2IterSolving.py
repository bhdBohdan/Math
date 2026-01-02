import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def norm_3(a):
    return np.sqrt(np.sum(a**2))

def Newton_iteration(f, invJ, x0, eps, kmax=1000):
    x_prev = x0.copy()
    k = 0
    while k < kmax:
        k += 1
        x_new = x_prev - invJ(x_prev).dot(f(x_prev))
        if norm_3(x_new - x_prev) < eps:
            return k, x_new
        x_prev = x_new
    raise Exception(f'Не вдалося досягти точності за {kmax} ітерацій')

# Система рівнянь f(x) = 0
def f(x_vec):
    x, y = x_vec[0], x_vec[1]
    f0 = x - np.cos(y) - 2
    f1 = np.cos(x - 1) + y - 0.8
    return np.array([f0, f1])

# Обернена матриця Якобі
def inverse_Jacobian_matrix(x_vec):
    x, y = x_vec[0], x_vec[1]
    df00, df01 = 1, np.sin(y)
    df10, df11 = -np.sin(x - 1), 1
    J = np.array([[df00, df01], [df10, df11]])
    return linalg.inv(J)

# Початкова точка з графічного аналізу
x0 = np.array([2.5, 0.5], dtype='float64')
eps = 1e-4

print("Метод Newton  :")
print(f"{'Точність':>10} | {'Ітерацій':>8} | {'Розв’язок x':>15} | {'Розв’язок y':>15}")
print("-" * 60)

for n in range(3, 8, 2): # Обмежимося eps=10^-7, бо метод повільний
    eps = 10**(-n)
    try:
        k, xk = Newton_iteration(f, inverse_Jacobian_matrix, x0, eps)
        print(f"10^{(-n):<3} | {k:>8} | {xk[0]:>15.10f} | {xk[1]:>15.10f}")
    except Exception as e:
        print(f"eps=10^-{n}: {e}")

# Візуалізація для підтвердження наближення
x_vals = np.linspace(1, 4, 100)
y_vals = np.linspace(-1, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
F1 = X - np.cos(Y) - 2
F2 = np.cos(X - 1) + Y - 0.8

plt.contour(X, Y, F1, [0], colors='r')
plt.contour(X, Y, F2, [0], colors='b')
plt.grid(True)
plt.title("Графічне знаходження кореня")
plt.show()

def simple_iteration(g, x0, eps, kmax=10000): # Збільшено kmax, бо збіжність повільна
    x_prev = x0.copy()
    k = 0
    while k < kmax:
        k += 1
        x_new = g(x_prev)
        if norm_3(x_new - x_prev) < eps:
            return k, x_new
        x_prev = x_new
    raise Exception(f'Точність {eps} не досягнута за {kmax} ітерацій')

# Ітераційні функції g(x, y)
def g_func(x_vec):
    x, y = x_vec[0], x_vec[1]
    g0 = np.cos(y) + 2
    g1 = 0.8 - np.cos(x - 1)
    return np.array([g0, g1])

# Початкове наближення
x0 = np.array([2.5, 0.5], dtype='float64')

print("Метод простих ітерацій (повільна збіжність):")

print(f"{'Точність':>10} | {'Ітерацій':>8} | {'Розв’язок x':>15} | {'Розв’язок y':>15}")
print("-" * 60)

for n in range(3, 8, 2): # Обмежимося eps=10^-7, бо метод повільний
    eps = 10**(-n)
    try:
        k, xk = simple_iteration(g_func, x0, eps)
        print(f"10^{(-n):<3} | {k:>8} | {xk[0]:>15.10f} | {xk[1]:>15.10f}")
    except Exception as e:
        print(f"eps=10^-{n}: {e}")