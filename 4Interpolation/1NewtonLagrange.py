import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Дані з таблиці
x_nodes = np.array([3.2, 3.6, 5.8, 5.9, 6.2])
y_nodes = np.array([5.3, 6.0, 2.4, -1.0, -3.2])
x_target = 4.0

# 1. Лагранж
def lagrange(x, x_n, y_n):
    z = 0
    for i in range(len(x_n)):
        p = 1
        for j in range(len(x_n)):
            if i != j:
                p *= (x - x_n[j]) / (x_n[i] - x_n[j])
        z += p * y_n[i]
    return z

# 2. Ньютон
def newton_divided_diff(x_n, y_n):
    n = len(y_n)
    coef = np.zeros([n, n])
    coef[:, 0] = y_n
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x_n[i + j] - x_n[i])
    return coef[0, :]

def newton_poly(x, x_n, coef):
    n = len(coef)
    p = coef[n-1]
    for i in range(n - 2, -1, -1):
        p = coef[i] + (x - x_n[i]) * p
    return p

# 3. Кубічний сплайн
# bc_type='natural' означає, що друга похідна на кінцях дорівнює 0
cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')

# Обчислення в точці x = 4
val_lagrange = lagrange(x_target, x_nodes, y_nodes)
coef_newton = newton_divided_diff(x_nodes, y_nodes)
val_newton = newton_poly(x_target, x_nodes, coef_newton)
val_spline = cs(x_target)

print(f"Результати для x = {x_target}:")
print(f"Лагранж: {val_lagrange:.6f}")
print(f"Ньютон:  {val_newton:.6f}")
print(f"Сплайн:  {val_spline:.6f}")

# Підготовка даних для графіків
x_plt = np.linspace(min(x_nodes), max(x_nodes), 500)
y_lagrange = [lagrange(i, x_nodes, y_nodes) for i in x_plt]
y_newton = [newton_poly(i, x_nodes, coef_newton) for i in x_plt]
y_spline = cs(x_plt)

methods = [
    ("Многочлен Лагранжа", y_lagrange, val_lagrange, 'orange'),
    ("Многочлен Ньютона", y_newton, val_newton, 'green'),
    ("Кубічний сплайн", y_spline, val_spline, 'blue')
]

# Вивід трьох окремих графіків
for title, y_data, target_val, color in methods:
    plt.figure(figsize=(8, 4))
    plt.plot(x_plt, y_data, label=title, color=color, linewidth=2)
    plt.scatter(x_nodes, y_nodes, color='red', label='Вузли')
    plt.scatter(x_target, target_val, color='black', marker='X', s=100, label=f'x=4 (y={target_val:.3f})')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()