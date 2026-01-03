import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Cauchy_ODE_NS_solver(f,g,u0,v0,a,b,numerical_method, nstrt, kiter, exact = None,**kwargs):
    """ застосування чисельного методу до задачi Кошi """
    # обчислення чисельних розв'язкiв на послiдовносi сiток
    # з подвоєнням кiлькостi вузлiв
    n = nstrt
    x_grids = []
    u_grids = []
    v_grids = []
    
    for k in range(kiter):
        x = np.linspace(a, b, n + 1)
        uk, vk = numerical_method(f, g, u0, v0, a, b, n, **kwargs)
        x_grids.append(x)
        u_grids.append(uk)
        v_grids.append(vk)
        n *= 2
        
    # Створюємо DataFrame для порівняння значень у вузлах самої розрідженої сітки (x0)
    solutions = pd.DataFrame(index=x_grids[0])
    ist = 1
    for k in range(kiter):
        solutions[f'y_{k}'] = u_grids[k][::ist]
        solutions[f'z_{k}'] = v_grids[k][::ist]
        ist *= 2
        
    return u_grids, v_grids, x_grids, solutions
    
def Euler_NS(f,g,u0,v0,a,b,n, **kwargs):
    """метод Ейлера для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    u[0], v[0] = u0, v0
    for i in range(n):
        u[i+1] = u[i] + h * f(x[i], u[i], v[i], **kwargs)
        v[i+1] = v[i] + h * g(x[i], u[i], v[i], **kwargs)
    return u, v

def RK4_NS(f,g,u0,v0,a,b,n, **kwargs):
    """ метод Рунге-Кутта четвертого порядку
    для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    u[0], v[0] = u0, v0
    for i in range(n):
        k1 = f(x[i], u[i], v[i])
        m1 = g(x[i], u[i], v[i])
        k2 = f(x[i] + h/2, u[i] + h/2*k1, v[i] + h/2*m1)
        m2 = g(x[i] + h/2, u[i] + h/2*k1, v[i] + h/2*m1)
        k3 = f(x[i] + h/2, u[i] + h/2*k2, v[i] + h/2*m2)
        m3 = g(x[i] + h/2, u[i] + h/2*k2, v[i] + h/2*m2)
        k4 = f(x[i] + h, u[i] + h*k3, v[i] + h*m3)
        m4 = g(x[i] + h, u[i] + h*k3, v[i] + h*m3)
        u[i+1] = u[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        v[i+1] = v[i] + h/6 * (m1 + 2*m2 + 2*m3 + m4)
    return u, v


# --- Нова логіка для конкретної задачі ---

# 1. Функції правої частини
def f_sys(x, y, z): return y - 3*z
def g_sys(x, y, z): return 3*y + z

# 2. Точний розв'язок (з другого зображення)
def y_exact(x): return np.exp(x) * (2 * np.cos(3*x) + np.sin(3*x))
def z_exact(x): return np.exp(x) * (2 * np.sin(3*x) - np.cos(3*x))

# 3. Параметри
a, b = 0, 0.1
y0, z0 = 2, -1
h_main = 0.01
n_start = int((b - a) / h_main)
epsilon = 0.01

# 4. Обчислення (kiter=2 достатньо для оцінки за Рунге між кроками h та h/2)
u_e, v_e, x_e, sol_e = Cauchy_ODE_NS_solver(f_sys, g_sys, y0, z0, a, b, Euler_NS, n_start, 2)
u_rk, v_rk, x_rk, sol_rk = Cauchy_ODE_NS_solver(f_sys, g_sys, y0, z0, a, b, RK4_NS, n_start, 2)

# 5. Оцінка похибки за принципом Рунге: R = |Uh - Uh/2| / (2^p - 1)
# Для Ейлера p=1, для RK4 p=4
err_runge_euler = np.max(np.abs(sol_e['y_0'] - sol_e['y_1'])) / (2**1 - 1)
err_runge_rk4 = np.max(np.abs(sol_rk['y_0'] - sol_rk['y_1'])) / (2**4 - 1)

# Додаємо точний розв'язок для порівняння
x_nodes = sol_rk.index
sol_rk['y_exact'] = y_exact(x_nodes)
sol_rk['z_exact'] = z_exact(x_nodes)

print(f"Оцінка похибки за Рунге (Ейлер): {err_runge_euler:.6f} (ε={epsilon})")
print(f"Оцінка похибки за Рунге (RK4): {err_runge_rk4:.10f} (ε={epsilon})")

# Вивід таблиці порівняння (RK4 та Точний розв'язок)
print("\nПорівняння RK4 з точним розв'язком:")
print(sol_rk[['y_1', 'y_exact', 'z_1', 'z_exact']].tail())

# Візуалізація
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Графіки компонент
ax[0].plot(x_rk[1], u_rk[1], 'r-', label='y (RK4)')
ax[0].plot(x_rk[1], v_rk[1], 'b-', label='z (RK4)')
ax[0].plot(x_rk[1], y_exact(x_rk[1]), 'k--', alpha=0.6, label='y (Exact)')
ax[0].plot(x_rk[1], z_exact(x_rk[1]), 'g--', alpha=0.6, label='z (Exact)')
ax[0].set_title("Розв'язки компонент y(x) та z(x)")
ax[0].legend()
ax[0].grid(True)

# Фазова траєкторія
ax[1].plot(u_rk[1], v_rk[1], 'm-', label='Phase Trajectory')
ax[1].scatter(y0, z0, color='black', zorder=5, label='Start (2, -1)')
ax[1].set_xlabel('y')
ax[1].set_ylabel('z')
ax[1].set_title("Фазовий портрет системи")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()