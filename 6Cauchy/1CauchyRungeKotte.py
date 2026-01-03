import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Cauchy_ODE_solver(f,u0,a,b,numerical_method, nstrt, kiter, exact = None):
        # обчислення чисельних розв'язкiв на послiдовносi сiток
    # з подвоєнням кiлькостi вузлiв
    n = nstrt
    x = []
    u = []
    for k in range(kiter):
        x.append(np.linspace(a, b, n+1))
        u.append(numerical_method(f,u0,a,b,n))
        n *= 2
    # запис у DataFrame значень усiх чисельних розв'язкiв
    # у початковому масивi точок x[0]
    solutions = pd.DataFrame(index=x[0][1::])
    ist = 1
    for k in range(kiter):
        solutions[f'u^{k}'] = u[k][ist::ist]
        ist *= 2
    if exact == None:
        return u,x,solutions
    # обчислення точного розв'язку у початковому масивi точок x[0]
    # i запис у DataFrame
    solutions['u'] = exact(x[0][1::])
    # обчислення абсолютних похибок i запис у новий DataFrame
    errors = pd.DataFrame(index=x[0][1::])
    for k in range(kiter):
        errors[f'e^{k}']=np.abs(solutions['u']-solutions[f'u^{k}'])
    # обчислення очiкуваної швидкостi збiжностi чисельних розв'язкiв до точного
    eoc = pd.DataFrame(index=x[0][1::])
    for k in range(kiter-1):
        eoc[f'r^{k}'] = errors[f'e^{k}'] / errors[f'e^{k+1}']

    return u,x, solutions, errors, eoc


def Euler_method(f,u0,a,b,n):
    """ метод Ейлера """
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    u = np.empty(n+1, dtype='float64')
    u[0] = u0
    for i in range(1,n+1):
        u[i] = u[i-1] + h*f(x[i-1],u[i-1])
    return u

def RK4_method(f,u0,a,b,n):
    """ метод Рунге-Кутта четвертого порядку
    """
    h=(b-a)/n
    x=np.linspace(a, b, n+1)
    u=np.empty(n+1)
    u[0]=u0
    for i in range(n):
        k1 = f(x[i], u[i])
        k2 = f(x[i] + h/2, u[i] + h/2*k1)
        k3 = f(x[i] + h/2, u[i] + h/2*k2)
        k4 = f(x[i+1], u[i] + h*k3)
        u[i+1] =u [i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return u

def f_func(x, u):
    return 1 + u / (x * (x + 1))

def exact_sol(x):
    return (x**2 + x * np.log(x)) / (x + 1)

# 2. Параметри задачі
u0 = 0.5
a, b = 1, 2
n_start = 10  # початкова кількість кроків
k_iterations = 5  # кількість подвоєнь сітки

# 3. Обчислення для методу Ейлера
u_e, x_e, sol_e, err_e, r_e = Cauchy_ODE_solver(f_func, u0, a, b, Euler_method, n_start, k_iterations, exact_sol)

# 4. Обчислення для методу Рунге-Кутти 4-го порядку
u_rk, x_rk, sol_rk, err_rk, r_rk = Cauchy_ODE_solver(f_func, u0, a, b, RK4_method, n_start, k_iterations, exact_sol)

# --- Вивід результатів ---

print("--- МЕТОД ЕЙЛЕРА (Похибки) ---")
print(err_e.head())
print("\nШвидкість збіжності (для Ейлера очікуємо ~2):")
print(r_e.mean())

print("\n--- МЕТОД РУНГЕ-КУТТИ 4 (Похибки) ---")
print(err_rk.head())
print("\nШвидкість збіжності (для RK4 очікуємо 2^4 = 16):")
print(r_rk.mean())


# 5. Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(x_e[-1], u_e[-1], 'r--', label='Euler (n_max)')
plt.plot(x_rk[-1], u_rk[-1], 'b:', label='RK4 (n_max)')
x_fine = np.linspace(a, b, 200)
plt.plot(x_fine, exact_sol(x_fine), 'k', alpha=0.5, label='Exact solution')
plt.title("Порівняння чисельних методів")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()

err_e.plot(figsize=(10, 6), logy=True, title="Похибки методу Ейлера (логарифмічна шкала)")
plt.show()
err_rk.plot(figsize=(10, 6), logy=True, title="Похибки методу Рунге-Кутти 4-го порядку (логарифмічна шкала)")
plt.show()