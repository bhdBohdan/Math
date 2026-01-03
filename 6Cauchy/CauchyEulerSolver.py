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

def f(x, u):
    return u**2+2*x-x**4

a=0
b=1
u0=0
n = 5
x = np.linspace(a, b, n+1)
u = Euler_method(f,u0,a,b,n)
for i in range(n+1):
    print(f"u({x[i]:.2})={u[i]:.6}")

us = [u]
xs = [x]
kiter = 5
for k in range(kiter):
    n*=2
    xs.append(np.linspace(a, b, n+1))
    us.append(Euler_method(f,u0,a,b,n))

xp=np.linspace(a, b, 256)
uxp=xp**2

# fig = plt.figure(figsize=(8, 5))
# plt.plot(xp, uxp, label='uxp')
# for k in range(kiter):
#     plt.plot(xs[k], us[k], label=f'us_{k}')
# ax = fig.gca()
# ax.legend()
# plt.show()

solutions=pd.DataFrame(index=xs[0][1::])
ist = 1
for k in range(kiter):
    solutions[f'us^{k}'] = us[k][ist::ist]
    ist *= 2

solutions['u'] = xs[0][1::]**2
print(solutions)
solutions.plot(figsize=(8,5))
plt.show()

errors=pd.DataFrame(index=xs[0][1::])
for k in range(kiter):
    errors[f'e^{k}'] = np.abs(solutions['u'] - solutions[f'us^{k}'])
print(errors)
errors.plot(figsize=(8,5), logy=True)
plt.show()