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

def f2(x, u):
    return -u*np.cos(x) + np.exp(-np.sin(x))
def exact(x):
    return (x+1)*np.exp(-np.sin(x))

a = 0
b = 10
u0 = 1
n = 10
kiter = 6

u,x,solutions,errors,eoc = Cauchy_ODE_solver(f2,u0,a,b,RK4_method, 10,kiter, exact= exact)

print("Чисельні розв'язки на послідовних сітках:")
print(solutions)
solutions.plot(figsize=(8,5))
plt.show()

print("Абсолютні похибки чисельних розв'язків:")
print(errors)
errors.plot(figsize=(8,5), logy=True)
plt.show()