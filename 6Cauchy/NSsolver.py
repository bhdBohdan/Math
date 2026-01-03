import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Cauchy_ODE_NS_solver(f,g,u0,v0,a,b,numerical_method, nstrt, kiter, exact = None,**kwargs):
    """ застосування чисельного методу до задачi Кошi """
    # обчислення чисельних розв'язкiв на послiдовносi сiток
    # з подвоєнням кiлькостi вузлiв
    n = nstrt
    x = []
    u = []
    v = []
    for k in range(kiter):
        x.append(np.linspace(a, b, n+1))
        uk, vk = numerical_method(f,g,u0,v0,a,b,n,**kwargs)
        u.append(uk)
        v.append(vk)
        n *= 2
    # запис у DataFrame значень усiх чисельних розв'язкiв
    # у початковому масивi точок x[0]
    solutions = pd.DataFrame(index=x[0][1::])
    ist = 1
    for k in range(kiter):
        solutions[f'u^{k}'] = u[k][ist::ist]
        solutions[f'v^{k}'] = v[k][ist::ist]
        ist *= 2
    if exact == None:
        return u,v,x,solutions
    
def Euler_NS(f,g,u0,v0,a,b,n, **kwargs):
    """метод Ейлера для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    u = np.empty(n+1, dtype='float64')
    v = np.empty(n+1, dtype='float64')
    u[0] = u0
    v[0] = v0
    for i in range(n):
        u[i+1] = u[i] + h*f(x[i], u[i], v[i], **kwargs)
        v[i+1] = v[i] + h*g(x[i], u[i], v[i], **kwargs)
    return u, v

def RK4_NS(f,g,u0,v0,a,b,n, **kwargs):
    """ метод Рунге-Кутта четвертого порядку
    для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x=np.linspace(a, b, n+1)
    h=(b-a)/n
    u = np.empty(n+1, dtype='float64')
    v = np.empty(n+1, dtype='float64')
    u[0] = u0
    v[0] = v0
    for i in range(n):
        k1 = f(x[i], u[i], v[i], **kwargs)
        m1 = g(x[i], u[i], v[i], **kwargs)
        k2 = f(x[i], u[i] + h/2*k1, v[i] + h/2*m1, **kwargs)
        m2 = g(x[i], u[i] + h/2*k1, v[i] + h/2*m1, **kwargs)
        k3 = f(x[i], u[i] + h/2*k2, v[i] + h/2*m2, **kwargs)
        m3 = g(x[i], u[i] + h/2*k2, v[i] + h/2*m2, **kwargs)
        k4 = f(x[i], u[i] + h*k3, v[i] + h*m3, **kwargs)
        m4 = g(x[i], u[i] + h*k3, v[i] + h*m3, **kwargs)

        u[i+1]=u[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
        v[i+1]=v[i] + h/6*(m1 + 2*m2 + 2*m3 + m4)
    return u, v

def f(x, u, v, **kwargs ):
    return kwargs['alpha']*u + kwargs['beta']*u*v

def g(x, u, v, **kwargs ):
    return kwargs['gamma']*v + kwargs['delta']*u*v

a= 0
b= 15
u0 = 80
v0 = 30
params = {'alpha': 0.25, 'beta': - 0.01, 'gamma': -1, 'delta': 0.01}

n = 32
kiter = 4
u, v, x,solutions = Cauchy_ODE_NS_solver(f,g,u0,v0,a,b,Euler_NS, n, kiter, **params)

print(solutions[['u^0','v^0','u^2','v^2','u^3','v^3']].head(4))
solutions.plot(xlabel='x', title='Розв\'язки задачi Кошi, отриманi методом Ейлера')
plt.show()

fig = plt.figure(figsize=(8, 5))
plt.title('Фазовi траєкторiї розв\'язкiв задачi Кошi, отриманих методом Ейлера')
plt.scatter(u0,v0, marker='o', label='start')
for k in range(kiter):
    plt.plot( solutions[f'u^{k}'],solutions[f'v^{k}'], label=f'u^{k},v^{k}')
plt.xlabel("u")
plt.ylabel("v")
ax = fig.gca()
ax.legend()
plt.show()

u_rk, v_rk, x,solutions_rk = Cauchy_ODE_NS_solver(f,g,u0,v0,a,b,RK4_NS, n, kiter, **params)
print(solutions_rk[['u^0','v^0','u^2','v^2','u^3','v^3']].head(4))

fig = plt.figure(figsize=(8, 5))
plt.title('Фазовi траєкторiї розв\'язкiв задачi Кошi, отриманих методом Ейлера')
plt.scatter(u0,v0, marker='o', label='start')
for k in range(kiter):
    plt.plot( solutions[f'u^{k}'],solutions[f'v^{k}'], label=f'u^{k},v^{k}')
plt.xlabel("u")
plt.ylabel("v")
ax = fig.gca()
ax.legend()

fig = plt.figure(figsize=(8, 5))
plt.title('Фазовi траєкторiї розв\'язкiв задачi Кошi, отриманих методом Рунге-Кутти')
plt.scatter(u0,v0, marker='o', label='start')
for k in range(kiter):
    plt.plot( solutions_rk[f'u^{k}'],solutions_rk[f'v^{k}'], label=f'u^{k},v^{k}')
plt.xlabel("u")
plt.ylabel("v")
ax = fig.gca()
ax.legend()
plt.show()



