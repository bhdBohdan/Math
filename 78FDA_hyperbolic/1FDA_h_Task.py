import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from mpl_toolkits import mplot3d

from AnimationFunction import animate_string

def FDA_H_solver( f,mu,nu,phi,psi,nxst,ntst,mx,mt,kiter,etalon= None,**kwargs):
    """ Розв'язування початково-крайової задачi для хвильового рiвняння
    методом скiнчених рiзниць
    """
    Nx = nxst
    Nt = ntst
    if Nx < 4 or Nt <= 1:
        raise Exception('invalid grid')
    #обчислення розв'язкiв на кожнiй сiтцi
    solutions = pd.DataFrame()
    stx = 1
    stt = 1
    for k in range(kiter):
        h = kwargs['l']/Nx
        tau = kwargs['T']/Nt
        tau2 = tau**2
        p =(kwargs['a']*tau/h)**2

        x=np.linspace(0, kwargs['l'], Nx+1)
        t=np.linspace(0, kwargs['T'], Nt+1)
        tau2fij = np.zeros((Nt+1,Nx+1), dtype='float64')
        for j in range(Nt):
            for i in range(1,Nx):
                tau2fij[j,i] = tau2*f(x[i],t[j],**kwargs)

        u = np.zeros((Nt+1,Nx+1), dtype='float64')

        u[0,:] = phi(x,**kwargs)
        u[1,:] = u[0,:] + tau * psi(x,**kwargs)
        for j in range(2,Nt+1):
            u[j,0] = mu(t[j],**kwargs)
            u[j,Nx] = nu(t[j],**kwargs)
            for i in range(1,Nx):
                u[j,i] = p*(u[j-1, i-1] + u[j-1,i+1])+2*(1-p)*u[j-1,i] - u[j-2,i]+tau2fij[j-1,i]
       
        #збереження значень k-го розв'язку у вузлах початкової сiтки
        solutions[f'u^{k}']=u[stt:(Nt+1):stt,stx:Nx:stx].reshape((nxst-1)*(ntst))
        Nx *= mx
        Nt *= mt
        stx *= mx
        stt *= mt
    if type(etalon) == type(None):
        return u,solutions
    # запис у DataFrame еталонного розв'язку
    solutions['u'] = etalon
    # обчислення абсолютних вiдхилень вiд еталонного розв'язку
    errors = pd.DataFrame()
    for k in range(kiter):
        errors[f'e^{k}'] = np.abs(solutions['u']-solutions[f'u^{k}'])
    # обчислення очiкуваної швидкостi збiжностi чисельних розв'язкiв до еталонного
    eoc = pd.DataFrame()
    for k in range(kiter-1):
        eoc[f'r^{k}'] = errors[f'e^{k}'] / errors[f'e^{k+1}']
    return u, solutions, errors, eoc

def grid_tabulator(f,Nx,Nt,**kwargs):
    x=np.linspace(0, kwargs['l'], Nx+1)
    t=np.linspace(0, kwargs['T'], Nt+1)
    u = np.zeros(Nt*(Nx-1), dtype=float)
    ji = 0
    for j in range(1,Nt+1):
        for i in range(1,Nx):
            u[ji] = f(x[i],t[j],**kwargs)
            ji += 1
    return u


def d3_plotter(u,T,l,Nt,Nx,ks=1):
    """ Побудова 3D-графiкiв функцiї u, заданої у вузлах сiтки (Nt+1)x(Ny+1)
    на прямокутнику зi сторонами T i l
    ks==1 -- побудова одного графiку
    ks==2 -- побудова двох графiкiв
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"Розв'язок початково-крайової задачi при Nt={Nt}, Nx={Nx}")
    t=np.linspace(0, T, Nt+1)
    x=np.linspace(0, l, Nx+1)
    Y, X = np.meshgrid(x, t)
    ax = fig.add_subplot(1, ks, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    surf = ax.plot_surface(Y, X, u, rstride=1, cstride=1, cmap='viridis',linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    if ks==2:
        ax = fig.add_subplot(1, ks, 2, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        surf = ax.plot_wireframe(Y, X, u, rstride=5, cstride=5)
        plt.show()

# --- функції для задачі 5)---

def f(x, t, **kwargs):
    return -240 / (5*x + t + 2)**3

def phi(x, **kwargs):
    return 5 / (5*x + 2)

def psi(x, **kwargs):
    return -5 / (5*x + 2)**2

def mu(t, **kwargs):
    return 5 / (t + 2)

def nu(t, **kwargs):
    return 5 / (t + 7)

def exact_solution(x, t, **kwargs):
    return 5 / (5*x + t + 2)



# --- Виконання ---

if __name__ == "__main__":
    # Параметри з вашого малюнка
    params = {'a': 1.0, 'l': 1.0, 'T': 1.0}
    
    # Кроки h=0.1, tau=0.1 відповідно до умови m,n = 0...10
    Nx_start = 10
    Nt_start = 10
    
    kiter = 3 # Кількість ітерацій для перевірки збіжності (подвоєння сітки)
    mx, mt = 2, 2
    
    # Створення еталонного розв'язку
    etalon = grid_tabulator(exact_solution, Nx_start, Nt_start, **params)
    
    # Розв'язання
    u, solutions, errors, eoc = FDA_H_solver(f, mu, nu, phi, psi, 
                                             Nx_start, Nt_start, mx, mt, kiter, 
                                             etalon, **params)
    

    
    print("\nТаблиця абсолютних відхилень від еталонного розв'язку:")
    print(errors.head(5))

    for k in range(kiter):
        print(f"e1^{k} = {np.max(errors[f'e^{k}']):.2}", end=' ')

   

    print("Таблиця розв'язків на вузлах сітки:")
    print(solutions.head(5))

    print("Таблиця похибок (перші 5 вузлів):")
    print(errors.head(5))

    print("\nТаблиця очікуваної швидкості збіжності чисельних розв'язків до еталонного:")
    print(eoc.head(5))

    print("\nМаксимальні похибки на кожній ітерації:")
    for k in range(kiter):
        print(f"e^{k} = {np.max(errors[f'e^{k}']):.2e}")

    # Візуалізація фінального (найточнішого) розв'язку
    final_Nt = Nt_start * mt**(kiter-1)
    final_Nx = Nx_start * mx**(kiter-1)
    d3_plotter(u, params['T'], params['l'], final_Nt, final_Nx)
    
    # Графік похибки
    errors.plot(title="Абсолютні відхилення (log scale)", xlabel='Номер вузла', logy=True)
    plt.show()
    animate_string(u, params['l'], params['T'], final_Nx, final_Nt)
