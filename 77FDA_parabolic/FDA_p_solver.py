import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

def FDA_P_solver(f,mu,nu,phi,nxst,ntst,mx,mt,kiter,etalon=None,**kwargs):
    """ Розв'язування початково-крайової задачi для рiвняння теплопровiдностi
    методом скiнчених рiзниць (явна схема)
    """
    Nx = nxst
    Nt = ntst
    if Nx <= 4 or Nt <= 1:
        raise Exception('invalid grid')
    #обчислення розв'язкiв на кожнiй сiтцi
    solutions = pd.DataFrame()
    stx = 1
    stt = 1
    for k in range(kiter):
        h = kwargs['l']/Nx
        tau = kwargs['T']/Nt
        alpha =kwargs['a']**2 *tau/(h*h)
        x=np.linspace(0, kwargs['l'], Nx+1)
        t=np.linspace(0, kwargs['T'], Nt+1)
        taufij = np.zeros((Nt,Nx+1), dtype='float64')
        for j in range(Nt):
            for i in range(1,Nx):
                taufij[j,i] = tau*f(x[i],t[j],**kwargs)
        u = np.zeros((Nt+1,Nx+1), dtype='float64')
        u[0,:] = phi(x,**kwargs)
        for j in range(1,Nt+1):
            u[j,0] = mu(t[j],**kwargs)
            u[j,Nx] = nu(t[j],**kwargs)
            for i in range(1,Nx):
                u[j,i]=alpha*(u[j-1,i-1]+u[j-1,i+1])+(1-2*alpha)*u[j-1,i]+taufij[j-1,i]
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
    surf = ax.plot_surface(Y, X, u, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    if ks==2:
        ax = fig.add_subplot(1, ks, 2, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        surf = ax.plot_wireframe(Y, X, u, rstride=5, cstride=5)
    plt.show()


def exact_solution(x,t,**kwargs):
    return kwargs['alpha']*t**2 + x

def f(x,t,**kwargs):
    return 2*kwargs['alpha']*t
def phi(x,**kwargs):
    return x
def mu(t,**kwargs):
    return kwargs['alpha']*t**2
def nu(t,**kwargs):
    return kwargs['alpha']*t**2+kwargs['l']

if __name__ == "__main__":
    
    params = {'a':1, 'alpha': 1., 'l': 3.,'T':2}
    Nx_start = 6
    Nt_start = 8
    kiter = 5
    mx = 1
    mt = 2
    Nx = Nx_start
    Nt = Nt_start
    etalon = grid_tabulator(exact_solution,Nx,Nt,**params)
    u,solutions,errors,eoc=FDA_P_solver(f,mu,nu,phi,Nx,Nt,mx,mt,kiter,etalon,**params)

    d3_plotter(u, params['T'],params['l'],Nt*mt**(kiter-1),Nx*mx**(kiter-1))

    print("Розв'язки на сітках з подвоєнням кількості вузлів:")
    print(solutions.head(5))

    print("Абсолютні похибки розв'язків:")
    print(errors.head(5))

    print("Очікувана швидкість збіжності:")
    print(eoc.head(5))

    errors.plot(xlabel='номер внутрiшнього вузла(наскрiзна нумерацiя)',
    title='Абсолютна похибка чисельних розв\'язкiв \n на спiльнiй множинi вузлiв 6х8 заcлогарифмiчною шкалою',
    logy=True)
    for k in range(kiter):
        print(f"e1^{k} = {np.max(errors[f'e^{k}']):.2}", end=' ')

    Nx1 = 18
    Nt1 = 8
    etalon1 = grid_tabulator(exact_solution,Nx1,Nt1,**params)
    u1,solutions1,errors1,eoc1 =FDA_P_solver(f,mu,nu,phi,Nx1,Nt1,mx,mt,kiter,etalon1,**params)
    d3_plotter(u1,params['T'],params['l'],Nt1*mt**(kiter-1),Nx1*mx**(kiter-1))
