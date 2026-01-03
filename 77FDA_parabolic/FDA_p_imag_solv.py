import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

from FDA_p_solver import grid_tabulator, d3_plotter

def FDA_Pim_solver(f,mu,nu,phi,nxst,ntst,mx,mt,kiter,etalon=None,**kwargs):
    """ Розв'язування початково-крайової задачi для рiвняння теплопровiдностi
    методом скiнчених рiзниць (неявна схема)
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
        ptau = h**2 / kwargs['a']**2
        p = ptau / tau
        c,d,b = set_matrix(p,Nt,Nx)
        x=np.linspace(0, kwargs['l'], Nx+1)
        t=np.linspace(0, kwargs['T'], Nt+1)
        ptaufij = np.zeros((Nt+1,Nx+1), dtype='float64')
        for j in range(1,Nt+1):
            for i in range(1,Nx):
                ptaufij[j,i] = ptau * f(x[i],t[j],**kwargs)

        u = np.zeros((Nt+1,Nx+1), dtype='float64')
        u[0,:] = phi(x,**kwargs)
        g = np.zeros(Nx+1, dtype='float64')
        for j in range(1,Nt+1):
            set_vector(g,u[j-1,:],p,mu,nu,t[j],ptaufij[j, :],**kwargs)
            u[j,:] = TDMA_solver(c, d , b, g)
        #збереження значень k-го розв'язку у вузлах початкової сiтки

        solutions[f'u^{k}']=u[stt:(Nt+1):stt,stx:Nx:stx].reshape((nxst-1)*ntst)
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

def set_matrix(p,Nt,Nx):
    """ функцiя задає дiагоналi матрицi СЛАР """
    c = np.full(Nx+1, -(p+2), dtype=float)
    c[0] = 1.
    c[Nx] = 1.
    a = np.full(Nx+1, 1, dtype=float)
    a[0] = 0
    a[Nx] = 0
    b = np.full(Nx+1, 1, dtype=float)
    b[0] = 0
    b[Nx] = 0
    return c,a,b

def set_vector(g,uprev,p,mu,nu,t,ptaufj,**kwargs):
    """ обчислення вектора вiльних членiв СЛАР """
    for i in range(1,len(g)):
        g[i] = - p*uprev[i] - ptaufj[i]
    g[0] = mu(t,**kwargs)
    g[-1] = nu(t,**kwargs)

def TDMA_solver(c, a, b, g):
    n = len(c)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -b[0] / c[0]
    beta[0] = g[0] / c[0]

    for i in range(1, n):
        w = c[i] + a[i] * alpha[i-1]
        alpha[i] = -b[i] / w
        beta[i] = (g[i] - a[i] * beta[i-1]) / w

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x


# --- Параметри задачі ---

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
    u,solutions,errors,eoc=FDA_Pim_solver(f,mu,nu,phi,Nx,Nt,mx,mt,kiter,etalon,**params)

    print("Розв'язки на сітках з подвоєнням кількості вузлів:")
    print(solutions.head(5))
    print("Абсолютні похибки розв'язків:")
    print(errors.head(5))

    errors.plot(xlabel='номер внутрiшнього вузла(наскрiзна нумерацiя)',
    title='Абсолютна похибка чисельних розв\'язкiв \n на спiльнiй множинi вузлiв 6х8 заcлогарифмiчною шкалою',
    logy=True)

    print("Максимальні похибки на кожній ітерації:")
    for k in range(kiter):
        print(f"e^{k} = {np.max(errors[f'e^{k}']):.2e}")

    d3_plotter(u, params['T'],params['l'],Nt*mt**(kiter-1),Nx*mx**(kiter-1))