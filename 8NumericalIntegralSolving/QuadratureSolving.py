import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

def IE_solver(f,K,quadrature,kiter,etalon=None,**kwargs):
    """ Розв'язування iнтегральних рiвнянь Фредгольма II-го роду
    методом механiчних квадратур
    з рiвномiрним розбиттям промiжку iнтегрування
    """
    x = []
    u = []
    m = kwargs['mstrt']
    #обчислення розв'язкiв на кожнiй сiтцi xk
    for k in range(kiter):
        xk, wk = quadrature(kwargs['a'], kwargs['b'], m)
        x.append(xk)
        tk = xk.size
        h = (kwargs['b'] - kwargs['a']) / m
        wk *= kwargs['lmbd'] * h
        fk = f(xk,**kwargs)
        B = np.zeros((tk,tk),dtype=float)
        for i in range(tk):
            B[i,:] = -wk * K(xk[i],xk,**kwargs)
            B[i,i] += 1

        uk = np.linalg.solve(B, fk)
        u.append(uk)
        m *= 2
    # запис у DataFrame значень усiх чисельних розв'язкiв
    # у вузлах квадратурної формули для початкового розбиття x[0]
    solutions = pd.DataFrame(index=x[0])
    ist = 1
    for k in range(kiter):
        solutions[f'u^{k}'] = u[k][::ist]
        ist *= 2

    if type(etalon) == type(None):
        return u,x,solutions
    
    # запис у DataFrame еталонного розв'язку
    solutions['ux'] = etalon
    # обчислення абсолютних вiдхилень вiд еталонного розв'язку i запис у DataFrame
    errors = pd.DataFrame(index=x[0])
    for k in range(kiter):
        errors[f'e^{k}']=np.abs(solutions['ux'] -solutions[f'u^{k}'])
    # обчислення швидкостi збiжностi чисельних розв'язкiв до еталонного
    eoc = pd.DataFrame(index=x[0])
    for k in range(kiter-1):
        eoc[f'r^{k}'] = errors[f'e^{k}'] / errors[f'e^{k+1}']
    return u,x, solutions, errors, eoc


def trapezoidal_quadrature(a,b,m,n=1):
    """ обчислення вузлiв i коефiцiєнтiв
    великої квадратурної формули трапецiй
    [a,b] -- промiжок iнтегрування
    m -- кiлькiсть пiдiнтервалiв
    n==1 -- порядок квадратурної ф-ли
    """
    if n!= 1:
        raise Exception('invalid quadrature')
    total_nodes = m+1
    x=np.linspace(a, b, total_nodes)
    w=np.ones(total_nodes, dtype=float)
    w[0]=0.5
    w[total_nodes-1]=0.5
    return x,w


#-------------------------------------------------
"""
           1
u(x) - 1/2 ∫ (x + 1) * e^(-α*x*y) * u(y) dy = e^(-α*x)
           0
"""

def K(xi,y,**kwargs):
    alp = kwargs['alpha']
    return np.exp(-y*xi*alp)*(xi+1)

def f(x,**kwargs):
    alp = kwargs['alpha']
    return np.exp(-x*alp)*(1+np.exp(-alp)/(2*alp)) - 1/(2*alp)

def exact_solution(**kwargs):
    x = np.linspace(params['a'], params['b'], params['mstrt']+1)
    y = np.exp(-x*kwargs['alpha'])
    return pd.Series(y, index=x, dtype=float)

if __name__ == "__main__":
    params = {'a':0, 'b': 1, 'lmbd':0.5, 'alpha':5,'mstrt': 4, 'n': 1}

    kiter = 5
    exact = exact_solution(**params)
    ut,xt,solutions,errors,eoc = IE_solver(f,K,trapezoidal_quadrature,kiter,exact,**params)

    print("Таблиця розв'язків :")
    print(solutions)

    print("\nТаблиця абсолютних відхилень від еталонного розв'язку:")
    print(errors)

    print("\nТаблиця очікуваної швидкості збіжності чисельних розв'язків до еталонного:")
    print(eoc)
    for k in range(kiter):
        print(f"e1^{k} = {np.max(errors[f'e^{k}']):.2}", end=' ')


    solutions.plot(title="Розв'язки інтегрального рівняння в вузлах квадратурної формули", xlabel='x', ylabel='u(x)')
    errors.plot(title="Графік абсолютних відхилень від еталонного розв'язку", xlabel='x', logy=True)
    plt.show()