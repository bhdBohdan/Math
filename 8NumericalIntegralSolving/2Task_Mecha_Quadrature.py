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




# --- Параметри з вашого завдання (зображення 2) ---

# Ядро: K(x, t) = tg(x / (5 + t))
def K(xi, t, **kwargs):
    return np.tan(xi / (5 + t))

# Права частина: f(x) = ln(1 + x)
def f(x, **kwargs):
    return np.log(1 + x)

if __name__ == "__main__":
    # lambda = 0.1, a = 0, b = 1
    params = {
        'a': 0, 
        'b': 1, 
        'lmbd': 0.1, 
        'mstrt': 10  # початкова кількість розбиттів
    }

    kiter = 4 # кількість подвоєнь сітки для перевірки збіжності
    
    # Викликаємо солвер без еталона (оскільки точний розв'язок невідомий)
    u_list, x_list, solutions = IE_solver(f, K, trapezoidal_quadrature, kiter, etalon=None, **params)

    print("Таблиця чисельних розв'язків на різних сітках (вузли x):")
    print(solutions)

    # Візуалізація результатів


    print("Таблиця розв'язків :")
    print(solutions)

    solutions.plot(title="Розв'язки інтегрального рівняння в вузлах квадратурної формули", xlabel='x', ylabel='u(x)', logy=True,
                   style=['k-', 'g--', 'b:', 'r-.'], linewidth=2, markersize=8)
    plt.show()
    
    
    # plt.title("Розв'язок інтегрального рівняння $u(x) = \ln(1+x) + 0.1 \int tg(x/(5+t))u(t)dt$")
    # plt.xlabel('x')
    # plt.ylabel('u(x)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()