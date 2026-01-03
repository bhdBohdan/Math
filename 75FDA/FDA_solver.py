import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def f_func(x): return -np.sin(2 + x**2) - x * np.cos(x)
def p_func(x): return x
def q_func(x): return -1
def u_star(x): return x * np.cos(x)

def FDA_solver(f,q,nstrt,kiter,etalon=None,**kwargs):
    """ Розв'язування крайових задач для ЗДР
    методом скiнчених рiзниць
    """
    if nstrt <= 2:
        raise Exception('invalid grid')
    n = nstrt
    x = []
    u = []
    #обчислення розв'язкiв на кожнiй сiтцi xk
    for k in range(kiter):
        xk = np.linspace(kwargs['a'], kwargs['b'], n+1)
        x.append(xk)
        c,a,b,d = set_matrix_diagonals_and_vector(f,q,xk,n,**kwargs)
        u.append(tridiagonal_matrix_algorithm(a,b,c,d))
        n *= 2
    # запис у DataFrame значень усiх чисельних розв'язкiв
    # у вузлах початкової сiтки x[0][1::]
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
    errors = pd.DataFrame(index=x[0][1:nstrt:])
    for k in range(kiter):
        errors[f'e^{k}']=np.abs(solutions['ux'].iloc[1:nstrt:]-solutions[f'u^{k}'].loc[1:nstrt:])
    # обчислення очiкуваної швидкостi збiжностi чисельних розв'язкiв до еталонного
    eoc = pd.DataFrame(index=x[0][1:nstrt:])
    for k in range(kiter-1):
        eoc[f'r^{k}'] = errors[f'e^{k}'] / errors[f'e^{k+1}']
    return u,x, solutions, errors, eoc

def set_matrix_diagonals_and_vector(f,q,x,n,**kwargs):
    """ функцiя задає 3 дiагоналi матрицi i вектор вiльних членiв СЛАР """
    h = np.abs(x[1]-x[0])
    h2 = h*h
    c = np.empty(n+1, dtype='float64' )

    for i in range(1,n):
        c[i] = h2*q(x[i])-2

    c[0] = -kwargs['alpha0'] + h*kwargs['beta0']
    c[n] = kwargs['alpha1'] + h*kwargs['beta1']
    a = np.ones(n+1, dtype='float64')
    a[0] = 0
    a[n] = -kwargs['alpha1']
    b = np.ones(n+1, dtype='float64')
    b[0] = kwargs['alpha0']
    b[n] = 0
    d = np.empty(n+1, dtype='float64' )
    for i in range(1,n):
        d[i] = h2*f(x[i])
    d[0] = h*kwargs['gamma0']
    d[n] = h*kwargs['gamma1']

    return c,a,b,d

def tridiagonal_matrix_algorithm(a,b,c,g):
    """ метод прогонки для розв'язування СЛАР
    з 3-дiагональною матрицею
    вектор с-головна дiагональ
    вектори a i b - нижня i верхня дiагоналi, паралельнi головнiй
    вектор g - вiльнi члени
    """
    n1=c.size
    alpha = np.empty(n1, dtype=float )
    beta = np.empty(n1, dtype=float )
    if c[0] !=0 :
        alpha[0] =-b[0]/c[0]
        beta [0] = g[0]/c[0]
    else:
        raise Exception('c[0]==0')
    for i in range(1,n1):
        w=a[i]*alpha[i-1]+c[i]
        if w != 0 :
            alpha[i] =-b[i]/w
            beta[i] = (g[i] - a[i]*beta[i-1])/w
        else:
            raise Exception('w==0')
        
    x = np.empty(n1, dtype=float )
    n = n1-1
    x[n] = beta[n]
    for i in range(n-1,-1,-1):
        x[i] = alpha[i]*x[i+1] + beta[i]
    return x

def main():
    def f1(x,**params):
        return 1
    def q1(x,**params):
        return 1

    params = {'a':0, 'b': np.pi/2,
    'alpha0':0, 'alpha1':1,
    'beta0': 1, 'beta1': 0,
    'gamma0':0, 'gamma1':1}

    def exact_solution(n,**kwargs):
        x = np.linspace(kwargs['a'], kwargs['b'], nstrt+1)
        y = 1 - np.cos(x)
        return pd.Series(y, index=x, dtype='float64')

    nstrt = 5
    etalon = exact_solution(nstrt,**params)
    kiter = 5
    u,x,solutions,errors,eoc = FDA_solver(f1,q1,nstrt,kiter,etalon,**params)
    print("Розв'язки на сітках з подвоєнням кількості вузлів:")
    print(solutions.head(5)
    )

    solutions.plot(xlabel='x', title='Розв\'язки задачi Кошi у вузлах початкової сiтки')
    plt.show()

    print("Абсолютні похибки розв'язків:")
    print(errors.head(5))
    #errors.plot(xlabel='x', title='Абсолютні похибки розв\'язків у вузлах початкової сiтки')
    errors.plot(xlabel='x',
    title='Похибки чисельних розв\'язкiв крайової задачi у внутрiшнiх \n вузлах, початкової сiтки (за логарифмiчною шкалою)',logy=True)
    plt.show()  

    for k in range(kiter):
        print(f"e1^{k} = {np.max(errors[f'e^{k}']):.2}", end=' ')

if __name__ == "__main__":
    main()