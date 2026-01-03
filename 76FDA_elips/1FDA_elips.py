import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from mpl_toolkits import mplot3d

def FDA_E_solver(f,phi1,phi2,psi1,psi2,nxst,nyst,mx,my,kiter,etalon=None,**kwargs):
    """ Розв'язування задачi Дiрiхле для рiвняння Пуасона
    на прямокутнику методом скiнчених рiзниць
    """
    Nx = nxst
    Ny = nyst
    if Nx <= 4 or Ny <=4 :
        raise Exception('invalid grid')
    solutions = pd.DataFrame()
    stx = 1
    sty = 1
    for k in range(kiter):
        hx = kwargs['a']/Nx
        hy = kwargs['b']/Ny
        alpha = 1/(hx*hx)
        beta = 1/(hy*hy)
        c = set_matrix(alpha,beta,Nx,Ny)
        x=np.linspace(0, kwargs['a'], Nx+1)
        y=np.linspace(0, kwargs['b'], Ny+1)
        ps1 = psi1(x,**kwargs)
        ps2 = psi2(x,**kwargs)
        ph1 = phi1(y,**kwargs)
        ph2 = phi2(y,**kwargs)
        g = set_vector(f,ph1,ph2,ps1,ps2,x,y,alpha,beta,Nx,Ny,**kwargs)
        u = linalg.solve(c, g)
        uij=u.reshape(((Nx-1),(Ny-1)))
        solutions[f'u^{k}'] = uij[stx-1:Nx:stx,sty-1:Ny:sty].reshape((nxst-1)*(nyst-1))
       
        Nx *= mx
        Ny *= my
        stx *= mx
        sty *= my
    #формування матрицi значень розв'язку у всiх вузлах сiтки
    Nx //= mx
    Ny //= my
    ue = np.empty((Nx+1,Ny+1), dtype='float32')
    ue[0,:] = ph1
    ue[Nx,:]= ph2
    ue[:,Ny]= ps2
    ue[:,0] = ps1
    for i in range(1,Nx):
        for j in range(1,Ny):
            ue[i,j] = uij[i-1,j-1]

    if type(etalon) == type(None):
        return ue,solutions
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
    return ue, solutions, errors, eoc

def set_matrix(alpha,beta,Nx,Ny):
    """ функцiя задає матрицю СЛАР """
    N = (Nx-1)*(Ny-1)
    c=np.zeros((N,N), dtype='float64')
    #реалiзацiя В-блокiв
    c[0,0] = -2*(alpha+beta)
    c[0,1] = beta
    c[N-1,N-1] = -2*(alpha+beta)
    c[N-1,N-2] = beta
    for i in range(1,N-1):
        c[i,i-1] = beta
        c[i,i] = -2*(alpha+beta)
        c[i,i+1] = beta
    #реалiзацiя А-блокiв
    nstop = (Ny-1)*(Nx-2)
    for k in range(Ny-2, nstop, Ny-1):
        c[k,k+1] = 0
        c[k+1,k] = 0
    for k in range(nstop):
        c[k,k+Ny-1] = alpha
        c[k+Ny-1,k] = alpha
    return c

def set_vector(f, ph1, ph2, ps1, ps2, x, y, alpha,beta, Nx, Ny,**kwargs):
    """ функцiя задає вектор вiльних членiв СЛАР """
    N = (Nx-1)*(Ny-1)
    g=np.empty(N, dtype='float64')
    k=0
    for i in range(1,Nx):
        for j in range(1,Ny):
            g[k] = f(x[i],y[j],**kwargs)
            k += 1
    k=0
    g[k] -= beta*ps1[1]+alpha*ph1[1]
    k=Ny-2
    g[k] -= beta*ps2[1]+alpha*ph1[Ny-1]
    k=(Nx-2)*(Ny-1)
    g[k] -= beta*ps1[Nx-1]+alpha*ph2[1]
    k=N-1
    g[k] -= beta*ps2[Nx-1]+alpha*ph2[Ny-1]
    for j in range(2,Ny-1):
        k=j-1
        g[k] -= alpha*ph1[j]
        k=(Nx-2)*(Ny-1)+j-1
        g[k] -= alpha*ph2[j]

    for i in range(2,Nx-1):
        k=(i-1)*(Ny-1)
        g[k] -= beta*ps1[i]
        k=i*(Ny-1)-1
        g[k] -= beta*ps2[i]

    return g

def grid_tabulator(f,Nx,Ny,**kwargs):
    x=np.linspace(0, kwargs['a'], Nx+1)
    y=np.linspace(0, kwargs['b'], Ny+1)
    u = np.zeros((Nx-1)*(Ny-1), dtype=float)
    ji = 0
    for i in range(1,Nx):
        for j in range(1,Ny):
            u[ji] = f(x[i],y[j],**kwargs)
            ji += 1
    return u


def d3_plotter(u,a,b,Nx,Ny,ks=1):
    """ Побудова 3D-графiкiв функцiї u, заданої у вузлах сiтки (Nx+1)x(Ny+1)
    на прямокутнику зi сторонами a i b
    ks==1 -- побудова одного графiку
    ks==2 -- побудова двох графiкiв
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.title(f"Розв'язок початково-крайової задачi при Nx={Nx}, Ny={Ny}")
    y=np.linspace(0, b, Ny+1)
    x=np.linspace(0, a, Nx+1)
    Y,X = np.meshgrid(y,x)
    ax = fig.add_subplot(1, ks, 1, projection='3d')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    surf = ax.plot_surface(Y,X, u, rstride=1, cstride=1, cmap='viridis',linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    if ks==2:
        ax = fig.add_subplot(1, ks, 2, projection='3d')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        surf = ax.plot_wireframe(Y, X, u, rstride=5, cstride=5)
        plt.show()


def f_poisson(x, y, **kwargs):
    return 10 * x

def phi_left(y, **kwargs): # x = 0
    return np.zeros(y.size)

def phi_right(y, **kwargs): # x = a
    return 5 * y**2

def psi_bottom(x, **kwargs): # y = 0
    return np.zeros(x.size)

def psi_top(x, **kwargs): # y = b
    return 5 * x

def exact_sol(x, y, **kwargs):
    return 5 * x * y**2

# 2. Параметри розрахунку
params = {'a': 1.0, 'b': 1.0}
Nx_start = 10
Ny_start = 10
kiter = 4 # Кількість ітерацій подвоєння сітки
mx, my = 2, 2

# 3. Обчислення еталонного розв'язку для порівняння
etalon = grid_tabulator(exact_sol, Nx_start, Ny_start, **params)

# 4. Запуск розв'язувача
u_final, solutions, errors, eoc = FDA_E_solver(
    f_poisson, phi_left, phi_right, psi_bottom, psi_top, 
    Nx_start, Ny_start, mx, my, kiter, etalon, **params
)

# 5. Виведення результатів
print("Розв'язки на сітках з подвоєнням кількості вузлів:")
print(solutions.head(4))
print("Абсолютні похибки розв'язків:")
print(errors.head(4))
errors.plot(xlabel='номер внутрiшнього вузла(наскрiзна нумерацiя)',
title='Абсолютна похибка чисельних розв\'язкiв \n на спiльнiй множинi вузлiв 5х4 заcлогарифмiчною шкалою',
logy=True)  


print("Максимальні похибки на кожній ітерації:")
for k in range(kiter):
    print(f"e^{k} = {np.max(errors[f'e^{k}']):.2e}")

# 6. Візуалізація результату на останній (найгустішій) сітці
Nx_last = Nx_start * (mx**(kiter-1))
Ny_last = Ny_start * (my**(kiter-1))
d3_plotter(u_final, params['a'], params['b'], Nx_last, Ny_last, ks=2)