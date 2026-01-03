import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d

# --- Функції для розв'язання, надані користувачем ---

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

def set_matrix(p, Nt, Nx):
    c = np.full(Nx+1, -(p+2), dtype=float)
    c[0], c[Nx] = 1., 1.
    a = np.full(Nx+1, 1, dtype=float)
    a[0], a[Nx] = 0, 0
    b = np.full(Nx+1, 1, dtype=float)
    b[0], b[Nx] = 0, 0
    return c, a, b

def set_vector(g, uprev, p, mu, nu, t, ptaufj, **kwargs):
    for i in range(1, len(g)-1):
        g[i] = - p * uprev[i] - ptaufj[i]
    g[0] = mu(t, **kwargs)
    g[-1] = nu(t, **kwargs)

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

def d3_plotter(u, T, l, Nt, Nx, title="Розв'язок"):
    fig = plt.figure(figsize=(10, 7))
    t = np.linspace(0, T, Nt+1)
    x = np.linspace(0, l, Nx+1)
    X, T_grid = np.meshgrid(x, t)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T_grid, u, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.title(title)
    fig.colorbar(surf)
    plt.show()



# --- Параметри задачі 5) ---

def exact_solution(x, t, **kwargs):
    return (t**3) * np.exp(-3 * x)

def f(x, t, **kwargs):
    return 3 * (t**2) * np.exp(-3 * x) * (1 - 3 * t)

def phi(x, **kwargs):
    return np.zeros_like(x)

def mu(t, **kwargs):
    return t**3

def nu(t, **kwargs):
    return (t**3) * np.exp(-3)

if __name__ == "__main__":
    # Параметри з умови: a=1, область [0,1], час [0, 0.02]
    params = {'a': 1.0, 'l': 1.0, 'T': 0.02}
    
    # Кроки h = 0.1, tau = 0.005
    h, tau = 0.1, 0.005
    Nx_start = int(params['l'] / h)    # 10
    Nt_start = int(params['T'] / tau)  # 4
    
    # Виконуємо розрахунок (kiter=1, бо нам треба конкретні h та tau)
    u_expl, solutions_expl  = FDA_P_solver(f, mu, nu, phi, Nx_start, Nt_start, 1, 1, 1, **params)
    u_impl, solutions_impl  = FDA_Pim_solver(f, mu, nu, phi, Nx_start, Nt_start, 1, 1, 1, **params)
    
    print("Явна схема розв'язку:")
    print(solutions_expl)
    print("\nНеявна схема розв'язку:")
    print(solutions_impl)

    solutions_expl.plot(xlabel='номер внутрiшнього вузла(наскрiзна нумерацiя)',
    title='Розв\'язки чисельних схем \n на сiтцi 10x4', logy=False)  
    solutions_impl.plot(xlabel='номер внутрiшнього вузла(наскрiзна нумерацiя)',
    title='Розв\'язки чисельних схем \n на сiтцi 10x4', logy=False)

    # Побудова графіків
    d3_plotter(u_expl, params['T'], params['l'], Nt_start, Nx_start, "Явна схема (h=0.1, tau=0.005)")
    d3_plotter(u_impl, params['T'], params['l'], Nt_start, Nx_start, "Неявна схема (h=0.1, tau=0.005)")
    
    # Порівняння з точним розв'язком в останній момент часу
    x_coords = np.linspace(0, params['l'], Nx_start + 1)
    u_exact_final = exact_solution(x_coords, params['T'])
    
    print(f"{'x':<10} | {'Точний':<15} | {'Явна (Err)':<15} | {'Неявна (Err)':<15}")
    print("-" * 65)
    for i, x in enumerate(x_coords):
        err_e = abs(u_expl[-1, i] - u_exact_final[i])
        err_i = abs(u_impl[-1, i] - u_exact_final[i])
        print(f"{x:<10.2f} | {u_exact_final[i]:<15.2e} | {err_e:<15.2e} | {err_i:<15.2e}")


error_explicit = np.abs(u_expl[-1, :] - u_exact_final)
error_implicit = np.abs(u_impl[-1, :] - u_exact_final)
plt.figure(figsize=(10, 6))
plt.semilogy(x_coords, error_explicit, 'r-o', label='Похибка явної схеми', markersize=4)
plt.semilogy(x_coords, error_implicit, 'b-s', label='Похибка неявної схеми', markersize=4)
plt.xlabel('Координата x')
plt.ylabel('Абсолютна похибка $|u_{num} - u_{exact}|$')
plt.title(f'Порівняння похибок на момент часу T={params["T"]} (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5) # сітка для обох масштабів
plt.legend()
plt.show()