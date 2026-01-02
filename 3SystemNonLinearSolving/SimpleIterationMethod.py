import matplotlib.pyplot as plt
import numpy as np

def simple_iteration(g, x0, eps, kmax = 1000):
    x_prev = x0.copy()
    k = 1
    x_new = g(x_prev)
    while norm_3(x_new-x_prev) > eps and k<kmax:
        k += 1
        x_prev = x_new
        x_new = g(x_prev)
    if k == kmax :
        raise Exception(f'Методом iтерацiй точнiсть eps={eps} не досягнута за k={k}→iтерацiй',k)
    return k, x_new

def D3_plotter(f, D, N0, N1, plotting="s012"):
    x0=np.linspace(D[0,0], D[0,1], N0+1)
    x1=np.linspace(D[1,0], D[1,1], N1+1)
    f0 = np.empty((N0+1,N1+1), dtype=float)
    f1 = np.empty((N0+1,N1+1), dtype=float)
    f2 = np.empty((N0+1,N1+1), dtype=float)
    for j in range(N1+1):
        for i in range(N0+1):
            f01 = f([x0[i],x1[j]])
            f0[j,i] = f01[0]
            f1[j,i] = f01[1]
            f2[j,i] = 0
    X1, X0 = np.meshgrid(x1, x0)
   # ============ f0, f1 i f2 ===============
    if plotting in ["s012", "sw012", "w012"]:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x1')
        ax.set_ylabel('x0')
        ax.set_zlabel('x2')
        ax.set_title(f"Графiки функцiй f0, f1 i f2")
        
        # Використовуємо 'in' або перевіряємо обидва варіанти
        if "s" in plotting: # Якщо в рядку є 's', малюємо поверхню
            ax.plot_surface(X1, X0, f0, cmap='viridis', alpha=0.6)
            ax.plot_surface(X1, X0, f1, cmap='magma', alpha=0.6)
            ax.plot_surface(X1, X0, f2, color='grey', alpha=0.3)
            
        if "w" in plotting: # Якщо в рядку є 'w', малюємо каркас
            ax.plot_wireframe(X1, X0, f0, rstride=5, cstride=5, color='orange')
            ax.plot_wireframe(X1, X0, f1, rstride=5, cstride=5, color='blue')
            
        plt.show()
    
def plot_intersection_graphics(f, D, N0, N1):
    offset0 = np.abs(D[0,0]-D[0,1])/N0/2
    offset1 = np.abs(D[1,0]-D[1,1])/N1/2
    X = np.linspace(D[0,0]-offset0, D[0,1]+offset0, N0)
    Y = np.linspace(D[1,0]-offset1, D[1,1]+offset1, N1)
    X, Y = np.meshgrid(X, Y)
    # Compute function values
    F0 = np.zeros_like(X)
    F1 = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f_val = f([X[i, j], Y[i, j]])
            F0[i, j], F1[i, j] = f_val[0], f_val[1]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f"Перетин з площиною x3=0 поверхонь, заданих ф-цiми f1 i f2")
    ax.contour(X, Y, F0, levels=[0], colors='orange')
    ax.contour(X, Y, F1, levels=[0], colors='blue')
    ax.grid(True)
    plt.show()

def domain(a,b,c,d):
    return np.array([[a,b],[c,d]])
def norm_3(a):
    return np.sqrt(np.sum(a**2))

def f(x):
    f0 = 4*x[0] - np.sin(x[1]) + 1
    f1 = np.cos(x[0]) - 2*x[1] + 3
    return np.array([f0,f1])
def g(x):
    g0 = (np.sin(x[1])-1)/4
    g1 = (np.cos(x[0])+3)/2
    return np.array([g0,g1])

D = domain(-0.25,0,1,2)
N0=10
N1=10

D3_plotter(f, D, N0, N1,plotting="sw012")
plot_intersection_graphics(f, D, N0, N1)
x0=np.array([-0.03, 2.0], dtype='float64')
try:
    for n in range(3,12,2):
        eps=10**(-n)
        k, xk = simple_iteration(g, x0, eps)
        print(f"Чисельний розв'язок с-ми р-нь x=[{xk[0]:18.14}, {xk[1]:19.16}]\n з точнiстю eps={eps}, к-сть iтерацiй k={k} ")
except Exception as e:
    print(e.args[0])