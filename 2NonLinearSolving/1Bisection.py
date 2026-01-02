from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return x**4 + 4*x**3 + 4.8*x**2 + 16*x + 1

def bisection(f,a,b,eps):
    """ Метод бiсекцiї для знаходження чисельного розв'язку рiвняння f(x) = 0,
    локалiзованого на iнтервалi (a,b),
    f -- непервна на вiдрiзку [a,b] функцiя,
    eps -- задана точнiсть
    """
    k = 0
    ba = np.abs(b-a)
    if ba < eps:
        return (a+b)/2, k+1
    while ba > eps:
        fa = f(a)
        k += 1
        x = (a+b)/2
        fx = f(x)
        if fx==0 :
            return x, k
        if fa*fx < 0:
            b = x
        else:
            a = x
        ba = np.abs(b-a)
    return (a+b)/2, k+1

a=-1
b=0
for n in [ 5, 7, 9, 11]:
    eps=10**(-n)
    x, k = bisection(f, a, b, eps)
    print(f"Розв'язок р-ня x={x} з точнiстю eps={eps}, к-сть iтерацiй k={k}")

def plot_graphics(function, a, b, n, sol=None):
    """ Побудова графiкiв функцiй на вiдрiзку [a, b]
    functions -- список функцiй для побудови
    a, b -- кiнцi вiдрiзку
    n -- кiлькiсть точок на вiдрiзку [a, b]
    """
    x = np.linspace(a, b, n)
    plt.figure(figsize=(10, 6))
 
    y = function(x)
    plt.plot(x, y, label="x**4 + 4*x**3 + 4.8*x**2 + 16*x + 1")
    plt.scatter(sol, function(sol), color='red', zorder=5, label='Знайдений корінь' if sol is not None else '')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title('Графіки функцій')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
    
plot_graphics(f, a, b, 256, x )
