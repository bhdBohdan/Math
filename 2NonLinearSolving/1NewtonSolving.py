import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 + 4*x**3 + 4.8*x**2 + 16*x + 1

def df(x):
    """Похідна функції f(x)"""
    return 4*x**3 + 12*x**2 + 9.6*x + 16

def ddf(x):
    """Друга похідна функції f(x)"""
    return 12*x**2 + 24*x + 9.6

def Newton_iteration_solver(f,first_derivative,second_derivative, a, b, x0, eps, n= 256):
    """ Пiдготовка даних для апостерiорної похибки i виклик функцiї,
    яка обчислює чисельний розв'язок рiвняння f(x)=0 методом Ньютона,
    де f -- задана на вiдрiзку [a, b] двiчi непервно-диференцiйована функцiя,
    first_derivative i second_derivative -- перша i друга похiднi заданої␣
    ,→функцiї
    x0 -- початкове наближення
    eps -- задана точнiсть
    n -- кiлькiсть точок на вiдрiзку [a, b]
    """
    if x0 < a or x0> b :
        raise Exception(f'Метод Ньютона не застосовано,\n x0={x0} поза [a,b]',x0)
    if f(x0) * second_derivative(x0) <= 0 :
        raise Exception(f'Метод Ньютона не застосовано,\n x0={x0} не задовольняє умову (3)',x0)
    x = np.linspace(a, b, n)
    m1 = np.min(np.abs( first_derivative(x)))
    m2 = np.max(np.abs( second_derivative(x)))
    e = np.sqrt(2 * m1 * eps / m2)
    return Newton_iteration(f,first_derivative, x0, e)

def Newton_iteration(f,f_deriv, x0, eps):
    """ Метод Ньютона для знаходження чисельного розв'язку рiвняння f(x)=0,
    де f -- задана двiчi непервно-диференцiйована функцiя на вiдрiзку [a, b],
    f_deriv -- похiдна заданої функцiї на вiдрiзку [a, b]
    x0 -- початкове наближення
    eps -- задана точнiсть
    """
    x_prev = x0
    k = 1
    x_new = x_prev - f(x_prev)/f_deriv(x_prev)
    if np.abs(x_new - x_prev) < eps:
        return x_new,k
    while np.abs(x_new - x_prev) > eps:
        k += 1
        x_prev = x_new
        x_new = x_prev - f(x_prev) / f_deriv(x_prev)
    return x_new, k

eps = 10**(-9)
a= -1
b = 0
x0 = 0
try:
    x, k = Newton_iteration_solver(f,df,ddf, a, b, x0, eps)
except Exception as e:
    print(e.args[0],e.args[1])
else:
    print(f"Розв'язок р-ня x={x} з точнiстю eps={eps}, к-сть iтерацiй k={k}")

flist = [f,df,ddf]
names = ["f(x)","f'(x)","f''(x)"]

def plot_graphics(functions, a, b, n):
    """ Побудова графiкiв функцiй на вiдрiзку [a, b]
    functions -- список функцiй для побудови
    a, b -- кiнцi вiдрiзку
    n -- кiлькiсть точок на вiдрiзку [a, b]
    """
    x = np.linspace(a, b, n)
    plt.figure(figsize=(10, 6))
    for i, func in enumerate(functions):
        y = func(x)
        plt.plot(x, y, label=names[i])
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title('Графіки функцій')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
    
plot_graphics(flist, a, b, 256)