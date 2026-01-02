import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 2*np.cos(x) - 4

def df(x):
    return 2*x - 2*np.sin(x)

def d2f(x):
    return 2 - 2*np.cos(x)

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


# Обчислення для двох коренів
# Для [2, 3] перевіримо умову Фур'є: f(3)*f''(3) > 0.
# f(3) > 0, f''(3) = 2 - 2*cos(3) > 0. Отже x0 = 3.

def plot_graphics(f, a, b, n):
    """функцiя для побудови графiкiв функцiй,
    отриманих через iтерабельну колекцiю f,
    за їхнiми значеннями в n точках на вiдрiзку [a,b]
    """
    x = np.linspace(a, b, n)
    fig = plt.figure()
    ax = fig.gca()
    for i in range(len(f)):
        y = f[i](x)
        ax.plot(x,y)
    ax.legend(range(len(f)))
    ax.axhline(color="grey", ls="--")
    plt.show()


eps = 10**(-9)

# Список для збереження знайдених коренів
roots = []

# Шукаємо перший корінь (додатний) на [2, 3]
try:
    x1, k1 = Newton_iteration_solver(f, df, d2f, 2, 3, 3.0, eps)
    roots.append((x1, k1))
except Exception as e:
    print(f"Помилка при пошуку 1-го кореня: {e.args[0]}")

# Шукаємо другий корінь (від'ємний) на [-3, -2]
try:
    x2, k2 = Newton_iteration_solver(f, df, d2f, -3, -2, -3.0, eps)
    roots.append((x2, k2))
except Exception as e:
    print(f"Помилка при пошуку 2-го кореня: {e.args[0]}")

# Виведення результатів
for i, (root, iterations) in enumerate(roots):
    print(f"Корінь {i+1}: x = {root:.10f}, ітерацій: {iterations}")

# Візуалізація
plot_graphics([f], -5, 5, 512)