import matplotlib.pyplot as plt
import numpy as np

def norm_3(a):
    return np.sqrt(np.sum(a**2))

def simple_iteration(g, x0, eps, kmax=5000): # Збільшено kmax для повільної збіжності
    x_prev = x0.copy()
    k = 0
    while k < kmax:
        k += 1
        x_new = g(x_prev)
        if norm_3(x_new - x_prev) < eps:
            return k, x_new
        x_prev = x_new
    raise Exception(f'Точність {eps} не досягнута за {kmax} ітерацій. Процес розбігається.')

# Початкові функції для графіку
def f(x, y):
    f0 = x**2 * y**2 - 3*x**3 - 6*y**3 + 8
    f1 = x**4 - 9*y + 2
    return f0, f1

# СТАБІЛЬНІ ітераційні функції
def g(x_vec):
    x, y = x_vec[0], x_vec[1]
    # Виражаємо x з першого рівняння через 3x^3
    new_x = np.cbrt((x**2 * y**2 - 6*y**3 + 8) / 3)
    # Виражаємо y з другого рівняння
    new_y = (x**4 + 2) / 9
    return np.array([new_x, new_y])

def plot_intersection_graphics():
    x = np.linspace(0, 2, 500)
    y = np.linspace(0, 2, 500)
    X, Y = np.meshgrid(x, y)
    F0, F1 = f(X, Y)

    plt.figure(figsize=(8, 6))
    # Виправлення легенди для contour
    c1 = plt.contour(X, Y, F0, levels=[0], colors='orange')
    c2 = plt.contour(X, Y, F1, levels=[0], colors='blue')
    
    # Створення легенди вручну
    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    plt.legend([h1[0], h2[0]], ['f1(x,y)=0', 'f2(x,y)=0'])
    
    plt.title("Графічне визначення початкового наближення")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# 1. Графічний аналіз
plot_intersection_graphics()

# 2. Обчислення (початкова точка з графіку приблизно [1.3, 0.5])
x0 = np.array([1.3, 0.5], dtype='float64')

print(f"{'Точність':>10} | {'Ітерацій':>8} | {'Розв’язок x':>15} | {'Розв’язок y':>15}")
print("-" * 60)

for n in range(3, 12, 2):
    eps = 10**(-n)
    try:
        k, xk = simple_iteration(g, x0, eps)
        print(f"10^{(-n):<3} | {k:>8} | {xk[0]:>15.10f} | {xk[1]:>15.10f}")
    except Exception as e:
        print(f"eps=10^-{n}: {e}")