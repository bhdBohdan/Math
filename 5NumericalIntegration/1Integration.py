import numpy as np

def f(x):
    return np.log(1 + x**2)

def F_exact(x):
    return x * np.log(1 + x**2) - 2 * (x - np.arctan(x))

def midpoint_rectangle_rule(a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        result += f(a + h * (i + 0.5))
    return result * h

def trapezoid_rule(a, b, n):
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

def simpson_rule(a, b, n):
    if n % 2 != 0: n += 1  # Кількість кроків має бути парною
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 4 * f(a + i * h)
    return result * (h / 3)

def integrate_with_precision(method, a, b, eps, order):
    n = 2
    iterations = 1
    i_prev = method(a, b, n)
    
    while True:
        n *= 2
        iterations += 1
        i_curr = method(a, b, n)
        
        # Оцінка похибки за правилом Рунге
        error = abs(i_curr - i_prev) / (2**order - 1)
        if error < eps:
            return i_curr, n, iterations
        
        i_prev = i_curr
        if iterations > 20: # Запобіжник
            break
    return i_curr, n, iterations

# Параметри
a, b = 0, 1.2
eps = 1e-4
exact_val = F_exact(b) - F_exact(a)

methods = [
    ("Прямокутників (середніх)", midpoint_rectangle_rule, 2),
    ("Трапецій", trapezoid_rule, 2),
    ("Парабол (Сімпсона)", simpson_rule, 4)
]

print(f"Точне значення інтегралу: {exact_val:.10f}\n")
print(f"{'Метод':<25} | {'Значення':<12} | {'Крок (n)':<8} | {'Ітер.':<6} | {'Відхилення':<12}")
print("-" * 75)

for name, func, order in methods:
    val, n, it = integrate_with_precision(func, a, b, eps, order)
    diff = abs(val - exact_val)
    print(f"{name:<25} | {val:<12.8f} | {n:<8} | {it:<6} | {diff:<12.2e}")