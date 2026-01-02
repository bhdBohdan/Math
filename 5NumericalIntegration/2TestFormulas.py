import numpy as np
import time
import unittest

# --- Оптимізовані функції ---

def rectangle_formula(f, a, b, m):
    """Чисельне iнтегрування за великою формулою середнiх прямокутникiв"""
    h = (b - a) / m
    x = np.linspace(a + h/2, b - h/2, m)
    return h * f(x).sum()

def trapezoidal_formula(f, a, b, m):
    """Чисельне iнтегрування за великою формулою трапецiй"""
    h = (b - a) / m
    x = np.linspace(a, b, m + 1)
    y = f(x)
    return h * (0.5 * (y[0] + y[-1]) + y[1:-1].sum())

def simpson_formula(f, a, b, m):
    """Чисельне iнтегрування за великою формулою парабол (Сiмпсона)"""
    h = (b - a) / m
    x = np.linspace(a, b, 2 * m + 1)
    y = f(x)
    # Коефіцієнти: 1, 4, 2, 4, 2, ..., 4, 1
    # h/6 береться від кроку h = (b-a)/m
    res = (y[0] + y[-1] + 
           4 * y[1:-1:2].sum() + 
           2 * y[2:-2:2].sum())
    return (h / 6) * res


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x**2
        self.a = 0
        self.b = 1
        self.m = 1000
        self.expected = 1/3
        self.eps = 1e-6

    def test_rectangle(self):
        res = rectangle_formula(self.f, self.a, self.b, self.m)
        self.assertAlmostEqual(res, self.expected, delta=self.eps)

    def test_trapezoidal(self):
        res = trapezoidal_formula(self.f, self.a, self.b, self.m)
        self.assertAlmostEqual(res, self.expected, delta=self.eps)

    def test_simpson(self):
        res = simpson_formula(self.f, self.a, self.b, self.m)
        self.assertAlmostEqual(res, self.expected, delta=1e-12)

# Запуск тестів:
unittest.main(argv=[''], exit=False)

def benchmark():
    m_large = 1_000_000
    a, b = 0, 1.2
    func = lambda x: np.log(1 + x**2)
    
    print(f"Бенчмарк для m = {m_large}\n" + "-"*40)
    
    for name, method in [("Rect", rectangle_formula), 
                         ("Trap", trapezoidal_formula), 
                         ("Simp", simpson_formula)]:
        start = time.time()
        res = method(func, a, b, m_large)
        end = time.time()
        print(f"{name:<5}: {end-start:.6f} сек. | Результат: {res:.8f}")

benchmark()