import unittest
import numpy as np
from scipy.interpolate import CubicSpline

# --- Функції для тестування ---

def lagrange_interpolation(x, x_n, y_n):
    z = 0
    for i in range(len(x_n)):
        p = 1
        for j in range(len(x_n)):
            if i != j:
                p *= (x - x_n[j]) / (x_n[i] - x_n[j])
        z += p * y_n[i]
    return z

def newton_coeffs(x_n, y_n):
    n = len(y_n)
    coef = np.zeros([n, n])
    coef[:, 0] = y_n
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x_n[i + j] - x_n[i])
    return coef[0, :]

def newton_interpolation(x, x_n, coef):
    n = len(coef)
    p = coef[n-1]
    for i in range(n - 2, -1, -1):
        p = coef[i] + (x - x_n[i]) * p
    return p

def spline3_interpolation(x_target, x_nodes, y_nodes):
    cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
    return cs(x_target)

# --- Клас тестів ---

class TestInterpolationMethods(unittest.TestCase):
    
    def setUp(self):
        """Ініціалізація даних перед кожним тестом"""
        self.x_nodes = np.array([3.2, 3.6, 5.8, 5.9, 6.2])
        self.y_nodes = np.array([5.3, 6.0, 2.4, -1.0, -3.2])
        self.eps = 1e-7 # Допустима похибка для float

    def test_lagrange_at_nodes(self):
        """Перевірка Лагранжа: значення в узлах мають збігатися з y_nodes"""
        for i in range(len(self.x_nodes)):
            res = lagrange_interpolation(self.x_nodes[i], self.x_nodes, self.y_nodes)
            self.assertAlmostEqual(res, self.y_nodes[i], delta=self.eps)

    def test_newton_at_nodes(self):
        """Перевірка Ньютона: значення в узлах мають збігатися з y_nodes"""
        coefs = newton_coeffs(self.x_nodes, self.y_nodes)
        for i in range(len(self.x_nodes)):
            res = newton_interpolation(self.x_nodes[i], self.x_nodes, coefs)
            self.assertAlmostEqual(res, self.y_nodes[i], delta=self.eps)

    def test_spline_at_nodes(self):
        """Перевірка Сплайна: значення в узлах мають збігатися з y_nodes"""
        for i in range(len(self.x_nodes)):
            res = spline3_interpolation(self.x_nodes[i], self.x_nodes, self.y_nodes)
            self.assertAlmostEqual(res, self.y_nodes[i], delta=self.eps)

    def test_linear_consistency(self):
        """Тест на лінійній функції: всі методи мають дати точний результат для прямої"""
        x_lin = np.array([0, 1, 2, 3])
        y_lin = np.array([0, 1, 2, 3]) # y = x
        x_test = 1.5
        
        # Лагранж
        self.assertAlmostEqual(lagrange_interpolation(x_test, x_lin, y_lin), 1.5)
        
        # Ньютон
        c_lin = newton_coeffs(x_lin, y_lin)
        self.assertAlmostEqual(newton_interpolation(x_test, x_lin, c_lin), 1.5)
        
        # Сплайн
        self.assertAlmostEqual(spline3_interpolation(x_test, x_lin, y_lin), 1.5)

if __name__ == '__main__':
    unittest.main()