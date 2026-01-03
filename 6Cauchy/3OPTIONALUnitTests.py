import unittest
import numpy as np

def Euler_method(f,u0,a,b,n):
    """ метод Ейлера """
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    u = np.empty(n+1, dtype='float64')
    u[0] = u0
    for i in range(1,n+1):
        u[i] = u[i-1] + h*f(x[i-1],u[i-1])
    return u

def RK4_method(f,u0,a,b,n):
    """ метод Рунге-Кутта четвертого порядку
    """
    h=(b-a)/n
    x=np.linspace(a, b, n+1)
    u=np.empty(n+1)
    u[0]=u0
    for i in range(n):
        k1 = f(x[i], u[i])
        k2 = f(x[i] + h/2, u[i] + h/2*k1)
        k3 = f(x[i] + h/2, u[i] + h/2*k2)
        k4 = f(x[i+1], u[i] + h*k3)
        u[i+1] =u [i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return u

def Euler_NS(f,g,u0,v0,a,b,n, **kwargs):
    """метод Ейлера для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    u = np.empty(n+1, dtype='float64')
    v = np.empty(n+1, dtype='float64')
    u[0] = u0
    v[0] = v0
    for i in range(n):
        u[i+1] = u[i] + h*f(x[i], u[i], v[i], **kwargs)
        v[i+1] = v[i] + h*g(x[i], u[i], v[i], **kwargs)
    return u, v

def RK4_NS(f,g,u0,v0,a,b,n, **kwargs):
    """ метод Рунге-Кутта четвертого порядку
    для задачi Кошi для системи двох ЗДР 1-го порядку
    """
    x=np.linspace(a, b, n+1)
    h=(b-a)/n
    u = np.empty(n+1, dtype='float64')
    v = np.empty(n+1, dtype='float64')
    u[0] = u0
    v[0] = v0
    for i in range(n):
        k1 = f(x[i], u[i], v[i], **kwargs)
        m1 = g(x[i], u[i], v[i], **kwargs)
        k2 = f(x[i], u[i] + h/2*k1, v[i] + h/2*m1, **kwargs)
        m2 = g(x[i], u[i] + h/2*k1, v[i] + h/2*m1, **kwargs)
        k3 = f(x[i], u[i] + h/2*k2, v[i] + h/2*m2, **kwargs)
        m3 = g(x[i], u[i] + h/2*k2, v[i] + h/2*m2, **kwargs)
        k4 = f(x[i], u[i] + h*k3, v[i] + h*m3, **kwargs)
        m4 = g(x[i], u[i] + h*k3, v[i] + h*m3, **kwargs)

        u[i+1]=u[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
        v[i+1]=v[i] + h/6*(m1 + 2*m2 + 2*m3 + m4)
    return u, v

class TestODEMethods(unittest.TestCase):

    def setUp(self):
        # Параметри для тестування одиничних рівнянь
        self.f_single = lambda x, u: u
        self.u0_single = 1.0
        self.a_single, self.b_single = 0, 1
        self.n_single = 1000
        self.exact_single = np.exp(1)  # u(1) = e^1

        # Параметри для тестування систем (NS)
        self.f_sys = lambda x, u, v: v
        self.g_sys = lambda x, u, v: -u
        self.u0_sys, self.v0_sys = 0.0, 1.0
        self.a_sys, self.b_sys = 0, np.pi / 2
        self.n_sys = 1000
        # Точний розв'язок при x = pi/2: u = sin(pi/2) = 1, v = cos(pi/2) = 0
        self.exact_u_sys = 1.0
        self.exact_v_sys = 0.0

    def test_euler_method_accuracy(self):
        """Перевірка точності методу Ейлера для одного рівняння"""
        res = Euler_method(self.f_single, self.u0_single, self.a_single, self.b_single, self.n_single)
        # Метод Ейлера має 1-й порядок точності, тому при n=1000 очікуємо похибку ~1e-3
        self.assertAlmostEqual(res[-1], self.exact_single, places=2)

    def test_rk4_method_accuracy(self):
        """Перевірка точності методу РК4 для одного рівняння"""
        res = RK4_method(self.f_single, self.u0_single, self.a_single, self.b_single, self.n_single)
        # РК4 має 4-й порядок точності, очікуємо дуже високу точність
        self.assertAlmostEqual(res[-1], self.exact_single, places=7)

    def test_euler_ns_accuracy(self):
        """Перевірка точності методу Ейлера для системи рівнянь"""
        u_res, v_res = Euler_NS(self.f_sys, self.g_sys, self.u0_sys, self.v0_sys, 
                               self.a_sys, self.b_sys, self.n_sys)
        self.assertAlmostEqual(u_res[-1], self.exact_u_sys, places=2)
        self.assertAlmostEqual(v_res[-1], self.exact_v_sys, places=2)

    def test_rk4_ns_accuracy(self):
        """Перевірка точності методу РК4 для системи рівнянь"""
        u_res, v_res = RK4_NS(self.f_sys, self.g_sys, self.u0_sys, self.v0_sys, 
                             self.a_sys, self.b_sys, self.n_sys)
        self.assertAlmostEqual(u_res[-1], self.exact_u_sys, places=7)
        self.assertAlmostEqual(v_res[-1], self.exact_v_sys, places=7)

    def test_output_shapes(self):
        """Перевірка, чи повертають функції масиви правильної довжини (n+1)"""
        n = 50
        res_single = Euler_method(self.f_single, self.u0_single, 0, 1, n)
        self.assertEqual(len(res_single), n + 1)
        
        u_ns, v_ns = Euler_NS(self.f_sys, self.g_sys, 0, 1, 0, 1, n)
        self.assertEqual(len(u_ns), n + 1)
        self.assertEqual(len(v_ns), n + 1)

if __name__ == '__main__':
    unittest.main()