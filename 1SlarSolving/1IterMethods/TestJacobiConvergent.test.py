import unittest
import numpy as np

from Jacobi import is_Jacobi_convergent

class TestJacobiConvergence(unittest.TestCase):

    def test_convergent_matrix(self):
        # Матриця з суворим діагональним переважанням
        A = np.array([
            [10, 2, 1],
            [1, 5, 1],
            [2, 3, 10]
        ])
        self.assertTrue(is_Jacobi_convergent(A))

    def test_divergent_matrix(self):
        # Матриця, де діагональ замала (незбіжна)
        A = np.array([
            [1, 10, 1],
            [10, 1, 1],
            [1, 1, 1]
        ])
        self.assertFalse(is_Jacobi_convergent(A))

    def test_zero_diagonal(self):
        # Матриця з нулем на діагоналі
        A = np.array([
            [0, 1],
            [1, 2]
        ])
        self.assertFalse(is_Jacobi_convergent(A))

    def test_identity_matrix(self):
        # Одинична матриця завжди збіжна (B буде нульовою)
        A = np.eye(4)
        self.assertTrue(is_Jacobi_convergent(A))

if __name__ == '__main__':
    unittest.main()