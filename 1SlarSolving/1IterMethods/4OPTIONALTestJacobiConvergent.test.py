import unittest
import numpy as np


# До 4 . is_Jacobi_convergent що виконує перевiрку збiжностi iтерацiйного процесу Якобi для довiльної матрицi.

def is_Jacobi_convergent(A):
    """
    Перевіряє достатню умову збіжності методу Якобі:
    наявність діагонального переважання або ||B|| < 1.
    """
    n = A.shape[0]
    # Перевірка на нулі на діагоналі
    if any(np.diag(A) == 0):
        return False
    
    # Створюємо матрицю B (як у Jacobi_modification)
    B = np.zeros_like(A, dtype=float)
    for i in range(n):
        B[i, :] = -A[i, :] / A[i, i]
        B[i, i] = 0
        
    # Перевіряємо три основні матричні норми
    norm_inf = np.linalg.norm(B, ord=np.inf) # Макс. сума модулів елементів рядка
    norm_1 = np.linalg.norm(B, ord=1)     # Макс. сума модулів елементів стовпця
    norm_2 = np.linalg.norm(B, ord='fro') # Фробеніусова норма (як апроксимація спектральної)

    return norm_inf < 1 or norm_1 < 1 or norm_2 < 1

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