import numpy as np

def set_matrix():
    # Ваша вихідна матриця A
    A = np.array([
        [1, 5, 3, -4],
        [3, 1, -2, 0],
        [5, -7, 0, 10],
        [0, 3, -5, 0]
    ], dtype=float)
    
    # Перетворення: A_new = A.T @ A
    return A.T @ A
    return  A

def set_vector():
    A = np.array([
        [1, 5, 3, -4],
        [3, 1, -2, 0],
        [5, -7, 0, 10],
        [0, 3, -5, 0]
    ], dtype=float)
    b = np.array([5, 2, 8, -2], dtype=float)
    
    # Перетворення: b_new = A.T @ b
    return A.T @ b
    return b

def set_x0():
    """ функцiя для задання вектора початкового наближення розв'язку"""
    return np.array([[1], [1], [1], [1]],dtype='float64')


