import numpy as np
from scipy import linalg

def linalg_solver(set_matrix, set_vector):

    a = set_matrix()
    b = set_vector()
    aT = a.T
    aTa = aT.dot(a)
    aTb = aT.dot(b)
    x = linalg.solve(aTa, aTb)
    return x

def set_matrix():
    matrix = np.array([
        [1,  1,  2],
        [3, -1, -1],
        [2,  3, -1],
        [1,  2,  3],
        [3, -2, -1]
    ], dtype=float)
    return matrix

def set_vector():
    vector = np.array([
        [-3],
        [-6],
        [ 0],
        [-3],
        [-7]
    ], dtype=float)
    return vector


try:
    x = linalg_solver(set_matrix, set_vector)
    print(f"Псевдорозв'язок СЛАР \n x={x.T}")
except Exception as e:
    print("Задана СЛАР має безлiч псевдорозв'язкiв, оскiльки отримано сингулярну матрицю")
