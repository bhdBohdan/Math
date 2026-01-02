import numpy as np

from Setup import set_matrix, set_vector

#МЕТОД ПРОСТОЇ ІТЕРАЦІЇ
def simple_iteration_solver(set_matrix, set_vector, eps):
    """ Керування виконанням методу простої iтерацiї при чисельному розв'язуваннi␣
    ,→СЛАР
    у випадку, коли за початкове наближення є вектор вiльних членiв"""
    b = set_matrix()
    d = set_vector()
    k_esterix, vector_norm = expected_number_of_iterations(b, d, eps)
    kmax = k_esterix + 1
    k_iteract = 0
    k_iteract, x = simple_iteration(b,d,d,eps,kmax,vector_norm)
    return vector_norm, k_iteract, x

def simple_iteration(b,d,x0,eps,kmax,norm):
    """ послiдовно обчислює наближення розв'язку СЛАР x=b*x+d
    методом простої iтерацiї до виконання принаймнi однiєї з умов:
    1) рiзниця двох послiдовних наближень у заданiй нормi norm є меншою або␣
    ,→рiвною заданому значенню eps
    2) кiлькiсть виконаних iтерацiй дорiвнює kmax
    x0 -- початкове наближення """

    x_prev = x0.copy()
    k = 1
    x_new = np.matmul(b,x0)+d

    while norm(x_new-x_prev) > eps and k < kmax:
        k += 1
        x_prev = x_new
        x_new=np.matmul(b,x_prev) + d

    if k == kmax :
        raise Exception(f'Методом iтерацiй точнiсть eps={eps} не досягнута за k={k},→iтерацiй',k)
    return k, x_new

def matrix_norm_calculator(matrix):
    nb1 = norm_1(matrix)
    nb2 = norm_2(matrix)
    nb_min = min(nb1,nb2)
    if nb_min >= 1 :
        raise Exception(f'Метод простої iтерацiї не застосовано,\n Норма,→матрицi={nb_min}',nb_min)
    if nb1 < nb2 :
        vector_norm = norm_1v
    else:
        vector_norm = norm_2v
    return nb_min, vector_norm

def expected_number_of_iterations(matrix, vector, eps):
    nb_min, vector_norm = matrix_norm_calculator(matrix)
    nd = vector_norm(vector)
    return int(np.log2(eps*(1-nb_min)/nd) / np.log2(nb_min) ), vector_norm

def norm_1v(a):
    """обчислення норми_1 вектора a"""
    return np.max(np.abs(a))
def norm_2v(a):
    """обчислення норми_2 вектора a"""
    return np.sum(np.abs(a))
def norm_1(a):
    """обчислення норми_1 матрицi a"""
    m = 0
    for i in range(a.shape[0]):
        s = np.sum(np.abs(a[i,:]))
        if s > m:
            m = s
    return m
def norm_2(a):
    """обчислення норми_2 матрицi a"""
    m=0
    for j in range(a.shape[1]):
        s = np.sum(np.abs(a[:,j]))
        if s > m:
            m = s
    return m

def print_vector_as_sequece(a):
    """друк вектора у виглядi послiдовностi чисел"""
    x=a.reshape(len(a))
    for i in range(len(a)):
        print(f"{x[i]:19.16}", end=' ')


for n in {5,7,9,11,13,15}:
    eps=10**(-n)
    try:
         rn,k, xk = simple_iteration_solver(set_matrix, set_vector, eps)
    except Exception as e:
        print(e.args[0])
    else :
        print(f"Чисельний розв'язок СЛАР \n x=[",end=' ' )
        print_vector_as_sequece(xk)
        print(f"]\n точнiсть eps={eps}, к-сть iтерацiй k={k}" )
