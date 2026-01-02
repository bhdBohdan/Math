import numpy as np
from scipy import linalg

from Setup import set_x0, set_matrix, set_vector
from SimpleIter import matrix_norm_calculator, print_vector_as_sequece

def Seidel_solver(set_matrix, set_vector,set_x0, eps):
    """ Задає алгоритм застосування методу Зейделя при чисельному розв'язуваннi␣
    ,→СЛАР (2.125),
    як початкове наближення можна задати довiльний вектор"""
    a = set_matrix()
    d = set_vector()
    f,h = Seidel_Jacobi_modification(a, d)
    finv = linalg.inv(f)
    b = -finv.dot(h)
    dt = finv.dot(d)
    x0 = set_x0()
    x1 = np.matmul(b,x0) + dt
    x1x0 = x1-x0
    k_esterix, vector_norm = expected_number_of_iterations_2(b, x1x0, eps)
    kmax = k_esterix + 1
    k_iteract = 0
    k_iteract, x = Seidel_iteration(a,d,x0,eps,kmax,vector_norm)
    return vector_norm, k_iteract, x

def Seidel_Jacobi_modification(a, b):
    n = a.shape[0]
    # Робимо копії, щоб не псувати оригінали
    a_mod = a.copy()
    b_mod = b.copy()
    
    for i in range(n):
        if a_mod[i, i] == 0:
            raise Exception(f"Нуль на діагоналі в рядку {i}!")
        
        div = a_mod[i, i]
        b_mod[i] /= div
        a_mod[i, :] /= -div
        a_mod[i, i] = 0  # Тепер це матриця B
        
    # Розщеплення на нижню (f) та верхню (h) трикутні матриці
    f = np.tril(a_mod) 
    h = a_mod - f
    return f, h

def Seidel_iteration(b,d,x0,eps,kmax, norm):
    """ послiдовно обчислює наближення розв'язку СЛАР
    методом Зейделя до виконання принаймнi однiєї з умов:
    1) рiзниця двох послiдовних наближень у заданiй нормi norm
    є меншою заданої величини eps
    2) кiлькiсть виконаних iтерацiй дорiвнює kmax
    x0 -- початкове наближення """
    n = b.shape[0]
    x_prev = x0.copy()
    k = 1
    x_n = np.empty(n)
    x_n[0] = ( b[0,1:] * x_prev[1:] ).sum() + d[0]
    for i in range(1,n):
        x_n[i] = ( b[i,:i] * x_n[:i] ).sum() + ( b[i,i+1:] * x_prev[i+1:] ).sum() + d[i]
    while norm(x_n - x_prev) > eps and k < kmax:
        x_prev=x_n.copy()
        k+=1
        x_n[0] = ( b[0,1:] * x_prev[1:]).sum() + d[0]
        for i in range(1,n):
            x_n[i] = ( b[i,:i] * x_n[:i] ).sum() + ( b[i,i+1:] * x_prev[i+1:] ).sum() + d[i]
    if k == kmax :
        raise Exception(f'Методом Зейделя точнiсть eps={eps} не досягнута за k={k} iтерацiй',k)
    return k, x_n

def expected_number_of_iterations_2(matrix, vector, eps):
    nb_min, vector_norm = matrix_norm_calculator(matrix)
    nd = vector_norm(vector)
    return int(np.log2(eps*(1-nb_min)/nd) / np.log2(nb_min)) + 1, vector_norm


for n in {5,7,9,11,13,15}:
    eps=10**(-n)
    try:
        rn,k, xk = Seidel_solver(set_matrix, set_vector, set_x0, eps)
    except Exception as e:
        print(e.args[0])
    else :
        print(f"Чисельний розв'язок СЛАР \n x=[",end=' ' )
        print_vector_as_sequece(xk)
        print(f"]\n точнiсть eps={eps}, к-сть iтерацiй k={k}" )


# a = set_matrix()
# d = set_vector()
# f,h = Seidel_Jacobi_modification(a, d)
# finv = linalg.inv(f)
# b = -finv.dot(h)
# dt = finv.dot(d)
# x0 = set_x0()
# x1 = np.matmul(b,x0) + dt
# x1x0 = x1-x0
# for n in {3,5,7,9,11,13,15}:
#     eps=10**(-n)
#     ke, nrm = expected_number_of_iterations_2(b, x1x0, eps)
#     print(f"точнiсть eps={eps}, очiкувана к-сть iтерацiй ke={ke}" )