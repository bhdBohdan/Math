import numpy as np
from scipy import linalg

# from Setup import set_x0, set_matrix, set_vector
from SimpleIter import matrix_norm_calculator, print_vector_as_sequece

def set_matrix():
    """ функцiя для задання матрицi СЛАР"""
    matrix=np.array([[10, 1, 2],[1,5,-1],[1,-2,10]],dtype='float64' )
    return matrix
def set_vector():
    """ функцiя для задання вектора вiльних членiв СЛАР"""
    vector=np.array([[18], [8], [27]],dtype='float64')
    return vector
def set_x0():
    """ функцiя для задання вектора початкового наближення розв'язку"""
    return np.array([[1], [2], [1]],dtype='float64')


def Seidel_solver(set_matrix, set_vector,set_x0, eps):
    """ Задає алгоритм застосування методу Зейделя при чисельному розв'язуваннi СЛАР (2.125),
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
    for i in range(a.shape[0]):
        if a[i,i] == 0 :
            raise Exception(f'Метод Зейделя не застосовано,\n a[i,i]==0 при i={i}',i)
        b[i] /= a[i,i]
        a[i,:] /= -a[i,i]

    f = a.copy()
    h = a.copy()
    for i in range(a.shape[0]):
        f[i,i+1:] = 0
        h[i,:i+1] = 0
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

if __name__ == "__main__":
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