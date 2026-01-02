import numpy as np

from Setup import set_x0, set_vector, set_matrix
from SimpleIter import matrix_norm_calculator, print_vector_as_sequece, simple_iteration

# МЕТОД ЯКОБІ
def Jacobi_solver(set_matrix, set_vector,set_x0, eps):
    """ Задає алгоритм застосування методу Якобi при чисельному розв'язуваннi СЛАР,
    як початкове наближення можна задати довiльний вектор"""
    b = set_matrix()
    d = set_vector()
    Jacobi_modification(b, d)
    x0 = set_x0()
    x1 = np.matmul(b,x0) + d
    x1x0 = x1-x0
    k_esterix, vector_norm = expected_number_of_iterations_2(b, x1x0, eps)
    kmax = k_esterix + 1
    k_iteract = 0
    k_iteract, x=simple_iteration(b,d,x0,eps,kmax,vector_norm)
    return vector_norm, k_iteract, x

def Jacobi_modification(a, b):
    """ модифiкацiя матрицi a i вектора вiльних членiв b СЛАР згiдно формули␣
    ,→(2.144)"""
    for i in range(a.shape[0]):
        if a[i,i] == 0 :
            raise Exception(f'Метод Якобi не застосовано,\n a[i,i]==0 при i={i}',i)
        b[i] /= a[i,i]
        a[i,:] /= -a[i,i]
        a[i,i] = 0

def expected_number_of_iterations_2(matrix, vector, eps):
    nb_min, vector_norm = matrix_norm_calculator(matrix)
    nd = vector_norm(vector)
    return int(np.log2(eps*(1-nb_min)/nd) / np.log2(nb_min)) + 1, vector_norm


for n in {5,7,9,11,13,15}:
    eps=10**(-n)
    try:
        rn, k,xk = Jacobi_solver(set_matrix, set_vector, set_x0, eps)
    except Exception as e:
        print(e.args[0])
    else :
        print(f"Чисельний розв'язок СЛАР \n x=[",end=' ' )
        print_vector_as_sequece(xk)
        print(f"]\n точнiсть eps={eps}, к-сть iтерацiй k={k}" )



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