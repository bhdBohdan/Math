import numpy as np

def set_matrix(n):
    return np.array([[1, 5, 3, -4],[3,1,-2, 0],[5, -7, 0, 10], [0, 3, -5, 0]],dtype=float ) # ВАРІАНТ 5

def set_vector(n):
    """ функцiя для задання вектора вiльних членiв конкретної СЛАР"""
    return np.array([5, 2, 8, -2],dtype=float)

def Gaussian_elimination(a,b):
    """ зведення матрицi a до верхньої трикутної
    та перетворення вектора b вiльних членiв методом Гаусса
    """
    n=b.size
    for k in range(1,n):
        for i in range(k,n):
            m=a[i,k-1]/a[k-1,k-1]
            for j in range(k,n):
                a[i,j]-= m * a[k-1,j]
            b[i]-= m * b[k-1]

def backward_substitution(a,b):
    n=b.size
    b[n-1]/=a[n-1,n-1]
    for k in range(1,n):
        for j in range(n-k,n):
            b[n-k-1]-=a[n-k-1,j]*b[j]
        b[n-k-1]/=a[n-k-1,n-k-1]

def GEM_solver(n, set_matrix, set_vector):
    """ Розв'язування СЛАР методом Гаусса
    """
    a=set_matrix(n)
    b=set_vector(n)
    Gaussian_elimination(a,b)
    backward_substitution(a,b)
    return b




n=3
x = GEM_solver(n, set_matrix, set_vector)
print(f'Чисельний розв\'язок СЛАР : \n x={x}')
