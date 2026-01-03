import numpy as np

from Setup import set_x0, set_matrix, set_vector
from SimpleIter import   print_vector_as_sequece, simple_iteration_solver 
from Jacobi import Jacobi_solver
from Seidel import Seidel_solver

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