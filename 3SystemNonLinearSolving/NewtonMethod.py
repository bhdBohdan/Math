import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd

from SimpleIterationMethod import norm_3

def Newton_iteration(f,invJ, x0, eps, kmax=1000):
    x_prev=x0.copy()
    k=1
    x_new = x_prev - invJ(x_prev).dot(f(x_prev))
    while norm_3(x_new-x_prev) > eps and k<kmax:
        k+=1
        x_prev = x_new
        x_new = x_prev - invJ(x_prev).dot(f(x_prev))
    if k == kmax :
        raise Exception(f'Методом iтерацiй точнiсть eps={eps} не досягнута за k={k}␣→iтерацiй',k)
    return k, x_new

def f(x):
    f0 = 4*x[0] - np.sin(x[1]) + 1
    f1 = np.cos(x[0]) - 2*x[1] + 3
    return np.array([f0,f1])
def inverse_Jacobian_matrix(x):
    df00 = 4
    df01 = - np.cos(x[1])
    df10 = - np.sin(x[0])
    df11 =-2
    invJ = linalg.inv(np.array([[df00, df01],[df10, df11]]))
    return invJ

x0=np.array([-0.03, 2.0], dtype='float64')

try:
    for n in range(3,12,2):
        eps=10**(-n)
        k, xk = Newton_iteration(f, inverse_Jacobian_matrix, x0, eps)
        print(f"Чисельний розв'язок с-ми р-нь x=[{xk[0]:20.17}, {xk[1]:20.17}]\n з␣точнiстю eps={eps}, к-сть iтерацiй k={k} ")
except Exception as e:
    print(e.args[0])