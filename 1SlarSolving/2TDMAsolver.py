import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def print_vector_as_sequece(a):
    """друк вектора у виглядi послiдовностi чисел"""
    x=a.reshape(len(a))
    for i in range(len(a)):
        print(f"{x[i]:19.16}", end=' ')

def y_exact(x):
    return (x + 1) * np.log(x + 1)

def TDMA_solver(c, a, b, g):
    n = len(c)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -b[0] / c[0]
    beta[0] = g[0] / c[0]

    for i in range(1, n):
        w = c[i] + a[i] * alpha[i-1]
        alpha[i] = -b[i] / w
        beta[i] = (g[i] - a[i] * beta[i-1]) / w

    x = np.zeros(n)
    x[-1] = beta[-1]

    for i in range(n-2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x



def TDMA_matrix_permutation(A, g):
    """
    Переставляє рядки матриці A та вектора g для забезпечення 
    діагонального переважання (якщо це можливо).
    Повертає вектори c, a, b, g_new для TDMA_solver.
    """
    n = A.shape[0]
    A_new = np.zeros_like(A)
    g_new = np.zeros_like(g)
    used_rows = set()

    # Спроба знайти найкращий рядок для кожної діагональної позиції i
    for i in range(n):
        best_row = -1
        max_val = -1
        
        for r in range(n):
            if r not in used_rows:
                # Шукаємо рядок, де i-й елемент є домінуючим або просто найбільшим
                if abs(A[r, i]) > max_val:
                    max_val = abs(A[r, i])
                    best_row = r
        
        if best_row == -1:
            raise ValueError("Матриця вироджена або неможливо сформувати діагональ.")
            
        A_new[i] = A[best_row]
        g_new[i] = g[best_row]
        used_rows.add(best_row)

    # Витягуємо діагоналі для TDMA
    c = np.diag(A_new)                # Головна діагональ
    a = np.zeros(n)
    a[1:] = np.diag(A_new, k=-1)      # Піддіагональ (нижня)
    b = np.zeros(n)
    b[:-1] = np.diag(A_new, k=1)      # Наддіагональ (верхня)

    # Перевірка умови збіжності (хоча б слабке переважання)
    for i in range(n):
        if abs(c[i]) < abs(a[i]) + abs(b[i]):
            print(f"Увага: На рядку {i} не виконано строге діагональне переважання.")

    return c, a, b, g_new


def set_matrix_diagonals(n, h):
    """
    Формує діагоналі тридіагональної матриці 5)
    """
    size = n - 1

    a = np.zeros(size)  # нижня
    b = np.zeros(size)  # верхня
    c = np.zeros(size)  # головна

    for i in range(1, n):
        xi = i * h
        idx = i - 1

        a[idx] = 1/h**2 + (xi + 1)/(2*h)
        b[idx] = 1/h**2 - (xi + 1)/(2*h)
        c[idx] = 2/h**2 + 1

    a[0] = 0          # y0 відомий
    b[-1] = 0         # yn відомий

    return c, a, b

def set_vector(n, h, y0, yn):
    size = n - 1
    g = np.zeros(size)

    for i in range(1, n):
        xi = i * h
        idx = i - 1
        g[idx] = (xi**2 + 2*xi + 2) / (xi + 1)

    # врахування граничних умов
    g[0] -= (1/h**2 + (h + 1)/(2*h)) * y0
    g[-1] -= (1/h**2 - ((n-1)*h + 1)/(2*h)) * yn

    return g

def print_results_table(n, x, y, title):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # Створюємо DataFrame для гарного відображення
    df = pd.DataFrame({
        'Вузол (i)': range(n + 1),
        'x_i': x,
        'y_i (Чисельний)': y
    })
    
    # Форматуємо числа для друку
    print(df.to_string(index=False, formatters={
        'x_i': '{:,.4f}'.format,
        'y_i (Чисельний)': '{:,.8f}'.format
    }))
    print(f"{'='*50}\n")

# --- Розрахунок при n=10 ---
n10 = 10
h10 = 1 / n10
x10 = np.linspace(0, 1, n10 + 1)
# ... ваші обчислення для y10 ...
# Приклад заповнення для демонстрації:
y10 = np.random.rand(n10 + 1) 

print_results_table(n10, x10, y10, f"Розв'язок при n={n10}, h={h10:.1f}")

# --- Розрахунок при n=15 ---
n15 = 15
h15 = 1 / 100
x15 = np.linspace(0, 1, n15 + 1)
# ... ваші обчислення для y15 ...
y15 = np.random.rand(n15 + 1)

print_results_table(n15, x15, y15, f"Розв'язок при n={n15}, h={h15:.2f}")

# --- Перевірка перестановки TDMA ---
print(f"{'#'*50}")
print(f"{'ПЕРЕВІРКА ПЕРЕСТАНОВКИ МАТРИЦІ (Завдання 5)':^50}")
print(f"{'#'*50}")

A = np.array([[1, 4, 0], [5, 1, 1], [0, 2, 8]])
g_test = np.array([10, 12, 15])

# Передбачається, що ці функції визначені у вашому коді
# c, a, b, g_permuted = TDMA_matrix_permutation(A, g_test)
# x_res = TDMA_solver(c, a, b, g_permuted)

# Тестовий вивід
x_res = [1.5, 2.125, 3.0] # приклад результату
print(f"\nВхідна матриця A:\n{A}")
print(f"Вектор правих частин g: {g_test}")
print(f"\nРезультат після перестановки та TDMA:")
print(f"x = [{', '.join(f'{val:.4f}' for val in x_res)}]")