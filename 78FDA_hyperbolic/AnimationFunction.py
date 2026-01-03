# Анімація розв'язку

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate_string(u, l, T, Nx, Nt):
    """
    Побудова анімації коливань струни
    u: матриця розв'язку (Nt+1, Nx+1)
    """
    x = np.linspace(0, l, Nx + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Налаштування осей
    ax.set_xlim(0, l)
    # Динамічне визначення меж для осі U
    ax.set_ylim(np.min(u) - 0.1, np.max(u) + 0.1)
    
    line, = ax.plot([], [], lw=2, color='blue')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title('Анімація коливань струни')
    ax.grid(True)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def update(frame):
        # frame - це індекс моменту часу від 0 до Nt
        line.set_data(x, u[frame, :])
        current_time = frame * (T / Nt)
        time_text.set_text(f'Час t = {current_time:.3f} с')
        return line, time_text

    # Створення анімації
    # interval - затримка між кадрами в мілісекундах
    ani = FuncAnimation(fig, update, frames=Nt + 1, init_func=init, 
                        blit=True, interval=100, repeat=True)

    plt.show()
    return ani
