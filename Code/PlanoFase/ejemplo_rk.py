import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import runge_kutta

# Definición de las funciones del sistema
# Ecuación de Rayleigh en forma cartesiana
def rayleigh_cartesiana(z):
    x = z[0]
    y = z[1]
    mu = 0.7  # Parámetro
    dx = y 
    dy = mu*(1-y**2)*y-x
    return np.array([dx, dy])

# Ecuación de Rayleigh en forma polar
def rayleigh_polar(z):
    r = z[0]
    theta = z[1]
    mu = 0.7  # Parámetro
    dr = mu/8*r*(4-r**2)
    dtheta = 1
    return np.array([dr, dtheta])

# Condiciones iniciales
y01 = np.array([0.01, -0.01])  # Para coordenadas cartesianas
y02 = np.array([0.1, 0])       # Para coordenadas polares

# Parámetros de integración
h = 0.1    # Tamaño del paso
n = 1000   # Número de pasos

# Resolver usando la función generalizada de Runge-Kutta
sol_cartesiana = runge_kutta(rayleigh_cartesiana, y01, h, n)
sol_polar = runge_kutta(rayleigh_polar, y02, h, n)

# Extraer las componentes de las soluciones
x1 = sol_cartesiana[0, :]
y1 = sol_cartesiana[1, :]

r = sol_polar[0, :]
theta = sol_polar[1, :]

# Convertir coordenadas polares a cartesianas para graficar
x2 = r * np.cos(theta)
y2 = r * np.sin(theta)

# Crear la gráfica
plt.figure(figsize=(10, 8))

# Plano fase para la solución en coordenadas cartesianas
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'b-', label='Solución cartesiana')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plano fase - Ecuación de Rayleigh (Coordenadas cartesianas)')
plt.legend()

# Plano fase para la solución en coordenadas polares (convertida a cartesianas)
plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r-', label='Solución polar')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plano fase - Ecuación de Rayleigh (Coordenadas polares)')
plt.legend()

plt.tight_layout()
plt.savefig('plano_fase_rayleigh.png')
plt.show()

print("La solución se ha calculado y graficado correctamente.") 