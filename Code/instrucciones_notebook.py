"""
INSTRUCCIONES PARA USAR LA FUNCIÓN GENERALIZADA DE RUNGE-KUTTA

A continuación se muestra cómo usar la función generalizada de Runge-Kutta que recibe como argumento
la función a evaluar. Puedes copiar y pegar estas secciones en tu notebook.

1. Primero, crea un archivo llamado 'runge_kutta_notebook.py' con el siguiente contenido:

```python
import numpy as np

def runge_kutta(func, y0, h, n):
    """
    Implementación del método de Runge-Kutta de cuarto orden para sistemas de ecuaciones diferenciales.
    
    Parámetros:
    - func: función que calcula las derivadas del sistema
    - y0: condiciones iniciales (array de numpy)
    - h: tamaño del paso
    - n: número de pasos
    
    Retorna:
    - y: matriz con la solución del sistema
    """
    y = np.zeros((len(y0), n+1))
    y[:,0] = y0
    
    for k in range(n):
        k1 = func(y[:,k])
        k2 = func(y[:,k] + (h/2)*k1)
        k3 = func(y[:,k] + (h/2)*k2)
        k4 = func(y[:,k] + h*k3)
        y[:,k+1] = y[:,k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return y
```

2. Luego, en tu notebook, importa la función:

```python
import numpy as np
import matplotlib.pyplot as plt
from runge_kutta_notebook import runge_kutta
```

3. Define las funciones que representan el sistema de ecuaciones diferenciales:

```python
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
```

4. Resuelve el sistema usando la función generalizada de Runge-Kutta:

```python
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
```

5. Visualiza los resultados:

```python
# Crear la gráfica
plt.figure(figsize=(12, 10))

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
plt.show()
```

6. Si quieres comparar con las funciones originales:

```python
# Cartesianas
def rk4(y0, h, n):
  y = np.zeros((2, n+1))
  y[:,0] = y0
  for k in range(n):
    k1 = rayleigh_cartesiana(y[:,k])
    k2 = rayleigh_cartesiana(y[:,k]+(h/2)*k1)
    k3 = rayleigh_cartesiana(y[:,k]+(h/2)*k2)
    k4 = rayleigh_cartesiana(y[:,k]+h*k3)
    y[:,k+1] = y[:,k]+(h/6)*(k1+2*k2+2*k3+k4)
  return y

# Polares
def rk4polar(y0, h, n):
  y = np.zeros((2, n+1))
  y[:,0] = y0
  for k in range(n):
    k1 = rayleigh_polar(y[:,k])
    k2 = rayleigh_polar(y[:,k]+(h/2)*k1)
    k3 = rayleigh_polar(y[:,k]+(h/2)*k2)
    k4 = rayleigh_polar(y[:,k]+h*k3)
    y[:,k+1] = y[:,k]+(h/6)*(k1+2*k2+2*k3+k4)
  return y

# Resolver usando las funciones originales
sol_cart_orig = rk4(y01, h, n)
sol_polar_orig = rk4polar(y02, h, n)

# Extraer las componentes de las soluciones
x1_orig = sol_cart_orig[0, :]
y1_orig = sol_cart_orig[1, :]

r_orig = sol_polar_orig[0, :]
theta_orig = sol_polar_orig[1, :]

# Convertir coordenadas polares a cartesianas para graficar
x2_orig = r_orig * np.cos(theta_orig)
y2_orig = r_orig * np.sin(theta_orig)

# Comparar los resultados
plt.figure(figsize=(12, 10))

# Comparación de soluciones cartesianas
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'b-', label='Función generalizada')
plt.plot(x1_orig, y1_orig, 'g--', label='Función original')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de soluciones cartesianas')
plt.legend()

# Comparación de soluciones polares
plt.subplot(2, 1, 2)
plt.plot(x2, y2, 'r-', label='Función generalizada')
plt.plot(x2_orig, y2_orig, 'm--', label='Función original')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de soluciones polares')
plt.legend()

plt.tight_layout()
plt.show()
```

Como puedes ver, la función generalizada de Runge-Kutta produce los mismos resultados que las funciones originales, 
pero con la ventaja de que puedes usar la misma función para cualquier sistema de ecuaciones diferenciales, 
simplemente pasando la función que define el sistema como argumento.
""" 