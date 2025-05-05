import numpy as np
import sympy as sp
import sympy as sp
from sympy import symbols, expand, Poly
import math
import ast
import matplotlib.pyplot as plt
###############################################################################
# Differential equations system
def f(z,f1,f2,parametros={},type = "cartesian"):
    """
    Function that calculates the derivatives of the system
    Parameters:
    - z = numpy array with the variables (x,y)
    - f1_original = expression that represents the derivative with respect to x
    - f2_original = expression that represents the derivative with respect to y

    Returns:
    - derivatives = numpy array with the derivatives
    """
    for key, value in parametros.items():
        exec(f"{key} = value")
    if type == "cartesian":
        x, y =z
        dx = eval(f1)
        dy = eval(f2)
        return np.array([dx,dy])
    elif type == "polar":
        r, theta = z
        dx = eval(f1)
        dy = eval(f2)
        return np.array([dx,dy])
###############################################################################
# RUNGE KUTTA 4° ORDER
def runge_kutta(f1, f2, y0, h, n,parametros={},type = "cartesian"):
    """
    Implementation of the fourth-order Runge-Kutta method for systems of differential equations.
    
    Parameters:
    - func: function that calculates the derivatives of the system
    - y0: initial conditions (numpy array)
    - h: step size
    - n: number of steps
    
    Returns:
    - y: matrix with the solution of the system
    """
    y = np.zeros((len(y0), n+1))
    y[:,0] = y0
    
    for k in range(n):
        k1 = f(y[:,k], f1, f2, parametros,type)
        k2 = f(y[:,k] + (h/2)*k1, f1, f2, parametros,type)
        k3 = f(y[:,k] + (h/2)*k2, f1, f2, parametros,type)
        k4 = f(y[:,k] + h*k3, f1, f2, parametros,type)
        y[:,k+1] = y[:,k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return y
###############################################################################
# JACOBIANO 
def calcular_jacobiano(f1, f2, vars):
    """
    Calculates the Jacobian matrix for a 2x2 system
    
    Parameters:
    f1, f2: Functions of the system (sympy expressions)
    vars: List of variables [x, y]
    
    Returns:
    Jacobian matrix 2x2 (sympy Matrix)
    """
    # Compute the partial derivatives
    df1x = sp.diff(f1, vars[0])
    df1y = sp.diff(f1, vars[1])
    df2x = sp.diff(f2, vars[0])
    df2y = sp.diff(f2, vars[1])
    
    return sp.Matrix([[df1x, df1y], 
                     [df2x, df2y]])
###############################################################################
# Polynomial Hilbert
def hilbert_system(A):
    D = ""
    x, y = symbols('x y')
    cont = 0
    # Dimensions of the matrix
    n = len(A)
    m = len(A[0])
    # Create the polynomial
    for i in range(n):
        for j in range(m):
            if A[i][j] != 0:
                cont += 1
                if i == 0 and j == 0:
                    D = str(A[i][j])
                else:
                    if A[i][j] > 0 and cont > 1:
                        D += "+"
                    D += str(A[i][j])    
                    if i > 0:
                        if i == 1:
                            D += "*x"
                        else:
                            D += "*x**" + str(i)
                    if j > 0:
                        if j == 1:
                            D += "*y"
                        else:
                            D += "*y**" + str(j)
    return D
###############################################################################
# Averaging
def average_system(A,B):
    n = max(len(A),len(B))
    m = max(len(A[0]),len(B[0]))
    R = np.zeros((n, m), dtype=float)
    for i in range(len(A)):
        for j in range(len(A[0])):
            # If i is even and j is odd
            if i % 2 == 0 and (j+1) % 2 == 0:
                R[i][j]=1/(2**(i+j+1)*math.factorial(int((i+j+1)/2)))
                R[i][j]=R[i][j]*(math.factorial(i)*math.factorial(j+1))/(math.factorial(int(i/2))*math.factorial(int((j+1)/2)))*A[i][j]
    for i in range(len(B)):
        for j in range(len(B[0])):
            # If i is odd and j is even
            if (i+1) % 2 == 0 and j % 2 == 0:
                R[i][j]=1/(2**(i+j+1)*math.factorial(int((i+j+1)/2)))
                R[i][j]=R[i][j]*(math.factorial(i+1)*math.factorial(j))/(math.factorial(int((i+1)/2))*math.factorial(int(j/2)))*B[i][j]    
    return R
###############################################################################
# Matrix to polynomial
def polinomio_a_matriz(polinomio_str):
    # Definir símbolos
    x, y = symbols('x y')
    
    # Convertir el string en una expresión simbólica
    expr = eval(polinomio_str)
    
    # Convertir la expresión en un polinomio de sympy
    poly = Poly(expr, x, y)
    
    # Obtener los términos (monomios) del polinomio
    terminos = poly.terms()
    
    # Encontrar los grados máximos de x e y
    max_grado_x = max(term[0][0] for term in terminos)
    max_grado_y = max(term[0][1] for term in terminos)
    
    # Crear una matriz de ceros con dimensiones adecuadas
    matriz_coeficientes = np.zeros((max_grado_x + 1, max_grado_y + 1), dtype=int)
    
    # Llenar la matriz con los coeficientes
    for term in terminos:
        grado_x, grado_y = term[0]
        coeficiente = term[1]
        matriz_coeficientes[grado_x, grado_y] = coeficiente
    
    return matriz_coeficientes
###############################################################################
# Polynomial coefficients averaging
def polynomial_averaging(R):
    D = ""
    r = symbols('r')
    cont = 0
    # Dimensions of the matrix
    n = len(R)
    m = len(R[0])
    R_transform = [0 for k in range(n+m-1)]
    for i in range(n):
        for j in range(m):
            R_transform[i+j] += R[i][j]
    # Create the polynomial
    p = len(R_transform)
    for i in range(p):
        if R_transform[i] != 0:
            cont += 1
            if i == 0:
                D = str(R_transform[i])
            elif i > 0:
                if R_transform[i] > 0 and cont > 1:
                    D += "+"
                D += str(R_transform[i])
                if i == 1:
                    D += "*r"
                else:
                    D += "*r**" + str(i)
    
    
    R_transform = R_transform[::-1]
    return D,R_transform
###############################################################################
# Graficar polinomio promedio
# Definimos una función general para graficar polinomios
# Función para graficar polinomios en una ventana separada
def graficar_polinomio(coeficientes, rango=(0, 2), raices=None, puntos=500, titulo="Gráfico del Polinomio promedio"):
    """
    Grafica un polinomio en una ventana separada.
    
    Parámetros:
    - coeficientes: lista o arreglo de coeficientes (de mayor a menor grado)
    - rango: tupla con el rango de valores de r a graficar
    - puntos: número de puntos a evaluar
    - titulo: título del gráfico
    """
    # Crear una nueva figura para esta gráfica
    plt.figure()  # Esto crea una nueva ventana separada
    
    # Crear el rango de valores
    r = np.linspace(rango[0], rango[1], puntos)
    
    # Evaluar el polinomio
    y = np.polyval(coeficientes, r)
    
    # Graficar
    # Marcar las raíces reales en la gráfica
    # Encontrar y mostrar las raíces del polinomio
    plt.scatter(raices, [0] * len(raices), color='red', label="Raíces")
    plt.plot(r, y, label=f"Polinomio {np.poly1d(coeficientes)}")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(titulo)
    plt.xlabel('r')
    plt.ylabel('f(r)')
    plt.grid(True)
    plt.ylim([np.min(y) - 0.1 * abs(np.min(y)), np.max(y) + 0.1 * abs(np.max(y))])
    plt.legend()
    
    # Mostrar la gráfica
    plt.show()
###############################################################################
# Función para guardar la gráfica cuando se presiona una tecla
def on_key_press(event):
    if event.key == 's':  # Guardar si se presiona 's'
        filename = "plano_fase.png"
        plt.savefig(filename, dpi=300)
        print(f"Gráfica guardada como {filename}")
###############################################################################