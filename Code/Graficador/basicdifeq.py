import numpy as np
import sympy as sp
import sympy as sp
from sympy import symbols, expand, Poly
import math
###############################################################################
# Differential equations system
def f(z,f1,f2,parametros):
    """
    Función que calcula las derivadas del sistema
    Parámetros:
    - z = array de numpy con las variables (x,y)
    - f1_original = expresión que representa la derivada respecto a x
    - f2_original = expresión que representa la derivada respecto a y

    Retorna:
    - derivadas = array de numpy con las derivadas
    """
    for key, value in parametros.items():
        exec(f"{key} = value")
    x, y =z
    dx = eval(f1)
    dy = eval(f2)
    return np.array([dx,dy])
###############################################################################
# RUNGE KUTTA 4° ORDER
def runge_kutta(f1, f2, y0, h, n,parametros):
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
        k1 = f(y[:,k], f1, f2, parametros)
        k2 = f(y[:,k] + (h/2)*k1, f1, f2, parametros)
        k3 = f(y[:,k] + (h/2)*k2, f1, f2, parametros)
        k4 = f(y[:,k] + h*k3, f1, f2, parametros)
        y[:,k+1] = y[:,k] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return y
###############################################################################
# JACOBIANO
def calcular_jacobiano(f1, f2, vars):
    """
    Calcula la matriz Jacobiana para un sistema 2x2
    
    Parámetros:
    f1, f2: Funciones del sistema (sympy expressions)
    vars: Lista de variables [x, y]
    
    Retorna:
    Matriz Jacobiana 2x2 (sympy Matrix)
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
    n = len(A)
    m = len(A[0])
    R = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(n):
        for j in range(m):
                        # If i is even and j is odd
            if i % 2 == 0 and (j+1) % 2 == 0:
                R[i][j]=1/(2**(i+j+1)*math.factorial(int((i+j+1)/2)))
                R[i][j]=R[i][j]*(math.factorial(i)*math.factorial(j+1))/(math.factorial(int(i/2))*math.factorial(int((j+1)/2)))*A[i][j]
            # If i is odd and j is even
            elif (i+1) % 2 == 0 and j % 2 == 0:
                R[i][j]=1/(2**(i+j+1)*math.factorial(int((i+j+1)/2)))
                R[i][j]=R[i][j]*(math.factorial(i+1)*math.factorial(j))/(math.factorial(int((i+1)/2))*math.factorial(int(j/2)))*B[i][j]
            else:
                R[i][j]=0
    return R
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
    return D