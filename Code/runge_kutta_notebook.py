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