from lector import Lector
import numpy as np
from copy import copy


np.set_printoptions(suppress=True)

datos = Lector("datos.txt")

class No_Acotado(Exception):
    pass

class No_Solucion(Exception):
    pass
class Simplex:
    def __init__(self):
        # Inicializamos las variables para evitar errores
        self.c = None
        self.A = None
        self.b = None
        
        # Dimensiones de la matriz A
        self.m = None
        self.n = None
        
        self.iter_m = None
        self.iter_n = None

        # Variable de la iteración
        self.iteraciones = 0
        
    def optimizar(self, problema):
        self.c = problema.c # Coeficientes de la función objetivo
        self.A = problema.A # Coeficientes de las restricciones
        self.b = problema.b # Termino independiente de las restricciones
        self.m = self.A.shape[0] # Número de restricciones
        self.n = self.A.shape[1] # Número de variables
        
        print('Inicio Simplex Primal \n') 
        # Fase I
        self.fase1()
        if np.round(self.iter_z, 5) > 0: #Nos indica que el poblema no tiene solución puesto que hay una variable artificial en la base al final de la Fase I
            raise No_Solucion(f"El problema no tiene solución, iteraciones realizadas: {self.iteraciones} \n")
        print(f'Solución básica factible encontrada, iteración {self.iteraciones} \n')
        # Fase 2
        self.fase2()

        # Final
        self.end()
        
    def generar_PA(self):
        self.iter_c = np.array([0] * self.n + [1] * self.m)
        self.iter_A = np.concatenate((self.A, np.identity(self.m)), axis=1)
        self.iter_b = self.b
        
        self.iter_Beta = np.arange(self.n, self.n + self.m) # Variables basicas
        self.iter_N = np.arange(self.n) # Variables no basicas
        
        self.iter_m = self.iter_A.shape[0] # Número de restricciones
        self.iter_n = self.iter_A.shape[1] # Número de variables
        
        self.iter_B = np.identity(self.iter_m) # Matriz de coeficientes de restricciones basicas
        self.iter_An = self.iter_A[:, self.iter_N] # Matriz de coeficientes de restricciones no basicas
        
        self.iter_cB = self.iter_c[self.iter_Beta] # Coeficientes de las variables basicas en la funcion objetivo
        self.iter_cN = self.iter_c[self.iter_N] # Coeficientes de las variables no basicas en la funcion objetivo
        
        self.iter_B_inv = self.iter_B # Inversa de la matriz de coeficientes de restricciones basicas (inicialmente es la identidad)
        
        self.iter_xB = self.iter_B_inv @ self.iter_b # Valores de las variables basicas
        self.iter_xN = np.array([0] * self.iter_n) # Valores de las variables no basicas
        
        self.iter_z = self.iter_cB @ self.iter_xB # Valor de la funcion objetivo

    def generar_sol(self):
        self.iter_c = np.array(self.c)
        self.iter_A = np.array(self.A)
        self.iter_b = self.b
        
        self.iter_Beta = self.iter_Beta # Variables basicas
        self.iter_N = np.setdiff1d(np.arange(self.n), self.iter_Beta) # Variables no basicas
        
        self.iter_m = self.iter_A.shape[0] # Número de restricciones
        self.iter_n = self.iter_A.shape[1] # Número de variables
        
        self.iter_B = self.iter_B # Matriz de coeficientes de restricciones basicas
        self.iter_An = self.iter_A[:, self.iter_N] # Matriz de coeficientes de restricciones no basicas
        
        self.iter_cB = self.iter_c[self.iter_Beta] # Coeficientes de las variables basicas en la funcion objetivo
        self.iter_cN = self.iter_c[self.iter_N] # Coeficientes de las variables no basicas en la funcion objetivo
        
        self.iter_B_inv = self.iter_B_inv # Inversa de la matriz de coeficientes de restricciones basicas (inicialmente es la identidad)
        
        self.iter_xB = self.iter_B_inv @ self.iter_b # Valores de las variables basicas
        self.iter_xN = np.array([0] * self.iter_n) # Valores de las variables no basicas
        
        self.iter_z = self.iter_cB @ self.iter_xB # Valor de la funcion objetivo
    
    def actualizar_inversa(self):
        
        d = self.iter_dB # Dirección básica de la variable que entra 
        
        d_safe = copy(d[self.iter_var_salida_indice]) # Guardamos el valor de la variable que sale ya que cambiaremos todos los valores de la dirección
        
        d /= -d[self.iter_var_salida_indice] # actualizamos todos los valores de d (p = i incluido)
        
        d[self.iter_var_salida_indice] = -1 / d_safe # Actualizamos la dirección solo para la variable que sale (p = i), utilizando el valor guardado
        
        E = np.identity(self.iter_m) 
        E[:, self.iter_var_salida_indice] = d # Actualizamos la columna correspondiente a la variable que sale
        
        self.iter_B_inv = E @ self.iter_B_inv # Actualizamos la inversa de la matriz de coeficientes de restricciones básicas
        
    def bland_argmin(self, array, entrada = True):
        min_valor = float('inf') # Cota superior para el valor mínimo
        min_indice_array = [] # Array de posiciones dentro de la array dada para los valores minimos
        min_indice = int # Índice menor de min_indice_array
        min_name = int # Nombre de la variable que entra o sale
        
        for i in range(len(array)):
            if array[i] < min_valor:
                min_valor = array[i]
                min_indice_array = [i]
            elif array[i] == min_valor:
                min_indice_array.append(i)
        
        if len(min_indice_array) == 1:
            min_indice = min_indice_array[0]
            
        elif entrada:
            min_name = self.iter_N[[min_indice_array[0]]]
            min_indice = min_indice_array[0]
            for i in min_indice_array:
                if self.iter_N[i] < min_name:
                    min_name = self.iter_N[i]
                    min_indice = min_indice_array[i]
        else:
            min_name = self.iter_Beta[[min_indice_array[0]]] 
            min_indice = min_indice_array[0]
            for i in min_indice_array:
                if self.iter_Beta[i] < min_name:
                    min_name = self.iter_Beta[i]
                    min_indice = min_indice_array[i]
        return min_indice
        

    def iter(self):
        self.iter_r = self.iter_cN - self.iter_cB @ self.iter_B_inv @ self.iter_An # Coeficientes reducidos

        while np.any(self.iter_r < 0):
            self.iter_dB = - self.iter_B_inv @ self.iter_An[:, np.argmin(self.iter_r)] # Direccion basica factible
            self.iter_theta_array = np.array([- self.iter_xB[i] / self.iter_dB[i] if self.iter_dB[i] < 0 else np.inf for i in range(self.iter_m)]) # Calculamos la lista de thetas posibles
            self.iter_theta = np.min(self.iter_theta_array) # Elegimos el theta minimo
                    
            # Identificar la no acotación del problema
            if np.all(self.iter_dB >= 0): # Problemas con la no acotación
                raise No_Acotado("El problema no tiene solución acotada")
                    
            self.iter_var_entrada_indice = self.bland_argmin(self.iter_r) # Recuperamos el indice en la matriz pertinente de la variable que entra (no el sub-indice de la variable)
            self.iter_var_entrada = self.iter_N[self.iter_var_entrada_indice] # Recuperamos la variable que entra
            self.iter_var_salida_indice = self.bland_argmin(self.iter_theta_array, entrada = False) # Recuperamos el indice en la matriz pertinente de la variable que sale (no el sub-indice de la variable)
            self.iter_var_salida = self.iter_Beta[self.iter_var_salida_indice] # Elegimos la variable que sale
            
            self.iter_Beta[np.where(self.iter_Beta == self.iter_var_salida)], self.iter_N[np.where(self.iter_N == self.iter_var_entrada)] = self.iter_var_entrada, self.iter_var_salida # Actualizamos las variables basicas y no basicas

            self.iter_xB += self.iter_theta * self.iter_dB # Actualizamos el valor de las variables basicas
            self.iter_xB[self.iter_var_salida_indice] = self.iter_theta # Actualizamos el valor de la variable que entra
            
            if np.any(self.iter_xB < 0): # Si xB es negativo, el problema no tiene solucio
                raise No_Solucion("El problema no tiene solución")
            
            # No cambiamos xN ya que en todo caso será un vector de n - m ceros
            
            self.iter_B = self.iter_A[:, self.iter_Beta] # Actualizamos la matriz de coeficientes de restricciones basicas
            self.iter_An = self.iter_A[:, self.iter_N] # Actualizamos la matriz de coeficientes de restricciones no basicas
            
            self.actualizar_inversa()
            
            self.iter_cB = self.iter_c[self.iter_Beta] # Coeficientes de las variables basicas en la funcion objetivo
            self.iter_cN = self.iter_c[self.iter_N] # Coeficientes de las variables no basicas en la funcion objetivo
            
            self.iter_z += self.iter_theta * np.min(self.iter_r) # Valor de la variable objetivo
            self.iter_r = self.iter_cN - self.iter_cB @ self.iter_B_inv @ self.iter_An # recalculamos los coeficientes reducidos

            self.iteraciones += 1

            print(f'Iteración {self.iteraciones}:, q = {self.iter_var_salida + 1}, B(p) = {self.iter_var_entrada + 1} , theta* = {self.iter_theta} , z = {self.iter_z}') # Sumamos 1 para que las variables empiecen en 1
        
    def fase1(self):
        print('Inicio FASE I ')
        self.generar_PA()
        self.iter()
        
    
    def fase2(self):
        print('Inicio FASE II ')
        self.generar_sol()
        self.iter()
        print(f'Solución óptima encontrada, iteración {self.iteraciones}, z = {self.iter_z} ')
        
    def end(self):
        print('FIN Simplex primal \n')
        print('Solución óptima: ')
        print(f'vb = {self.iter_Beta + 1}') # Sumamos 1 para que las variables empiecen en 1
        print(f'xb = {np.round(self.iter_xB, 4)}')
        print(f'z = {np.round(self.iter_z, 4)}')
        print(f'r = {np.round(self.iter_r, 4)}')
        
resol = Simplex().optimizar # Función para resolver un problema