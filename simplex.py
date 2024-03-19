from lector import Lector
import numpy as np

datos = Lector("datos.txt")

class Simplex:
    def __init__(self):
        # Inicializamos las variables para evitar errores
        self.c = None
        self.A = None
        self.b = None
        
        # Dimensiones de la matriz A
        self.m = None
        self.n = None
        
        self.fase1_m = None
        self.fase1_n = None
        

    def optimizar(self, problema):
        self.c = problema.c # Coeficientes de la función objetivo
        self.A = problema.A # Coeficientes de las restricciones
        self.b = problema.b # Lado derecho de las restricciones
        self.m = self.A.shape[0] # Número de restricciones
        self.n = self.A.shape[1] # Número de variables
        
        # Fase 1
        print(self.fase1())
        
    def generar_PA(self):

        self.fase1_c = np.array([0] * self.n + [1] * self.m)
        self.fase1_A = np.concatenate((self.A, np.identity(self.m)), axis=1)
        self.fase1_b = self.b
        
        self.fase1_Beta = np.arange(self.n, self.n + self.m) # Variables basicas
        self.fase1_N = np.arange(self.n) # Variables no basicas
        
        self.fase1_m = self.fase1_A.shape[0] # Número de restricciones
        self.fase1_n = self.fase1_A.shape[1] # Número de variables
        
        self.fase1_B = np.identity(self.fase1_m) # Matriz de coeficientes de restricciones basicas
        self.fase1_An = self.fase1_A[:, self.fase1_N] # Matriz de coeficientes de restricciones no basicas
        
        self.fase1_cB = self.fase1_c[self.fase1_Beta] # Coeficientes de las variables basicas en la funcion objetivo
        self.fase1_cN = self.fase1_c[self.fase1_N] # Coeficientes de las variables no basicas en la funcion objetivo
        
        self.fase1_B_inv = self.fase1_B # Inversa de la matriz de coeficientes de restricciones basicas (inicialmente es la identidad)
        
        self.fase1_xB = self.fase1_B_inv @ self.fase1_b # Valores de las variables basicas
        self.fase1_xN = np.array([0] * self.fase1_n) # Valores de las variables no basicas
        
        self.fase1_z = self.fase1_cB @ self.fase1_xB # Valor de la funcion objetivo
        
    def actualizar_inversa(self):
        self.fase1_B_inv = np.linalg.inv(self.fase1_B)
        
    def fase1(self):
        self.generar_PA()
        self.fase1_r = self.fase1_cN - self.fase1_cB @ self.fase1_B_inv @ self.fase1_An # Coeficientes reducidos
        
        while np.any(self.fase1_r < 0):
            self.fase1_dB = - self.fase1_B_inv @ self.fase1_An[:, np.argmin(self.fase1_r)] # Direccion basica factible
            self.fase1_theta_array = np.array([- self.fase1_xB[i] / self.fase1_dB[i] if self.fase1_dB[i] < 0 else np.inf for i in range(self.fase1_m)]) # Calculamos la lista de thetas posibles
            self.fase1_theta = np.min(self.fase1_theta_array) # Elegimos el theta minimo
            
            self.fase1_var_entrada = self.fase1_N[np.argmin(self.fase1_r)] # Recuperamos la variable que entra
            self.fase1_var_salida_indice = np.argmin(self.fase1_theta_array) # Recuperamos el indice de la variable que sale
            self.fase1_var_salida = self.fase1_Beta[self.fase1_var_salida_indice] # Elegimos la variable que sale
            
            self.fase1_Beta[np.where(self.fase1_Beta == self.fase1_var_salida)], self.fase1_N[np.where(self.fase1_N == self.fase1_var_entrada)] = self.fase1_var_entrada, self.fase1_var_salida # Actualizamos las variables basicas y no basicas

            self.fase1_xB += self.fase1_theta * self.fase1_dB # Actualizamos el valor de las variables basicas
            self.fase1_xB[self.fase1_var_salida_indice] = self.fase1_theta # Actualizamos el valor de la variable que entra
            
            if np.any(self.fase1_xB) < 0: # Si xB es negativo, el problema no tiene solucion
                return "El problema no tiene solución"
            
            # No cambiamos xN ya que en todo caso será un vector de n - m ceros
            
            self.fase1_B = self.fase1_A[:, self.fase1_Beta] # Actualizamos la matriz de coeficientes de restricciones basicas
            self.fase1_An = self.fase1_A[:, self.fase1_N] # Actualizamos la matriz de coeficientes de restricciones no basicas
            
            self.actualizar_inversa()
                    
            self.fase1_cB = self.fase1_c[self.fase1_Beta] # Coeficientes de las variables basicas en la funcion objetivo
            self.fase1_cN = self.fase1_c[self.fase1_N] # Coeficientes de las variables no basicas en la funcion objetivo
            
            self.fase1_z += self.fase1_theta * np.min(self.fase1_r) # Valor de la variable objetivo
            
            self.fase1_r = self.fase1_cN - self.fase1_cB @ self.fase1_B_inv @ self.fase1_An # recalculamos los coeficientes reducidos
        
        return self.fase1_r, self.fase1_z, self.fase1_N
        
        
        
Simplex().optimizar(datos.problemas["12-1"])

    
    