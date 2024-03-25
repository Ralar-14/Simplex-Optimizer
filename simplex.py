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
        
        self.iter_m = None
        self.iter_n = None
        

    def optimizar(self, problema):
        self.c = problema.c # Coeficientes de la función objetivo
        self.A = problema.A # Coeficientes de las restricciones
        self.b = problema.b # Lado derecho de las restricciones
        self.m = self.A.shape[0] # Número de restricciones
        self.n = self.A.shape[1] # Número de variables
        
        # Fase 1
        self.fase1()
        
        # Fase 2
        return self.fase2()
        
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
        self.iter_B_inv = np.linalg.inv(self.iter_B)

    def iter(self):
        iteraciones = 0
        self.iter_r = self.iter_cN - self.iter_cB @ self.iter_B_inv @ self.iter_An # Coeficientes reducidos

        while np.any(self.iter_r < 0):
            self.iter_dB = - self.iter_B_inv @ self.iter_An[:, np.argmin(self.iter_r)] # Direccion basica factible
            self.iter_theta_array = np.array([- self.iter_xB[i] / self.iter_dB[i] if self.iter_dB[i] < 0 else np.inf for i in range(self.iter_m)]) # Calculamos la lista de thetas posibles
            self.iter_theta = np.min(self.iter_theta_array) # Elegimos el theta minimo
            
            self.iter_var_entrada = self.iter_N[np.argmin(self.iter_r)] # Recuperamos la variable que entra
            self.iter_var_salida_indice = np.argmin(self.iter_theta_array) # Recuperamos el indice de la variable que sale
            self.iter_var_salida = self.iter_Beta[self.iter_var_salida_indice] # Elegimos la variable que sale
            
            self.iter_Beta[np.where(self.iter_Beta == self.iter_var_salida)], self.iter_N[np.where(self.iter_N == self.iter_var_entrada)] = self.iter_var_entrada, self.iter_var_salida # Actualizamos las variables basicas y no basicas

            self.iter_xB += self.iter_theta * self.iter_dB # Actualizamos el valor de las variables basicas
            self.iter_xB[self.iter_var_salida_indice] = self.iter_theta # Actualizamos el valor de la variable que entra
            
            if np.any(self.iter_xB < 0): # Si xB es negativo, el problema no tiene solucion
                return "El problema no tiene solución"
            
            # No cambiamos xN ya que en todo caso será un vector de n - m ceros
            
            self.iter_B = self.iter_A[:, self.iter_Beta] # Actualizamos la matriz de coeficientes de restricciones basicas
            self.iter_An = self.iter_A[:, self.iter_N] # Actualizamos la matriz de coeficientes de restricciones no basicas
            
            self.actualizar_inversa()
                    
            self.iter_cB = self.iter_c[self.iter_Beta] # Coeficientes de las variables basicas en la funcion objetivo
            self.iter_cN = self.iter_c[self.iter_N] # Coeficientes de las variables no basicas en la funcion objetivo
            
            self.iter_z += self.iter_theta * np.min(self.iter_r) # Valor de la variable objetivo
            
            self.iter_r = self.iter_cN - self.iter_cB @ self.iter_B_inv @ self.iter_An # recalculamos los coeficientes reducidos
            iteraciones += 1
        return self.iter_r, self.iter_z, self.iter_N
        
    def fase1(self):
        self.generar_PA()
        return self.iter()
    
    def fase2(self):
        self.generar_sol()
        return self.iter()
        
        
        
               
print(Simplex().optimizar(datos.problemas["12-1"]))

    
    