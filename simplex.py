from lector import Lector
import numpy as np

datos = Lector("datos.txt")

class Simplex:
    def __init__(self):
        # Inicializamos las variables para evitar errores
        self.c = None
        self.A = None
        self.b = None

        # Variables de la fase 1
        self.fase1_c = None
        self.fase1_A = None
        self.fase1_b = None
        self.fase1_B = None
        
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
        self.fase1()
        
    def fase1(self):

        self.fase1_c = np.array([0] * self.n + [1] * self.m)
        self.fase1_A = np.concatenate((self.A, np.identity(self.m)), axis=1)
        self.fase1_b = self.b
        
        self.fase1_B = np.arange(self.n, self.n + self.m)
        self.fase1_N = np.arange(self.n)
        
        self.fase1_m = self.fase1_A.shape[0]
        self.fase1_n = self.fase1_A.shape[1]
        
    def fase2(self):
        self.fase2_An = self.fase1_A[:, self.fase1_N]
        
        
        
        
Simplex().optimizar(datos.problemas["49-3"])

    
    