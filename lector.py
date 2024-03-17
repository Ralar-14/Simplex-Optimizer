# Clase que lee los datos de un archivo de texto para luego aplicar Simplex

import numpy as np

# Lectura de los datos de un archivo de texto

# Clase que contiene una diccionario de problemas (objetos de la clase Problema)
class Lector:
    
    class Problema:
        def __init__(self, problema = None):
            # Inicializamos los atributos del problema en None para evitar errores
            self.problem = problema
            self.c = None
            self.A = None
            self.b = None
            self.z = None
            self.vb = None
        
        def __repr__(self):
            return f"c={self.c}\nA={self.A}\nb={self.b}\nz*={self.z}\nvb*={self.vb}"
        
        
    def __init__(self, path):
        self.path = path
        self.problemas = {}
        self.read_data()
    
    # Método que lee los datos del archivo de texto
    def read_data(self):
        with open(self.path, "r") as file:
            data = file.read()
            
        # Dividimos el archivo en diferentes problemas
        problemas = data.split("\n\n\n")
        
         
        for problema in problemas:
            new_problema = self.Problema(problema)
            
            # Separa las secciones del problema
            secciones = problema.split("\n\n")
            
            for seccion in secciones:
                if seccion.startswith("Problema_"):
                    new_problema.problem = seccion[9:]
                    
                elif seccion.startswith("c="):
                    new_problema.c = np.fromstring(seccion[2:], sep=' ')

                elif seccion.startswith("A="):
                    filas_a = seccion[2:].strip().split("\n")
                    new_problema.A = np.array([np.fromstring(fila, sep=' ') for fila in filas_a])

                elif seccion.startswith("b="):
                    new_problema.b = np.fromstring(seccion[2:], sep=' ')
                    
                elif seccion.startswith("z*="):
                    # Pese a que z es un escalar, se almacena como un array de un solo elemento para mantener la consistencia
                    new_problema.z = np.fromstring(seccion[3:], sep=' ')
                    
                elif seccion.startswith("vb*="):
                    new_problema.vb = np.fromstring(seccion[4:], sep=' ')
                    
            # Agregamos el problema a la lista de problemas
            self.problemas[new_problema.problem] = new_problema

# Uso de la clase
# lector_problema = Lector("datos.txt")
# print(lector_problema.problemas["49-3"])