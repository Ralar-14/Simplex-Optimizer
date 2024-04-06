from simplex import *
from time import time
import os
import sys

inicio = time()

#Escribe aqui el numero de problema a solucionar
Problemas = ['12-1', '12-2', '12-3', '12-4', '49-1', '49-2', '49-3', '49-4']
#Escribe aqui el nombre de la carpeta en la que se guardaran las soluciones

Carpeta = 'soluciones'
#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

# Crear una carpeta para guardar las soluciones
if not os.path.exists(Carpeta):
    os.mkdir(Carpeta)

for Problema in Problemas:
    archivo = f'{Carpeta}/{Problema}.txt'

    with open(archivo, "w", encoding='utf-8') as file:
    # Redirigir la salida estándar (stdout) hacia el archivo
        sys.stdout = file
        try:
            resol(datos.problemas[Problema])  
        except (No_Acotado, No_Solucion) as e:
            print(f'\n{"-"*50}')
            print(f'\nError al resolver el problema {Problema}')
            if isinstance(e, No_Acotado):
                print(f'El problema no tiene solución acotada')
            elif isinstance(e, No_Solucion):
                print(f'El problema no tiene solución')
            file.close()

#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

sys.stdout = sys.__stdout__
print(f"\nTiempo de ejecución: {time()-inicio} segundos")