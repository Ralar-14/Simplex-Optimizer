from simplex import *

#Escriu aqui el numero de problema a solucionar
Problema = '49-1'


arxiu = f'{Problema}.txt'
with open(arxiu, "w", encoding='utf-8') as archivo:
    # Redirigir la salida estándar (stdout) hacia el archivo
    import sys
    sys.stdout = archivo
    resol(datos.problemas[Problema])
