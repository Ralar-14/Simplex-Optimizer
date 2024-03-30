from simplex import *

#Escriu aqui el numero de problema a solucionar
Problema = '12-3'

archivo = f'{Problema}.txt'

with open(archivo, "w", encoding='utf-8') as file:
    # Redirigir la salida estándar (stdout) hacia el archivo
    import sys
    sys.stdout = file
    try:
        resol(datos.problemas[Problema])  
              
    except:
        print(f'Error al resolver el problema {Problema}')
