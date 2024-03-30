from simplex import *

#Escriu aqui el numero de problema a solucionar
Problema = '49-4'

#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------

archivo = f'{Problema}.txt'

with open(archivo, "w", encoding='utf-8') as file:
# Redirigir la salida estándar (stdout) hacia el archivo
    sys.stdout = file
    try:
        resol(datos.problemas[Problema])  
    except (No_Acotado, No_Solucion) as e:
        print(f'\n{"-"*50}')
        print(f'\nError al resoldre el problema {Problema}')
        if isinstance(e, No_Acotado):
            print(f'El problema no té solució acotada')
        elif isinstance(e, No_Solucion):
            print(f'El problema no té solució')

#--------------------------------------------------------------
#--------------------------------------------------------------
#--------------------------------------------------------------
