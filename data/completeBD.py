import pandas as pd
import os

# Ruta del directorio que contiene los archivos Parquet
directorio = '/data/samuelrt/data'

# Inicializar una lista para almacenar los DataFrames
dfs = []

# Leer cada archivo Parquet en el directorio
for archivo in os.listdir(directorio):
    if archivo.endswith('.parquet'):
        ruta_archivo = os.path.join(directorio, archivo)
        df = pd.read_parquet(ruta_archivo)
        
        # Agregar el DataFrame a la lista
        dfs.append(df)

# Obtener la lista de estaciones de cada DataFrame
estaciones_por_archivo = [set(df['station']) for df in dfs]

# Encontrar la intersección de estaciones (comunes a todos los archivos)
estaciones_comunes = set.intersection(*estaciones_por_archivo)

# Filtrar los DataFrames para incluir solo las estaciones comunes
datos_filtro = pd.concat([df[df['station'].isin(estaciones_comunes)] for df in dfs], ignore_index=True)

# Guardar el DataFrame filtrado en un nuevo archivo Parquet
ruta_salida = '/data/samuelrt/data/data_17_18.parquet'
datos_filtro.to_parquet(ruta_salida, index=False)
