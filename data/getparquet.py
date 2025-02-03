import pandas as pd
import os
import numpy as np
import json

var =['PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
 'UMIDADE RELATIVA DO AR, HORARIA (%)',
 'VENTO, VELOCIDADE HORARIA (m/s)','PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']

col_names = ['pressure','temperature','humidity','wind','precipitation']

# Especifica la ruta de la carpeta
#carpeta = r'/prj/posgrad/samuelrt/Documentos/analiseexploratorio/dados'
carpeta = r'/data/samuelrt/data/codes/2017'
with open('/data/samuelrt/data/codes/nomes_station.json', 'r') as archivo:
    # Cargar el contenido del archivo en un diccionario
    station_code = json.load(archivo)

# Lista todos los archivos con extensión .csv
archivos_csv = [(archivo,archivo.split('_')[3]) for archivo in os.listdir(carpeta) if archivo.endswith('.CSV')]

df_principal = pd.DataFrame()

for csv,code in archivos_csv:
    if code in station_code.keys():
        print(station_code[code])
        df = pd.read_csv(carpeta+'/'+csv,sep =';',skiprows=8,encoding='cp1252')
        df.loc[:,'espacio'] = ' '
        fecha = pd.Series()
        fecha = df['DATA (YYYY-MM-DD)'] + df['espacio'] + df['HORA (UTC)']
        df = df[var]
        for c in df.columns:
            df[c] = df[c].astype(str).str.replace(',', '.')
            # Convertir a tipo float
            df[c] = df[c].astype(float)
        df.columns = col_names
        for c in df.columns:
            df[c] = df[c].where(df[c]>= 0, None)
        df.interpolate(inplace=True)
        if df.isna().sum().sum()==0:
            df.loc[:,'station'] = station_code[code][0]
            df.loc[:,'latitude'] = station_code[code][1]
            df.loc[:,'longitude'] = station_code[code][2]
            df.loc[:,'date'] = fecha.values
            df_principal = pd.concat([df_principal,df], ignore_index=False)
    else: continue
df_principal.to_parquet('DATA_BASE.parquet',index=False,engine='pyarrow')
print(df.shape)
