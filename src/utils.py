import re
import pandas as pd

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', str(t)).strip()
    return t

def to_date(x):
    x = convertir_fecha(x) if isinstance(x, str) else x
    return pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')

# Formatear fechas

def convertir_fecha(fecha_str):
    meses = {
        'enero': '01',
        'febrero': '02',
        'marzo': '03',
        'abril': '04',
        'mayo': '05',
        'junio': '06',
        'julio': '07',
        'agosto': '08',
        'septiembre': '09',
        'octubre': '10',
        'noviembre': '11',
        'diciembre': '12'   
    }
    
    partes = fecha_str.split(' de ')
    dia = partes[0].zfill(2)  # Asegura 2 dígitos para días 1-9
    mes = meses[partes[1]]
    año = partes[2]
   
    return f"{dia}/{mes}/{año}"  # Formato DD/MM/AAAA