import os
import pandas as pd
from pathlib import Path

# reconocer directorio del proyecto
# PROJECT_DIR = Path.cwd().parent
PROJECT_DIR = Path(__file__).resolve().parent.parent
print(f"Directorio del proyecto: {PROJECT_DIR}")

def main():

    # buscar todos los archivos en el directorio raw/archive
    list_files = os.listdir(os.path.join(PROJECT_DIR, "data", "raw", "archive"))
    print(f"Archivos en el directorio de datos raw/archive: {list_files}")

    for file in list_files:
        print(f"Limpieza de datos para el archivo: {file}")
        limpiar_datos(file)





def limpiar_datos(table_name):

    table_name = table_name
    
    df = leer_datos(table_name)
    df = reemplazar_valores(df, to_replace=' ', value='N/A')

    df = eliminar_columnas_nulas(df, umbral=0.5)
    df = imputar_valores_faltantes(df)

    # guardar datos limpios
    output_path = os.path.join(PROJECT_DIR, "data", "clean", f"{table_name}")
    df.to_csv(output_path, index=False)

def reemplazar_valores(df, to_replace, value):
    for col in df.columns:
        # 
        try:
            df[col].replace(to_replace, value, inplace=True)
            print(f"Columna {col}: se reemplazaron los valores '{to_replace}' por '{value}'")
        except Exception as e:
            continue
    return df


def imputar_valores_faltantes(df):

    df = imputacion_numerica(df)
    df = imputacion_categorica(df)

    return df


def imputacion_categorica(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        print(f"Columna {col}: valores nulos imputados con la moda '{mode_value}'")
    return df

def imputacion_numerica(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in num_cols:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)
        print(f"Columna {col}: valores nulos imputados con la media {mean_value:.2f}")
    return df


def eliminar_columnas_nulas(df, umbral=0.5):
    list_cols = df.columns.tolist()
    # loop sobre una lista de python
    cols_to_drop = list() # lista vacía para almacenar nombres de columnas a eliminar
    for col in list_cols:
        n_nulls = df[col].isnull().sum()
        l_values = len(df[col])

        # condicional para identificar columnas con más del 50% de valores nulos
        if n_nulls/l_values > umbral:
            cols_to_drop.append(col)
            print(f"La columna {col} tiene más del {umbral*100}% de valores nulos.")

    print(f"Columnas a eliminar: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

def leer_datos(table_name):

    data_path = os.path.join(PROJECT_DIR, "data", "raw", "archive", table_name)
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    main()