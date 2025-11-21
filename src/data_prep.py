import os
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    # define the files to use
    files_to_use = [
        "eruptions.csv",
        "volcano.csv",
        "events.csv"
    ]

    dict_df = {}
    for file in files_to_use:
        print(f"Preparando datos para el archivo: {file}")
        df_tmp = leer_datos(file)
        df_tmp = preparar_datos(df_tmp, file)
        dict_df[file] = df_tmp        # dict_df[file] = df


    df = create_final_dataset(dict_df)
    df.dropna(inplace=True) # manipulate this to have a better imputation strategy

    data_path = os.path.join(PROJECT_DIR, "data", "processed")
    os.makedirs(data_path, exist_ok=True)
    
    output_path = os.path.join(data_path, "final_volcanic_data.csv")
    features = [
        'latitude',
        'longitude',
        'elevation',
        'eruption_start_year',
        'vei',
    ]

    cat = [
        'eruption_category',
        'primary_volcano_type',
        'evidence_category',
        'tectonic_settings',
        'major_rock_1',
    ]

    df['vei'] = pd.to_numeric(df['vei'], errors='coerce')
    df['vei'] = df['vei'].round(0)
    df = df.dropna(subset=['vei'])

    # df['vei_class'] = df['vei'].apply(lambda x: 'low' if x <= 2 else ('medium' if x <=4 else 'high'))    
    df['vei_class'] = df['vei'].apply(lambda x: 'low' if x <= 2 else 'high')

    df = df[features + cat + ['vei_class']]
    df.to_csv(output_path, index=False)

    # df_std = standardize_data(df)
    # output_path_std = os.path.join(data_path, "final_volcanic_data_standardized.csv")
    # df_std.to_csv(output_path_std, index=False)
    


def standardize_data(df):

    df = scale_numerical_variables(df)
    df = convert_to_lowercase(df)
    # df = encode_categorical_variables(df)
    

    return df


def convert_to_lowercase(df):

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].str.lower()
    print(f"Se convirtieron las columnas categóricas a minúsculas. {df.shape}")
    return df


def encode_categorical_variables(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # cols_to_encode = [col for col in cat_cols if df[col].nunique() < 20]
    for col in cat_cols:
        # TODO: apply one-hot encoding (dummy variables) if number of unique values is less than a threshold
        # Needs to do a dedicated analysis of each categorical variable to decide the best encoding strategy
        # if col in cols_to_encode:
            # df[col] = df[col].astype('category').cat.codes # convert to categorical codes (Label Encoding)
        df = pd.get_dummies(df, columns=[col], prefix=col)

    print(f"Se codificaron las columnas categóricas. {df.shape}")
    return df


def scale_numerical_variables(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in num_cols:
        # TODO: apply Min-Max scaling if needed
        mean_value = df[col].mean()
        std_value = df[col].std()
        df[col] = (df[col] - mean_value) / std_value
    print(f"Se escalaron las columnas numéricas. {df.shape}")
    return df


def create_final_dataset(dict_df):
    # joing eruptions and events by 'eruption_number' and 'volcano_number'
    df_eruptions = dict_df['eruptions.csv']
    df_events = dict_df['events.csv']

    df_merged = pd.merge(df_eruptions, df_events, on=['eruption_number', 'volcano_number'], how='left', suffixes=('_eruption', '_event'))

    # join with volcanoes by 'volcano_number'
    df_volcanoes = dict_df['volcano.csv'] 

    df_final = pd.merge(df_merged, df_volcanoes, on='volcano_number', how='left', suffixes=('', '_volcano'))
    
    # drop high correlated columns if any
    to_drop = ['start_year', 'latitude_volcano', 'longitude_volcano', 'volcano_name_event', 'volcano_name']  
    df_final.drop(columns=to_drop, inplace=True)
    

    df_final.drop_duplicates(inplace=True)
    print(f"Final dataset shape: {df_final.shape}")

    return df_final

def preparar_datos(df, table_name):

    if 'eruptions' in table_name:
        df = feature_eruptions(df)
    elif 'volcano' in table_name:
        df = feature_volcanoes(df)
    elif 'events' in table_name:
        df = feature_events(df)
    else:
        print(f"No hay funciones de preparación definidas para el archivo {table_name}")
    
    
    return df


def feature_events(df):
    cols_to_use = ['volcano_number', 'volcano_name', 'eruption_number', 'eruption_start_year', 'event_type']
    df = df[cols_to_use]


    return df


def feature_volcanoes(df):
    cols_to_use = ['volcano_number', 'volcano_name', 'primary_volcano_type', 'latitude', 'longitude', 'elevation', 'evidence_category', 
                   'last_eruption_year', 'tectonic_settings', 'major_rock_1']
    df = df[cols_to_use]

    # fix last_eruption_year replace 'Unknown' with NaN and convert to numeric
    df['last_eruption_year'] = pd.to_numeric(df['last_eruption_year'], errors='coerce')


    return df


def feature_eruptions(df):

    cols_to_use =['volcano_number', 'volcano_name', 'eruption_number', 'eruption_category', 'vei', 'start_year', 'latitude', 'longitude']
    df = df[cols_to_use]

    # fix vei values
    df['vei'] = pd.to_numeric(df['vei'], errors='coerce')
    df['vei'] = df['vei'].round(0)

    # análisis específico para datos de erupciones
    return df


def leer_datos(file):
    input_path = os.path.join(PROJECT_DIR, "data", "clean", file)
    df = pd.read_csv(input_path)
    print(f"Datos leídos para el archivo {file} con forma {df.shape}")
    return df


if __name__ == "__main__":
    main()