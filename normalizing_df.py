import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path



def normalize_data(df):
    data_normalizada = df.copy()
    num_cols = data_normalizada.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    data_normalizada[num_cols] = scaler.fit_transform(data_normalizada[num_cols])
    return data_normalizada


if __name__ == "__main__":

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_normalized.csv") 

    data_done = data_ready.copy()
    #data_done = data_done[data_done['gender'].notna()]

    # ESTA VARIABLE SI FUNCIONA, SE DEBERIA AGREGAR ANTES!! NO EN ESTE DOCUMENTO
    mapeo_grupos = {1:[1], 2:[2, 3], 3:[7], 4:[4, 5, 6]}
    data_done["opinion_migration"] = None
    for nueva_cat, lista_originales in mapeo_grupos.items():
        for original in lista_originales:
            data_done.loc[data_done["marielboatlift"] == original, "opinion_migration"] = nueva_cat

    data_done["opinion_migration"] = pd.to_numeric(data_done["opinion_migration"], errors='coerce')

    normalize_data(data_done).to_csv("data_normalized_fr.csv")

# Cosas que pueden fallar en esta funcion:
    # que se cambien columnas sin nan
    #que se cambien columnas con letras
    # columnas numericas no se estan normalizando