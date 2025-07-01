from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np

# esta funcion usa train_test_split para dividir la data y ya transformarla a numpy (quiza deberia ser otra
#funcion, no se), y tambien hace el round pq con la data normalizada, no funciona sin eso

def split_data(df, treatment, outcome, covariates_names, percentage_test): # quite question pq la impresion ya no esta, pero quiza viene bien para evaluar si anda todo bien
    
    d = df[treatment].astype(int)
    y = df[outcome]
    x_cov = df[covariates_names] # dataframe pq son muchas
    
    x_train, x_test, d_train, d_test, y_train, y_test = train_test_split(
        x_cov,
        d, 
        y, 
        test_size=percentage_test, 
        stratify=y, 
        random_state= 2000 # el random state tambien puede ser siempre el mismo (creo), por lo que podriamos ponerlo con SCR y simlplemente llamarlo. habria que agregarlo a la funcion claro.
        )
    
    
    return x_train, x_test, d_train, d_test, y_train, y_test

    """
    # en este caso, creo que seria mas que un print, un test que compruebe que las proporciones no estan tan alejadas unas de otras.
    print(f'Question {question}')
    original_proportion = y.value_counts(normalize=True)
    original_treatment_proportion = d.value_counts(normalize=True)
    print("Original proportions:")
    print(original_proportion, original_treatment_proportion)

    train_proportion = y_train.value_counts(normalize=True)
    train_treatment_proportion = d_train.value_counts(normalize=True)
    print("\nTrain proportions:")
    print(train_proportion, train_treatment_proportion)

    test_proportion = y_test.value_counts(normalize=True)
    test_treatment_proportion = d_test.value_counts(normalize=True)
    print("\nTest proportions:")
    print(test_proportion, test_treatment_proportion)
    """





if __name__ == "__main__":

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_with_new_variables_from_original.csv") 

    data_done = data_ready.copy()
    #data_done = data_done[data_done['gender'].notna()] # quitar, se debe agregar en el data cleaning

    # mi idea seria exportar esta funcion de split_data cada vez que la necesito y eta funcion me agarra directamente las 4 preguntas


    t_list = ['Q1_1_treat', 'Q1_2_treat', 'Q2_1_treat', 'Q2_2_treat']
    y_list = ['Q1_1', 'Q1_2', 'Q2_1', 'Q2_2']

    covariates_names = ["employment", "education", "age", "female", "houseowner", "financialwellbeing", "optimism_bias", "scale_sufficiency", "openness", "conscientiousness",
                "extraversion", "agreeableness", "neuroticism", "rationality_score", "growthmind",
                'Positive_Reciprocity', 'Negative_Reciprocity', 'Altruism', 'Trust', 'Risk_Preferences', 'Time_Preferences', "Social_Anxiety",
                "Public_SelfConsciousness", "Private_SelfConsciousness", "ProcrastinationExAnte", "ProcrastinationExPost", "empathic_concern_score", "perspective_taking_score"]

    for i,j,k,l in zip(t_list, y_list, covariates_names, range(4)):
        split_data(data_done, i, j, k, l+1)

