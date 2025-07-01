import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#import patsy

SRC = Path(__file__).parent.resolve()
data = SRC / "waves123_augmented_consent.dta"
# porbe con la data que envio david y luego con la data que enviaron que es menos "modificada" y parece que ninguna de estas variables se tocan
# porque corre el documento sin problema

waves_data = pd.read_stata(data, convert_categoricals=False)

##### AGREGO ESTO ACA PORQUE SI NO DESPUES ES UNA PARIDERA CAMBIAR TODO, PERO REALMENTE DEBERIA IR AL FINAAAL
#########################################...........................############################

waves_data = pd.read_stata("waves123_augmented_consent.dta", convert_categoricals=False)

cols_x = [f'xlist_{i}' for i in range(1, 9)]
cols_y = [f'ylist_{i}' for i in range(1, 9)]

waves_data['x_final'] = waves_data[cols_x].idxmax(axis=1).str.extract('(\d+)').astype(float)
waves_data['y_final'] = waves_data[cols_y].idxmax(axis=1).str.extract('(\d+)').astype(float)

# salir mal: puede que no este agarrando bien los datos, es decir la primera con 1. cuando x8=1, debe x_final y y ser siempre =8

conditions_x = [
    waves_data['x_final'].isin([1, 2]),
    waves_data['x_final'].isin([3, 4]),
    waves_data['x_final'] == 5,
    waves_data['x_final'] == 6,
    waves_data['x_final'] == 7,
    waves_data['x_final'] == 8
]

choices_x = ["2.5", "1.5", "0.5", "-0.5", "-1.5", "-2.5"]

waves_data['zone_x'] = np.select(conditions_x, choices_x, default=np.nan)

conditions_y = [
    waves_data['y_final'] == 1,
    waves_data['y_final'] == 2,
    waves_data['y_final'] == 3,
    waves_data['y_final'].isin([4, 5]),
    waves_data['y_final'].isin([6, 7]),
    waves_data['y_final'] == 8
]

choices_y = ["-2.5", "-1.5", "-0.5", "0.5", "1.5", "2.5"]

waves_data['zone_y'] = np.select(conditions_y, choices_y, default=np.nan)

waves_data['zone_x'] = waves_data['zone_x'].astype(float)
waves_data['zone_y'] = waves_data['zone_y'].astype(float)

# Esto puede tener mil errores...... AL LORO
conditions = [
    (waves_data['zone_x'] > 0.5) & (waves_data['zone_y'] < 0.5), # equiality averse
    (waves_data['zone_x'] > 0.5) & (waves_data['zone_y'] >= -0.5) & (waves_data['zone_y']<=0.5), #kiss up
    (waves_data['zone_x'] > 0.5) & (waves_data['zone_y'] > 0.5), # altruistic
    (waves_data['zone_x'] >= -0.5) & (waves_data['zone_x'] <= 0.5) & (waves_data['zone_y'] < -0.5), # kick down
    (waves_data['zone_x'] >= -0.5) & (waves_data['zone_x'] <= 0.5) & (waves_data['zone_y'] >= -0.5) & (waves_data['zone_y'] <= 0.5), # selfish
    (waves_data['zone_x'] >= -0.5) & (waves_data['zone_x'] <= 0.5) & (waves_data['zone_y'] > 0.5), # maximin 
    (waves_data['zone_x'] < -0.5) & (waves_data['zone_y'] < -0.5), # spiteful
    (waves_data['zone_x'] < -0.5) & (waves_data['zone_y'] >= -0.5) & (waves_data['zone_y'] <= 0.5), # envious
    (waves_data['zone_x'] < -0.5) & (waves_data['zone_y'] > 0.5), # inequality averse
]

# Y los valores a asignar para cada combinación
values = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Crear nueva columna
waves_data['preference_type'] = np.select(conditions, values, default=np.nan)

#########################################...........................############################

dataset = waves_data.copy()

testing = dataset.copy()
testing = testing[(testing['Finished_w3'].isna())] # testing['marielboatlift'] esta vacio, finished w3 significa que participaron

# Only participants with w3 finished, no suspicious (no se que es), WTA&WTP lower than 250, and defined gender.
dataset = dataset[dataset['marielboatlift'].notna()] 


"""
dataset = dataset[
    (dataset['Finished_w3'] == 1) & #re
    (dataset['anysuspicious'] == 0) & #re
    (dataset['WTA'] <= 250) & #re
    (dataset['WTP'] <= 250) & #re
    (dataset['gender'].notna()) #re
    ]
"""
N = len(dataset)
print("Number of participants (rows):", N)

# Algo de investment advices que no pillo mucho...
dataset['stepsinvestpre23'] = np.where((dataset['stepsinvestpre22'] == 1) | (dataset['stepsinvestpost22'] == 1), 1, 0)

# Original coding:
# confidence == 4 -> "among the 50 best"  -> bracket 1  (best)
# confidence == 3 -> "ranked 51–100"      -> bracket 2
# confidence == 2 -> "ranked 101–400"     -> bracket 3
# confidence == 1 -> "ranked below 400"   -> bracket 4  (worst)
# confidence == 99 -> "no clue"           -> NA bracket 0

dataset['conf_new'] = 0
dataset.loc[dataset['confidence'] == 4, 'conf_new'] = 1
dataset.loc[dataset['confidence'] == 3, 'conf_new'] = 2
dataset.loc[dataset['confidence'] == 2, 'conf_new'] = 3
dataset.loc[dataset['confidence'] == 1, 'conf_new'] = 4
dataset.loc[dataset['confidence'] == 99, 'conf_new'] = 0
dataset['conf_new'] = pd.to_numeric(dataset['conf_new'], errors='coerce') # revisar este codigo e intentar hacerlo de nuevo pq no creo que esta linea sea necesaria


# Converting ranking_w2 into numeric var
ranking_map = {
    "Jan-50": 1,
    "51-150": 2,
    "151-350": 3,
    "351-1026": 4
    }
dataset['ranking_num'] = dataset['ranking_w2'].map(ranking_map).astype(int)

# Optimism bias, difference bewteen actual score and confidence. The bigger, the less accurate
dataset['optimism_bias'] = dataset['ranking_num'] - dataset['conf_new']

# Arranging other variables
dataset['scale_sufficiency'] = dataset['scale_sufficiency']*100
dataset['donate_lottery_w1'] = dataset['donate_lottery_w1']/100

# Creating binary variables with meat, housing, mobility, bonusdonation, donate lotery, scale sufficiency, EPC
dataset['meat_w1'] = np.where(dataset['meat_w1'] >= 4.475, 0, 1)
dataset['housing_w1'] = np.where(dataset['housing_w1'] >= 20.09, 0, 1)
dataset['mobility_hyp_w1'] = np.where(dataset['mobility_hyp_w1'] >= 26.52, 1, 0)
dataset['bonusdonation_w1'] = np.where(dataset['bonusdonation_w1'] > 0.297, 1, 0)
dataset['donate_lottery_w1'] = np.where(dataset['donate_lottery_w1'] > 0.198, 1, 0)
dataset['scale_sufficiency'] = np.where(dataset['scale_sufficiency'] > 55.08, 1, 0)
dataset['EPC_ABC'] = np.where(dataset['EPC_ABC'] > 0.2259, 1, 0)

# Big Five

# The negative ones are reversed so a mean of them make sense.
dataset['bigfive_1'] = 6 - dataset['bigfive_1']
dataset['bigfive_3'] = 6 - dataset['bigfive_3']
dataset['bigfive_4'] = 6 - dataset['bigfive_4']
dataset['bigfive_5'] = 6 - dataset['bigfive_5']
dataset['bigfive_7'] = 6 - dataset['bigfive_7']

dataset['openness'] = dataset[['bigfive_5', 'bigfive_10']].mean(axis=1)
dataset['conscientiousness'] = dataset[['bigfive_3', 'bigfive_8']].mean(axis=1)
dataset['extraversion'] = dataset[['bigfive_1', 'bigfive_6']].mean(axis=1)
dataset['agreeableness'] = dataset[['bigfive_2', 'bigfive_7']].mean(axis=1)
dataset['neuroticism'] = dataset[['bigfive_4', 'bigfive_9']].mean(axis=1)

# Rationality score
# How many correct "rationality" questions they have out of 3.
dataset['rationality_score'] = (
    (dataset['rationality1'] == 2).astype(int) +
    (dataset['rationality2'] == 8).astype(int) +
    (dataset['rationality3'] == 50).astype(int)
    )

# Growth mindset
dataset['growthmind'] = dataset['growthmind'].replace({1: 4, 2: 3, 3: 2, 4: 1}).astype(float)

# Lambda calculations
wta_and_wtp_is_zero = dataset[(dataset['WTP'] == 0) & (dataset['WTA'] == 0)]


dataset['gain'] = 20
dataset['lambda_risky'] = np.where(dataset['riskymeasure'] != 0, dataset['gain'] / dataset['riskymeasure'], np.nan) # no lo entiendo
dataset['lambda_riskyless'] = np.where(dataset['WTP'] != 0, dataset['WTA'] / dataset['WTP'], np.nan) # no lo haria asi, if wtp wta es cero, ponemos 1. luego hacemos esta operacion

# Renaming columns
dataset.rename(columns={
    'procrastinate': 'ProcrastinationExAnte', #Procrastination
    'timepreferences': 'ProcrastinationExPost', #Procrastination
    'selfcons_1': 'Social_Anxiety', #Self-consciousness
    'selfcons_2': 'Public_SelfConsciousness', #Self-consciousness
    'selfcons_3': 'Private_SelfConsciousness' #Self-consciousness
    }, inplace=True)

# Empathy recoding
def recode_empathy(val):
    return {23: 4, 4: 3, 3: 2, 2: 1, 1: 0}.get(val, val)

for i in range(1, 5):
    dataset[f'empathy_{i}'] = dataset[f'empathy_{i}'].apply(recode_empathy)

dataset['empathic_concern_score'] = dataset['empathy_1'] + dataset['empathy_2']
dataset['perspective_taking_score'] = dataset['empathy_3'] + dataset['empathy_4']

# Rename columns, no las entiendo del todo
dataset.rename(columns={
    'descriptive_1': 'Positive_Reciprocity',
    'descriptive_3': 'Negative_Reciprocity',
    'descriptive2_4': 'Altruism',
    'descriptive_2': 'Trust',
    'descriptive2_5': 'Risk_Preferences',
    'descriptive2_51': 'Time_Preferences'
    }, inplace=True)

# Z-score. (x-mu/tita)
preferences_vars = [
    'Altruism', 'Trust', 'Risk_Preferences', 'Time_Preferences',
    'Positive_Reciprocity', 'Negative_Reciprocity'
    ]
for var in preferences_vars:
    dataset[f'{var}_z'] = scale(dataset[var].astype(float))

# Mariel Boatlift
# two dummies
dataset['knowledge_mariel'] = (dataset['marielboatlift'] == 1).astype(int) # if answer was right, dummy==1
dataset['no_knowledge_mariel'] = (dataset['marielboatlift'] != 1).astype(int) # if answer was wrong, dummy==1


dataset['Q1_1'] = dataset[['question1treat_1', 'question1control_1']].bfill(axis=1).iloc[:, 0] # y (outcome)
dataset['Q1_1_treat'] = np.where(dataset['question1treat_1'].notna(), 1, 0) # dummy D (treatment)
#dataset['Q1_1'] = dataset['question1treat_1'].fillna(dataset['question1control_1']) # another way of doing this, more intuitive
                                  
dataset['Q1_2'] = dataset[['question1treat_2', 'question1control_2']].bfill(axis=1).iloc[:, 0]
dataset['Q1_2_treat'] = np.where(dataset['question1treat_2'].notna(), 1, 0)
#dataset['Q1_2'] = dataset['question1treat_2'].fillna(dataset['question1control_2'])

dataset['Q2_1'] = dataset[['question2treat_1', 'question2control_1']].bfill(axis=1).iloc[:, 0]
dataset['Q2_1_treat'] = np.where(dataset['question2treat_1'].notna(), 1, 0)
#dataset['Q2_1'] = dataset['question2treat_1'].fillna(dataset['question2control_1'])

dataset['Q2_2'] = dataset[['question2treat_2', 'question2control_2']].bfill(axis=1).iloc[:, 0]
dataset['Q2_2_treat'] = np.where(dataset['question2treat_2'].notna(), 1, 0)
#dataset['Q2_2'] = dataset['question2treat_2'].fillna(dataset['question2control_2'])



# ESTO LO AGREGO YO, realmente hay que mirarlo...
dataset['trustscientists_1'] = dataset['trustscientists_1'].replace(np.NaN, -1)
dataset['trustscientists_2'] = dataset['trustscientists_2'].replace(np.NaN, -1)
dataset['trustscientists_3'] = dataset['trustscientists_3'].replace(np.NaN, -1)
dataset['trustinst_1'] = dataset['trustinst_1'].replace(np.NaN, -1)
dataset['trustinst_2'] = dataset['trustinst_2'].replace(np.NaN, -1)
dataset['trustinst_3'] = dataset['trustinst_3'].replace(np.NaN, -1)
dataset['trustinst_4'] = dataset['trustinst_4'].replace(np.NaN, -1)

# en la variable de residence2 hay un error realmente, el unico que puso other, es de hecho de luxemburgo...
dataset["country_residence"] = dataset["residence2"].replace(4, 0)

dataset["marielboatlift"].value_counts()

conditions = [
    dataset['marielboatlift'].isin([1]), # had right beliefs
    dataset['marielboatlift'].isin([2, 4]), # they thought it will change a little
    dataset['marielboatlift'].isin([3, 5, 6, 7]) # think they change a lot, or do not know
]

choices = [0, 1, 2]

dataset['exante_beliefs'] = np.select(conditions, choices, default=np.nan)

mapeo = {
    7: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 3,
    1: 4
}

dataset['marielboatlift_1'] = dataset['marielboatlift'].map(mapeo)
dataset['marielboatlift_2'] = np.where(dataset['marielboatlift'].isin([1.0, np.nan]), 1.0, 0.0)


### ESTO HAY QUE MIRARLO Y CORREGIRLO!!!
dataset["trust_in_science"] = (0.33*dataset["trustscientists_1"]) + (0.33*dataset["trustscientists_2"]) + (0.33*dataset["trustscientists_3"])
dataset["trust_in_institutions"] = (0.25*dataset["trustinst_1"]) + (0.25*dataset["trustinst_2"]) + (0.25*dataset["trustinst_3"]) + (0.25*dataset["trustinst_4"])

# genero masculinity trait... asumo que este chico no volteo los valores porque la matriz de correlacion es extrana
#dataset['masculinity_46_inv'] = 1.00 - dataset['masculinity_46'] # revisar pq creo que no es 1-, era 1- al estar normalizada
#dataset['masculinity_40_inv'] = 1.00 - dataset['masculinity_40']
#dataset['masculinity_41_inv'] = 1.00 - dataset['masculinity_41']

#dataset["CW_var"] = (0.2*dataset["masculinity_46_inv"]) + (0.2*dataset["masculinity_42"]) + (0.2*dataset["masculinity_45"]) + (0.2*dataset["masculinity_1"]) + (0.2*dataset["masculinity_44"])
# hacemos tambien el AA aunque la variable de 41 esta mal
#dataset["AA_var"] = (0.2*dataset["masculinity_40_inv"]) + (0.2*dataset["masculinity_41_inv"]) + (0.2*dataset["masculinity_39"]) + (0.2*dataset["masculinity_43"]) + (0.2*dataset["masculinity_47"])
#dataset["persistency"] = (0.2*dataset["masculinity_39"]) + (0.2*dataset["masculinity_43"])

# creo mi female como yo creo que deberia ser: junto los NA con los females (en vez de con los males)
dataset['female_created'] = np.where(dataset['female2'].isin([1.0, np.nan]), 1.0, 0.0)

# voy a generar un medidor de influence in general de society
dataset["influence_society"] = (0.33*dataset["influencegeneral_6"]) + (0.33*dataset["influencegeneral_7"]) + (0.33*dataset["influencegeneral_8"])

# armamos tambien una escala de world issues, una conjunta de todas y una donde migracion fue la mas alta
columns_wi = ["worldissues_6", "worldissues_1", "worldissues_2", "worldissues_3", "worldissues_4", "worldissues_5", "worldissues_7", "worldissues_8"]
dataset['migration_importance'] = dataset.apply(lambda row: 1 if row['worldissues_6'] == row[columns_wi].max() else 0, axis=1)
# los 6 y 7 veo que su distribucion es deeeeeemasiado diferente al resto
dataset["world_issues"] = (0.125*dataset["worldissues_1"]) + (0.125*dataset["worldissues_2"]) + (0.125*dataset["worldissues_3"]) + (0.125*dataset["worldissues_4"]) + (0.125*dataset["worldissues_5"]) + (0.125*dataset["worldissues_6"]) + (0.125*dataset["worldissues_7"]) + (0.125*dataset["worldissues_8"])


# data_1['extremist'] = np.where(data_1['climatedesobedience_1'].isin([0.0, 0.1, 0.9, 1.0]), 1, 0)
dataset["policy_preferences"] = (0.20*dataset["climatedesobedience_1"]) + (0.20*dataset["carbontax1"]) + (0.20*dataset["cartax_w1"]) + (0.20*dataset["meattax_w1"]) + (0.20*dataset["housingtax_w1"])

# quitare "renewenergy" y agregare longlisbeh_w1_*  como medidor de acciones verdes
dataset["longlistbeh_score"] = (0.1*dataset["longlistbeh_w1_1"]) + (0.1*dataset["longlistbeh_w1_2"]) + (0.1*dataset["longlistbeh_w1_3"]) + (0.1*dataset["longlistbeh_w1_4"]) + (0.1*dataset["longlistbeh_w1_5"]) + (0.1*dataset["longlistbeh_w1_6"]) + (0.1*dataset["longlistbeh_w1_7"]) + (0.1*dataset["longlistbeh_w1_8"]) + (0.1*dataset["longlistbeh_w1_9"]) + (0.1*dataset["longlistbeh_w1_10"])

# vendria bien quitar todas las columnas que no se necesitan ?
dataset.to_csv("data_cleaned.csv")
