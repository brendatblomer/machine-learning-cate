import pandas as pd 
from pathlib import Path
#from sklearn import linear_model

SRC = Path(__file__).parent.resolve()
data_david_path = SRC / "waves123_augmented2.dta"
data_original_path = SRC / "waves123_augmented_consent.dta"


data_ready = pd.read_csv(SRC/"data_with_new_variables_from_original.csv")

data_david = pd.read_stata(data_david_path, convert_categoricals=False) 

data_original = pd.read_stata(data_original_path, convert_categoricals=False) 


def summary_of_datasets(df, name):
    print (f'Dataset:{name}')
    print (f'Number of observations:{len(df)}')

#summary_of_datasets(data_ready, "data_ready")
#summary_of_datasets(data_david, "data_david")
#summary_of_datasets(data_original, "data_original")

# Lets compare dataset original and david

cols_david = set(data_david.columns)
cols_original = set(data_original.columns)

unique_cols_david = cols_david - cols_original
unique_cols_original = cols_original - cols_david
shared_columns_david_original = cols_david & cols_original

#print(len(unique_cols_david))
#print(len(unique_cols_original))
#print(len(shared_columns_david_original))