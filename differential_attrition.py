import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt



def dif_attrition_categorical_var(df, finished_var, var):
    
    tabla = pd.crosstab(df[var], df[finished_var])
    chi2, p_val, dof, expected = chi2_contingency(tabla)
    return p_val

#por si se quiere un test mas exacto
def p_fisher_for_binary_var(df, finished_var, var):
    
    tabla = pd.crosstab(df[var], df[finished_var])
    _, p_fisher = fisher_exact(tabla)
    return p_fisher
       

if __name__ == "__main__":

    
    SRC = Path(__file__).parent.resolve()
    data = SRC / "waves123_augmented_consent.dta"
    waves_data = pd.read_stata(data, convert_categoricals=False)

    data_original = waves_data.copy()

    categorical_var_list = ["female3", "education","age", "financialwellbeing", "urban"]
    finished_wave_list = ['Finished_w2', 'Finished_w3']

    # for the test we need nan to become 0 because there are no 0 but nan
    data_original['Finished_w3'] = data_original['Finished_w3'].fillna(0)
    data_original['Finished_w2'] = data_original['Finished_w2'].fillna(0)

    for i in finished_wave_list:
        print(f'{i}:')
        for j in categorical_var_list:
            dif_attrition_categorical_var(data_original, i, j)
        #graph_distributions_treated_vs_not(data_original, i, j)

#completaron = data_original[data_original['Finished_w3'] == 1]
#no_completaron = data_original[data_original['Finished_w3'] == 0]

# Comparar edad (numérica)
#t_stat, p_val = ttest_ind(completaron['age'], no_completaron['age'])
#print(f"T-test age: p = {p_val:.4f}")
