import pandas as pd 
from pathlib import Path
import statsmodels.api as sm


def normal_regression(df, treatment, outcome, question):
    
    rhs = sm.add_constant(df[treatment])
    model = sm.OLS(df[outcome], rhs)
    results = model.fit(cov_type="HC3")
    
    # realmnte seria un return results y luego la tabla summary se genera en otro .py
    print(f'Question {question}: {results.summary()}')

def normal_regression_with_controls(df, treatment, outcome, controls, question):
    
    df_copy = df.copy()
    for i in controls:
        rhs = sm.add_constant(df_copy[[treatment, i]])
        model = sm.OLS(df_copy[outcome], rhs)
        results = model.fit(cov_type="HC3")
        
        # realmnte seria un return results y luego la tabla summary se genera en otro .py
        print(f'Question {question} with covariate {i}: {results.summary()}')

def normal_regression_with_interactions(df, treatment, outcome, covariables, question):
    
    df_copy = df.copy()
    for i in covariables:
        interaction_term = df_copy[treatment]*df_copy[i]
        df_copy["interaction"] = interaction_term
        rhs = sm.add_constant(df_copy[[treatment, "interaction", i]])
        model = sm.OLS(df_copy[outcome], rhs)
        results = model.fit(cov_type="HC3")
        
        # realmnte seria un return results y luego la tabla summary se genera en otro .py
        print(f'Question {question} with covariate {i}: {results.summary()}')
            
        
if __name__ == "__main__":

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_with_new_variables_from_original.csv") 

    data_done = data_ready.copy()
    data_done = data_done[data_done['gender'].notna()]


    y_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
    d_list =["Q1_1_treat", "Q1_2_treat", "Q2_1_treat", "Q2_2_treat"]
    covariates_names = ["employment", "education", "age", "female", "houseowner", "financialwellbeing", "optimism_bias", "scale_sufficiency", "openness", "conscientiousness",
                "extraversion", "agreeableness", "neuroticism", "rationality_score", "growthmind",
                'Positive_Reciprocity', 'Negative_Reciprocity', 'Altruism', 'Trust', 'Risk_Preferences', 'Time_Preferences', "Social_Anxiety",
                "Public_SelfConsciousness", "Private_SelfConsciousness", "ProcrastinationExAnte", "ProcrastinationExPost", "empathic_concern_score", "perspective_taking_score"]
    controls_list = ["female3", "education", "financialwellbeing"]


    for d, y, q in zip(d_list, y_list, range(4)):
        normal_regression(data_done, d, y, q+1)

    for d, y, q in zip(d_list, y_list, range(4)):
        normal_regression_with_interactions(data_done, d, y, covariates_names, q+1) 
        
    for d, y, q in zip(d_list, y_list, range(4)):
        normal_regression_with_controls(data_done, d, y, controls_list, q+1) 