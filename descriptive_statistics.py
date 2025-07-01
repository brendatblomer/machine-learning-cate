import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import linear_model

def graph_distributions_treated_vs_not(df, covariate, treated):
    """ 
    Graphs a distrubution of the treated and non treated given a covariate.
    Helps to see if experiment is random.
    """

    sns.histplot(data=df, x=covariate, hue=treated)
    plt.title(f"Distribution of {covariate} given treatment")

def graph_soft_distributions_treated_vs_not(df, covariate, treated):
    """ 
    Graphs a soft/aprox distrubution of the treated and non treated given a covariate.
    Helps to see if experiment is random.
    """

    sns.kdeplot(data=df, x=covariate, hue=treated, common_norm=False)
    plt.title(f"Distribution of {covariate} given treatment")

def graph_distributions_boxplot(df, covariate, outcome, treated):
    """
    Graphs boxplots given a covariate.
    """

    sns.boxplot(data=df, x=outcome, y=covariate, hue=treated)
    plt.title(f'Distribution of {covariate} given {outcome}')

def graph_distributions_violinplot(df, covariate, outcome, treated):
    """
    Graphs violinplots given a covariate.
    """

    sns.violinplot(data=df, x=outcome, y=covariate, hue=treated, split=True, inner="quart")
    plt.title(f'Distribution of {covariate} given {outcome}')

#este se tarda demasiadon (mirar pq)
def graph_heatmap_corr_cov(df, covariates):
    """
    Graphs a heatmap that shows the correlations between variables selected.
    """

    corr = df[covariates].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, annot_kws={"size":5})
    plt.title("Correlation Matrix of All Covariates")
    


if __name__ == "__main__":

    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_normalized.csv") 

    data_done = data_ready.copy()
    #data_done = data_done[data_done['gender'].notna()] # MIRAR PORQUE NO QUIERO ELIMINAR A ESTA GENTE...

    y_list = ["Q1_1", "Q1_2", "Q2_1", "Q2_2"]
    d_list =["Q1_1_treat", "Q1_2_treat", "Q2_1_treat", "Q2_2_treat"]

    list_of_cov = ["age", "education", "conscientiousness", "female", "extraversion"]
    list_of_all_cov = [
        "employment", "education", "age", 
        "female", "houseowner", "financialwellbeing", 
        "optimism_bias", "openness", "conscientiousness", 
        "extraversion", "agreeableness", "neuroticism", 
        "rationality_score", "growthmind", "lambda_risky", "Social_Anxiety", 
        "Public_SelfConsciousness", "Private_SelfConsciousness", 
        "ProcrastinationExAnte", "ProcrastinationExPost", 
        "empathic_concern_score", "perspective_taking_score"]

    for i in list_of_cov:
        plt.figure()
        graph_distributions_treated_vs_not(data_done, i, "Q1_1_treat")
        plt.tight_layout()
        plt.savefig(f"bld/graph_hist_{i}.png", dpi=300)
        plt.close()

    for i in list_of_cov:
        plt.figure()
        graph_soft_distributions_treated_vs_not(data_done, i, "Q1_1_treat")
        plt.tight_layout()
        plt.savefig(f"bld/graph_kde_{i}.png", dpi=300)
        plt.close()

    plt.figure()
    graph_heatmap_corr_cov(data_done, list_of_all_cov)
    plt.tight_layout()
    plt.savefig("bld/heatmap_covariates.png", dpi=300)
    plt.close()

    # mirando si hace sentido sumarlas, parece que si
    #empathy_list = ['empathy_1', 'empathy_2', 'empathy_3', 'empathy_4']
    #graph_heatmap_corr_cov(data_done, empathy_list)

    #big_five_list =['bigfive_1', 'bigfive_2', 'bigfive_3', 'bigfive_4', 'bigfive_5', 'bigfive_6', 'bigfive_7', 'bigfive_8', 'bigfive_9', 'bigfive_10']
