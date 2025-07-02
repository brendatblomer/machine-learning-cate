import numpy as np
from sklearn import tree
import pandas as pd
from econml.dml import CausalForestDML as CausalForest
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from econml.cate_interpreter import SingleTreeCateInterpreter
from econml.dml import LinearDML
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from econml.metalearners import TLearner
from econml.dml import DML
from econml.validate.drtester import DRTester



def generating_causal_forest(model_outcome, model_treatment, n_trees, min_sample_per_leaf, max_samples, random_seed, outcome, treatment, x_cov):
    model = CausalForest(
        model_y=model_outcome,
        model_t=model_treatment,
        n_estimators=n_trees,
        discrete_treatment=True,
        criterion="het",
        min_samples_leaf=min_sample_per_leaf,
        max_samples=max_samples, #QUE ERA ESTO? Al saber, cambiar arriba el nombre a algo logico
        random_state=random_seed
    ).fit(outcome, treatment, X=x_cov)
    return model


# ESTE SE IRIA A OTRO PY o no?
def graph_distribution_indiv_treatment(model, x_cov, n_question):
   ate_cf = model.ate(x_cov)
   cate = model.effect(x_cov)
   sns.histplot(cate, bins=15, color='lightgray', edgecolor='black')
   plt.title(f'Question {n_question}: Histogram of estimated CATE')
   plt.xlabel("Estimated CATE")
   plt.ylabel("Frequency")
   plt.axvline(ate_cf, color="green", linestyle="--", label=f'ATE={ate_cf:.2f}')
   plt.legend()
   plt.savefig(f'bld/predicted_cate_question{n_question}.png', dpi=300)
   plt.close()


# ESTE IRIA CON LS GRFICAS FINALES TB o no?
# no se si estos se deberian hacer con train solo o con el total, ahora se andan haciendo con el total
def graph_importance_variables(model, x_cov, question):
    importances = model.feature_importances_
    features = x_cov.columns
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.savefig(f'bld/variable_importance_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()

def graph_representative_tree(model, x_cov, covariates_names, question, tree_depth, min_per_leaf, random_seed):
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=tree_depth, min_samples_leaf=min_per_leaf, random_state=random_seed)
    intrp.interpret(model, x_cov)
    intrp.plot(feature_names=covariates_names)
    plt.savefig(f'bld/tree_question{question}.png', dpi=300, bbox_inches='tight')
    plt.close()

# esta funcion esta fatal, la hice rapido para tener resultados, luego si eso corregir
def printing_some_characteristics(df, model, x_cov, question):
    cate = model.effect(x_cov)
    cate_mean = np.mean(cate)
    grupo_bajo = cate <= cate_mean
    grupo_alto = cate > cate_mean
    grupo_bajo_df = df[grupo_bajo]
    grupo_alto_df = df[grupo_alto]

    print(f'Question{question}')
    print("Big 5")
    print("Mean openness for group 1:", round(grupo_bajo_df["openness"].mean(), 2))
    print("Mean openness for group 2:", round(grupo_alto_df["openness"].mean(), 2))
    print("Mean conscientiousness for group 1:", round(grupo_bajo_df["conscientiousness"].mean(), 2))
    print("Mean conscientiousness for group 2:", round(grupo_alto_df["conscientiousness"].mean(), 2))
    print("Mean extraversion for group 1:", round(grupo_bajo_df["extraversion"].mean(), 2))
    print("Mean extraversion for group 2:", round(grupo_alto_df["extraversion"].mean(), 2))
    print("Mean agreeableness for group 1:", round(grupo_bajo_df["agreeableness"].mean(), 2))
    print("Mean agreeableness for group 2:", round(grupo_alto_df["agreeableness"].mean(), 2))
    print("Mean neuroticism for group 1:", round(grupo_bajo_df["neuroticism"].mean(), 2))
    print("Mean neuroticism for group 2:", round(grupo_alto_df["neuroticism"].mean(), 2))
    print("Not Big 5")
    print("Mean trust_in_science for group 1:", round(grupo_bajo_df["trust_in_science"].mean(), 2))
    print("Mean trust_in_science for group 2:", round(grupo_alto_df["trust_in_science"].mean(), 2))
    print("Mean policy_preferences for group 1:", round(grupo_bajo_df["policy_preferences"].mean(), 2))
    print("Mean policy_preferences for group 2:", round(grupo_alto_df["policy_preferences"].mean(), 2))
    print("Mean age for group 1:", round(grupo_bajo_df["age"].mean(), 2))
    print("Mean age for group 2:", round(grupo_alto_df["age"].mean(), 2))
    print("Mean female for group 1:", round(grupo_bajo_df["female_created"].mean(), 2))
    print("Mean female for group 2:", round(grupo_alto_df["female_created"].mean(), 2))
    print("Mean education for group 1:", round(grupo_bajo_df["education"].mean(), 2))
    print("Mean education for group 2:", round(grupo_alto_df["education"].mean(), 2))

    