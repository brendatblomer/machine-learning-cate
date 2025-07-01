import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def placebo_testing(df, treatment, outcome, question):

    real_model = LinearRegression().fit(df[[treatment]], df[outcome])
    real_ate = real_model.coef_[0]
    
    np.random.seed(23) # ver la otra forma de poner una seed pq esta puede que cambien ligeramente los resultados
    placebo_ates = []
    data_temp = df.copy()
    
    for i in range(1000):
        data_temp['d_placebo'] = np.random.permutation(df[treatment])  # Aleatoriza el tratamiento
        model = LinearRegression().fit(data_temp[['d_placebo']], data_temp[outcome])
        placebo_ates.append(model.coef_[0])
    
    p_value = np.mean(np.abs(placebo_ates) >= abs(real_ate))
    text_pvalue = f'Empirical p-value: {p_value:.4f}'

    # esto se pasaria a otra .py luego
    fig = plt.figure()
    plt.hist(placebo_ates, bins=30, color='gold', edgecolor='black', alpha=0.8)
    plt.axvline(real_ate, color='darkblue', linestyle='--', label='Real ATE')
    plt.title(f' Question {question}: ATE Distribution under aleatory treatment Placebo')
    plt.xlabel('ATE (false)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.text(0.5, -0.15, text_pvalue, transform=plt.gca().transAxes,fontsize=8, ha='center', va='top')

    return fig

if __name__ == "__main__":
    
    SRC = Path(__file__).parent.resolve()
    data_ready = pd.read_csv(SRC/"data_normalized.csv") 

    data_done = data_ready.copy()
    #data_done = data_done[data_done['gender'].notna()]

    # estas listas se usan mucho, mover luego a donde estaria SCR
    t_list = ['Q1_1_treat', 'Q1_2_treat', 'Q2_1_treat', 'Q2_2_treat']
    y_list = ['Q1_1', 'Q1_2', 'Q2_1', 'Q2_2']


    for i,j,k in zip(t_list, y_list, range(4)):
        fig = placebo_testing(data_done, i, j, k+1)
        fig.savefig(f"bld/placebo_test_question_{k+1}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)