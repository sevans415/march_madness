import pandas as pd
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler
import numpy as np

import march_madness_games as mmg

# This module contains functions that are used to cross validate the model and split iinto train and test sets
# They create a list of games with the appropriate list of variables 


# train_test_split takes in a window (number of years to include in training set) and the year that we want to test on and returns 
# a x_train, y_train, x_test, and y_test. 


def train_test_split(window, test_yr, seeds_arr, slots_arr, tourney_arr, column_names, predictor_dfs):
    x_train, y_train = mmg.generate_multiple_years_of_games(range(test_yr - window, test_yr),
                                     seeds_arr, 
                                     slots_arr, 
                                     tourney_arr, 
                                     column_names, 
                                     predictor_dfs
                                     )

    x_test, y_test = mmg.generate_multiple_years_of_games([test_yr],
                                     seeds_arr, 
                                     slots_arr, 
                                     tourney_arr, 
                                     column_names, 
                                     predictor_dfs
                                     )

    return x_train, y_train, x_test, y_test



# cross_val_c takes in a window (number of years to include in training set) and outputs the scores for each value of C for each    # test year. 

def cross_val_c(window, seeds_arr, slots_arr, tourney_arr, column_names, predictor_dfs):
    col_names = ['0.0001', '0.001', '0.01', '.1', '1', '10', '100']
    test_yr_range = range(2003 + window, 2016)
    scores = pd.DataFrame(index=test_yr_range, columns = col_names)
    
    c_vals = range(-4, 3)

    for yr in test_yr_range:
        x_train, y_train, x_test, y_test = train_test_split(window, 
                                                            yr, 
                                                            seeds_arr, 
                                                            slots_arr, 
                                                            tourney_arr, 
                                                            column_names, 
                                                            predictor_dfs)
        scaler = StandardScaler().fit(x_train.iloc[:, 2:])
        for c in c_vals:
            model = LogReg(C = 10**c)
            model.fit(scaler.transform(x_train.iloc[:, 2:]), y_train.values.T[0])
            scores.ix[yr, c+2] = model.score(scaler.transform(x_test.iloc[:, 2:]), y_test.values.T[0])
    return scores
