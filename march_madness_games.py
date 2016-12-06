import pandas as pd
import numpy as np
import march_madness_classes as mmc
import march_madness_models as mmm

# This module contains functions that are used to generate a list of playoff games
# They create a list of games with the appropriate list of variables 

# generate_multiple_years_of_games is the primary function used form this module, the others are helper functions
# pass in list of the years you want
# pass in the seed list (all of the years)
# pass in the games (which has team 1 v team 2) -- actual results of the tournament
# pass in the names of the predictors
# pass in the dataframes (list of matricies with the years and the teams as axes)
# returns 2 DataFrames (pred, resp) --> X values (difference between min index team x and max index team x) , y values (indicator that min_index team won.

# note: if we pass the include_scoring_dif flag - the y_value returned is the score difference in the game
### PRIMAR
def generate_multiple_years_of_games(years, 
                                     seed_list_arr, 
                                     slot_list_arr, 
                                     tourney_data_arr, 
                                     predictors, 
                                     predictor_dfs, 
                                     scoring_dif=False):
    min_year = 1985

    preds = pd.DataFrame({})
    resps = np.array([])
    
    # iterate years
    for year in years:
        year_index = int(year) - min_year
        # generate 1 year of data
        pred_df, resp_arr = generate_single_year_of_games(year, 
                                                          seed_list_arr[year_index], 
                                                          slot_list_arr[year_index], 
                                                          tourney_data_arr[year_index],
                                                          predictors,
                                                          predictor_dfs,
                                                          scoring_dif=scoring_dif)
      
        # add to list we are keeping 
        preds = pd.concat([preds, pred_df])
        resps = np.concatenate((resps, resp_arr))
     
    # format 
    preds = preds.reset_index(drop=True)
    if scoring_dif:
        resp_df = pd.DataFrame(data=resps, columns=["min_index_x"])
    else:
        resp_df = pd.DataFrame(data=resps, columns=["min_index_win"])
        
    return preds, resp_df
        
    
### HELPER FUNCTIONS ####################################################################    
 
# extracts the value of a team's x value in a given year from the matrix    
def get_predictor(team_id, year, df):
    return df.loc[year, str(team_id)]

# extracts the difference of 2 teams' x values in a given year from the matrix    
def get_predictor_dif(team_id_1, team_id_2, year, df):
    return df.loc[year, str(team_id_1)] - df.loc[year, str(team_id_2)]

# create 1 rows of a team's X values in a given year
def get_predictors(team_id, year, df_arr):
    row = np.zeros(len(df_arr))
    i = 0
    for df in df_arr:
        # extract predictor i 
        row[i] = get_predictor(team_id, year, df)
        i = i + 1
    return row

# create 1 rows of a the differences between 2 teams' X values in a given year
def get_predictors_dif(team_id_1, team_id_2, year, df_arr):
    row = np.zeros(len(df_arr))
    i = 0
    for df in df_arr:
        # extract difference i
        row[i] = float(get_predictor_dif(team_id_1, team_id_2, year, df))
        i = i + 1
    return row

# function to extract the y_values of the team with the min index winning
def extract_response(tourney_game_df, scoring_dif=False):
    # quantitative response
    if scoring_dif:
        min_index_x = np.zeros(tourney_game_df.shape[0])
    
    # categorical response
    min_index_win = np.zeros(tourney_game_df.shape[0])
    
    i = 0
    for index, game in tourney_game_df.iterrows():
        # qualitative response
        if int(game["Prediction"]) == min(int(game["Strongseed Team"]), int(game["Weakseed Team"])):
            min_index_win[i] = 1 
        
        # quantitative resposne
        if scoring_dif:
            if min_index_win[i] == 1:
                min_index_x[i] = int(game["Prediction Scoring Dif"])
            else:
                min_index_x[i] = -1 * int(game["Prediction Scoring Dif"])
        
        i = i + 1
      
    # quantitative or qualitative predictor
    if scoring_dif:
        return min_index_x
    else:
        return min_index_win

# generates dataframe of the predictors
def extract_predictors(tourney_game_df, predictor_list, predictor_dfs, year):
    # buffer to hold our values
    pred_matrix = np.zeros((tourney_game_df.shape[0], len(predictor_list)))
    
    # fill predictor matrix
    for i in range(tourney_game_df.shape[0]):   
        # min and max index teams
        min_index_team = min(int(tourney_game_df.loc[i, "Strongseed Team"]), int(tourney_game_df.loc[i, "Weakseed Team"]))
        max_index_team = max(int(tourney_game_df.loc[i, "Strongseed Team"]), int(tourney_game_df.loc[i, "Weakseed Team"]))                  
        # fill matrix
        pred_matrix[i,  0] = min_index_team
        pred_matrix[i,  1] = max_index_team
        pred_matrix[i, 2:] = get_predictors_dif(min_index_team, max_index_team, year, predictor_dfs)

    # gen dataframe                       
    pred_df = pd.DataFrame(data = pred_matrix, columns = predictor_list)
                           
    return pred_df

# get the results of a tournament  from the raw data
def get_tourney_results(seeds, slots, raw_data, scoring_dif=False):
    tourney = mmc.Tournament(seeds, 
                         slots, 
                         mmm.ActualTournament(raw_data, include_scoring_dif=scoring_dif), 
                         include_scoring_dif=scoring_dif)
    
    return tourney.entire_bracket

# get single years worth of games
def generate_single_year_of_games(year, seed_list, slot_list, tourney_data, predictors, predictor_dfs, scoring_dif=False):
    # get results of the games
    tourney_results = get_tourney_results(seed_list, slot_list, tourney_data, scoring_dif=scoring_dif)
    
    # get predictors
    pred_df = extract_predictors(tourney_results, predictors, predictor_dfs, year)
    
    # get response
    resp_arr = extract_response(tourney_results, scoring_dif=scoring_dif)
    
    return pred_df, resp_arr

# filtration function
def filter_season(data, season):
    return data[data["Season"] == season]
    
def filter_into_seasons(data):
    # buffer to hold list of seasons
    season_arr = []
    
    # min and max
    max_season = data["Season"].max()
    min_season = data["Season"].min()
    
    # filter
    for season in range(min_season, max_season + 1):
        season_arr.append(filter_season(data, season))
        
    return season_arr