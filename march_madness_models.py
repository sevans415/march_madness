import pandas as pd
import numpy as np
import march_madness_games as mmg
import random as rand

################# MODELS FOR OUR TOURNAMENTS #####################

# predictor high seed
class BasicPredictor(object):
    # init function
    def __init__(self):
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        return team_1

# actual tournament results
class ActualTournament(object):
    # init function
    def __init__(self, data, include_scoring_dif=False):
        self.include_scoring_dif = include_scoring_dif                                            
        self.tourney = data
        return
    
    def predict(self, team_1, team_2):
        game_played_team_1_win = self.tourney[(self.tourney["Wteam"] == int(team_1)) & (self.tourney["Lteam"] == int(team_2))]
        game_played_team_2_win = self.tourney[(self.tourney["Lteam"] == int(team_1)) & (self.tourney["Wteam"] == int(team_2))]
        
        # extract winner and loser
        if game_played_team_1_win.shape[0] == 1:
            winning_team = team_1
            scoring_dif = game_played_team_1_win["Wscore"] - game_played_team_1_win["Lscore"]                                
        elif game_played_team_2_win.shape[0] == 1:
            winning_team = team_2
            scoring_dif = game_played_team_2_win["Wscore"] - game_played_team_2_win["Lscore"]       
        else:
            print "Error"
            return -1
        
        # return socre and scoring dif if we want                                       
        if self.include_scoring_dif:
            return (winning_team, scoring_dif.values[0])
        else:
            return winning_team                                   
                                                    
# predictor using markov chain stationary distribution
class MarkovPredictor(object):
    # init function
    def __init__(self, data):
        self.data = data
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        team_1 = int(team_1)
        team_2 = int(team_2)
        
        # lookup the pi values in the lookup table
        team_1_pi_i = self.data.loc[self.data["Team"] == team_1, "pi_i"].values[0]
        team_2_pi_i = self.data.loc[self.data["Team"] == team_2, "pi_i"].values[0]
        
        if team_1_pi_i > team_2_pi_i:
            return team_1
        else:
            return team_2

# predictor using some model for predicting head to head games
class ModelPredictor(object):
    # init function
    def __init__(self, model, scaler, dfs_arr, year, simulation=False):
        self.model = model
        self.dfs_arr = dfs_arr
        self.year = year
        self.simulation = simulation
        self.scaler = scaler
        return
    
    # head to head predicitons
    def predict(self, team_1, team_2):
        team_1 = int(team_1)
        team_2 = int(team_2)
        
        # min and max index
        min_index_team = min(team_1, team_2)
        max_index_team = max(team_1, team_2)
        
        # get the x values
        row = mmg.get_predictors_dif(min_index_team, max_index_team, self.year, self.dfs_arr)

        # predict under model
        y_hat = self.model.predict(self.scaler.transform(row.reshape(1,-1)))
        p_hat = self.model.predict_proba(self.scaler.transform(row.reshape(1,-1)))[0,1]

        # if simulation, return min_index team with prob p_hat
        if self.simulation:
            random_unif = rand.uniform(0,1)
     
            # return min_index with probability p_hat
            if random_unif <= p_hat:
                return min_index_team
            else:
                return max_index_team
        
        # if not a simulation, return the prediction of the model
        else:
            if y_hat == 1:
                return min_index_team
            else:
                return max_index_team    
            
# predict based on expected points from the simulation          
class ExpectedPointsPredictor(object):
    # pass in a dataframe with the expected points of each team from a simulation
    def __init__(self, points_df):
        self.points_df = points_df
        return
    
    # predict based on looking up expected points
    def predict(self, team_1, team_2):
     
        team_1_points = self.points_df.loc[self.points_df.index == int(team_1), "pred_points"].values[0]
        team_2_points = self.points_df.loc[self.points_df.index == int(team_2), "pred_points"].values[0]
        
        # predict max(points 1, points 2)
        if team_1_points > team_2_points:
            return team_1
        else:
            return team_2      