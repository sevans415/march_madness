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

        
# MODEL PREDICTOR ------------------------------------------------------------------------

# however, you are able to do some biasing of the predictions
# higher_seed_bias=False      ----> if True, will predict higher seed (upset) with probability p + higher_seed_bias_delta
# higher_seed_bias_delta=.05  ----> tuned to how much bias we want towards upsets/top seed winning
        
# we are also able to do "cooling" of our model ----> cooling cooresponds to changing the bias depening on the round
# pass in a dict of the form {1:r1, 2:r2, 3:r3, 4:r4, 5:r5, 6:r6}
# when we update bias the probability, we do p + higher_seed_bias_delta * r_i depending on the round

# predictor using some model for predicting head to head games
class ModelPredictor(object):
    # init function
    def __init__(self, 
                 model, 
                 scaler, 
                 dfs_arr, 
                 year, 
                 seeds_df=None,
                 simulation=False,
                 higher_seed_bias=False,
                 higher_seed_bias_delta=.05,
                 cooling=None
                 ):
        
        self.model = model
        self.dfs_arr = dfs_arr
        self.year = year
        self.simulation = simulation
        self.scaler = scaler
        self.seeds_df = seeds_df
        self.higher_seed_bias = higher_seed_bias
        self.higher_seed_bias_delta = higher_seed_bias_delta
        self.cooling=cooling
        
        # used to check what round we are in
        self.game_count = 0
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
        
        # predict probability team 1 win under model
        p_hat = self.model.predict_proba(self.scaler.transform(row.reshape(1,-1)))[0,1]
        
        # if bias towards the lower seed
        if self.higher_seed_bias:
            if self.seeds_df is None:
                print "you must pass in the seed dataframe to use higher seed bias"
                return
            else:
                # check which round we are in
                if self.game_count < 32:
                    cur_round = 1
                elif self.game_count < 32 + 16:
                    cur_round = 2
                elif self.game_count < 32 + 16 + 8:
                    cur_round = 3
                elif self.game_count < 32 + 16 + 8 + 4:
                    cur_round = 4
                elif self.game_count < 32 + 16 + 8 + 4 + 2:
                    cur_round = 5
                elif self.game_count < 32 + 16 + 8 + 4 + 2 + 1:
                    cur_round = 6
                else:
                    print "issue with game count"
                    return 
                
                # if there is a "cooling" (i.e. change probability of upset given round)
                if self.cooling is None:
                    # update the delta
                    cooling_factor = 1
                else:
                    cooling_factor = self.cooling.get(cur_round)
                
                # get the seeds to see which team is the underdog
                min_index_seed_str = self.seeds_df.loc[self.seeds_df["Team"] == min_index_team, "Seed"].values[0]
                max_index_seed_str = self.seeds_df.loc[self.seeds_df["Team"] == max_index_team, "Seed"].values[0]
                
                # convert the seeds to ints for comparieson
                min_index_seed = int(min_index_seed_str[1:3])
                max_index_seed = int(max_index_seed_str[1:3])  
                
                # confirm not a play in game, iterate
                if self.game_count == 0 and len(min_index_seed_str) == 4 and len(max_index_seed_str) == 4:
                    # dont iterate
                    self.game_count = self.game_count
                else:
                    # iterate if not a play in game
                    self.game_count = self.game_count + 1
                
                # Update p_hat given the underdog status on one of the teams
                if min_index_seed < max_index_seed:
                    # update p_hat to predict max_index more often
                    p_hat = p_hat - self.higher_seed_bias_delta * cooling_factor
                
                # if max index team is the lower seed
                elif max_index_seed < min_index_seed:
                    # update p_hat to predict min_index more often
                    p_hat = p_hat + self.higher_seed_bias_delta * cooling_factor
                
                # if they are the same seed, leave it alone
                else:
                    p_hat = p_hat
        
        # if simulation, return min_index team with prob p_hat
        if self.simulation:
            random_unif = rand.uniform(0,1)
     
            # return min_index with probability p_hat
            if random_unif <= p_hat:
                return min_index_team
            else:
                return max_index_team
        
        # if not a simulation, return the prediction of the model (or adjusted model)
        else:
            if p_hat > .5:
                return min_index_team
            else:
                return max_index_team    
# 
# EXPECTED POINTS PREDICTOR ---------------------------------------------------------------------------------------------------
# predict based on expected points from the simulation   
# looks up the expected number of points 2 teams will score,
# predicts arg_max(E[points_1], E[points_2])

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