import pandas as pd
import numpy as np

# tournament class
class Tournament(object):
    # init function
    def __init__(self, seeds, slots, model):
        self.seeds = seeds
        self.slots = slots
        self.model = model
        
        games = []
       
        round_1_slots = slots[slots["Slot"].str.contains("R1")]
        
        # generate first round games
        for index, slot in round_1_slots.iloc[:32, :].iterrows():
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"] 

            # teams
            team_1 = seeds.loc[seeds["Seed"] == team_1_seed, "Team"].values[0]
            team_2 = seeds.loc[seeds["Seed"] == team_2_seed, "Team"].values[0]
            
            # predict winner under our model
            cur_game_pred_team = self.model.predict(team_1, team_2)

            # predict winner seed under our model
            if cur_game_pred_team ==  team_1:
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            # append games
            games.append((slot["Slot"], 
                          team_1_seed, 
                          team_1, 
                          team_2_seed, 
                          team_2, 
                          cur_game_pred_team, 
                          cur_game_pred_seed))

        # convert to datafram
        self.round_1_df = pd.DataFrame(data=np.array(games), 
                                       columns=["Slot", 
                                                "Strongseed", 
                                                "Strongseed Team", 
                                                "Weakseed", 
                                                "Weekseed Team", 
                                                "Prediction", 
                                                "Prediction Seed"])
        
        self.round_2_df = pd.DataFrame()
        self.round_3_df = pd.DataFrame()
        self.round_4_df = pd.DataFrame()
        self.round_5_df = pd.DataFrame()
        self.round_6_df = pd.DataFrame()
            
    
    # run a particular round
    def generate_round_games(self, round_n):
        games = []
        
        n_games_in_prev_round = {2: 32, 3: 16, 4: 8, 5:4, 6:2}
        
        prev_round_df_dic = {2: self.round_1_df,
                         3: self.round_2_df,
                         4: self.round_3_df,
                         5: self.round_4_df,
                         6: self.round_5_df}
    
        # slots of previous round
        round_n_slots = self.slots[self.slots["Slot"].str.contains("R{}".format(round_n))]
        
        # prev round df
        prev_round_df = prev_round_df_dic.get(round_n)
        
        # generate first round games
        for index, slot in round_n_slots.iloc[:n_games_in_prev_round.get(round_n), :].iterrows():
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"]
            
            # teams
            team_1 = prev_round_df.loc[prev_round_df["Slot"] == team_1_seed, "Prediction"].values[0]
            team_2 = prev_round_df.loc[prev_round_df["Slot"] == team_2_seed, "Prediction"].values[0]

            # predict winner under our model
            cur_game_pred_team = self.model.predict(team_1, team_2)

            # predict winner seed under our model
            if int(cur_game_pred_team) ==  int(team_1):
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            # append games
            games.append((slot["Slot"], 
                          team_1_seed, 
                          team_1, 
                          team_2_seed, 
                          team_2, 
                          cur_game_pred_team, 
                          cur_game_pred_seed))

        # convert to datafram
        cur_round_df = pd.DataFrame(data=np.array(games), 
                                       columns=["Slot", 
                                                "Strongseed", 
                                                "Strongseed Team", 
                                                "Weakseed", 
                                                "Weekseed Team", 
                                                "Prediction", 
                                                "Prediction Seed"])
        
        if round_n == 2:
            self.round_2_df = cur_round_df
        elif round_n == 3:
            self.round_3_df = cur_round_df
        elif round_n == 4:
            self.round_4_df = cur_round_df
        elif round_n == 5:
            self.round_5_df = cur_round_df
        elif round_n == 6:
            self.round_6_df = cur_round_df  
            
     
    # simulate an entire tournament
    def simulate_tournament(self):  
        for n in range(2,7):
            self.generate_round_games(n)
            
        self.entire_bracket = pd.concat([self.round_1_df, 
                                              self.round_2_df,
                                              self.round_3_df,
                                              self.round_4_df,
                                              self.round_5_df,
                                              self.round_6_df])
        self.entire_bracket.reset_index(inplace = True, drop=True)
        
    # score model vs true results
    def score_model(self, actual_results, print_res=False, scoring='ESPN'):
        # extract the df from the actual results tournament
        actual_results_df      = actual_results.entire_bracket
        actual_results_round_1 = actual_results.round_1_df
        actual_results_round_2 = actual_results.round_2_df
        actual_results_round_3 = actual_results.round_3_df
        actual_results_round_4 = actual_results.round_4_df
        actual_results_round_5 = actual_results.round_5_df
        actual_results_round_6 = actual_results.round_6_df
        
        # count correct answers
        tot_correct = actual_results_df[actual_results_df["Prediction"] == self.entire_bracket["Prediction"]].shape[0]
        r_1_correct = actual_results_round_1[actual_results_round_1["Prediction"] == self.round_1_df["Prediction"]].shape[0]
        r_2_correct = actual_results_round_2[actual_results_round_2["Prediction"] == self.round_2_df["Prediction"]].shape[0]
        r_3_correct = actual_results_round_3[actual_results_round_3["Prediction"] == self.round_3_df["Prediction"]].shape[0]
        r_4_correct = actual_results_round_4[actual_results_round_4["Prediction"] == self.round_4_df["Prediction"]].shape[0]
        r_5_correct = actual_results_round_5[actual_results_round_5["Prediction"] == self.round_5_df["Prediction"]].shape[0]  
        r_6_correct = actual_results_round_6[actual_results_round_6["Prediction"] == self.round_6_df["Prediction"]].shape[0]
         
        # total games
        tot_games = actual_results_df.shape[0]
        r_1_games = actual_results_round_1.shape[0]
        r_2_games = actual_results_round_2.shape[0]
        r_3_games = actual_results_round_3.shape[0]
        r_4_games = actual_results_round_4.shape[0]
        r_5_games = actual_results_round_5.shape[0]
        r_6_games = actual_results_round_6.shape[0]
        
        # accuracy
        tot_accuracy = float(tot_correct) / tot_games      
        r_1_accuracy = float(r_1_correct) / r_1_games
        r_2_accuracy = float(r_2_correct) / r_2_games
        r_3_accuracy = float(r_3_correct) / r_3_games
        r_4_accuracy = float(r_4_correct) / r_4_games
        r_5_accuracy = float(r_5_correct) / r_5_games
        r_6_accuracy = float(r_6_correct) / r_6_games
        
        # score depending on scoring system
        if scoring=='ESPN':
            r_1_points = r_1_correct * 10
            r_2_points = r_2_correct * 20
            r_3_points = r_3_correct * 40
            r_4_points = r_4_correct * 80
            r_5_points = r_5_correct * 160
            r_6_points = r_6_correct * 320
        
        tot_points = r_1_points + r_2_points + r_3_points + r_4_points + r_5_points + r_6_points
        
        # if we want to print the results
        if print_res:
            print "Total Points  : {}\n".format(tot_points)
            print "Total Accuracy: {} / {} = {}".format(tot_correct, tot_games, tot_accuracy)
            print "R1    Accuracy: {} / {} = {}".format(r_1_correct, r_1_games, r_1_accuracy)
            print "R2    Accuracy: {} / {} = {}".format(r_2_correct, r_2_games, r_2_accuracy)
            print "R3    Accuracy: {} / {} = {}".format(r_3_correct, r_3_games, r_3_accuracy)
            print "R4    Accuracy: {} / {} = {}".format(r_4_correct, r_4_games, r_4_accuracy)
            print "R5    Accuracy: {} / {} = {}".format(r_5_correct, r_5_games, r_5_accuracy)
            print "R6    Accuracy: {} / {} = {}".format(r_6_correct, r_6_games, r_6_accuracy)

        return (tot_points, tot_accuracy)
 
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
    def __init__(self, data):
        self.tourney = data
        return
    
    def predict(self, team_1, team_2):
        game_played_team_1_win = self.tourney[(self.tourney["Wteam"] == int(team_1)) & (self.tourney["Lteam"] == int(team_2))]
        game_played_team_2_win = self.tourney[(self.tourney["Lteam"] == int(team_1)) & (self.tourney["Wteam"] == int(team_2))]
        
        if game_played_team_1_win.shape[0] == 1:
            return team_1
        elif game_played_team_2_win.shape[0] == 1:
            return team_2
        else:
            print "Error"
            return -1
 
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
