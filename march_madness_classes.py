import pandas as pd
import numpy as np
import march_madness_models as mmm

### This module contains classes that are used for structuring the tournament 

### CLASSES
# 1) Tournament -> Creates a tournament structure based on the seeds, slots, and model of a given year

# 2) Simulator -> Runs a tournament under some model n times, uses expected points scored as metric to optimize points scored

########################################################################################################

### Tournament Class
### arguments = seeds, slots, and model
### proceeds to generate the entire tournament based on the model that is passed
### creates a dataframe of all games with predictions

### PROPERTIES
# 1) round_i_df --> df of games in round i

# 2) entire_bracket --> df of all games in the tournament

### METHODS
# 1) init --> runs the entire tournament, taking results of prev round 

# 2) score_tournament --> takes in the actual results of the tournament year and scores it according to ESPN

# 3) get_predicted_points --> returns a dataframe of how many points each team scores under the model

class Tournament(object):
    # init function
    def __init__(self, seeds, slots, model, include_scoring_dif=False):
        self.seeds = seeds
        self.slots = slots
        self.model = model
        self.include_scoring_dif = include_scoring_dif
        
        games = []
       
        # slots
        round_1_slots = slots[slots["Slot"].str.contains("R1")]
        
        # generate first round games
        for index, slot in round_1_slots.iloc[:32, :].iterrows():
      
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"] 
            
            # lookup team id
            team_1 = seeds.loc[seeds["Seed"] == team_1_seed, "Team"].values
            team_2 = seeds.loc[seeds["Seed"] == team_2_seed, "Team"].values
            
            # play in game
            if len(team_1) == 0:
                # get seeds 
                team_1_a_seed = team_1_seed + "a"
                team_1_b_seed = team_1_seed + "b"
                
                # lookup team id
                team_1_a = seeds.loc[seeds["Seed"] == team_1_a_seed, "Team"].values[0]
                team_1_b = seeds.loc[seeds["Seed"] == team_1_b_seed, "Team"].values[0]
                
                # predict winner of play ing
                if include_scoring_dif:
                    team_1, x = self.model.predict(team_1_a, team_1_b)
                else:
                    team_1 = self.model.predict(team_1_a, team_1_b)
            
            # not a play in game
            else:
                # extract value
                team_1 = team_1[0]
             
            # play in game
            if len(team_2) == 0:
                # get seeds 
                team_2_a_seed = team_2_seed + "a"
                team_2_b_seed = team_2_seed + "b"
                
                # lookup team id
                team_2_a = seeds.loc[seeds["Seed"] == team_2_a_seed, "Team"].values[0]
                team_2_b = seeds.loc[seeds["Seed"] == team_2_b_seed, "Team"].values[0]
                
                # predict winner of play in
                if include_scoring_dif:
                    team_2, x = self.model.predict(team_2_a, team_2_b)
                else:
                    team_2 = self.model.predict(team_2_a, team_2_b)
                
            # not a play in game
            else:
                # exrtract value
                team_2 = team_2[0]
                
            # predict winner under our model
            if include_scoring_dif:
                cur_game_pred_team, cur_game_pred_scoring_dif = self.model.predict(team_1, team_2)
            else:
                cur_game_pred_team = self.model.predict(team_1, team_2)
            
            # predict winner seed under our model
            if cur_game_pred_team ==  team_1:
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            if self.include_scoring_dif:
                # append games
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed,
                              cur_game_pred_scoring_dif))
            else:
                # append games
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed))

        # convert to dataframe
        if self.include_scoring_dif:
            self.round_1_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed",
                                                    "Prediction Scoring Dif"])
        else:
            self.round_1_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed"])
        
        self.round_2_df = pd.DataFrame()
        self.round_3_df = pd.DataFrame()
        self.round_4_df = pd.DataFrame()
        self.round_5_df = pd.DataFrame()
        self.round_6_df = pd.DataFrame()
        
        # run entire tournament
        self.run_tournament()
            
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
        
        # generate games in a round
        for index, slot in round_n_slots.iloc[:n_games_in_prev_round.get(round_n), :].iterrows():
            # get seeds
            team_1_seed = slot["Strongseed"]
            team_2_seed = slot["Weakseed"]
            
            # teams
            team_1 = prev_round_df.loc[prev_round_df["Slot"] == team_1_seed, "Prediction"].values[0]
            team_2 = prev_round_df.loc[prev_round_df["Slot"] == team_2_seed, "Prediction"].values[0]

            # predict winner under our model
            if self.include_scoring_dif:
                cur_game_pred_team, cur_game_pred_scoring_dif = self.model.predict(team_1, team_2)
            else:
                cur_game_pred_team = self.model.predict(team_1, team_2)                                                                         
            # predict winner seed under our model
            if int(cur_game_pred_team) == int(team_1):
                cur_game_pred_seed = team_1_seed
            else:
                cur_game_pred_seed = team_2_seed

            # append games, include scoring dif if necessary
            if self.include_scoring_dif:
                games.append((slot["Slot"], 
                              team_1_seed, 
                              team_1, 
                              team_2_seed, 
                              team_2, 
                              cur_game_pred_team, 
                              cur_game_pred_seed,
                              cur_game_pred_scoring_dif))
            else:
                games.append((slot["Slot"], 
                          team_1_seed, 
                          team_1, 
                          team_2_seed, 
                          team_2, 
                          cur_game_pred_team, 
                          cur_game_pred_seed))

        # convert to datafram
        if self.include_scoring_dif:
            cur_round_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
                                                    "Prediction", 
                                                    "Prediction Seed",
                                                    "Prediction Scoring Dif"])
        else:
            cur_round_df = pd.DataFrame(data=np.array(games), 
                                           columns=["Slot", 
                                                    "Strongseed", 
                                                    "Strongseed Team", 
                                                    "Weakseed", 
                                                    "Weakseed Team", 
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
    def run_tournament(self):  
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
    def score_tournament(self, actual_results, print_res=False, scoring='ESPN'):
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
    
    # individual team points scored
    def get_predicted_points_for_team(self, team, scoring='ESPN'):
        # counts number of wins in the projected bracket
        team = str(team)
        wins = self.entire_bracket[self.entire_bracket["Prediction"] == team].shape[0]
         
        # TODO: allow for other scoring systems    
        # 10 points for R1, 20 for R2    
        points = 0                           
        for i in range(int(wins)):
            points = points + 10 * 2 ** i
                         
        return points
                                   
    # points scored for all teams
    def get_predicted_points(self, scoring="ESPN"):
        # setup buffers
        teams = self.seeds["Team"]
        points = np.zeros(teams.shape[0])

        i = 0
        for team in teams:
            # get points of team i
            points[i] = self.get_predicted_points_for_team(team)
            i = i + 1
              
        return points
    
########################################################################################################

### Simulator Class
### arguments = seeds, slots, and model
### proceeds to generate the entire tournament based on the model that is passed n times
### calculates expected points scored by each team
### produces final bracket, predicting games based on argmax(E[points_i], E[points_j])

### METHODS
# 1) init --> sets up model, seeds, slots

# 2) simulate_tournament --> takes in n_iterations, predicts bracket (used probability from logistic model to predict each game... sends team 1 over team 2 with some proabbiltu p based on the logistic model of the head to head results of the game) n times, calculates expected score of each team                                                      

# 3) predict_tournament --> deterministically predicts bracket based on expected score of each team

# 4) score_tournament --> compares predicted bracket to the results of the actual tournament
    

# used to simulate a tournament n times and output the combined results
class Simulator(object):
    def __init__(self, seeds, slots, model):
        self.seeds = seeds
        self.slots = slots
        self.model = model
    
    # run the tournament
    def run_tournament(self, scoring="ESPN"):
        # generate and run a tournament
        tournament = Tournament(self.seeds, self.slots, self.model)
        
        # get the number of predicted points that each team accumulates
        predicted_points = tournament.get_predicted_points(scoring=scoring)
        
        return predicted_points
        
    # simulate the tournament n times
    def simulate_tournament(self, n_iterations, scoring="ESPN"):
        predicted_points = np.zeros(self.seeds.shape[0])
        
        # calculate total points score over the entire simulation
        for i in range(n_iterations):
            predicted_points = predicted_points + self.run_tournament(scoring="ESPN")
            
        # make a dataframe for safe keeping
        self.predicted_points = pd.DataFrame(data=predicted_points, index=self.seeds["Team"], columns=["pred_points"])
        
        return self.predicted_points
    
    # output the final predictions based on the simulation
    def predict_tournament(self):
        # setup model for prediction
        expected_points_model = mmm.ExpectedPointsPredictor(self.predicted_points)
         
        # run tourney using our model as a prediction
        self.tournament_prediction = Tournament(self.seeds, self.slots, expected_points_model)
        self.tournament_prediction.run_tournament()
        
        # return the created tournament object
        return self.tournament_prediction
    
    # score the tournament
    def score_tournament(self, actual_results, print_res=True, scoring="ESPN"):
        return self.tournament_prediction.score_tournament(actual_results, print_res=print_res, scoring=scoring)        