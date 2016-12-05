import pandas as pd
import numpy as np
import random as rand

# tournament class
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

            # teams
            team_1 = seeds.loc[seeds["Seed"] == team_1_seed, "Team"].values[0]
            team_2 = seeds.loc[seeds["Seed"] == team_2_seed, "Team"].values[0]
            
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
        # wins in a given round
        team = str(team)
        wins = self.entire_bracket[self.entire_bracket["Prediction"] == team].shape[0]
         
        points = 0                           
        for i in range(int(wins)):
            points = points + 10 * 2 ** i
                         
        return points
                                   
    # points scored for all teams
    def get_predicted_points(self, scoring="ESPN"):
        # setup buffers
        teams = self.seeds["Team"]
        points = np.zeros(teams.shape[0])

        # get the predicted points for a team
        i = 0
        for team in teams:
            points[i] = self.get_predicted_points_for_team(team)
            i = i + 1
              
        return points

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
        expected_points_model = ExpectedPointsPredictor(self.predicted_points)
         
        # run tourney using our model as a prediction
        self.tournament_prediction = Tournament(self.seeds, self.slots, expected_points_model)
        self.tournament_prediction.run_tournament()
        
        # return the created tournament object
        return self.tournament_prediction
    
    # score the tournament
    def score_tournament(self, actual_results, print_res=True, scoring="ESPN"):
        return self.tournament_prediction.score_tournament(actual_results, print_res=print_res, scoring=scoring)
        
        
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
        row = get_predictors_dif(min_index_team, max_index_team, self.year, self.dfs_arr)

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
           
############# FUNCTIONS USED TO MANIPULATE DF ###############################        
        
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


############### FUNCTIONS USED TO BUILD HEAD TO HEAD MODELS ######################

# extracting from the dataframe
def get_predictor(team_id, year, df):
    return df.loc[year, str(team_id)]

def get_predictor_dif(team_id_1, team_id_2, year, df):
    return df.loc[year, str(team_id_1)] - df.loc[year, str(team_id_2)]

def get_predictors(team_id, year, df_arr):
    row = np.zeros(len(df_arr))
    i = 0
    for df in df_arr:
        row[i] = get_predictor(team_id, year, df)
        i = i + 1
    return row

def get_predictors_dif(team_id_1, team_id_2, year, df_arr):
    row = np.zeros(len(df_arr))
    i = 0
    for df in df_arr:
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

# function to extract the y_values of the team with the min index winning
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
def get_tourney_results(seeds, slots, raw_data, scoring_dif=False):
    tourney = Tournament(seeds, 
                         slots, 
                         ActualTournament(raw_data, include_scoring_dif=scoring_dif), 
                         include_scoring_dif=scoring_dif)
    
    tourney.run_tournament()
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

# generate multiple years of games in a row
def generate_multiple_years_of_games(years, seed_list_arr, 
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
        