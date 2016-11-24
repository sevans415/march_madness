# march_madness

# Data
The predictors are at are at datasets/our_data.

Each of the data files are structures as a 2 dimensional matrix; with the rows representing a given year and each column representing a given team. Thus, wins[1985, 1000] is the number of wins by team 1000 in year 1985. In order to get an "x" vector to use a predictor for a given year, take the 1985 "row" (which is itself a pandas Series object) and use it as a predictor.

We also have additional helper data at datasets/our_data/helper_data. These data are not necessarily predictors, but are used in creating predictors. The data are structured as above.

We also have the markov chain data in datasets/our_data/markov_data. These data are used in our markov chain analysis in generating the markov chain/calculating the stationary distributions. There is a dataframe with the team id and the distribution for each year; however, note that I will be fixing these to be in the structure of the other data soon.

# March Madness Classes 
