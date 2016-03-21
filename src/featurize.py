import csv
import numpy as np
import json

def get_team_features(team_id, season, id_to_team, stats):
	assert str(team_id) in id_to_team
	team = id_to_team[str(team_id)]
	# Hardcode: Changed Middle Tenneessee St. to Middle Tennessee in kenpom.csv
	if team not in stats or str(season) not in stats[team]:
		return None
	team_stats = stats[team][str(season)]
	features = [
		team_stats['Wins'],
		team_stats['Losses'],
		team_stats['Pyth'],
		team_stats['AdjustO'],
		team_stats['AdjustD'],
		team_stats['AdjustT'],
		team_stats['Luck'],
		team_stats['SOS Pyth'],
		team_stats['SOS OppO'],
		team_stats['SOS OppD'],
		team_stats['NCSOS Pyth'],
	]
	return features

def get_training_data(year_range=range(2002, 2017)):
	with open('../data/id_to_team.json') as f:
	    id_to_team = json.load(f)
	with open('../data/stats.json') as f:
	    stats = json.load(f)

	x = []
	y = []
	results = ['../march-machine-learning-mania-2016-v2/TourneyCompactResults.csv', '../march-machine-learning-mania-2016-v2/RegularSeasonCompactResults.csv']
	for f in results:
		with open(f) as csvfile:
			reader = csv.reader(csvfile)
			headers = next(reader)
			for row in reader:
				season = row[0]
				if int(season) not in year_range:
					continue

				wteam, lteam = row[2], row[4]
				wscore, lscore = row[3], row[5]
				wloc = row[6]
				mov = int(wscore) - int(lscore)
				# print "{0}: {1} ({2}) beat {3} {4}-{5} (+{6})".format(season, id_to_team[wteam], wloc, id_to_team[lteam], wscore, lscore, mov)

				wfeatures = get_team_features(wteam, season, id_to_team, stats)
				lfeatures = get_team_features(lteam, season, id_to_team, stats)
				if not wfeatures or not lfeatures:
					continue

				#First team location features [Home, Away, Neutral] 
				wteam_first = [0,0,0]
				lteam_first = [0,0,0]
				if wloc == "H":
					wteam_first = [1,0,0]
					lteam_first = [0,1,0]
				elif wloc == "A":
					wteam_first = [0,1,0]
					lteam_first = [1,0,0]
				elif wloc == "N":
					wteam_first = [0,0,1]
					lteam_first = [0,0,1]

				x.append(wfeatures+lfeatures+wteam_first)
				x.append(lfeatures+wfeatures+lteam_first)

				# margin of victory output
				# norm_mov = (mov+20.0) / 40.0
				# norm_mov = norm_mov if norm_mov <= 1.0 else 1.0
				# y.append(norm_mov)
				# y.append(-norm_mov)

				# win/loss
				y.append(1)
				y.append(0)

	return np.array(x).astype(float), np.array(y).astype(float).reshape((len(y), 1))