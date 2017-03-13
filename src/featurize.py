import csv
import numpy as np
import json

def get_team_features(team_id, season, id_to_team, stats):
	assert str(team_id) in id_to_team
	team = id_to_team[str(team_id)]
	# Hardcoded: Change Middle Tenneessee St. to Middle Tennessee in kenpom.csv
	# Change Little Rock to Arkansas Little Rock
	if team not in stats or str(season) not in stats[team]:
		return None
	team_stats = stats[team][str(season)]
	if 'counter' not in team_stats:
		# print team, season
		return None
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
		team_stats['fgm'],
		team_stats['fga'],
		team_stats['fgm3'],
		team_stats['fga3'],
		team_stats['ftm'],
		team_stats['fta'],
		team_stats['oreb'],
		team_stats['dr'],
		team_stats['ast'],
		team_stats['to'],
		team_stats['stl'],
		team_stats['blk'],
		team_stats['pf'],
		team_stats['opp_fgm'],
		team_stats['opp_fga'],
		team_stats['opp_fgm3'],
		team_stats['opp_fga3'],
		team_stats['opp_ftm'],
		team_stats['opp_fta'],
		team_stats['opp_oreb'],
		team_stats['opp_dr'],
		team_stats['opp_ast'],
		team_stats['opp_to'],
		team_stats['opp_stl'],
		team_stats['opp_blk'],
		team_stats['opp_pf'],
	]
	return features

def get_matchup_features(season, t1_id, t2_id, t1_loc, id_to_team, stats):
	t1, t2 = id_to_team[t1_id], id_to_team[t2_id]

	t1_features = get_team_features(t1_id, season, id_to_team, stats)
	t2_features = get_team_features(t2_id, season, id_to_team, stats)
	if not t1_features or not t2_features:
		# print season, t1, t2, 'no features available'
		return None, None

	#First team location features [Home, Away, Neutral] 
	t1_first = [0,0,0]
	t2_first = [0,0,0]
	if t1_loc == "N":
		t1_first = [0,0,1]
		t2_first = [0,0,1]
	elif t1_loc == "H":
		t1_first = [1,0,0]
		t2_first = [0,1,0]
	elif t1_loc == "A":
		t1_first = [0,1,0]
		t2_first = [1,0,0]

	x1 = t1_features + t2_features + t1_first
	x2 = t2_features + t1_features + t2_first
	return x1, x2

def get_training_data(year_range, id_to_team, stats, results):
	x = []
	y = []
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

				x1, x2 = get_matchup_features(season, wteam, lteam, wloc, id_to_team, stats)
				if not x1 or not x2:
					continue
				x.append(x1)
				x.append(x2)

				# margin of victory output
				# norm_mov = (mov+20.0) / 40.0
				# norm_mov = norm_mov if norm_mov <= 1.0 else 1.0
				# y.append(norm_mov)
				# y.append(-norm_mov)

				y.append(mov)
				y.append(-mov)

				# win/loss
				# y.append(1)
				# y.append(0)

	return np.array(x).astype(float), np.array(y).astype(float).reshape((len(y), 1))