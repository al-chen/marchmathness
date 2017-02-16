import csv
import json
import datetime
import kenpom_scraper
import os.path

def get_kenpom_stats(f):
	stats = {}
	with open(f) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			team = row['Team'].lower().rstrip()
			year = row['Year']
			if team not in stats:
				stats[team] = {}
			stats[team][year] = row
	return stats

def check_spelling(teams, spellings_file='../data/TeamSpellings.csv'):
	spelling_to_id = {}
	with open(spellings_file) as csvfile:
		reader = csv.reader(csvfile)
		headers = next(reader)
		for row in reader:
			spelling, team_id = row
			spelling_to_id[spelling] = team_id
		differences = sorted(teams - set(spelling_to_id.keys()))
		assert len(differences) == 0, "{0} differences in team spellings".format(str(len(differences))) + '\n' + str(differences)
	return spelling_to_id

def get_id_to_team_mappings(teams, spelling_to_id):
	team_to_id = {}
	for team in teams:
		team_to_id[team] = spelling_to_id[team]
	id_to_team = {v: k for k, v in team_to_id.items()}
	return id_to_team

def add_advanced_stats(stats, id_to_team):
	stat_names = ['fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'oreb', 'dr', 'ast', 'to', 'stl', 'blk', 'pf',
				  'opp_fgm', 'opp_fga', 'opp_fgm3', 'opp_fga3', 'opp_ftm', 'opp_fta', 'opp_oreb', 'opp_dr', 'opp_ast', 'opp_to', 'opp_stl', 'opp_blk', 'opp_pf']
	results = ['../march-machine-learning-mania-2016-v2/TourneyDetailedResults.csv', '../march-machine-learning-mania-2016-v2/RegularSeasonDetailedResults.csv']
	for f in results:
		with open(f) as csvfile:
			reader = csv.reader(csvfile)
			headers = next(reader)
			for row in reader:
				season = row[0]

				wteam, lteam = row[2], row[4]
				wscore, lscore = row[3], row[5]

				wteam_name, lteam_name = id_to_team[wteam], id_to_team[lteam]
				
				for team_name in [wteam_name, lteam_name]:
					if team_name not in stats or season not in stats[team_name]:
						continue
					team_stats = stats[team_name][season]
					if team_name == wteam_name:
						fgm, fga, fgm3, fga3, ftm, fta, oreb, dr, ast, to, stl, blk, pf = [float(x) for x in row[8:21]]
						opp_fgm, opp_fga, opp_fgm3, opp_fga3, opp_ftm, opp_fta, opp_oreb, opp_dr, opp_ast, opp_to, opp_stl, opp_blk, opp_pf = [float(x) for x in row[21:34]]
					else:
						fgm, fga, fgm3, fga3, ftm, fta, oreb, dr, ast, to, stl, blk, pf = [float(x) for x in row[21:34]]
						opp_fgm, opp_fga, opp_fgm3, opp_fga3, opp_ftm, opp_fta, opp_oreb, opp_dr, opp_ast, opp_to, opp_stl, opp_blk, opp_pf = [float(x) for x in row[8:21]]

					stat_nums = [fgm, fga, fgm3, fga3, ftm, fta, oreb, dr, ast, to, stl, blk, pf, 
								 opp_fgm, opp_fga, opp_fgm3, opp_fga3, opp_ftm, opp_fta, opp_oreb, opp_dr, opp_ast, opp_to, opp_stl, opp_blk, opp_pf]

					if 'counter' not in team_stats:
						team_stats['counter'] = 0
						for name in stat_names:
							team_stats[name] = 0.0

					team_stats['counter'] += 1
					for name, num in zip(stat_names, stat_nums):
						team_stats[name] += num

	for team in stats.keys():
		years = stats[team].keys()
		for year in years:
			team_stats = stats[team][year]
			if 'counter' in team_stats:
				ctr = team_stats['counter']
				for stat in stat_names:
					team_stats[stat] /= float(ctr)
			# else:
			# 	print team, year

if __name__ == "__main__":
	now = datetime.datetime.now()
	kp_filename = '../data_' + str(now.year) + '/kenpom_' + now.strftime("%Y-%m-%d") + '.csv'
	if not os.path.isfile(kp_filename):
		kenpom_scraper.scrape(kp_filename)
	stats = get_kenpom_stats(kp_filename)
	teams = set(stats.keys())
	print teams
	print len(stats), len(teams)
	# spelling_to_id = check_spelling(teams)
	# id_to_team = get_id_to_team_mappings(teams, spelling_to_id)

	# add_advanced_stats(stats, id_to_team)

	# with open('../data/id_to_team.json', 'w') as f:
	# 	json.dump(id_to_team, f)
	# with open('../data/stats_advanced.json', 'w') as f:
	# 	json.dump(stats, f)