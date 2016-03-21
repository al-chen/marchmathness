import csv
import json

def get_team_stats(f='../data/kenpom.csv'):
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

if __name__ == "__main__":
	stats = get_team_stats('../data/kenpom.csv')
	teams = set(stats.keys())
	spelling_to_id = check_spelling(teams)
	id_to_team = get_id_to_team_mappings(teams, spelling_to_id)
	
	with open('../data/id_to_team.json', 'w') as f:
		json.dump(id_to_team, f)
	with open('../data/stats.json', 'w') as f:
		json.dump(stats, f)