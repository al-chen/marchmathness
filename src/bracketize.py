import csv
import json
import predict
import pickle

def seeds_to_bracket(seeds, seed_to_team_id, id_to_team):
	bracket = []
	for seed in seeds:
		if seed not in seed_to_team_id:
			a = seed + 'a'
			b = seed + 'b'
			playin_teams = id_to_team[seed_to_team_id[a]] + '/' + id_to_team[seed_to_team_id[b]]
			bracket.append(playin_teams)
		else:
			bracket.append(id_to_team[seed_to_team_id[seed]])
	return bracket

def greedy_predict(seeds, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, verbose=False):
	if rounds == 0:
		threshold = 1
	else:
		threshold = len(seeds) / pow(2,rounds)

	history = [seeds]
	while len(seeds) > threshold:
		seeds_copy = seeds[:]
		for i in range(0, len(seeds), 2):
			if i+1 >= len(seeds): # odd number of seeds
				continue
			seed1, seed2 = seeds[i], seeds[i+1]
			t1_id, t2_id = seed_to_team_id[seed1], seed_to_team_id[seed2]
			t1, t2 = id_to_team[t1_id], id_to_team[t2_id]
			t1_prob, t2_prob = predict.predict_matchup(nn, scalerX, scalerY, str(chosen_season), t1_id, t2_id, "N", id_to_team, stats)
			
			if t1_prob[0] > t2_prob[0]:
				seeds_copy.remove(seed2)
				if verbose:
					print "{0} defeats {1} with probability {2}".format(t1, t2, str(t1_prob[0]))
			else:
				seeds_copy.remove(seed1)
				if verbose:
					print "{0} defeats {1} with probability {2}".format(t2, t1, str(t2_prob[0]))
		seeds = seeds_copy
		history.append(seeds)
	return history

def dynamic_ev(seeds_done, seeds_todo, points, level, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, dic={}):
	if len(seeds_todo) <= 1:
		return seeds_done, points, dic

	seed1, seed2 = seeds_todo[0], seeds_todo[1]
	if (seed1, seed2) in dic:
		t1_prob = dic[(seed1, seed2)]
		t2_prob = 1.0 - t1_prob
	elif (seed2, seed1) in dic:
		t2_prob = dic[(seed2, seed1)]
		t1_prob = 1.0 - t2_prob
	else:
		t1_id, t2_id = seed_to_team_id[seed1], seed_to_team_id[seed2]
		t1, t2 = id_to_team[t1_id], id_to_team[t2_id]
		t1_prob, t2_prob = predict.predict_matchup(nn, scalerX, scalerY, str(chosen_season), t1_id, t2_id, "N", id_to_team, stats)
		t1_prob, t2_prob = t1_prob[0], t2_prob[0]
		dic[(seed1,seed2)] = t1_prob
		dic[(seed2,seed1)] = t2_prob

	seeds_todo.pop(0)
	seeds_todo.pop(0)

	correct_pts = 10.0 * pow(2,level-1)
	sd1, p1, dic = dynamic_ev(seeds_done + [seed1], seeds_todo, points + correct_pts*t1_prob, level, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, dic=dic)
	sd2, p2, dic = dynamic_ev(seeds_done + [seed2], seeds_todo, points + correct_pts*t2_prob, level, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, dic=dic)
	if p1 >= p2:
		return sd1, p1, dic
	return sd2, p2, dic

def value_predict(seeds, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0):
	sd = seeds
	p = 0.0
	d = {}
	level = 1
	history = [seeds[:]]
	while len(sd) > 1:
		sd, p, d = dynamic_ev([], sd, p, level, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, dic=d)
		level += 1
		history.append(sd[:])
	return history

if __name__ == "__main__":
	with open('../data/id_to_team.json') as f:
		id_to_team = json.load(f)

	with open('../data/stats.json') as f:
		stats = json.load(f)

	nn_filename = '../data/saved_nn'
	nnObject = open(nn_filename,'r')
	nn = pickle.load(nnObject)

	scalerXObject = open(nn_filename + '_scalerX','r')
	scalerX = pickle.load(scalerXObject)

	scalerYObject = open(nn_filename + '_scalerY','r')
	scalerY = pickle.load(scalerYObject)

	chosen_season = 2016
	seed_to_team_id = {}
	with open('../march-machine-learning-mania-2016-v2/TourneySeeds.csv') as csvfile:
		reader = csv.reader(csvfile)
		headers = next(reader)
		for row in reader:
			season, seed, team = row
			if int(season) != chosen_season:
				continue
			seed_to_team_id[seed] = team

	regional_seeds = ['01', '16', '08', '09', '05', '12', '04', '13', '06', '11', '03', '14', '07', '10', '02', '15']
	seeds = regional_seeds * 4
	for i, char in zip(range(4), ['W', 'X', 'Y', 'Z']):
		for j in range(16):
			idx = i*16 + j
			seeds[idx] = char + seeds[idx]

	playin_seeds = []
	for seed in seed_to_team_id.keys():
		if len(seed) > 3:
			playin_seeds.append(seed)
	playin_seeds = sorted(playin_seeds)

	playin_predictions = greedy_predict(playin_seeds, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=1)
	playin_winners = playin_predictions[-1]
	for pw in playin_winners:
		playin_seed = pw[:-1]
		seed_to_team_id[playin_seed] = seed_to_team_id[pw]

	print 'Greedy Bracket'
	tournament = greedy_predict(seeds, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0, verbose=False)
	for t in tournament:
		print seeds_to_bracket(t, seed_to_team_id, id_to_team)

	print 'Value Bracket'
	tournament = value_predict(seeds, nn, scalerX, scalerY, chosen_season, id_to_team, stats, rounds=0)
	for t in tournament:
		print seeds_to_bracket(t, seed_to_team_id, id_to_team)