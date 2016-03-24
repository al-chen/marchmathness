import csv
import json
import predict
import pickle

class Bracket(object):
	def __init__(self, nn, scalerX, scalerY, id_to_team, stats, season=0):
		self.nn = nn
		self.scalerX = scalerX
		self.scalerY = scalerY
		self.id_to_team = id_to_team
		self.stats = stats

		if season == 0:
			self.chosen_season = 2016
		else:
			self.chosen_season = season

		self.seed_to_team_id = {}
		self.set_season(self.chosen_season)

	def set_season(self, chosen_season):
		self.chosen_season = chosen_season

		self.seed_to_team_id.clear()
		with open('../march-machine-learning-mania-2016-v2/TourneySeeds.csv') as csvfile:
			reader = csv.reader(csvfile)
			headers = next(reader)
			for row in reader:
				season, seed, team = row
				if int(season) != self.chosen_season:
					continue
				self.seed_to_team_id[seed] = team

	def set_playin_winners(self, playin_winners):
		for pw in playin_winners:
			playin_seed = pw[:-1]
			self.seed_to_team_id[playin_seed] = self.seed_to_team_id[pw]

	def get_playin_seeds(self):
		assert self.seed_to_team_id is not None
		playin_seeds = []
		for seed in self.seed_to_team_id.keys():
			if len(seed) > 3:
				playin_seeds.append(seed)
		playin_seeds = sorted(playin_seeds)
		return playin_seeds

	def simulate_playins(self, verbose=False):
		playin_seeds = self.get_playin_seeds()
		playin_predictions = self.greedy_predict(playin_seeds, rounds=1, verbose=verbose)
		playin_winners = playin_predictions[-1]
		self.set_playin_winners(playin_winners)

	def set_seed_to_team_id(self, seed_to_team_id):
		self.seed_to_team_id = seed_to_team_id

	def seeds_to_bracket(self, seeds):
		assert self.seed_to_team_id is not None
		bracket = []
		for seed in seeds:
			if seed not in self.seed_to_team_id:
				a = seed + 'a'
				b = seed + 'b'
				playin_teams = self.id_to_team[self.seed_to_team_id[a]] + '/' + self.id_to_team[self.seed_to_team_id[b]]
				bracket.append(playin_teams)
			else:
				bracket.append(self.id_to_team[self.seed_to_team_id[seed]])
		return bracket

	def greedy_predict(self, seeds, rounds=0, verbose=False):
		if rounds == 0:
			threshold = 1
		else:
			threshold = len(seeds) / pow(2,rounds)

		history = [seeds[:]]
		while len(seeds) > threshold:
			seeds_copy = seeds[:]
			for i in range(0, len(seeds), 2):
				if i+1 >= len(seeds): # odd number of seeds
					continue
				seed1, seed2 = seeds[i], seeds[i+1]
				t1_id, t2_id = self.seed_to_team_id[seed1], self.seed_to_team_id[seed2]
				t1, t2 = self.id_to_team[t1_id], self.id_to_team[t2_id]
				t1_prob, t2_prob = predict.predict_matchup(self.nn, self.scalerX, self.scalerY, str(self.chosen_season), t1_id, t2_id, "N", self.id_to_team, self.stats)
				
				if t1_prob[0] > t2_prob[0]:
					seeds_copy.remove(seed2)
					if verbose:
						print "{0} defeats {1} with probability {2}".format(t1, t2, str(t1_prob[0]))
				else:
					seeds_copy.remove(seed1)
					if verbose:
						print "{0} defeats {1} with probability {2}".format(t2, t1, str(t2_prob[0]))
			seeds = seeds_copy
			history.append(seeds[:])
		return history

	def memoized_ev(self, seeds_done, seeds_todo, points, level, dic, cutoff):
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
			t1_id, t2_id = self.seed_to_team_id[seed1], self.seed_to_team_id[seed2]
			t1, t2 = self.id_to_team[t1_id], self.id_to_team[t2_id]
			t1_prob, t2_prob = predict.predict_matchup(self.nn, self.scalerX, self.scalerY, str(self.chosen_season), t1_id, t2_id, "N", self.id_to_team, self.stats)
			t1_prob, t2_prob = t1_prob[0], t2_prob[0]
			dic[(seed1,seed2)] = t1_prob
			dic[(seed2,seed1)] = t2_prob

		seeds_todo.remove(seed1)
		seeds_todo.remove(seed2)

		correct_pts = 10.0 * pow(2,level-1)
		if t1_prob >= cutoff:
			sd1, p1, dic = self.memoized_ev(seeds_done + [seed1], seeds_todo[:], points + correct_pts*t1_prob, level, dic, cutoff)
			sd2, p2, dic = [], 0, dic
		elif t2_prob >= cutoff:
			sd1, p1, dic = [], 0, dic
			sd2, p2, dic = self.memoized_ev(seeds_done + [seed2], seeds_todo[:], points + correct_pts*t2_prob, level, dic)
		else:
			sd1, p1, dic = self.memoized_ev(seeds_done + [seed1], seeds_todo[:], points + correct_pts*t1_prob, level, dic, cutoff)
			sd2, p2, dic = self.memoized_ev(seeds_done + [seed2], seeds_todo[:], points + correct_pts*t2_prob, level, dic, cutoff)

		if p1 >= p2:
			return sd1, p1, dic
		return sd2, p2, dic

	def value_predict(self, seeds, cutoff=0.8):
		sd = seeds
		p = 0.0
		d = {}
		level = 1
		history = [seeds[:]]
		while len(sd) > 1:
			sd, p, d = self.memoized_ev([], sd, p, level, d, cutoff)
			level += 1
			history.append(sd[:])
		return history

if __name__ == "__main__":
	with open('../data/id_to_team.json') as f:
		id_to_team = json.load(f)

	with open('../data/stats_advanced.json') as f:
		stats = json.load(f)

	nn_filename = '../data/advanced_nn_100epochs'
	nnObject = open(nn_filename,'r')
	nn = pickle.load(nnObject)

	scalerXObject = open(nn_filename + '_scalerX','r')
	scalerX = pickle.load(scalerXObject)

	scalerYObject = open(nn_filename + '_scalerY','r')
	scalerY = pickle.load(scalerYObject)

	seeds = ['W01', 'W16', 'W08', 'W09', 'W05', 'W12', 'W04', 'W13', 'W06', 'W11', 'W03', 'W14', 'W07', 'W10', 'W02', 'W15', 
			 'X01', 'X16', 'X08', 'X09', 'X05', 'X12', 'X04', 'X13', 'X06', 'X11', 'X03', 'X14', 'X07', 'X10', 'X02', 'X15', 
			 'Y01', 'Y16', 'Y08', 'Y09', 'Y05', 'Y12', 'Y04', 'Y13', 'Y06', 'Y11', 'Y03', 'Y14', 'Y07', 'Y10', 'Y02', 'Y15', 
			 'Z01', 'Z16', 'Z08', 'Z09', 'Z05', 'Z12', 'Z04', 'Z13', 'Z06', 'Z11', 'Z03', 'Z14', 'Z07', 'Z10', 'Z02', 'Z15']

	b = Bracket(nn, scalerX, scalerY, id_to_team, stats, season=2016)
	b.simulate_playins(verbose=False)
	
	greedy_tourney = b.greedy_predict(seeds[:], rounds=0, verbose=False)
	print 'greedy predictions'
	for t in greedy_tourney:
		print b.seeds_to_bracket(t)
	
	value_tourney = b.value_predict(seeds[:], cutoff=0.8)
	print 'value predictions'
	for t in value_tourney:
		print b.seeds_to_bracket(t)
