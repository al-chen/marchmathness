import csv
import numpy as np
import featurize
import json
import pickle

nn_filename = '../data/saved_nn'
nnObject = open(nn_filename,'r')
nn = pickle.load(nnObject)

scalerXObject = open(nn_filename + '_scalerX','r')
scalerX = pickle.load(scalerXObject)

scalerYObject = open(nn_filename + '_scalerY','r')
scalerY = pickle.load(scalerYObject)

with open('../data/id_to_team.json') as f:
	id_to_team = json.load(f)
with open('../data/stats.json') as f:
	stats = json.load(f)

with open('../march-machine-learning-mania-2016-v2/SampleSubmission.csv') as csvfile:
	reader = csv.reader(csvfile)
	headers = next(reader)
	for row in reader:
		game_id = row[0]
		season, t1_id, t2_id = game_id.split('_')
		t1, t2 = id_to_team[t1_id], id_to_team[t2_id]
		t1_features = featurize.get_team_features(t1_id, season, id_to_team, stats)
		t2_features = featurize.get_team_features(t2_id, season, id_to_team, stats)
		if not t1_features or not t2_features:
			print season, t1, t2, 'no features available'
			break
		loc = [0,0,1]
		x1 = t1_features + t2_features + loc
		x2 = t2_features + t1_features + loc

		x = []
		x.append(x1)
		x.append(x2)
		x = scalerX.transform(x)

		print t1, t2
		x1, x2 = x
		t1_activation, t2_activation = nn.activate(x1), nn.activate(x2)
		if scalerY:
			t1_activation, t2_activation = scalerY.inverse_transform(t1_activation), scalerY.inverse_transform(t2_activation)

		sum_win_prob = t1_activation + t2_activation
		t1_activation /= sum_win_prob
		t2_activation /= sum_win_prob
		print t1_activation, t2_activation
