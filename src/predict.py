import csv
import numpy as np
import featurize
import json
import pickle
from marchine import MMNN
from sklearn import preprocessing

def predict_matchup(mn, season, t1_id, t2_id, t1_loc):
	x1, x2 = featurize.get_matchup_features(season, t1_id, t2_id, t1_loc, mn.id_to_team, mn.stats)
	if not x1 or not x2:
		print 'Could not get features', mn.id_to_team[t1_id], mn.id_to_team[t2_id]
		print x1
		print x2
		return None, None

	x = []
	x.append(x1)
	x.append(x2)
	x = mn.scalerX.transform(x)
	
	x1, x2 = x
	t1_activation, t2_activation = mn.nn.activate(x1), mn.nn.activate(x2)
	if mn.mov_label and mn.scalerY is not None: # Margin of victory used as labels
		t1_activation, t2_activation = mn.scalerY.inverse_transform(t1_activation.reshape(1,-1))[0], mn.scalerY.inverse_transform(t2_activation.reshape(1,-1))[0]
		avg = (t1_activation - t2_activation) / 2.0
		t1_activation, t2_activation = avg, -avg
	else:
		sum_win_prob = t1_activation + t2_activation
		t1_activation /= sum_win_prob
		t2_activation /= sum_win_prob

	return t1_activation, t2_activation

def predict_submission(mn_filename, readable_output, kaggle_output, submission):
	mn = MMNN()
	mn.load(mn_filename)
	with open(readable_output, 'w') as rs:
		writer_rs = csv.writer(rs, lineterminator='\n')
		writer_rs.writerow(['Team 1 Team 2', 'Pred1', 'Pred2', '', 'Team 2 Team 1', 'Pred2', 'Pred1'])

		with open(kaggle_output, 'w') as ms:
			writer_ms = csv.writer(ms, lineterminator='\n')
			writer_ms.writerow(['Id', 'Pred'])

			with open(submission) as csvfile:
				reader = csv.reader(csvfile)
				headers = next(reader)

				for row in reader:
					game_id = row[0]
					season, t1_id, t2_id = game_id.split('_')
					t1, t2 = mn.id_to_team[t1_id], mn.id_to_team[t2_id]

					t1_activation, t2_activation = predict_matchup(mn, season, t1_id, t2_id, 'N')
					if not t1_activation or not t2_activation:
						print 'Could not get features', t1, t2
						break

					# print t1, t2
					# print t1_activation, t2_activation

					writer_ms.writerow([game_id, str(t1_activation[0])])
					writer_rs.writerow([t1 + ' ' + t2, str(t1_activation[0]), str(t2_activation[0]), '', t2 + ' ' + t1, str(t2_activation[0]), str(t1_activation[0])])

def process_winrates(filename):
	with open(filename) as csvfile:
		reader = csv.reader(csvfile)
		headers = next(reader)
		matchups, y = [], []
		for row in reader:
			matchup, mov = row
			matchups.append(matchup)
			y.append(mov)

		y = np.array(y).astype(float)
		maxy, miny = max(y), min(y)
		print maxy, miny
		for i in range(len(y)):
			if y[i] >= 0.0:
				y[i] = y[i] / float(maxy) * 0.5 + 0.5
			else:
				y[i] = y[i] / float(abs(miny)) * 0.5 + 0.5
		print max(y), min(y)

		y = np.array(y).astype(float).reshape((len(y), 1))
		mms = preprocessing.MinMaxScaler().fit(y)
		winrates = mms.transform(y)
		print np.average(winrates)

	with open('../asdf.csv', 'w') as f:
		writer = csv.writer(f, lineterminator='\n')
		writer.writerow(['Id', 'Pred'])
		for matchup, wr in zip(matchups, winrates):
			writer.writerow([matchup] + list(wr))
		
if __name__ == "__main__":
	predict_submission('../data_2017/saved_nn', '../readable_output.csv', '../my_submission.csv', '../march-machine-learning-mania-2017/sample_submission.csv')
	# process_winrates('../my_submission.csv')
