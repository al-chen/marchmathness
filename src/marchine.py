import csv
import numpy as np
import featurize
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle
import json

from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

class MMNN(): # March Madness Neural Network
	def __init__(self):
		self.year_range = range(2002, 2018)
		self.hiddenclass = TanhLayer
		self.outclass = SigmoidLayer
		self.maxEpochs = 1
		self.verbose = True
		self.hidden_size = 0

	def load_ids(self, filename):
		with open(filename) as f:
			self.id_to_team = json.load(f)

	def load_stats(self, filename):
		with open(filename) as f:
			self.stats = json.load(f)

	def set_year_range(self, year_range):
		self.year_range = year_range

	def set_results(self, results):
		self.results = results

	def get_data(self):
		if not self.year_range or not self.id_to_team or not self.stats or not self.results:
			print 'Not enough information set to get data'
		self.data, self.labels = featurize.get_training_data(self.year_range, self.id_to_team, self.stats, self.results)

	def preprocess(self):
		self.scalerX = preprocessing.StandardScaler().fit(self.data)
		self.processed_data = self.scalerX.transform(self.data)

		# self.scalerY = None
		# self.processed_labels = self.labels

		self.scalerY = preprocessing.MinMaxScaler().fit(self.labels)
		self.processed_labels = self.scalerY.transform(self.labels)

	def test(self):
		x, y = shuffle(self.processed_data, self.processed_labels, random_state=42)
		x_train, x_test = x[:int(0.9*len(x))], x[int(0.9*len(x)):]
		y_train, y_test = y[:int(0.9*len(y))], y[int(0.9*len(y)):]

		self.train(x_train, y_train)

		test_ds = SupervisedDataSet(x_test.shape[1], 1)
		test_ds.setField('input', x_test)
		test_ds.setField('target', y_test)
		preds = self.nn.activateOnDataset(test_ds)
		counter = 0
		success = 0
		for i in range(0, len(preds)-1, 2):
			counter += 1
			if (preds[i][0] > preds[i+1][0] and y_test[i][0] > y_test[i+1][0]) or (preds[i][0] < preds[i+1][0] and y_test[i][0] < y_test[i+1][0]):
				success += 1
		print 'Accuracy on test set:', float(success) / counter

	def train(self, x_train=None, y_train=None):
		if x_train is None and y_train is None:
			x_train, y_train = shuffle(self.processed_data, self.processed_labels)

		ds = SupervisedDataSet(x_train.shape[1], 1)
		assert(x_train.shape[0] == y_train.shape[0])
		ds.setField('input', x_train)
		ds.setField('target', y_train)

		if self.hidden_size == 0:
			hs = x_train.shape[1]
		self.nn = buildNetwork(x_train.shape[1], hs, 1, bias=True, hiddenclass=self.hiddenclass, outclass=self.outclass)
		trainer = BackpropTrainer(self.nn, ds, verbose=self.verbose)
		trainer.trainUntilConvergence(maxEpochs=self.maxEpochs)

	def save(self, filename):
		f = open(filename,'wb')
		pickle.dump(self.__dict__,f,2)
 		f.close()

	def load(self, filename):
		f = open(filename,'rb')
		tmp_dict = pickle.load(f)
		f.close()          
		self.__dict__.update(tmp_dict) 


if __name__ == "__main__":
	mn = MMNN()

	mn.load_ids('../data_2017/id_to_team.json')
	mn.load_stats('../data_2017/stats_advanced.json')
	mn.set_year_range(range(2002,2013))
	mn.set_results(['../march-machine-learning-mania-2017/TourneyDetailedResults.csv', '../march-machine-learning-mania-2017/RegularSeasonDetailedResults.csv'])
	mn.maxEpochs = 10
	print 'Getting data...'
	mn.get_data()
	print 'Preprocessing...'
	mn.preprocess()
	# print 'Testing...'
	# mn.test()
	print 'Training...'
	mn.train()

	mn.save('../data_2017/saved_nn')
