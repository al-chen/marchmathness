import csv
import numpy as np
import featurize
from sklearn import preprocessing
from sklearn.utils import shuffle
import pickle

from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def setup(x_train, y_train, hidden_size=0, hiddenclass=TanhLayer, outclass=SigmoidLayer, maxEpochs=50, verbose=False):
	ds = SupervisedDataSet(x_train.shape[1], 1)
	assert(x_train.shape[0] == y_train.shape[0])
	ds.setField('input', x_train)
	ds.setField('target', y_train)

	if hidden_size == 0:
		hidden_size = x_train.shape[1]
	nn = buildNetwork(x_train.shape[1], hidden_size, 1, bias=True, hiddenclass=hiddenclass, outclass=outclass)
	trainer = BackpropTrainer(nn, ds, verbose=verbose)
	trainer.trainUntilConvergence(maxEpochs=maxEpochs)
	return nn

def test(year_range=range(2002,2017), hidden_size=0, hiddenclass=TanhLayer, outclass=SigmoidLayer, maxEpochs=50, verbose=True):
	data, labels = featurize.get_training_data(year_range=year_range)
	scalerX = preprocessing.StandardScaler().fit(data)
	scalerY = preprocessing.StandardScaler().fit(labels)

	# x, y = shuffle(scalerX.transform(data), scalerY.transform(labels), random_state=42)
	x, y = shuffle(scalerX.transform(data), labels, random_state=42)

	x_train, x_test = x[:0.9*len(x)], x[0.9*len(x):]
	y_train, y_test = y[:0.9*len(y)], y[0.9*len(y):]

	nn = setup(x_train, y_train, hidden_size=hidden_size, hiddenclass=hiddenclass, outclass=outclass, maxEpochs=maxEpochs, verbose=verbose)

	test_ds = SupervisedDataSet(x_test.shape[1], 1)
	test_ds.setField('input', x_test)
	test_ds.setField('target', y_test)
	preds = nn.activateOnDataset(test_ds)
	counter = 0
	success = 0
	for i in range(0, len(preds), 2):
		counter += 1
		if (preds[i][0] > preds[i+1][0] and y_test[i][0] > y_test[i+1][0]) or (preds[i][0] < preds[i+1][0] and y_test[i][0] < y_test[i+1][0]):
			success += 1
	print 'Accuracy on test set:', float(success) / counter
	return nn

def train(year_range=range(2002,2017), hidden_size=0, hiddenclass=TanhLayer, outclass=SigmoidLayer, maxEpochs=50, verbose=False):
	data, labels = featurize.get_training_data(year_range=year_range)
	scalerX = preprocessing.StandardScaler().fit(data)

	# scalerY = preprocessing.StandardScaler().fit(labels)
	# x_train, y_train = shuffle(scalerX.transform(data), scalerY.transform(labels))
	scalerY = None
	x_train, y_train = shuffle(scalerX.transform(data), labels)

	nn = setup(x_train, y_train, hidden_size=hidden_size, hiddenclass=hiddenclass, outclass=outclass, maxEpochs=maxEpochs, verbose=verbose)
	return nn, scalerX, scalerY

if __name__ == "__main__":
	print 'starting training'
	nn, scalerX, scalerY = train(year_range=range(2002,2017), hidden_size=0, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, maxEpochs=100, verbose=True)
	
	nn_filename = '../data/saved_nn'
	nnObject = open(nn_filename, 'w')
	pickle.dump(nn, nnObject)
	nnObject.close()

	scalerXObject = open(nn_filename + '_scalerX', 'w')
	pickle.dump(scalerX, scalerXObject)
	nnObject.close()

	scalerYObject = open(nn_filename + '_scalerY', 'w')
	pickle.dump(scalerY, scalerYObject)
	nnObject.close()
