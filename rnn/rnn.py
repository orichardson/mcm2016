import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

import pandas as pd

import sys
sys.path.insert(0, "../modules")

import preprocess

def rnn_model():
	in_neurons=1008	
	out_neurons=441
	hidden_neurons = 30
	model = Sequential()
	model.add(LSTM(input_dim=in_neurons, output_dim=hidden_neurons, return_sequences=False))
	model.add(Dense(input_dim=hidden_neurons, output_dim=out_neurons))
	model.add(Activation("sigmoid"))
	model.compile(loss="mean_squared_error", optimizer="rmsprop")

def alltrue(x):
	return True

def prepare_data(X, Y, train=5, condition=alltrue):
	"""X, Y are assumed to be panda dataframes."""
	X_id2year = preprocess.id_to_year(X)
	Y_id2year = preprocess.id_to_year(Y)

	schools = preprocess.same_schools(X, Y, X_id2year=X_id2year, Y_id2year=Y_id2year, condition=condition)
	X_data = [], Y_data = []
	for school in schools:
		X_data_pt = pd.DataFrame()
		column = 0
		for year in X_id2year[school]:	
			if column == 5:
				column = 0
				X_data.append(X_data_pt)
				X_data_pt = pd.DataFrame()
			X_data_pt = pd.concat([X_data_pt, X[(X['unitid'] == school) & (X['academicyear'] == year)]], axis=1)
			column += 1

		Y_data_pt = pd.DataFrame()
		column = 0
		for year in Y_id2year[school]:	
			if column == 5:
				column = 0
				Y_data.append(Y_data_pt)
				Y_data_pt = pd.DataFrame()
			Y_data_pt = pd.concat([Y_data_pt, Y[(Y['unitid'] == school) & (Y['academicyear'] == year)]], axis=1)
			column += 1

	return np.array(X_data), np.array(Y_data)	
		
		
def train_test_split(X, Y, train=5):
	X_test, Y_test, X_train, Y_train = preprocess.create_test(X, Y, 0.1, contiguous=lambda x: train < len(x) and preprocess.is_contiguous(x))
	X_train, Y_train = preprocess.prepare_data(X_train, Y_train, train=train, contiguous=lambda x: train < len(x) and preprocess.is_contiguous(x))
	X_test, Y_test = preprocess.prepare_data(X_test, Y_test, train=train, contiguous=lambda x: train < len(x) and preprocess.is_contiguous(x))
	return X_train, Y_train, X_test, Y_test

	
