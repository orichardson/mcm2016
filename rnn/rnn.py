import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

import pandas as pd

import sys
sys.path.insert(0, "../modules")

import preprocess
import collections

def rnn_model():
	in_neurons=1008	
	out_neurons=441
	hidden_neurons = 30
	model = Sequential()
	model.add(LSTM(input_dim=in_neurons, output_dim=hidden_neurons, return_sequences=False))
	model.add(Dense(input_dim=hidden_neurons, output_dim=out_neurons))
	model.add(Activation("sigmoid"))
	model.compile(loss="mean_squared_error", optimizer="rmsprop")


def flatten_catdat_single(X):
	"""convert school labels (categorical) into a one hot matrix concatenated along rows"""
	num_schools = len(set(X['unitid']))
	last_spot = 0
	spots = collections.OrderedDict()
	one_hot_matrix = pd.DataFrame(np.zeros(shape=(len(X.iloc[0,:]), num_schools))
	for i in range(len(X.iloc[0, :])):
		inst_id = X.iloc[i,:]['unitid']
		if inst_id not in spots:
			spots[inst_id] = len(spots)

		one_hot_matrix.iloc[i,spots[inst_id]] = 1
	one_hot_matrix.columns = spots.keys()	
	X = X.drop('unitid')
	return pd.concat([one_hot_matrix, X], axis=1)
			
	
def flatten_catdat(X, Y):
	"""convert categorical labels of schools into a one hot matrix, concatenated along rows"""
	"""assumes X,Y are datasets with the same schools in them"""
	return flatten_catdat_single(X), flatten_catdat_single(Y)
					

def in_window(year, year_0, window):
	return year <= (year_0 - 1) and year >= (year_0 - window)

def prepare_data(X, Y, window=5):
	"""assumes X, Y are datasets with the same schools in them, possibly flattened"""
	columns = [x for x in (set(X.columns) & set(Y.columns)) if isinstance(x, float)].append('academicyear')
	X.sort_values(columns, inplace=True, kind='quicksort')  
	Y.sort_values(columns, inplace=True, kind='quicksort')
	
	data = []
	previous = {}
	for i, year in Y['academicyear']:
		school = Y.columns[np.where(Y.iloc[i, :len(columns)] == 1) ]	
		previous[(school, year)] = i
		
		Y_data = Y.iloc[i, :] - Y.iloc[previous[(school, year - 1)], :]
		Y_data[school] = 1
			
		X_data = X[X[school] == 1 & (X['academicyear'] <= year - 1) & (X['academicyear'] >= year - window)]
		data.append( (X_data, Y_data) )
		

	return np.array(data)
	












































	"""X, Y are assumed to be panda dataframes."""
	X_id2year = preprocess.id_to_year(X)
	Y_id2year = preprocess.id_to_year(Y)

	schools = preprocess.same_schools(X, Y, X_id2year=X_id2year, Y_id2year=Y_id2year, condition=condition)
	#X_data, Y_data = [], []
	X_data, Y_data = pd.DataFrame(), pd.DataFrame()	
	for school in schools:
		X_data_pt = pd.DataFrame()
		column = 0
		for year in X_id2year[school]:	
			if column == train - 1:
				column = 0
				X_data.append(X_data_pt)
				X_data_pt = pd.DataFrame()
			X_data_pt = pd.concat([X_data_pt, X[(X['unitid'] == school) & (X['academicyear'] == year)]], axis=0)
			column += 1

		Y_data_pt = pd.DataFrame()
		column = 0
		for year in Y_id2year[school]:	
			if column == 5:
				column = 0
				Y_data.append(Y_data_pt)
				Y_data_pt = pd.DataFrame()
			Y_data_pt = pd.concat([Y_data_pt, Y[(Y['unitid'] == school) & (Y['academicyear'] == year)]], axis=0)
			column += 1

	return np.array(X_data), np.array(Y_data)	
		
		
def train_test_split(X, Y, train=5):
	X_test, Y_test, X_train, Y_train = preprocess.create_test(X, Y, 0.1, contiguous=lambda x: train < len(x) and preprocess.is_contiguous(x))
	X_train, Y_train = prepare_data(X_train, Y_train, train=train, condition=lambda x: train < len(x) and preprocess.is_contiguous(x))
	X_test, Y_test = prepare_data(X_test, Y_test, train=train, condition=lambda x: train < len(x) and preprocess.is_contiguous(x))
	return X_train, Y_train, X_test, Y_test

	
