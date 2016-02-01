import sys
sys.path.insert(0, 'modules')

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import preprocess
import numpy as np
import pandas as pd


class rnn_mcm_baker:
	
	def __init__(self):	
		print('Reading data...')
	
		X = pd.read_csv('data/preprocessed/sorted_common_X.csv', nrows=2000)
		Y = pd.read_csv('data/preprocessed/sorted_common_Y.csv', nrows=2000)
		
		self.Ylen = len(Y.columns)
		self.Xlen = len(X.columns)
		print('Preparing data...')	
		self.prepare(X, Y)
		self.current = 0
		print('Done.')

	def prepare(self, X, Y):
		self.X_total = np.empty(shape=(0 , self.Xlen))	
		self.Y_total = np.empty(shape=(0, self.Ylen))
		last_school = None
		print(len(Y['academicyear']))
		for i, year in enumerate(Y['academicyear']):
			curr_school = Y.iloc[i, :]['unitid']
			if curr_school == last_school:
				Y_data = Y.iloc[i, :] - Y.iloc[i - 1, :]		
			else:
				Y_data = pd.DataFrame(np.zeros(shape=(1, self.Ylen)))
			
			X_data = X[(X['academicyear'] == (year - 1)) & (X['unitid'] == curr_school) ]
			print(X_data.shape)
			np.concatenate( (self.X_total, X_data))
			np.concatenate( (self.Y_total, Y_data))
			last_school = curr_school

		imp, scal, vart = Imputer(), StandardScaler(), VarianceThreshold()
		pipe = Pipeline(zip(['imp','vart', 'scal'], [imp,vart,scal]))
		self.X_total = pipe.fit_transform(self.X_total)
		self.Y_total = pipe.fit_transform(self.Y_total)

	def next_batch(self, batch_size):
		curr = self.current
		X = self.X_total[curr:(curr + batch_size), :]
		Y = self.Y_total[curr:(curr + batch_size), :]
		self.current += batch_size
		return X, Y
			
			

	
	



