import pandas as pd
import random
import numpy as np
import yaml
import collections
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.sparse import dok_matrix, hstack

## keep the headers, can use .iloc (integer locate)
def load_raw_data():
	x_files = [ pd.read_csv("../data/finances/delta_public_release_{0}.csv".format(p), encoding = "ISO-8859-1") for p in [ "87_99", "00_13" ]]
	
	y_files = []
	for i in range(1996, 2014):
		df = pd.read_csv("../data/scorecard/MERGED{0}_PP.csv".format(i), encoding = "ISO-8859-1")
		df.insert(len(df.columns), "academicyear", i)
		y_files.append(df)

	X_data = x_files[0].append(x_files[1])
	X_data.sort_values('unitid', inplace=True, kind='heapsort')
	Y_data = y_files[0].append(y_files[1:])
	Y_data = Y_data.rename(columns = { '\xef\xbb\xbfUNITID' : 'unitid' })
	Y_data.sort_values('unitid', inplace=True, kind='heapsort')
	
	return X_data, Y_data
	
## make term dictionary for V's 
def makeVDict():
	f = open('data/scorecard/data_dictionary.yaml')
	dd = yaml.safe_load(f)['dictionary']
	f.close()
	
	lookup = {}
	
	for di in dd.keys():
		if 'source' in dd[di]:
			lookup[dd[di]['source']] = (di, dd[di]['description'])

	return lookup


## figure out test data
def load(kind='scorecard', numeric_only=True, sort=False):
	df = pd.read_csv("data/processed/{0}_train.csv".format(kind))
	
	if numeric_only:
		df = df.select_dtypes(include=['int64', 'float64'])
	if sort:
		df.sort_values('unitid', inplace=True, kind='heapsort')

	return df

def fload():
	X = pd.read_csv("data/processed/finances_train.csv")
	Y = pd.read_csv("data/processed/scorecard_train.csv")
	X, Y = remove_undesirable(X, Y)
	X = X.select_dtypes(include=['int64', 'float64'])
	Y = Y.select_dtypes(include=['int64', 'float64'])
	return X, Y
	

# distribution of years

def id_to_year(data):
	id_year = data[["unitid", "academicyear"]]
	
	data_unitid = id_year.iloc[:, 0]
	id2year = { x : [] for x in data_unitid }
	
	for i, item in enumerate(data_unitid):
		id2year[item].append(id_year.iloc[i, 1])

	return id2year

def get_year_distrib(data):
	"""Determines how many universities we have x years of data for, for x = 1,...,28"""
	id2year = id_to_year(data)

	vals, counts = np.unique([len(x) for x in id_to_year.values()], return_counts=True)
	return dict(zip(list(vals), list(counts)))

def years_are_contiguous(data):
	"""Determines for how many universities have contiguous year data"""
	id2year = id_to_year(data)	

	vals, counts = np.unique([is_contiguous(sorted(x)) for x in id2year.values()], return_counts=True)
	return dict(zip(list(vals), list(counts)))

def is_contiguous(alist):
	"""Determines if a list consists of contiguous integers"""
	return map(lambda x: x - alist[0], alist) == range(len(alist))

def alltrue(x):
	return True	

def same_schools(X, Y, X_id2year=None, Y_id2year=None,condition=alltrue):
	if X_id2year is None:
		X_id2year = id_to_year(X)	
	if Y_id2year is None:
		Y_id2year = id_to_year(Y)
	
	X_schools = set([x for x in X_id2year if condition(X_id2year[x])])
	Y_schools = set([x for x in Y_id2year if condition(Y_id2year[x])])

	return X_schools & Y_schools
	
def create_test(X, Y, percent, contiguous=lambda x: is_contiguous(x)):
	same = list(same_schools(X, Y, condition=contiguous))
	cutoff = int(percent * len(same))
	random.shuffle(same)
	restricted = set(same[:cutoff])	 
	unrestricted = set(same[cutoff:])	
	
	return X[X['unitid'].isin(restricted)], Y[Y['unitid'].isin(restricted)], X[X['unitid'].isin(unrestricted)], Y[Y['unitid'].isin(unrestricted)]

		
def flatten_catdat(X, category):
	"""convert school labels (categorical) into a one hot matrix concatenated along rows"""
	label_enc = LabelEncoder()
	X[category] = label_enc.fit_transform(X[category])
	hot_enc = OneHotEncoder()
	flattened = hot_enc.fit_transform(X[[category]].as_matrix())
	X.drop(category, axis=1, inplace=True)
	encoder = Pipeline([ ('label_enc', label_enc), ('hot', hot_enc)]) 
	return flattened, X, encoder
	
def remove_undesirable(X, Y):	
	both = ['unitid', 'academicyear']
	X_keep = both + list( x.rstrip() for x in open('../data/finances/xvars.txt') )
	Y_keep = both + list( y.rstrip() for y in open('../data/scorecard/yvars.txt') )
	return X[ X_keep ], Y[ Y_keep ]
	
def prepare_data(X, Y, window=5):
	"""assumes X, Y are datasets with the same schools in them, with all categorical variables flattened and the school variables in the very front"""
	common = same_schools(X, Y)
	X = X[X['unitid'].isin(common)]	
	Y = Y[Y['unitid'].isin(common)]
	flattened, X, pipe = flatten_catdat(X, 'unitid')
	
	Y.sort_values(['unitid', 'academicyear'], inplace=True, kind='quicksort', ascending=[True, True])
	
	data = []
	previous = {}
	print('fuck')
	for i, year in enumerate(Y['academicyear']):
		school = Y.iloc[i, :]['unitid']
		#school = Y.columns[np.where(Y.iloc[i, :len(columns)] == 1) ]	
		previous[(school, year)] = i
		
		try:
			Y_data = Y.iloc[i, :] - Y.iloc[previous[(school, year - 1)], :]
		except:
			Y_data = Y.iloc[i, :] - Y.iloc[i, :]
		X_school = pipe.transform(school)
		idx = np.where(X_school == 1)[0][0]
		indices = np.where(flattened[:, idx] == 1)[0].tolist()	
		X_data = X.iloc[:, indices][(X['academicyear'] <= year - 1) & (X['academicyear'] >= year - window)]
		# removes school data from the Y value
		data.append((hstack([X_school, dok_matrix(np.matrix(X_data.values).flatten())]), np.matrix(Y_data)))
	print(len(data))
		

	return np.array(data)
	

