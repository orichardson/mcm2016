import pandas as pd
import random
import numpy as np

## keep the headers, can use .iloc (integer locate)
def load_raw_data():
	x_files = [ pd.read_csv("finances/delta_public_release_{0}.csv".format(p), encoding = "ISO-8859-1") for p in [ "87_99", "00_13" ]]
	
	y_files = []
	for i in range(1996, 2014):
		df = pd.read_csv("scorecard/MERGED{0}_PP.csv".format(i), encoding = "ISO-8859-1")
		df.insert(len(df.columns), "academicyear", i)
		y_files.append(df)

	X_data = x_files[0].append(x_files[1])
	X_data.sort_values('unitid', inplace=True, kind='heapsort')
	Y_data = y_files[0].append(y_files[1:])
	Y_data = Y_data.rename(columns = { '\xef\xbb\xbfUNITID' : 'unitid' })
	Y_data.sort_values('unitid', inplace=True, kind='heapsort')
	
	return X_data, Y_data

## figure out test data
def load(kind='scorecard', numeric_only=True):
	df = pd.read_csv("processed/{0}_train.csv".format(kind))
	
	if numeric_only:
		df = df.select_dtypes(include=['int64', 'float64'])

	return df
	

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


def same_schools(X, Y, contiguous=True):
	X_id2year = id_to_year(X)	
	Y_id2year = id_to_year(Y)

	if contiguous:
		X_schools = set(x for x in X_id2year if is_contiguous(X_id2year[x]))
		Y_schools = set(y for y in Y_id2year if is_contiguous(Y_id2year[y]))
	else:
		X_schools = set(x for x in X_id2year if 1 <= len(X_id2year[x]) and len(X_id2year[x]) <= 28)
		Y_schools = set(y for y in Y_id2year if 1 <= len(Y_id2year[y]) and len(Y_id2year[y]) <= 28)		

	return X_schools & Y_schools
	
def create_test(X, Y, percent, contiguous=True):
	same = list(same_schools(X, Y, contiguous))
	cutoff = int(percent * len(same))
	random.shuffle(same)
	restricted = set(same[:cutoff])	 
	unrestricted = set(same[cutoff:])	
	
	return X[X['unitid'].isin(restricted)], Y[Y['unitid'].isin(restricted)], X[X['unitid'].isin(unrestricted)], Y[Y['unitid'].isin(unrestricted)]

		
		
	
