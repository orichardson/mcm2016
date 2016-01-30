import pandas as pd
import itertools
import numpy as np

## keep the headers, can use .iloc (integer locate)
def load_data():
	x_files = [ pd.read_csv("finances/delta_public_release_{0}.csv".format(p)) for p in [ "87_99", "00_13" ]]
	
	y_files = []
	for i in range(1996, 2014):
		df = pd.read_csv("scorecard/MERGED{0}_PP.csv".format(i))
		df.insert(len(df.columns), "academicyear", i)
		y_files.append(df)

	X_data = x_files[0].append(x_files[1])
	Y_data = y_files[0].append(y_files[1:])
	
	return X_data, Y_data

## figure out test data

# distribution of years

def get_year_distrib(data):
	id_year = data[["unitid", "academicyear"]]
	
	data_unitid = id_year.iloc[:, 0]
	id_to_year = { x : [] for x in data_unitid }
	
	for i, item in enumerate(data_unitid):
		id_to_year[item].append(id_year.iloc[i, 1])

	vals, counts = np.unique([len(x) for x in id_to_year.values()], return_counts=True)
	return dict(zip(list(vals), list(counts)))



