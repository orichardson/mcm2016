# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:50:42 2016

@author: Oliver
"""

import pandas as pd, yaml


f = open('scorecard/data_dictionary.yaml')
data = yaml.safe_load(f)
f.close();

dd = data['dictionary']

x = [(di, dd[di]['source'] if 'source' in dd[di] else dd[di]['calculate'], dd[di]['description']) for di in dd.keys()]


ys = None

for yr in range(1996, 2013):
	df = pd.read_csv('scorecard/MERGED{0}_PP.csv'.format(yr))
	ysi = set(df.columns)
    
	if not ys:
		ys = ysi
								
	ys &= ysi;
	del df
				
z = [[xi[0], xi[1], xi[2]] for xi in x if xi[1] in ys]
towrite = pd.DataFrame(z)
towrite.to_csv('thing.csv')

