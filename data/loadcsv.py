# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:50:42 2016

@author: Oliver
"""

import csv

def readYear(yr):    
    data_file = 'CollegeScorecard_Raw_Data/MERGED{0}_PP.csv'.format(yr)

    return csv.DictReader(open(data_file)) 