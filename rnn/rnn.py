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

