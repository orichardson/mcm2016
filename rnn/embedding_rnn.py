import pandas
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow


HIDDEN_SIZE = 20

def rnn_model(X, y):
	cell = rnn_cell.GRUCell(HIDDEN_SIZE)
	_, encoding = rnn.rnn(cell, X, dtype=tf.float64)
	return skflow.models.logistic_regression(encoding[-1], y)
