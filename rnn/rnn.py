import tensorflow.models.rnn.seq2seq as seq2seq
from tensorflow.models.rnn import rnn_cell

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer, StandardScaler

import skflow

#encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
#decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
window = 5

pipeline = Pipeline(zip([ "imputate", "vart", "scale", "svm" ], [ Imputer(), VarianceThreshold(), StandardScaler(), clf ]))
   

model = seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, rnn_cell.GRUCell(num_units), window, 441)
