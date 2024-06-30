from vocab import vocabulary
from hyperparameters import sentence

d_model =  8 # dims of embeddings
d_feed_forward = 2
vocab_size = len(vocabulary)
seq_length = len(sentence)

dropout = 0.1