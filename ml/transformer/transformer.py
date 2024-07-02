import torch
import torch.nn as nn
import math

torch.set_printoptions(precision=4, sci_mode=False)

import logging
logging.basicConfig(level=logging.INFO,format='%(message)s')

from hyperparameters import sentence

from config import (
    d_model, # size of embedding vector
    vocab_size,
    dropout,
    seq_length,
    d_feed_forward,
)
from vocab import vocabulary
from embedding import InputEmbeddings
from positional_encoding import PositionalEncoding
from layer_norm import LayerNormalization
from feed_forward_block import FeedForwardBlock





"""
 EMBEDDING LAYER:
"""
logging.info(f"Input sentence: {sentence}")
# logging.info(f'vocab of mate: {vocabulary.get(sentence[6].lower(), vocabulary["[UNK]"])}')
indices = [vocabulary.get(word.lower(), vocabulary["[UNK]"]) for word in sentence]
logging.debug(f'indices: {indices}')
input_indices = torch.tensor(indices, dtype=torch.long)
logging.debug(f'torch input indices: {input_indices}')
input_embedding = InputEmbeddings(d_model, vocab_size)
logging.debug(f'- Input after embedding: {input_embedding(input_indices)}')






"""
 POSITIONAL ENCODING:
"""
logging.debug(f"- seq_length: {seq_length}")
positional_encoding = PositionalEncoding(d_model, seq_length, dropout)
logging.debug(f"- Embedding including positional encoding: {positional_encoding(input_embedding(input_indices))}")






"""
 LAYER NORMALIZATION:

    HELPS WITH:
    1. Stabilize Training.
    2. Speeds up convergence -> Helps the model to learn more effectively by
                                maintaining more consistent gradients during back-propagation.
    3. Improves generalization -> By ensuring that the activations within the
                                    network remain in a similar range, the model can
                                    generalize better to new, unseen data.

    HOW IT WORKS:
    Takes:
    ((x - mean) / stdev) * alpha * beta

    ^introduces 2 learnable parameters that allow the normalized values to be scaled and shifted.

    So in this case with the input sentence being ['Your', 'cat', 'is', 'a', 'lovely', 'cat', 'mate'],
    each of these items would have the mean and stdev being calculated for their vectors.

    You calcualte mean and std individually, then in the batch (let's say 3) you multipy it with
    alpha and then add beta.

    Alpha / gamma is the same.
    Beta / bias is the same.
"""
layernorm = LayerNormalization()
logging.debug(f"- After layer normalization: {layernorm(positional_encoding(input_embedding(input_indices)))}")






"""
FEED FORWARD LAYER:
    Two matrices, multiplied by W1 and W2, with ReLu in between these linear layers and biases.
"""
feedforwardblock = FeedForwardBlock(d_model, d_feed_forward, dropout)
logging.debug(f"- After feed forward block: {feedforwardblock(layernorm(positional_encoding(input_embedding(input_indices))))}")







"""
MULTI-HEAD ATTENTION:
    The attention is used 3 times (as you can see in the arch graph pic):
    1. Query
    2. Key
    3. Value

    It's a duplication of the input 3 times.

    Where we transform the input (sequence, d_model) and output 3 vectors:
        Because we are in the encoder: these 3 matrices are exactly the same.
        It's different later in the decoder. 

        Then we multiply each one by a weight, so that:
        Q-vector * W^Q_matrix = Q' = Q1, Q2, Q3, Q4 ... etc
        K-vector * W^K_matrix = K'
        V-vector * W^v_matrix = V'

        Then we split these matrices into smaller matrices (same number as the number of heads we want):
            Q' = Q1, Q2, Q3, Q4 ... etc
            K' = K1, K2, K3, K4 ... etc
            V' = V1, V2, V3, V4 ... etc
        
        ^this split is done so that the Q1 matrix has the same dimension as the embedding dimension (d_model),
        this gives each head the access to the full sentence, but 

        Then we take these smaller matrices and use the Attention formula:
        23 min: https://www.youtube.com/watch?v=ISNdQcPhsts&t=340s&ab_channel=UmarJamil

        where we concatinate the heads:
            MultiHead (Q,K,V) = Concat(head1, ...head_n_heads,) * W°

"""

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # We need to divide the embedding vector by num_heads, so let's assert that:
        assert d_model & num_heads == 0, "d_model is not divisible by num_heads"
        self.d_k = d model // num_heads # dk = d_model / num_heads # TODO: what does this mean?
        self.w_q = nn.Linear(d_model, d_model) # this is the W^Q_matrix
        self.w_k = nn.Linear(d_model, d_model) # this is the W^K_matrix
        self.w_v = nn.Linear(d_model, d_model) # this is the W^V_matrix
        self.w_o = 
        self.dropout = dropout
        













                 
