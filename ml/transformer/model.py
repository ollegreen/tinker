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
    num_heads,
)
from vocab import vocabulary
from embedding import InputEmbeddings
from positional_encoding import PositionalEncoding
from layer_norm import LayerNormalization
from feed_forward_block import FeedForwardBlock
from attention import MultiHeadAttentionBlock
from residual_connection import ResidualConnection
from encoder import EncoderBlock, Encoder
from decoder import DecoderBlock, Decoder


"""
This file is used to practice rewriting the modules and functions available in the other files.
Essentially a place for active recall (and spaced repetition).
"""



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





########################################################################################
"""
 POSITIONAL ENCODING:
"""
logging.debug(f"- seq_length: {seq_length}")
positional_encoding = PositionalEncoding(d_model, seq_length, dropout)
logging.debug(f"- Embedding including positional encoding: {positional_encoding(input_embedding(input_indices))}")





########################################################################################
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





########################################################################################
"""
FEED FORWARD LAYER:
    Two matrices, multiplied by W1 and W2, with ReLu in between these linear layers and biases.
"""
feedforwardblock = FeedForwardBlock(d_model, d_feed_forward, dropout)
logging.debug(f"- After feed forward block: {feedforwardblock(layernorm(positional_encoding(input_embedding(input_indices))))}")






########################################################################################
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

attentionblock = MultiHeadAttentionBlock(d_model, num_heads, dropout)
# logging.debug(f"- After attention: {attentionblock(feedforwardblock(layernorm(positional_encoding(input_embedding(input_indices)))))}")


########################################################################################
"""
RESIDUAL CONNECTION LAYER

    You see between the attention and normalisation layer, that sometimes it switches over to skip it;
    this is the layer that handles that.
    It's basically a skip-connection.
"""

residualconnection = ResidualConnection(dropout)


########################################################################################
"""
Now here's the kicker:
    In the paper/architecture, we can see that we have the following components encapsulated in a block:
        1. Layer Norm
        2. Multihead attention
        3. FFN (MLP)
        4. Layer Norm

    This block is replicated n times. The original transformer had 6.

    **So what happens is that this process is that we take the output from the last layer in this encoder,
    and feed it back into the same layer, 6 times over, before we give it to the decoder layer.
"""



# encoderblock = EncoderBlock()


########################################################################################
"""
Then we simply aggregate all of these blocks into the encoder, which encapsulates all of them.
"""
# encoder = Encoder()



########################################################################################
"""
DECODER BLOCK:
    One thing worthy to note: we have the first masked self attention which is the same as the
    implementation we've done so far, just including the mask.
    * However:
        The 2nd one is a cross-attention block, which basically takes in the Query and the Key
        from the encoder block to take in and match then to calculate the relationship between them.

"""

class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float,
                 ) -> None:
        super().__init__()
        # this should have:
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, input, encoder_output, src_mask, target_mask):
        input = self.residual_connections[0](input, lambda input: self.self_attention_block(input, input, input, target_mask)) # <- #TODO: is this really correct?
        input = self.residual_connections[1](input, lambda input: self.cross_attention_block(input, encoder_output, encoder_output, src_mask)) #TODO: is this really correct?
        input = self.residual_connections[2](input, self.feed_forward_block)
        return input


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()


    def forward(self, input, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            input = layer(input, encoder_output, src_mask, target_mask)

        return self.layer_norm(input)




# decoderblock = DecoderBlock()

########################################################################################
"""
Then we simply aggregate all of these blocks into the decoder, which encapsulates all of them.
"""
# decoder = Decoder()


"""
LINEAR LAYER + SOFTMAX:
    The output after the decoder is a format of sequence by d_model.
    OBJECTIVE: we want to map these words back into the vocabulary
        * SOLUTION: Linear Layer; converts the embedding into a postion of the vocabulary.
        Also called: Projection Layer in the youtube walkthrough as it is projecting the enbedding to the vocab.
"""





"""
Transformer:
"""

