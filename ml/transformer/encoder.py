import torch
import torch.nn as nn

from attention import MultiHeadAttentionBlock
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection
from layer_norm import LayerNormalization


import logging
logging.basicConfig(level=logging.INFO,format="%(message)s")


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float, ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # use it twice


    def forward(self, input, src_mask):
        """
        In arch: we send the same input coming from the positional encoding to both the attention as well as
        the residual connection:
        """
        # the first part going to the attention block + res connection:
        input = self.residual_connections[0](input, lambda input: self.self_attention_block(input, input, input, src_mask))
        # the 2nd part of the feed forward + res connection:
        input = self.residual_connections[1](input, self.feed_forward_block)

        return input


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, input, mask):
        for layer in self.layers:
             input = layer(input, mask)

        return self.layer_norm(input)
