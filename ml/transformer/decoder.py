import torch
import torch.nn as nn

from attention import MultiHeadAttentionBlock
from feed_forward_block import FeedForwardBlock
from residual_connection import ResidualConnection
from layer_norm import LayerNormalization


import logging
logging.basicConfig(level=logging.INFO,format="%(message)s")

class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # 3 residual connections, because that's the vanilla transformer setup:
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, input, encoder_output, src_mask, target_mask):
        # ^ src_mask = source language (encoder): GeoHashes/English, target language (decoder): Transit Time/Swedish

        input = self.residual_connections[0](input, lambda input: self.self_attention_block(input, input, input, target_mask))
        
        # inputs: query coming from the decoder, key and value as coming from the encoder:
        input = self.residual_connections[1](input, lambda input: self.cross_attention_block(input, encoder_output, encoder_output, src_mask))
        input = self.residual_connections[2](input, self.feed_forward_block)
        return input

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.layer_norm = LayerNormalization()

    def forward(self, input, encoder_output, src_mask, target_mask):
        # what's happening here is that we're calling the forward method from the DecoderBlock, which means we need: input, encoder_output, src_mask, target_mask
        for layer in self.layers:
            input = layer(input, encoder_output, src_mask, target_mask)
        
        return self.layer_norm(input)

