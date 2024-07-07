import torch
import torch.nn as nn

from attention import MultiHeadAttentionBlock
from feed_forward_block import FeedForwardBlock
from encoder import EncoderBlock, Encoder
from decoder import DecoderBlock, Decoder
from embedding import InputEmbeddings
from positional_encoding import PositionalEncoding
from projection_layer import ProjectionLayer

import logging
logging.basicConfig(level=logging.INFO,format="%(message)s")




class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 src_embedding: InputEmbeddings,
                 target_embedding: InputEmbeddings,
                 src_position: PositionalEncoding,
                 target_position: PositionalEncoding,
                 projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_position = src_position
        self.target_position = target_position
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, src_mask, target, target_mask) # <- = forward method of decoder (order of inputs)

    def project(self, input):
        return self.projection_layer(input)

"""
BUILD TRANSFORMER - Given a set of parameters and hyperparameters:

"""

def build_transformer(src_vocab_size: int,
                      target_vocab_size: int,
                      src_seq_len: int,
                      target_seq_len: int,
                      d_model: int,
                      number_of_blocks: int = 6,
                      num_heads = 2,
                      dropout: float = 0.1,
                      d_ff: int = 2048, # hidden layer of feed-forward
                      ):

    # create embedding layer: 
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # Note: target_pos isn't really needed as they are the same as the src positional encoding, but just for vis/debugging.
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(number_of_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    decoder_blocks = []
    for _ in range(number_of_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer -> projecting to the target vocabulary:
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_pos, target_pos, projection_layer)

    # Intitialise the parameters:
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
