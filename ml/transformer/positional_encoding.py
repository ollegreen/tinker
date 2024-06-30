import torch
import torch.nn as nn
import math

import logging
logging.basicConfig(level=logging.INFO,format='%(message)s')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.dim_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        """
        1. Need the size of the vector (d_model), since we can return a positional encoding vector of the correct size.
        2. Need the lenght of the sentence (sequence length) to add the positions.
        """

        # Create a matrix of shape: (seq_length, sequence_length):
        positional_encoding = torch.zeros(seq_length, d_model)
        logging.debug(f"the positional encoding: {positional_encoding} at size: {len(positional_encoding)}")

        # Create a vector of shape (seq_length, 1) for the position:
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        logging.debug(f"the position vector: {position}")
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        logging.debug(f"div term: {div_term}")
        # Apply the sin to even positions:
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # : means every word will have sin, but only even dimensions
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # : means every word will have sin, but only odd dimensions

        positional_encoding = positional_encoding.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer("positional_encoding", positional_encoding)


    def forward(self, input):
        # add this positional encoding to every word:
        logging.debug(f"input: {input}")
        logging.debug(f"input.shape: {input.shape}")
        logging.debug(f"input shape[1]: {input.shape[1]}")
        logging.debug(f"what we're adding as a positional encoding: {(self.positional_encoding[:, :input.shape[1], :]).requires_grad_(False)}")
        input = input + (self.positional_encoding[:, :input.shape[1], :]).requires_grad_(False)
        logging.debug(f"what the embedding looks like after adding positional encoding: {input}")
        logging.debug(f"what the final output is that we're returning: the dropout; {self.dropout(input)}")
        return self.dropout(input)
