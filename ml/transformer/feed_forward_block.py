import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO,format='%(message)s')


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_feed_forward, bias=True) # Weight 1 + Bias 1
        logging.debug(f"linear layer 1: {self.linear_1:}")

        self.dropout = nn.Dropout(dropout)
        logging.debug(f"The dropout: {self.dropout}")
        """
        Dropout is a regularization technique used in neural networks to prevent overfitting during training.
        In PyTorch, the nn.Dropout layer randomly sets a fraction of input units to zero at
        each forward pass during training time.
        """

        self.linear_2 = nn.Linear(d_feed_forward, d_model, bias=True) # Weight 2 + Bias 2
        logging.debug(f"linear layer 2: {self.linear_1:}")

    def forward(self, input):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_feed_forward) -> (batch, seq_len, d_model)
        logging.debug(f"layer 1 without ReLu: {self.linear_1(input)}")
        linear_layer_1_with_relu = torch.relu(self.linear_1(input))
        logging.debug(f"layer 1 WITH ReLu: {linear_layer_1_with_relu}")
        logging.debug(f"layer 1 WITH ReLu and dropout applied: {self.dropout(linear_layer_1_with_relu)}")
        logging.debug(f"layer 2 after passing through layer 1: {self.linear_2(self.dropout(linear_layer_1_with_relu))}")
        return self.linear_2(self.dropout(linear_layer_1_with_relu))
