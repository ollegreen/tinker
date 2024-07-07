import torch
import torch.nn as nn

from layer_norm import LayerNormalization

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, input, sublayer):
        return input + self.dropout(sublayer(self.norm(input)))
