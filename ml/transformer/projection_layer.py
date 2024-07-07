import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO,format="%(message)s")


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input):
        # (Batch, seq_len, d_model) -> (batch, seq_len, vocab_size)

        return torch.log_softmax(self.proj(input), dim = -1)
