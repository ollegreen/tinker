import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # input dimensionality
        self.vocab_size = vocab_size # the total number of different words that are in the model's vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)
        # ^size of the dict of embeddings + embedding vector

    def forward(self, input):
        return self.embedding(input) * math.sqrt(self.d_model)
