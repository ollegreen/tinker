import torch
import torch.nn as nn
import math

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # We need to divide the embedding vector by num_heads, so let's assert that:
        assert d_model & num_heads == 0, "d_model is not divisible by num_heads"

        self.d_k = d_model // num_heads # dk = d_model / num_heads
        self.w_q = nn.Linear(d_model, d_model) # this is the W^Q_matrix
        self.w_k = nn.Linear(d_model, d_model) # this is the W^K_matrix
        self.w_v = nn.Linear(d_model, d_model) # this is the W^V_matrix
        self.w_o = nn.Linear(d_model, d_model) # this is the W^V_matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        logging.info(f"d_k = query.shape[-1]: {query.shape[-1]}")
        d_k = query.shape[-1]

        # -2, -1 means transpose the last two dimensions: (Batch, seq_length, d_model) ->  (Batch, d_model, seq_length)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Before applying softmax:
        # we will apply the mask to hide some interaction between words:
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # <- means for when mask=0, replace it with this small value

        # apply softmax:
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, num_heads, seq_length, seq_length)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (attention_scores @ value) = output from the attention layer
        logging.info(f"(attention_scores @ value): {(attention_scores @ value)}")
        return (attention_scores @ value), attention_scores # @38:58 explanation. the , attention_scores is for visualisation


    def forward(self, q, k, v, mask): # mask, if we don't want
        query = self.w_q(q) # <- called Q' in the video
        key =   self.w_k(k) # From (Batch, seq_length, d_model) -> To: (Batch, seq_length, d_model)
        value = self.w_v(v) # From (Batch, seq_length, d_model) -> To: (Batch, seq_length, d_model)
        logging.info(f"query = self.w_q(q) before function: {query}")

        # goes from: (Batch, seq_length, d_model) -> (batch, seq_length, num_heads, d_k) -> transpose: (batch, num_heads, seq_length, d_k)
        # with the transpose: num_heads are now looking at seq_length, d_k at each head, which is what we want.
        # which means that each head will now see each full sentence, but only a smaller part of the embedding.
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        # .view() -> Returns a new tensor with the same data as the self tensor but of a different shape.
        logging.info(f"query AFTER the .view() + reshape of tensor function: {query}")

        input, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        input, input.transpose(1, 2).contiguous().view(input.shape[0], -1, self.num_heads, * self.d_k)

        return self.w_o(input) # last multiplication step

