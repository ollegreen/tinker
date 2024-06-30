import torch
import torch.nn as nn

torch.set_printoptions(precision=4, sci_mode=False)

import logging
logging.basicConfig(level=logging.INFO,format='%(message)s')


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        """
        Need epsilon to add to the layer norm formula.
        If sigma is close to 0 -> then it would return a huge value in the output,
        which would be problematic for CPUs to compute.
        """
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        logging.debug(f"alpha: {self.alpha}")
        self.bias = nn.Parameter(torch.zeros(1)) # Additive
        logging.debug(f"beta: {self.alpha}")

    def forward(self, input):
        mean = input.mean(dim = -1, keepdim=True) # mean of the last dimension, after the batch
        logging.debug(f"mean: {mean}")
        std = input.std(dim = -1, keepdim=True)
        logging.debug(f"st dev: {std}")
        return self.alpha * (input - mean) / (std + self.epsilon) + self.bias
