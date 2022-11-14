# %%
import torch.nn as nn


class Residual(nn.Module):
    """A Wrapper for another PyTorch Module that implements a residual
    connection around that module."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


# %%
