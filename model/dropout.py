import torch
import torch.fx
from torch import nn, Tensor

from torchvision.utils import _log_api_usage_once

"""
From stochastic depth code: Run the script here to find the file and the code

from torchvision.ops import stochastic_dropout
print (stochastic_dropout.__file__)
"""

def stochastic_dropout(input: Tensor, p: float, training: bool = True) -> Tensor:

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(stochastic_dropout)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    noise = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    
    # Is multiplication by a factor > 1 necessary for non-dropout neurons?
    if survival_rate > 0.0:
        noise.div_(survival_rate)

    return input * noise

torch.fx.wrap("stochastic_dropout")

class StochasticDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_dropout(input, self.p, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p})"
        return s

