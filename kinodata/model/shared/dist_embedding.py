import torch
import numpy as np
from torch import Tensor
from torch.nn import Parameter, Module


def gaussian(x, mean, std):
    return torch.exp(-0.5 * torch.pow((x - mean) / std, 2)) / ((2 * torch.sqrt(torch.tensor(np.pi, dtype=x.dtype))) * std)


class GaussianDistEmbedding(Module):
    def __init__(self, size: int, max_dist: float) -> None:
        super().__init__()
        self.size = size
        self.d_cut = Parameter(
            torch.tensor([max_dist], dtype=torch.float32), requires_grad=False
        )

        means, stds = self._initial_params()
        self.means = Parameter(means)
        self.stds = Parameter(stds)

    def _initial_params(self):
        means = torch.linspace(0, self.d_cut.item(), self.size)
        stds = torch.ones(self.size)
        return means, stds

    def forward(self, d: Tensor) -> Tensor:
        #self.stds.data.clamp_(min=1e-8)
        return gaussian(d.view(-1, 1), self.means, self.stds)
