from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader


class Siren(nn.Module):
    def __init__(self, inpt_dim: int, nodes: List[int]):
        super().__init__()
        self.linear_layers = nn.ModuleList(
            nn.Linear(fan_in, fan_out) 
            for fan_in, fan_out in zip([inpt_dim] + nodes[:-1], nodes)
            )
    def forward(self, x):
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = torch.sin(x)
        x = self.linear_layers[-1](x)
        return x

def init_siren(siren):
    for idx, layer in enumerate(siren.linear_layers[:-1]):
        n = layer.in_features
        b = (6 / n)**0.5
        a = -b
        if idx == 0:
            # a *= 30
            # b *= 30
            a = - 30 / n
            b = 30 / n
        torch.nn.init.uniform_(layer.weight, a, b)
        # torch.nn.init.uniform_(layer.bias, -0.001, 0.001)

    # Last layer: special case
    layer = siren.linear_layers[-1]
    n = layer.in_features
    b = (6 / n)**0.5 * 0.1
    a = -b
    torch.nn.init.uniform_(layer.weight, a, b)
    torch.nn.init.uniform_(layer.bias, 0.79, 0.81)

def make_siren(inpt_dim, nodes):
    siren = Siren(inpt_dim, nodes)
    
    # init_siren(siren)

    return siren
