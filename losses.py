from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

def equation_loss(model: nn.Module, spatial_coords: torch.Tensor, timestamps, alpha):
    """Differentiably calculate loss for heat equation:
    loss_i = (dT_i/dt_i - alpha (ddT_i/dx_idx_i + ddT_i/dy_idy_i))**2
    where i is from 0 to N-1
    loss = sum_over_i{loss_i}

    Args:
        model (nn.Module): model that accepts (N, k+1), where k is the number of spatial dimensions
        spatial_coords (torch.Tensor): (N, k) tensor containing spatial coordinates
        timesteps (N, k): (N,) tensor containing timesteps
        alpha: thermal diffusivity

    Returns:
        number: loss
    """
    device = timestamps.device
    
    spatial_coords.requires_grad_(True)
    timestamps.requires_grad_(True)

    temperatures = model(torch.concat([spatial_coords, timestamps.view(-1, 1)], dim=1))
    # Display for debugging
    # print("Mean:", temperatures.mean())
    # print("Std:", temperatures.std())

    # spatial_coords_grad: Nx2
    # ( dT1/dx1  dT1/dy1 )
    # (   ...      ...   )
    # ( dTn/dxn  dTn/dyn )
    spatial_coords_grad, timesteps_grad = torch.autograd.grad(temperatures, [spatial_coords, timestamps], torch.ones_like(temperatures, device=device), create_graph=True)

    grad_outputs = torch.stack([torch.ones(len(temperatures), device=device), torch.zeros(len(temperatures), device=device)], dim=1)
    grad_outputs = torch.stack([grad_outputs, torch.flip(grad_outputs, dims=(1,))])

    # spatial_coords_grad_grad: 2xNx2
    # (
    #   ( ddT1/dx1dx1  ddT1/dy1dx1 )
    #   (     ...         ...      )
    #   ( ddTn/dxndxn  ddTn/dyndxn )
    # )
    # (
    #   ( ddT1/dx1dy1  ddT1/dy1dy1 )
    #   (     ...         ...      )
    #   ( ddTn/dxndyn  ddTn/dyndyn )
    # )
    spatial_coords_grad_grad, = torch.autograd.grad(spatial_coords_grad, spatial_coords, grad_outputs=grad_outputs, is_grads_batched=True, create_graph=True)

    # laplacians: N
    # (
    #   ddT1/dx1dx1 + ddT1/dy1dy1
    #              ...      
    #   ddTn/dxndxn + ddTn/dyndyn
    # )
    laplacians = spatial_coords_grad_grad[0, :, 0] + spatial_coords_grad_grad[1, :, 1]
    
    # heat equation
    # dT/dt = alpha (ddT/dxdx + ddT/dydy)
    loss = torch.mean(torch.square(timesteps_grad - alpha * laplacians))

    return loss

def condition_loss(model: nn.Module, independent_vars: torch.Tensor, temperatures: torch.Tensor):
    loss = torch.mean(torch.square(model(independent_vars)[..., 0] - temperatures))
    return loss
