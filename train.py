from typing import List
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from visualize import save_to_video
from siren import make_siren
from siren_orig import make_siren as make_siren_orig
from losses import equation_loss, condition_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_image(img_path: str):
    img = Image.open(img_path)
    img = img.convert("L")
    img = np.array(img)
    img = 255 - img
    img = (img - img.min()) / (img.max() - img.min())
    return torch.tensor(img, dtype=torch.float32, device=device)

def equation(model, batch_size, alpha, min_t, max_t):
    # Heat equation
    timestamps = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
    spatial_coords = torch.rand(batch_size, 2, device=device) * 2 - 1
    return equation_loss(model, spatial_coords, timestamps, alpha)

def initial(model, batch_size, min_t):
    # Initial condition for a circle
    independent_vars = torch.rand(batch_size, 3, device=device) * 2 - 1
    independent_vars[:, 2] = min_t
    temperatures = torch.zeros(batch_size, dtype=torch.float32, device=device)
    mask = torch.sqrt((independent_vars[:, 0] - 0.1)**2 + (independent_vars[:, 1] - 0.4)**2) < 0.5
    temperatures[mask] = 1
    return condition_loss(model, independent_vars, temperatures)

def initial_image(model, batch_size, min_t, img):
    # Initial condition (image)
    independent_vars = torch.rand(batch_size, 3, device=device) * 2 - 1
    independent_vars[:, 2] = min_t
    temperatures = torch.zeros(batch_size, dtype=torch.float32, device=device)
    temperatures = torch.nn.functional.grid_sample(img[None, None], independent_vars[:, :2].view(1, -1, 1, 2), align_corners=False).view(-1)
    return condition_loss(model, independent_vars, temperatures)

def boundary(model, batch_size, min_t, max_t, T):
    # Boundary condition
    independent_vars = torch.rand(batch_size, 3, device=device) * 2 - 1
    independent_vars[:, 2] = torch.rand(batch_size, device=device) * (max_t - min_t) + min_t
    independent_vars[:batch_size//4, 0] = -1
    independent_vars[batch_size//4:batch_size//2, 0] = 1
    independent_vars[batch_size//2:3*batch_size//4, 1] = -1
    independent_vars[3*batch_size//4:, 1] = 1
    temperatures = torch.ones(batch_size, dtype=torch.float32, device=device) * T
    return condition_loss(model, independent_vars, temperatures)


def train(args):
    # (H, W)
    img = read_image(args.image)
    
    # siren = make_siren(3, [128, 128, 128, 128, 1])
    siren = make_siren_orig(3, 128, 4, 1, first_omega_0=30)
    siren.to(device)

    optimizer = torch.optim.Adam(siren.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.02, threshold_mode='rel')

    # Heat equation
    # Initial conditions
    # Boundary conditions
    for epoch in range(100):

        lr = optimizer.param_groups[0]['lr']
        print(f"epoch = {epoch}")
        print(f"lr = {lr}")

        if lr < 1e-7:
            break
        total_loss = 0

        for i in range(1000):
            loss = 0
            
            loss += equation(siren, args.batch_size, args.alpha, args.min_t, args.max_t)
            # loss += initial(siren, batch_size, min_t)
            loss += initial_image(siren, args.batch_size, args.min_t, img)
            loss += boundary(siren, args.batch_size, args.min_t, args.max_t, args.boundary)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        total_loss /= 1000
        scheduler.step(total_loss)
        print(f"Loss: {total_loss}\n", flush=True)
        torch.save(siren.state_dict(), args.model_path)

    # siren.eval()
    # siren.to('cpu')
    # save_to_video(siren, min_t=min_t, max_t=max_t, timestep=0.1, spatial_resolution=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--model_path", type=str, default='simple_heat.pt')
    parser.add_argument("--min_t", type=float, default=0)
    parser.add_argument("--max_t", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024 * 4)
    parser.add_argument("--boundary", type=int, default=0)
    args = parser.parse_args()
        
    train(args)
