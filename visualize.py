from typing import List
import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from siren import Siren
from siren_orig import make_siren as make_siren_orig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_image(model, grid_x, grid_y, timestamp):
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x) * timestamp], dim=-1)
    temperature = torch.squeeze(model(grid.view(-1, 3)))
    return temperature.view(grid_x.shape).cpu().detach().numpy()

def draw_image(ax, model, grid_x, grid_y, timestamp, min_x=-1, max_x=1, min_y=-1, max_y=1, min_T=0, max_T=1):
    image = compute_image(model, grid_x, grid_y, timestamp)
    ax.imshow(image, vmin=min_T, vmax=max_T, cmap=plt.cm.Reds, interpolation='none', extent=[min_x, max_x, max_y, min_y])
    ax.xaxis.set_ticks_position("top")

def save_to_video(outfile, model, min_t, max_t, timestep, spatial_resolution, min_x=-1, max_x=1, min_y=-1, max_y=1, min_T=0, max_T=1):
    model.to(device)
    xes = torch.linspace(min_x, max_x, int(spatial_resolution * (max_x - min_x)), device=device)
    yes = torch.linspace(min_y, max_y, int(spatial_resolution * (max_y - min_y)), device=device)
    grid_x, grid_y = torch.meshgrid([xes, yes], indexing='xy')
    plt.show()

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    dpi = 2 * spatial_resolution / (fig.get_window_extent().width / fig.dpi)
    # print(fig.get_window_extent().width / fig.dpi)
    fig.set_dpi(dpi)

    # Update function for animation
    @torch.no_grad()
    def update(frame):
        ax.clear()
        t = frame * timestep + min_t
        draw_image(ax, model, grid_x, grid_y, t, min_x, max_x, min_y, max_y, min_T, max_T)
        ax.set_title(f'Time: {int(t)} seconds')

    # Create animation
    animation = FuncAnimation(fig, update, frames=int((max_t - min_t) // timestep), interval=timestep)
    writergif = PillowWriter(fps=1 / timestep)

    # Save animation to a video file
    animation.save(outfile, writer=writergif, dpi=dpi)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='animation.gif')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--min_t', type=float, required=True)
    parser.add_argument('--max_t', type=float, required=True)
    parser.add_argument('--spatial_resolution', type=int, default=200)
    args = parser.parse_args()

    # Load model
    # siren = Siren(1, [1])
    siren = make_siren_orig(3, 128, 4, 1)
    siren.load_state_dict(torch.load(args.model_path))
    siren.eval()

    siren.eval()
    siren.to('cpu')
    save_to_video(args.out_path, siren, min_t=args.min_t, max_t=args.max_t, timestep=0.1, spatial_resolution=args.spatial_resolution)
