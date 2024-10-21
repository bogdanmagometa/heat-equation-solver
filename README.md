# Heat equation solver

Solve heat equation by fitting neural nets (<a href="https://arxiv.org/abs/2006.09661">SIRENs</a>).

## Examples

### Image is initial condition, 0 as boundary condition

Train a neural net and save weights to `new_year_heat.pt`:
```bash
python3 train.py --image img/new_year.png --model_path new_year_heat.pt
```

Save visualization to gif:
```bash
python3 visualize.py --model_path new_year_heat.pt --min_t 0 --max_t 10 --spatial_resolution 300 --out_path animation.gif
```

![heat](./img/animation.gif)

### Circle as initial condition, positive constant as boundary condition

![heat](./img/animation_circle.gif)

<!-- ### Highly diffusive rod in low-diffusive environment

```bash
python3 train.py --image img/initial_heat.png --alpha diffusivity.png --model_path rod_heat.pt
python3 visualize.py --model_path new_year_heat.pt --min_t 0 --max_t 10 --spatial_resolution 300 --out_path animation_rod.gif
```
 -->

## Prerequisites

Python env:
- python 3.10
- torch 2.1
- numpy 1.26
- matplotlib 3.8

Other combinations might also work
