# GNN Weather Forecasting from Scratch

A minimal graph neural network for short-range weather forecasting over India, built without neural-lam or any ML weather framework. The goal was to understand the core message passing mechanics before working on the full GSoC integration.

## Structure

```
data/           ERA5 surface and pressure level data, processed graph tensors
graph/          Graph construction from ERA5 grid
model/          Message passing layer and full GNN
training/       Training loop, persistence baseline and inference
config.yaml     Hyperparameters
```

## Data

ERA5 reanalysis, India bounding box, 2021 to 2022. 6-hourly timesteps, 7 variables: u10, v10, sp, t850, t500, z850, z500. Flattened to 15,609 nodes (129 lat x 121 lon).

Graph edges built with KDTree k-nearest neighbours. Each node connects to its k closest neighbours by Euclidean distance in Cartesian space. Edge features are delta-lat, delta-lon and distance.

## Model

Standard encode-process-decode GNN. Encoder projects the 7 input features to a 64-dimensional hidden space. Processor runs N rounds of message passing where each round aggregates neighbour messages via scatter-add and updates node representations with a residual connection. Decoder projects back to 7 features.

The model takes one timestep as input and predicts the next.

## Results

Persistence baseline MSE (predicting t as t+1): **0.0934**

| Config | MSE |
|---|---|
| 6 layers k=8 | 0.0772 |
| 1 layer k=8 | 0.0750 |
| 3 layers k=8 | 0.0749 |
| 1 layer k=4 | 0.0757 |
| 1 layer k=16 | 0.0739 |
| 3 layers k=16 | 0.0742 |

Best result is 1 layer with k=16, beating persistence by about 21%. Wider connectivity consistently outperforms deeper stacking for this task. Adding more layers past 1 gives marginal or no improvement, likely because repeated local aggregation over a small regional domain starts to wash out spatial signal rather than sharpen it.

**Per-variable MAE (normalized by std, best config):**

| Variable | MAE | Normalized |
|---|---|---|
| u10 | 1.317 m/s | 0.434 |
| v10 | 1.059 m/s | 0.413 |
| sp | 907 Pa | 0.022 |
| t850 | 2.35 K | 0.037 |
| t500 | 2.05 K | 0.036 |
| z850 | 150.7 m²/s² | 0.021 |
| z500 | 375.1 m²/s² | 0.018 |

Thermodynamic variables (sp, t, z) are predicted well with normalized errors below 0.04. Wind components (u10, v10) are roughly 10x harder, with normalized errors around 0.4. This is consistent with the fact that wind is turbulent and directionally variable at short timescales, while pressure and temperature fields are smoother and slower to change.

## Running

```bash
# Build graph from ERA5 data
python graph/build_graph.py

# Train
python training/train.py

# Persistence baseline
python training/baseline.py

# Inference on a specific timestep
python training/inference.py
```

## Configuration

All hyperparameters live in config.yaml:

```yaml
graph:
  k: 16

model:
  hidden_dim: 64
  num_layers: 1

training:
  num_epochs: 10
  lr: 0.001
```

## Dependencies

```
torch
numpy
scipy
netCDF4
pyyaml
```