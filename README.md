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

ERA5 reanalysis, India bounding box, 2019 to 2022. 6-hourly timesteps, 7 variables: u10, v10, sp, t850, t500, z850, z500. Flattened to 15,609 nodes (129 lat x 121 lon).

Split chronologically: 2019/2020 for training, 2021 for validation, 2022 for test.

Graph edges built with KDTree k-nearest neighbours. Each node connects to its k closest neighbours by Euclidean distance in Cartesian space. Edge features are delta-lat, delta-lon and distance.

## Model

Standard encode-process-decode GNN. Encoder projects the 7 input features to a 64-dimensional hidden space. Processor runs N rounds of message passing where each round aggregates neighbour messages via scatter-add and updates node representations with a residual connection. Decoder projects back to 7 features.

The model takes one timestep as input and predicts the next.

## Results

**Ablation study (2-year run, no val split):**

Persistence baseline: **0.0934**

| Config | MSE |
|---|---|
| 6 layers k=8 | 0.0772 |
| 1 layer k=8 | 0.0750 |
| 3 layers k=8 | 0.0749 |
| 1 layer k=4 | 0.0757 |
| 1 layer k=16 | 0.0739 |
| 3 layers k=16 | 0.0742 |

Wider connectivity consistently outperforms deeper stacking. Adding layers past 1 gives marginal improvement, likely because repeated local aggregation over a small regional domain starts to wash out spatial signal.

**Final evaluation (4-year run, train/val/test split, 1 layer k=16):**

| Split | Persistence | GNN |
|---|---|---|
| Train | 0.1027 | 0.0789 |
| Val | 0.1037 | 0.0950 |
| Test | 0.1014 | 0.0945 |

Beats persistence on all splits. The smaller gap on val and test vs train is expected since the model was trained on 2019/2020 and evaluated on later years with different seasonal patterns.

**Per-variable MAE on test set (normalized by std):**

| Variable | MAE | Normalized |
|---|---|---|
| u10 | 1.258 m/s | 0.442 |
| v10 | 0.893 m/s | 0.361 |
| sp | 1047 Pa | 0.025 |
| t850 | 3.50 K | 0.031 |
| t500 | 3.15 K | 0.039 |
| z850 | 579.8 m²/s² | 0.079 |
| z500 | 974.3 m²/s² | 0.047 |

Thermodynamic variables (sp, t, z) are predicted well with normalized errors below 0.08. Wind components (u10, v10) are roughly 5x harder, which is consistent with wind being more turbulent and directionally variable at short timescales.

## Running

```bash
# Build graph from ERA5 data
python graph/build_graph.py

# Train
python training/train.py

# Persistence baseline
python training/baseline.py

# Inference and test evaluation
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
  num_epochs: 30
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