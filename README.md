# GNN Weather Forecasting — LAM and Global

Graph neural network for short-range weather forecasting, built from scratch using ERA5 reanalysis data. Supports two domains via a single config switch: a regional LAM setup over India, and a global setup using a hierarchical icosahedral mesh. Encode-process-decode architecture with autoregressive training.

This repo is part of a unified demo ([neural-lam-demo](https://github.com/Joltsy10/neural-lam-demo)) built as GSoC 2026 preparation for [Neural-LAM Project 4](https://github.com/mllam/neural-lam). The global graph geometry is handled by the companion repo [neural-lam-global-mesh](https://github.com/Joltsy10/neural-lam-global-mesh).

---

## Architecture

### LAM (Flat GNN)

Encode-process-decode GNN with flat KNN graph over the regional grid.

- **Encoder**: linear projection from node features to hidden dimension
- **Processor**: N stacked message passing layers, each aggregating neighbor messages via scatter-add, concatenating with node features, and updating via MLP
- **Decoder**: linear projection back to feature dimension
- **Residual prediction**: model predicts state delta (x_{t+1} - x_t). Improves thermodynamic variable accuracy by focusing the model on small incremental changes rather than reconstructing the full atmospheric state.

### Global (Hierarchical GNN)

Encode-process-decode GNN with hierarchical icosahedral mesh. Implements the HiLAM architecture from [Oskarsson et al. (2024)](https://arxiv.org/abs/2309.17370).

- **G2M encoder**: message passing from ERA5 grid nodes to finest mesh level
- **Processor**: hierarchical up-down sweep through mesh levels
  - Up sweep (finest to coarsest): same-level pass then up pass at each level
  - Down sweep (coarsest to finest): down pass then same-level pass at each level
  - Separate learned weights per level and per pass type
- **M2G decoder**: message passing from finest mesh level back to grid nodes
- **Residual prediction**: same delta formulation as LAM

| Hyperparameter | LAM | Global |
|---|---|---|
| hidden_dim | 128 | 128 |
| num_layers | 2 | — |
| mesh_level | — | 2 (162 mesh nodes) |
| k neighbors | 16 | — |
| g2m_angle_deg | — | 7.5 |
| edge features | distance + relative lat/lon | great-circle length + tangential displacement |

---

## Data

ERA5 reanalysis downloaded via CDS API, 6-hourly timesteps at 1° resolution.

**Variables (7):** u10, v10, sp, t850, t500, z850, z500

**LAM domain:** India bounding box 6–38N, 68–98E (15,609 grid nodes)

**Global domain:** Full sphere 181×360 (65,160 grid nodes)

**Split:**
- Train: 2019–2020 (2688 timesteps)
- Val: 2021 (1344 timesteps)
- Test: 2022 (1344 timesteps)

Pressure level downloads split by variable (temperature and geopotential separately) to stay within CDS cost limits.

---

## Training

- **Loss**: MSE over autoregressive rollout steps
- **Optimizer**: Adam, lr=0.001
- **Scheduler**: ReduceLROnPlateau, patience=3, factor=0.5
- **Gradient clipping**: max_norm=1.0
- **Epochs**: 30

---

## LAM Results

### Per-variable MAE at T+1 (6h)

| Variable | MAE | Normalized MAE |
|---|---|---|
| u10 | 0.9887 m/s | 0.347 |
| v10 | 0.9256 m/s | 0.374 |
| sp | 426.96 Pa | 0.010 |
| t850 | 1.3953 K | 0.012 |
| t500 | 0.8269 K | 0.010 |
| z850 | 193.28 m²/s² | 0.026 |
| z500 | 317.31 m²/s² | 0.015 |

Thermodynamic variables achieve normalized MAE below 0.03. Wind components are harder at ~0.35, consistent with the turbulent nature of near-surface winds.

### Rollout MAE vs Lead Time

![Rollout MAE](plots/rollout_mae.png)

GNN beats persistence at 6h and 12h. At 24h the model underperforms persistence due to the diurnal cycle, the atmosphere at T+24 closely resembles T+0, making persistence artificially strong.

### T850 Actual vs Predicted

![T850 Prediction](plots/t850_pred_vs_actual.png)

The model captures large-scale temperature structure: warm peninsula, cooler north, Himalayan cold signature. Predicted fields are slightly smoother than actual, typical of GNNs that underestimate fine-scale variability.

### T850 Prediction Error

![T850 Error](plots/t850_error.png)

Largest errors concentrated in the Himalayan and Tibetan Plateau region. The model has no elevation features and cannot represent the orographic barrier, systematically overpredicting temperature by 3–5K in this region.

---

## Global Results

Trained on full sphere ERA5 at 1° resolution (65,160 grid nodes), hierarchical icosahedral mesh at refinement level 2 (162 finest mesh nodes, 3 levels), 30 epochs, K=1 rollout.

### Training Curve

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.125068 | 0.118681 |
| 5 | 0.102108 | 0.102620 |
| 10 | 0.097316 | 0.099679 |
| 15 | 0.095172 | 0.099290 |
| 20 | 0.093765 | 0.099249 |
| 25 | 0.092749 | 0.097507 |
| 30 | 0.092223 | 0.095992 |

Best val loss: 0.095912 (epoch 28).

### Per-variable MAE at T+1 (6h) — Model vs Persistence

| Variable | Model MAE | Persistence MAE | Improvement |
|---|---|---|---|
| u10 | 1.2345 m/s | 1.4905 m/s | 17% |
| v10 | 1.2529 m/s | 1.6235 m/s | 23% |
| sp | 176.05 Pa | 190.06 Pa | 7% |
| t850 | 0.9919 K | 1.1271 K | 12% |
| t500 | 0.8773 K | 0.9422 K | 7% |
| z850 | 130.11 m²/s² | 149.11 m²/s² | 13% |
| z500 | 158.58 m²/s² | 180.90 m²/s² | 12% |

Model beats persistence on every variable. Wind components show the largest improvement (17–23%); persistence is particularly poor for wind since it cannot capture advection. Thermodynamic variables show smaller gains as they change more slowly.

Rollout normalized MAE at T+1: **0.1257** vs persistence **0.1448**.

### Global Error Visualization

Interactive 3D globe with elevation-based error heatmap with the nodes displaced outward proportional to prediction error, colored by magnitude. Higher spikes indicate regions of larger forecast error.

![Global Error Globe](plots/globe_error.png)

Run `visualize_globe.ipynb` for the interactive version.

### Notes

Refinement level 2 is a coarse mesh designed for rapid iteration; the processor operates on 12, 42, and 162 mesh nodes across the three levels. Accuracy is limited by mesh resolution rather than model capacity.

### Mesh Level 3 (Early Stopping)

Training was also run at refinement level 3 (642 finest mesh nodes, 4 levels, 177,160 G2M edges) with hidden_dim reduced to 64 to fit within 8GB VRAM.

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.112517 | 0.102669 |
| 2 | 0.098427 | 0.095169 |

Training was stopped after 2 epochs due to compute time (~7 hours per epoch on a laptop 4070). The val loss at epoch 2 (0.095169) already matches mesh level 2's best result (0.095912) despite being only 2 epochs in, confirming that finer mesh resolution improves convergence rate. Full convergence at mesh level 3 requires cluster hardware. Training can be resumed from the saved checkpoint with `resume=True` in train.py.

---

## Switching Domains

Set `domain: lam` or `domain: global` in `config.yaml`. Everything downstream — data loading, graph construction, model selection — branches automatically.

```yaml
domain: global  # or lam

graph:
  k: 16                  # lam only
  mesh_level: 2          # global only
  g2m_angle_deg: 7.5     # global only

model:
  hidden_dim: 128
  num_layers: 2          # lam only
  node_dim: 7
```

---

## How to Run

### Install dependencies
```bash
pip install torch numpy scipy pyyaml cdsapi xarray matplotlib plotly
```

### Download ERA5 data
```bash
python data/download_era5.py
```
Requires a CDS API key at `~/.cdsapirc`. Downloads surface and pressure level files separately to avoid CDS cost limits.

### Build graph
```bash
python graph/build_graph.py
```
LAM builds a flat KNN graph. Global calls the icosahedral bridge layer from [neural-lam-global-mesh](https://github.com/Joltsy10/neural-lam-global-mesh) and saves all `.pt` files in the format `utils.load_graph` expects.

### Train
```bash
python training/train.py
```

### Inference
```bash
python training/inference.py
```

### Persistence baseline
```bash
python training/baseline.py
```

### Visualize (2D)
```bash
jupyter notebook visualize.ipynb
```

### Visualize (3D Globe)
```bash
jupyter notebook visualize_globe.ipynb
```

---

## Project Structure

```
gnn-weather-from-scratch/
├── data/
│   ├── download_era5.py       — CDS API download, split by variable for global
│   ├── lam/                   — LAM ERA5 files + built graph .pt files
│   └── global/                — Global ERA5 files + built graph .pt files
├── graph/
│   └── build_graph.py         — Config-aware graph construction for LAM and global
├── model/
│   ├── gnn.py                 — Flat GNN for LAM
│   ├── hi_gnn.py              — Hierarchical GNN for global (HiLAM-style)
│   └── message_passing.py     — Shared bipartite-aware MessagePassingLayer
├── training/
│   ├── train.py               — Domain-aware training loop
│   ├── inference.py           — Rollout and evaluation (LAM and global)
│   └── baseline.py            — Persistence baseline (LAM and global)
├── plots/
├── visualize.ipynb            — 2D cartopy plots (LAM)
├── visualize_globe.ipynb      — Interactive 3D globe (Global)
└── config.yaml
```

---

## Relationship to Neural-LAM

The global hierarchical model here is a simplified standalone implementation of the architecture described in [Oskarsson et al. (2024)](https://arxiv.org/abs/2309.17370). The graph files produced by `build_graph.py` are in the exact format `utils.load_graph` expects in neural-lam, so the bridge to the full neural-lam codebase is direct.

The `MessagePassingLayer` in `message_passing.py` supports bipartite passes with separate source and destination node sets, which is required for G2M, M2G, and inter-level up/down passes in the hierarchical case.
